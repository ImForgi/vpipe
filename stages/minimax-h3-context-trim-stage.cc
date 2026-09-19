#include "stages/minimax-h3-context-trim-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/minimax-h3-context.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

namespace vpipe {

namespace {

constexpr ConfigKey kAttrs[] = {
  {.key = "trim_frames", .type = ConfigType::Int,
   .doc = "leading frames to drop -- the context length the clip was "
          "continued with. Ignored when iport2 (info) is wired, which "
          "carries the importer's own value",
   .def_int = 22},
  {.key = "conform_audio", .type = ConfigType::Bool,
   .doc = "cut or pad the PCM to exactly (frames - trim) / fps seconds, "
          "absorbing H3's +-8 ms audio rounding so it cannot accumulate over "
          "a chain",
   .def_bool = true},
  {.key = "fps", .type = ConfigType::Real,
   .doc = "frame rate used when a beat's sideband carries none",
   .def_real = 24.0},
  {.key = "luma_match", .type = ConfigType::Bool,
   .doc = "put the frames just after the cut back on the level this clip "
          "settles at -- they come off the pinned context carrying ITS "
          "exposure, which on a join reads as a flash",
   .def_bool = true},
  {.key = "luma_match_frames", .type = ConfigType::Int,
   .doc = "how many frames after the cut the correction reaches, released "
          "linearly across them (0 = off). A frame already on the settled "
          "level is left alone whatever this is",
   .def_int = 6},
  {.key = "declick_ms", .type = ConfigType::Real,
   .doc = "raised-cosine fade at the PCM cut, in milliseconds, so the "
          "soundtrack does not start mid-waveform (0 = off)",
   .def_real = 12.0},
};
const PortSpec kIports[] = {
  {.name = "image",
   .doc = "per-frame planar u8 RGB [3, H, W] from vae-decode oport0, "
          "{frame, frames, fps} on the sideband",
   .type = &typeid(TensorBeatPayload), .tags = "rgb-frames", .clock_group = 0},
  {.name = "audio",
   .doc = "OPTIONAL decoded PCM f32 [channels, samples] with sample_rate, "
          "from audio-vae-decode",
   .type = &typeid(TensorBeatPayload), .tags = "audio-pcm", .clock_group = 1},
  {.name = "info",
   .doc = "OPTIONAL minimax-h3-context-import oport2: its trim_frames is used "
          "instead of the config value",
   .type = &typeid(FlexDataPayload), .tags = "minimax-h3-context-info",
   .clock_group = 2},
};
const PortSpec kOports[] = {
  {.name = "image",
   .doc = "the frames after the overlap, {frame, frames} renumbered",
   .type = &typeid(TensorBeatPayload), .tags = "rgb-frames", .clock_group = 0},
  {.name = "audio",
   .doc = "the PCM after the overlap (conformed to the picture when "
          "conform_audio)",
   .type = &typeid(TensorBeatPayload), .tags = "audio-pcm", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "minimax-h3-context-trim",
  .doc       = "Drops the regenerated context overlap from the head of a "
               "decoded MiniMax-H3 clip -- frames and PCM by the same "
               "duration, the PCM conformed to the picture -- so a continued "
               "clip appends cleanly to the one before it.",
  .display_name = "MiniMax-H3 Context Trim",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

MiniMaxH3ContextTrimStage::MiniMaxH3ContextTrimStage(
    const SessionContextIntf* s, std::string id,
    std::vector<InEdge> iports, FlexData config)
  : TypedStage<MiniMaxH3ContextTrimStage>(s, std::move(id), std::move(iports),
                                          std::move(config))
{
  allocate_oports(spec().oports.size());
  _trim_cfg    = (int)attr_int("trim_frames");
  _conform     = attr_bool("conform_audio");
  _fps_cfg     = attr_real("fps");
  _luma_match  = attr_bool("luma_match");
  _luma_frames = (int)attr_int("luma_match_frames");
  _declick_ms  = attr_real("declick_ms");
  if (_trim_cfg < 0) {
    fail_config(fmt("MiniMaxH3ContextTrimStage('{}'): trim_frames must be >= 0",
                    this->id()));
  }
  if (!(_fps_cfg > 0.0)) {
    fail_config(fmt("MiniMaxH3ContextTrimStage('{}'): fps must be positive",
                    this->id()));
  }
  if (_luma_frames < 0) {
    fail_config(fmt("MiniMaxH3ContextTrimStage('{}'): luma_match_frames must "
                    "be >= 0", this->id()));
  }
  if (!(_declick_ms >= 0.0)) {
    fail_config(fmt("MiniMaxH3ContextTrimStage('{}'): declick_ms must be >= 0",
                    this->id()));
  }
  _trim = _trim_cfg;
  _fps  = _fps_cfg;
}

const StageSpec&
MiniMaxH3ContextTrimStage::spec() const noexcept
{
  return kSpec;
}

void
MiniMaxH3ContextTrimStage::apply_info_(const BeatPayloadIntf* beat)
{
  const auto* f = dynamic_cast<const FlexDataPayload*>(beat);
  if (f == nullptr || !f->data.is_object()) { return; }
  const auto o = f->data.as_object();
  if (o.contains("trim_frames")) {
    const int t = (int)o.at("trim_frames").as_int(_trim);
    if (t >= 0 && t != _trim) {
      session()->info(fmt("MiniMaxH3ContextTrimStage('{}'): trimming {} frames "
                          "(from the importer)", this->id(), t));
    }
    if (t >= 0) { _trim = t; }
  }
}

int
MiniMaxH3ContextTrimStage::luma_hold_() const
{
  // Nothing to level when nothing was cut: without a trim this clip does
  // not follow another one and its first frame is its own.
  if (!_luma_match || _luma_frames <= 0 || _trim <= 0) { return 0; }
  // Far enough to have both the frames being corrected and the body they
  // are judged against.
  return std::max(_luma_frames, h3ctx::kLumaBodyEnd);
}

std::vector<std::unique_ptr<BeatPayloadIntf>>
MiniMaxH3ContextTrimStage::release_held_()
{
  luma_match_();
  std::vector<std::unique_ptr<BeatPayloadIntf>> out;
  out.swap(_held);
  return out;
}

void
MiniMaxH3ContextTrimStage::luma_match_()
{
  const int n = (int)_held.size();
  if (n <= 0) { return; }
  std::vector<std::uint8_t*> px((std::size_t)n, nullptr);
  std::vector<std::int64_t>  len((std::size_t)n, 0);
  std::vector<double>        level((std::size_t)n, 0.0);
  for (int k = 0; k < n; ++k) {
    px[(std::size_t)k] = h3ctx::frame_pixels(
        _held[(std::size_t)k].get(), nullptr, nullptr, nullptr,
        &len[(std::size_t)k]);
    if (px[(std::size_t)k] == nullptr) { return; }   // not planar u8: leave it
    level[(std::size_t)k] =
        h3ctx::frame_luma(px[(std::size_t)k], len[(std::size_t)k]);
  }
  // The target is THIS clip's body, never the clip it continues. Aiming at
  // the previous clip instead corrects an error the pinning has already
  // removed, and the correction's release then shows as a ramp over the
  // frames after the join -- a second flash in place of the first.
  const h3ctx::LumaTrend body =
      h3ctx::luma_trend(level.data(), n, h3ctx::kLumaBodyBegin,
                        h3ctx::kLumaBodyEnd);
  if (!body.ok) { return; }
  double lo = 1.0, hi = 1.0;
  int    did = 0;
  for (int k = 0; k < _luma_frames && k < n; ++k) {
    const double g =
        h3ctx::luma_gain(body, level[(std::size_t)k], k, _luma_frames);
    if (std::abs(g - 1.0) < 1e-3) { continue; }
    h3ctx::apply_luma_gain(px[(std::size_t)k], len[(std::size_t)k], g);
    lo = did == 0 ? g : std::min(lo, g);
    hi = did == 0 ? g : std::max(hi, g);
    ++did;
  }
  if (did > 0 && !_luma_said) {
    _luma_said = true;
    session()->info(fmt(
        "MiniMaxH3ContextTrimStage('{}'): luma-matched the {} frame(s) after "
        "the cut onto this clip's own settled level, gain {} .. {}",
        this->id(), did, lo, hi));
  }
}

void
MiniMaxH3ContextTrimStage::reset_run_state()
{
  // Per-launch reset. Stopping a pipeline destroys the runtime, not this
  // stage, so all of this is the PREVIOUS run's: an info port already
  // latched and never listened to again, the trim value that run's
  // importer sent, and a frame counter that would carry on past the new
  // clip's first frames -- dropping every frame of a clip whose beats
  // carry no sideband index.
  _trim          = _trim_cfg;
  _fps           = _fps_cfg;
  _info_latched  = false;
  _short_said    = false;
  _fps_said      = false;
  _luma_said     = false;
  _luma_done     = false;
  _frame_counter = 0;
  _clip_frames   = 0;
  _held.clear();
}

Job
MiniMaxH3ContextTrimStage::process(RuntimeContext& ctx)
{
  const bool info_wired = ctx.num_iports() > 2 && ctx.iport_connected(2);
  // The importer speaks first -- it is a source, and it writes its info
  // beat before generate-video can have produced the frames below -- so
  // waiting for that beat makes the trim right from frame 0. It is waited
  // for even when beats are already queued, which is exactly when the
  // value is needed: the PCM in particular arrives as ONE beat
  // and is cut once; taking it before the info beat lands trims the whole
  // soundtrack by the stage's own default instead, and which happens
  // first is only ever decided by which clock domain the runtime
  // schedules first.
  //
  // The wait cannot hold the stage open: EOS on the info port releases it
  // (`read` returns null), and so does having nothing left to apply the
  // value to.
  bool data_live = false;
  for (unsigned p = 0; p < 2; ++p) {
    if (ctx.num_iports() > p && ctx.iport_connected(p) && !ctx.eos(p)) {
      data_live = true;
    }
  }
  if (info_wired && !_info_latched && !ctx.eos(2) && data_live) {
    auto b = co_await ctx.read(2);
    _info_latched = true;
    if (b) {
      apply_info_(b.get());
    } else {
      session()->warn(fmt(
          "MiniMaxH3ContextTrimStage('{}'): the info port closed without a "
          "beat; trimming the configured {} frames", this->id(), _trim));
    }
  }
  // A graph-sourced importer sends one info per clip: take any that
  // arrived since.
  while (info_wired && ctx.backlog(2) > 0) {
    auto b = co_await ctx.read(2);
    if (!b) { break; }
    apply_info_(b.get());
  }

  std::vector<unsigned> ports;
  for (unsigned p = 0; p < 2; ++p) {
    if (ctx.num_iports() > p && ctx.iport_connected(p)) { ports.push_back(p); }
  }
  std::vector<unsigned> live;
  for (unsigned p : ports) {
    if (!ctx.eos(p)) { live.push_back(p); }
  }
  if (live.empty()) {
    ctx.signal_done();
    co_return;
  }
  // Frames and PCM run on independent clocks: wait for either.
  co_await ctx.read_any(live);

  for (unsigned p : ports) {
    const std::uint32_t avail = ctx.backlog(p);
    for (std::uint32_t i = 0; i < avail; ++i) {
      auto beat = co_await ctx.read(p);
      if (!beat) { break; }
      auto* tb = dynamic_cast<TensorBeatPayload*>(beat.get());

      if (p == 0) {
        // ---- a decoded frame ---------------------------------------
        std::int64_t frame = -1, frames = -1;
        if (tb != nullptr && tb->sideband.is_object()) {
          const auto o = tb->sideband.as_object();
          if (o.contains("frame"))  { frame  = o.at("frame").as_int(-1); }
          if (o.contains("frames")) { frames = o.at("frames").as_int(-1); }
        }
        if (frame < 0) { frame = _frame_counter; }
        ++_frame_counter;
        if (frames > 0) { _clip_frames = frames; }
        // The decoder states the rate it decoded at; `fps` is the
        // fallback its own doc says it is.
        if (tb != nullptr && tb->sideband.is_object()) {
          const auto o = tb->sideband.as_object();
          if (o.contains("fps")) {
            const double f = o.at("fps").as_real(0.0);
            if (f > 0.0 && f != _fps) {
              if (!_fps_said) {
                _fps_said = true;
                session()->info(fmt(
                    "MiniMaxH3ContextTrimStage('{}'): trimming the sound at "
                    "the {} fps the frames carry, not the configured {}",
                    this->id(), f, _fps_cfg));
              }
              _fps = f;
            }
          }
        }
        if (frames >= 0 && frames <= _trim) {
          if (!_short_said) {
            _short_said = true;
            session()->warn(fmt(
                "MiniMaxH3ContextTrimStage('{}'): a {}-frame clip is not "
                "longer than the {}-frame trim; passing it through "
                "untrimmed", this->id(), frames, _trim));
          }
          // Untrimmed, but not unordered: a previous clip may still have
          // frames in hand, and writing this one now would put it in
          // front of them for good.
          if (!_held.empty()) {
            auto out = release_held_();
            for (auto& h : out) { co_await ctx.write(0, std::move(h)); }
          }
          co_await ctx.write(0, std::move(beat));
          continue;
        }
        if (frame < _trim) { continue; }   // the regenerated overlap
        if (tb != nullptr && tb->sideband.is_object()) {
          auto o = tb->sideband.as_object();
          o.insert_or_assign("frame", FlexData::make_int(frame - _trim));
          if (frames >= 0) {
            o.insert_or_assign("frames", FlexData::make_int(frames - _trim));
          }
        }
        // Kept frame 0 is a clip's first frame, and a run can carry more
        // than one clip (the importer sends an info beat per clip). So the
        // hold is scoped to the clip: whatever the previous one left in
        // hand goes out HERE, ahead of this frame, and this clip holds
        // again for itself. Counting what is held rather than trusting the
        // frame index also keeps the order right when a new info beat
        // moves `_trim` mid-run, which would otherwise let a later frame
        // past a hold that is still filling.
        const int hold = luma_hold_();
        if (frame - _trim == 0) {
          _luma_done = false;
          if (!_held.empty()) {
            auto out = release_held_();
            for (auto& h : out) { co_await ctx.write(0, std::move(h)); }
          }
        }
        if (hold > 0 && !_luma_done && (int)_held.size() < hold) {
          _held.push_back(std::move(beat));
          if ((int)_held.size() >= hold) {
            _luma_done = true;
            auto out = release_held_();
            for (auto& h : out) { co_await ctx.write(0, std::move(h)); }
          }
          continue;
        }
        co_await ctx.write(0, std::move(beat));
        continue;
      }

      // ---- the clip's PCM ------------------------------------------
      if (tb == nullptr || tb->dtype != TensorBeat::DType::F32 ||
          (tb->shape.size() != 1 && tb->shape.size() != 2) ||
          !tb->is_contiguous()) {
        session()->warn(fmt(
            "MiniMaxH3ContextTrimStage('{}'): audio beat {} is not f32 PCM "
            "[channels, samples]; passing it through", this->id(),
            beat->describe()));
        co_await ctx.write(1, std::move(beat));
        continue;
      }
      int sample_rate = 0;
      if (tb->sideband.is_object()) {
        const auto o = tb->sideband.as_object();
        if (o.contains("sample_rate")) {
          sample_rate = (int)o.at("sample_rate").as_int(0);
        }
      }
      if (sample_rate <= 0) {
        session()->warn(fmt(
            "MiniMaxH3ContextTrimStage('{}'): PCM beat without a sample_rate; "
            "passing it through untrimmed", this->id()));
        co_await ctx.write(1, std::move(beat));
        continue;
      }
      const std::int64_t channels = tb->shape.size() == 2 ? tb->shape[0] : 1;
      const std::int64_t samples  = tb->shape.back();
      // `_clip_frames` is the picture this soundtrack belongs to, when a
      // frame has already been seen: conforming to IT is the point --
      // deriving the frame count from the sample count instead would
      // conform the sound to its own rounding.
      const h3ctx::PcmTrim plan = h3ctx::plan_pcm_trim(
          samples, sample_rate, _fps, _trim, _clip_frames, _conform);
      if (plan.keep <= 0) {
        co_await ctx.write(1, std::move(beat));
        continue;
      }
      auto out = std::make_unique<TensorBeatPayload>();
      out->dtype = TensorBeat::DType::F32;
      out->shape = tb->shape.size() == 2
                       ? std::vector<std::int64_t>{channels, plan.keep}
                       : std::vector<std::int64_t>{plan.keep};
      out->resize_contiguous((std::size_t)(channels * plan.keep));   // zeros
      const std::int64_t copy = std::min(plan.keep, samples - plan.drop);
      const float* src = tb->as_f32();
      float* dst = out->as_f32();
      for (std::int64_t c = 0; copy > 0 && c < channels; ++c) {
        std::memcpy(dst + c * plan.keep, src + c * samples + plan.drop,
                    (std::size_t)copy * sizeof(float));
      }
      // The cut lands wherever it lands in the waveform, and a waveform
      // that starts at a non-zero sample is a click. A few ms of fade is
      // below the threshold of being heard as one.
      const std::int64_t fade = std::min<std::int64_t>(
          plan.keep,
          (std::int64_t)std::llround(_declick_ms * 0.001 * (double)sample_rate));
      if (plan.drop > 0 && fade > 1) {
        for (std::int64_t c = 0; c < channels; ++c) {
          float* p = dst + c * plan.keep;
          for (std::int64_t i = 0; i < fade; ++i) {
            const double t = (double)i / (double)(fade - 1);
            p[i] = (float)((double)p[i] * 0.5 * (1.0 - std::cos(M_PI * t)));
          }
        }
      }
      out->sideband = tb->sideband;
      if (out->sideband.is_object()) {
        out->sideband.as_object().insert_or_assign(
            "samples", FlexData::make_int(plan.keep));
      }
      session()->info(fmt(
          "MiniMaxH3ContextTrimStage('{}'): PCM {} -> {} samples ({} dropped "
          "at the head{})", this->id(), samples, plan.keep, plan.drop,
          plan.keep > samples - plan.drop ? ", tail padded" : ""));
      co_await ctx.write(1, std::move(out));
    }
  }

  bool all_eos = true;
  for (unsigned p : ports) {
    if (!ctx.eos(p)) { all_eos = false; break; }
  }
  if (all_eos || ctx.stop_requested()) { ctx.signal_done(); }
}

Job
MiniMaxH3ContextTrimStage::drain(RuntimeContext& ctx)
{
  // A clip shorter than the hold, or a stop part-way through one, ends
  // with frames still in hand. This is the only place that always runs:
  // the process() call that finds every iport at EOS returns before it
  // reaches its own tail, so a flush there is missed whenever the last
  // beat and the close land in different calls.
  if (!_held.empty()) {
    auto out = release_held_();
    for (auto& h : out) { co_await ctx.write(0, std::move(h)); }
  }
  co_return;
}

VPIPE_REGISTER_STAGE(MiniMaxH3ContextTrimStage)
VPIPE_REGISTER_SPEC(MiniMaxH3ContextTrimStage, kSpec)

}  // namespace vpipe
