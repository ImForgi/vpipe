#include "stages/minimax-h3-context-import-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstring>
#include <string>
#include <utility>

namespace vpipe {

namespace {

constexpr ConfigKey kAttrs[] = {
  {.key = "input_url", .type = ConfigType::String,
   .doc = "context file written by minimax-h3-context-export (.safetensors). "
          "Required unless iport0 carries a previous clip's latent",
   .is_path = true},
  {.key = "context_frames", .type = ConfigType::Int,
   .doc = "video frames carried from the end of the previous clip: 17k + 5 "
          "(5, 22, 39, 56, ...). 22 is the tested default -- about 0.9 s of "
          "motion",
   .def_int = 22},
  {.key = "audio_context_frames", .type = ConfigType::Int,
   .doc = "audio carried, in video frames: 0 = the same window as the video, "
          "-1 = no audio context. Longer than the video window reaches back "
          "further for audio-only continuity",
   .def_int = 0},
  {.key = "start_frame", .type = ConfigType::Int,
   .doc = "frame of the NEW clip the context starts at. 0 (default) is a "
          "continuation, whose regenerated overlap minimax-h3-context-trim "
          "drops; anything else is an interior guide and trims nothing",
   .def_int = 0},
  {.key = "duration_mode", .type = ConfigType::String,
   .doc = "what generate-video's `frames` means with this context: clip "
          "(default) = the generated clip, the overlap comes out of it; "
          "new_footage = the footage KEPT after the trim (clip lengthened to "
          "the nearest 17n+5); new_footage_min = the same rounded up. The "
          "footage modes cap at the 362 frames H3 was trained on",
   .def_str = "clip"},
  {.key = "audio_align", .type = ConfigType::String,
   .doc = "exact (default): the carried audio ends exactly where the carried "
          "picture ends, including H3's sub-latent audio rounding. round: "
          "snap it to the generated audio's latent grid",
   .def_str = "exact"},
};
const PortSpec kIports[] = {
  {.name = "latent",
   .doc = "OPTIONAL: a previous clip's sampled video latent (generate-video "
          "oport0) in the same graph. Wired, it replaces input_url and one "
          "context is emitted per latent",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
  {.name = "audio_latent",
   .doc = "OPTIONAL: that clip's audio latent (generate-video oport1)",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "h3_context",
   .doc = "f32 [24, n, H/16, W/16]: the last n latent frames of the clip, "
          "with {context_frames, start_frame, duration_mode} -> "
          "generate-video iport 11",
   .type = &typeid(TensorBeatPayload), .tags = "minimax-h3-context-video",
   .clock_group = 0},
  {.name = "h3_context_audio",
   .doc = "f32 [2, 32, a]: the matching audio latents with {audio_offset} "
          "(a = 0 when there is none) -> generate-video iport 12",
   .type = &typeid(TensorBeatPayload), .tags = "minimax-h3-context-audio",
   .clock_group = 0},
  {.name = "info",
   .doc = "FlexData {trim_frames, context_frames, start_frame, "
          "duration_mode, audio_latents, audio_offset, source_frames} -> "
          "minimax-h3-context-trim iport 2",
   .type = &typeid(FlexDataPayload), .tags = "minimax-h3-context-info",
   .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "minimax-h3-context-import",
  .doc       = "Source: the tail of a previous MiniMax-H3 clip (a context "
               "file from minimax-h3-context-export, or latents in the same "
               "graph) as a continuation context for generate-video, which "
               "pins it clean on the new clip's timeline so motion and sound "
               "carry across the join. FL2VA and Ref2VA, with or without LoRA.",
  .display_name = "MiniMax-H3 Context Import",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

MiniMaxH3ContextImportStage::MiniMaxH3ContextImportStage(
    const SessionContextIntf* s, std::string id,
    std::vector<InEdge> iports, FlexData config)
  : TypedStage<MiniMaxH3ContextImportStage>(s, std::move(id), std::move(iports),
                                            std::move(config))
{
  allocate_oports(spec().oports.size());
  _input_url            = attr_path("input_url", /*for_write=*/false);
  _context_frames       = (int)attr_int("context_frames");
  _audio_context_frames = (int)attr_int("audio_context_frames");
  _start_frame          = (int)attr_int("start_frame");
  const std::string dm  = attr_str("duration_mode");
  const std::string al  = attr_str("audio_align");

  if (!h3ctx::is_clip_frames(_context_frames)) {
    fail_config(fmt(
        "MiniMaxH3ContextImportStage('{}'): context_frames {} is not 17k + 5 "
        "(5, 22, 39, 56, ...); only those tails start on a VAE cycle boundary",
        this->id(), _context_frames));
  }
  if (_start_frame < 0) {
    fail_config(fmt(
        "MiniMaxH3ContextImportStage('{}'): start_frame must be >= 0",
        this->id()));
  }
  if (!h3ctx::parse_duration_mode(dm, &_duration)) {
    fail_config(fmt(
        "MiniMaxH3ContextImportStage('{}'): duration_mode must be clip | "
        "new_footage | new_footage_min (got '{}')", this->id(), dm));
  }
  if (!h3ctx::parse_audio_align(al, &_audio_align)) {
    fail_config(fmt(
        "MiniMaxH3ContextImportStage('{}'): audio_align must be exact | round "
        "(got '{}')", this->id(), al));
  }
}

const StageSpec&
MiniMaxH3ContextImportStage::spec() const noexcept
{
  return kSpec;
}

bool
MiniMaxH3ContextImportStage::plan_(const std::vector<std::int64_t>& video_shape,
                                   const std::vector<std::int64_t>& audio_shape,
                                   const std::string& source,
                                   h3ctx::TailPlan* plan) const
{
  // `kMaxSourceLatents` keeps a shape read out of a FILE from reaching
  // the int arithmetic below as something it cannot represent: a clip is
  // a few hundred latents and the model tops out at 362 frames, so
  // anything past this is a corrupt header rather than a long clip.
  constexpr std::int64_t kMaxSourceLatents = 1 << 20;
  if (video_shape.size() != 4 ||
      video_shape[0] != h3ctx::kVideoChannels ||
      video_shape[1] <= 0 || video_shape[1] > kMaxSourceLatents) {
    session()->warn(fmt(
        "MiniMaxH3ContextImportStage('{}'): {} is not a MiniMax-H3 video "
        "latent [{}, T, h, w]", this->id(), source, h3ctx::kVideoChannels));
    return false;
  }
  const bool has_audio = audio_shape.size() == 3 &&
                         audio_shape[0] == h3ctx::kStereo &&
                         audio_shape[1] == h3ctx::kAudioLatentChannels &&
                         audio_shape[2] > 0 &&
                         audio_shape[2] <= kMaxSourceLatents;
  std::string err;
  if (!h3ctx::plan_tail((int)video_shape[1],
                        has_audio ? (int)audio_shape[2] : 0, _context_frames,
                        _audio_context_frames, _start_frame, _audio_align,
                        plan, &err)) {
    session()->warn(fmt("MiniMaxH3ContextImportStage('{}'): {}: {}", this->id(),
                        source, err));
    return false;
  }
  if (plan->overhang_ignored) {
    session()->warn(fmt(
        "MiniMaxH3ContextImportStage('{}'): {} has {} audio latents for {} "
        "frames, which is not H3's own grid; the carried audio is aligned as "
        "if it were", this->id(), source, audio_shape[2],
        plan->source_frames));
  }
  if (_audio_context_frames >= 0 && !has_audio) {
    session()->info(fmt(
        "MiniMaxH3ContextImportStage('{}'): {} has no soundtrack; carrying "
        "video only", this->id(), source));
  }
  return true;
}

bool
MiniMaxH3ContextImportStage::emit_(
    const h3ctx::TailPlan& plan, const float* video_tail,
    const std::vector<std::int64_t>& video_shape, const float* audio_tail,
    const std::vector<std::int64_t>& audio_shape, const std::string& source,
    Beats* out) const
{
  if (video_tail == nullptr || video_shape.size() != 4) { return false; }
  // Video tail.
  {
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = TensorBeat::DType::F32;
    t->shape = video_shape;
    const std::size_t n = (std::size_t)(video_shape[0] * video_shape[1] *
                                        video_shape[2] * video_shape[3]);
    t->resize_contiguous(n);
    std::memcpy(t->as_f32(), video_tail, n * sizeof(float));
    FlexData sb = FlexData::make_object();
    auto o = sb.as_object();
    o.insert_or_assign("context_frames",
                       FlexData::make_int((std::int64_t)plan.context_frames));
    o.insert_or_assign("start_frame",
                       FlexData::make_int((std::int64_t)_start_frame));
    o.insert_or_assign(
        "duration_mode",
        FlexData::make_string(h3ctx::duration_mode_name(_duration)));
    o.insert_or_assign("source_frames",
                       FlexData::make_int((std::int64_t)plan.source_frames));
    t->sideband = std::move(sb);
    out->video = std::move(t);
  }
  // Audio tail -- always a beat, 0 latents when there is none, so
  // generate-video can read both ports unconditionally.
  {
    const bool have = plan.audio_latents > 0 && audio_tail != nullptr &&
                      audio_shape.size() == 3;
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = TensorBeat::DType::F32;
    t->shape = have ? audio_shape
                    : std::vector<std::int64_t>{h3ctx::kStereo,
                                                h3ctx::kAudioLatentChannels, 0};
    const std::size_t n =
        have ? (std::size_t)(audio_shape[0] * audio_shape[1] * audio_shape[2])
             : 0;
    t->resize_contiguous(n);
    if (n > 0) { std::memcpy(t->as_f32(), audio_tail, n * sizeof(float)); }
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("audio_offset",
                                    FlexData::make_real(plan.audio_offset));
    t->sideband = std::move(sb);
    out->audio = std::move(t);
  }
  {
    FlexData info = FlexData::make_object();
    auto o = info.as_object();
    o.insert_or_assign("trim_frames",
                       FlexData::make_int((std::int64_t)plan.trim_frames));
    o.insert_or_assign("context_frames",
                       FlexData::make_int((std::int64_t)plan.context_frames));
    o.insert_or_assign("start_frame",
                       FlexData::make_int((std::int64_t)_start_frame));
    o.insert_or_assign(
        "duration_mode",
        FlexData::make_string(h3ctx::duration_mode_name(_duration)));
    o.insert_or_assign("audio_latents",
                       FlexData::make_int((std::int64_t)plan.audio_latents));
    o.insert_or_assign("audio_offset", FlexData::make_real(plan.audio_offset));
    o.insert_or_assign("source_frames",
                       FlexData::make_int((std::int64_t)plan.source_frames));
    out->info = make_payload<FlexDataPayload>(std::move(info));
  }
  session()->info(fmt(
      "MiniMaxH3ContextImportStage('{}'): context from {} ({} frames): last {} "
      "frames ({} latents) + {} audio latents at offset {:.3f}, trim {}",
      this->id(), source, plan.source_frames, plan.context_frames,
      plan.video_latents, plan.audio_latents, plan.audio_offset,
      plan.trim_frames));
  return true;
}

bool
MiniMaxH3ContextImportStage::build_(
    const float* video, const std::vector<std::int64_t>& video_shape,
    const float* audio, const std::vector<std::int64_t>& audio_shape,
    const std::string& source, Beats* out) const
{
  if (video == nullptr) { return false; }
  h3ctx::TailPlan plan;
  if (!plan_(video_shape, audio == nullptr ? std::vector<std::int64_t>{}
                                           : audio_shape,
             source, &plan)) {
    return false;
  }
  std::vector<float> vtail, atail;
  std::vector<std::int64_t> vshape, ashape;
  if (!h3ctx::slice_video_tail(video, video_shape, plan.video_latents, &vtail,
                               &vshape)) {
    session()->warn(fmt("MiniMaxH3ContextImportStage('{}'): cannot slice the "
                        "video tail of {}", this->id(), source));
    return false;
  }
  if (plan.audio_latents > 0 &&
      !h3ctx::slice_audio_tail(audio, audio_shape, plan.audio_latents, &atail,
                               &ashape)) {
    session()->warn(fmt("MiniMaxH3ContextImportStage('{}'): cannot slice the "
                        "audio tail of {}", this->id(), source));
    return false;
  }
  return emit_(plan, vtail.data(), vshape,
               atail.empty() ? nullptr : atail.data(), ashape, source, out);
}

void
MiniMaxH3ContextImportStage::reset_run_state()
{
  // Stopping a pipeline destroys the runtime, not this stage, so without
  // this a Stop-then-Start would find the file already imported, emit no
  // context and let generate-video run the clip without it.
  _file_done = false;
}

Job
MiniMaxH3ContextImportStage::process(RuntimeContext& ctx)
{
  const bool from_graph = ctx.num_iports() > 0 && ctx.iport_connected(0);
  Beats beats;

  if (from_graph) {
    // One context per upstream clip.
    auto vb = co_await ctx.read(0);
    if (!vb) {
      ctx.signal_done();
      co_return;
    }
    std::unique_ptr<BeatPayloadIntf> ab;
    if (ctx.num_iports() > 1 && ctx.iport_connected(1)) {
      ab = co_await ctx.read(1);
    }
    const auto* v = dynamic_cast<const TensorBeatPayload*>(vb.get());
    const auto* a =
        ab ? dynamic_cast<const TensorBeatPayload*>(ab.get()) : nullptr;
    if (v == nullptr || v->dtype != TensorBeat::DType::F32 ||
        !v->is_contiguous()) {
      session()->warn(fmt(
          "MiniMaxH3ContextImportStage('{}'): iport0 carried {}, not an f32 "
          "latent; no context", this->id(), vb->describe()));
      co_return;
    }
    const bool audio_ok = a != nullptr && a->dtype == TensorBeat::DType::F32 &&
                          a->is_contiguous();
    if (!build_(v->as_f32(), v->shape, audio_ok ? a->as_f32() : nullptr,
                audio_ok ? a->shape : std::vector<std::int64_t>{},
                "the upstream clip", &beats)) {
      co_return;
    }
  } else {
    // One context from the file, then done: the stage closes its ports,
    // and a generate-video asked for a second clip finds none and skips.
    if (_file_done) {
      ctx.signal_done();
      co_return;
    }
    _file_done = true;
    if (_input_url.empty()) {
      session()->warn(fmt(
          "MiniMaxH3ContextImportStage('{}'): no input_url and iport0 is not "
          "wired; there is no context to import", this->id()));
      ctx.signal_done();
      co_return;
    }
    // The HEADER first: a context is under a second of a file that holds
    // a whole clip (~100 MB at 720p), so the shapes are read, the tail is
    // planned from them, and only that tail is read.
    h3ctx::ContextHeader head;
    std::string err;
    if (!h3ctx::read_context_header(_input_url, &head, &err)) {
      session()->warn(fmt("MiniMaxH3ContextImportStage('{}'): {}",
                          this->id(), err));
      ctx.signal_done();
      co_return;
    }
    const auto fmt_it = head.metadata.find("format");
    if (fmt_it != head.metadata.end() && fmt_it->second != h3ctx::kFormat) {
      session()->warn(fmt(
          "MiniMaxH3ContextImportStage('{}'): '{}' is format '{}', expected "
          "'{}'; reading it anyway", this->id(), _input_url, fmt_it->second,
          h3ctx::kFormat));
    }
    const std::string source = "'" + _input_url + "'";
    h3ctx::TailPlan plan;
    if (!plan_(head.video_shape, head.audio_shape, source, &plan)) {
      ctx.signal_done();
      co_return;
    }
    h3ctx::ContextFile tail;
    if (!h3ctx::read_context_tail(_input_url, plan.video_latents,
                                  plan.audio_latents, &tail, &err)) {
      session()->warn(fmt("MiniMaxH3ContextImportStage('{}'): {}",
                          this->id(), err));
      ctx.signal_done();
      co_return;
    }
    if (!emit_(plan, tail.video.data(), tail.video_shape,
               tail.audio.empty() ? nullptr : tail.audio.data(),
               tail.audio_shape, source, &beats)) {
      ctx.signal_done();
      co_return;
    }
  }

  co_await ctx.write(0, std::move(beats.video));
  co_await ctx.write(1, std::move(beats.audio));
  co_await ctx.write(2, std::move(beats.info));
}

VPIPE_REGISTER_STAGE(MiniMaxH3ContextImportStage)
VPIPE_REGISTER_SPEC(MiniMaxH3ContextImportStage, kSpec)

}  // namespace vpipe
