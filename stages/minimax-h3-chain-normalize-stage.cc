#include "stages/minimax-h3-chain-normalize-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/minimax-h3-context.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

namespace vpipe {

namespace {

// Both constants are the reference's, fixed there rather than exposed:
// the correction is capped, and anything under the floor is not worth
// the rounding it would cost every pixel.
constexpr double kMaxSoften = 0.85;
constexpr double kMinSoften = 0.02;
// Its baseline window is at least this many frames however short the
// configured seconds work out.
constexpr int kMinBaselineFrames = 8;
// And a take shorter than this is passed through untouched.
constexpr int kMinTakeFrames = 8;
// The baseline window is HELD, so it is bounded: an hour of frames is
// past anything a first clip can be, and it keeps the seconds-to-frames
// conversion inside what an int64 can hold.
constexpr double kMaxWindowSeconds = 3600.0;

constexpr ConfigKey kAttrs[] = {
  {.key = "skip_seconds", .type = ConfigType::Real,
   .doc = "skipped at the very start, and excluded from the baseline. H3 "
          "opens with an exposure fade-in -- luma climbs for the first "
          "~1.7 s -- and letting that into the baseline drags it dark, "
          "which softens the whole take",
   .def_real = 2.0},
  {.key = "baseline_seconds", .type = ConfigType::Real,
   .doc = "how much of the FIRST clip sets the house level everything "
          "after is measured against. skip_seconds + this should fit "
          "inside clip one: run past its end and the window takes the "
          "next clip's drift as the level",
   .def_real = 10.0},
  {.key = "fps", .type = ConfigType::Real,
   .doc = "frame rate used when a beat's sideband carries none",
   .def_real = 24.0},
  {.key = "strength", .type = ConfigType::Real,
   .doc = "how hard excess structure is pulled back; 1.0 is what the "
          "reference settled on by eye, 0 measures and reports without "
          "touching a pixel",
   .def_real = 1.0},
  {.key = "deadband", .type = ConfigType::Real,
   .doc = "frames within this ratio of the baseline are left alone, so "
          "the first clip is never softened against its own average",
   .def_real = 1.06},
  {.key = "ema", .type = ConfigType::Real,
   .doc = "smoothing on the correction, so it eases in rather than "
          "stepping at a window boundary",
   .def_real = 0.10},
  {.key = "colour_match", .type = ConfigType::Bool,
   .doc = "histogram-match every frame to a reference frame from the "
          "first clip. This is what fixes the grade wandering across a "
          "long chain -- the half no sharpening control touches",
   .def_bool = true},
};
const PortSpec kIports[] = {
  {.name = "image",
   .doc = "per-frame planar u8 RGB [3, H, W], the joined chain in order",
   .type = &typeid(TensorBeatPayload), .tags = "rgb-frames", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "image",
   .doc = "the same frames, levelled against the first clip",
   .type = &typeid(TensorBeatPayload), .tags = "rgb-frames", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "minimax-h3-chain-normalize",
  .doc       = "Levels a joined MiniMax-H3 chain against its own first "
               "clip: the fine texture that ratchets up at every join, "
               "and the colour grade that wanders with it. A port of "
               "ComfyUI-H3-Multishot's H3ChainNormalize.",
  .display_name = "MiniMax-H3 Chain Normalize",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

MiniMaxH3ChainNormalizeStage::MiniMaxH3ChainNormalizeStage(
    const SessionContextIntf* s, std::string id, std::vector<InEdge> iports,
    FlexData config)
  : TypedStage<MiniMaxH3ChainNormalizeStage>(s, std::move(id),
                                             std::move(iports),
                                             std::move(config))
{
  allocate_oports(spec().oports.size());
  _skip_s   = attr_real("skip_seconds");
  _base_s   = attr_real("baseline_seconds");
  _fps_cfg  = attr_real("fps");
  _strength = attr_real("strength");
  _deadband = attr_real("deadband");
  _ema      = attr_real("ema");
  _colour   = attr_bool("colour_match");
  if (!(_skip_s >= 0.0) || !(_base_s > 0.0) ||
      _skip_s + _base_s > kMaxWindowSeconds) {
    fail_config(fmt("MiniMaxH3ChainNormalizeStage('{}'): skip_seconds must "
                    "be >= 0, baseline_seconds > 0, and the two together no "
                    "more than {} s -- the window is held in memory, and "
                    "past that the frame count stops fitting the arithmetic",
                    this->id(), kMaxWindowSeconds));
  }
  if (!(_fps_cfg > 0.0)) {
    fail_config(fmt("MiniMaxH3ChainNormalizeStage('{}'): fps must be "
                    "positive", this->id()));
  }
  if (!(_strength >= 0.0)) {
    fail_config(fmt("MiniMaxH3ChainNormalizeStage('{}'): strength must be "
                    ">= 0", this->id()));
  }
  if (!(_deadband >= 1.0)) {
    fail_config(fmt("MiniMaxH3ChainNormalizeStage('{}'): deadband must be "
                    ">= 1.0 -- below it the first clip would be softened "
                    "against its own average", this->id()));
  }
  if (!(_ema > 0.0) || _ema > 1.0) {
    fail_config(fmt("MiniMaxH3ChainNormalizeStage('{}'): ema must be in "
                    "(0, 1]", this->id()));
  }
  _fps = _fps_cfg;
}

const StageSpec&
MiniMaxH3ChainNormalizeStage::spec() const noexcept
{
  return kSpec;
}

void
MiniMaxH3ChainNormalizeStage::reset_run_state()
{
  // Per-launch reset. Everything below describes THIS take: the frames
  // counted, the baseline the texture is measured against, the grade the
  // colour is held to, and the smoothed correction so far. A second
  // launch is a second take and must find none of it.
  _fps      = _fps_cfg;
  _seen     = 0;
  _settled  = false;
  _baseline = 0.0;
  _sigma    = 0.0;
  _peak     = 0.0;
  _said     = false;
  _odd_said = false;
  _have_ref = false;
  _ref_channels = 0;
  _baseline_samples.clear();
  _held.clear();
}

void
MiniMaxH3ChainNormalizeStage::settle_()
{
  if (_settled) { return; }
  _settled = true;
  if (_baseline_samples.empty()) { return; }
  // The MEDIAN of the window, as the reference takes it: the lower of
  // the two middles on an even count, which is what indexing a sorted
  // list at size/2 gives.
  auto& v = _baseline_samples;
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)mid, v.end());
  _baseline = v[mid];
  v.clear();
  v.shrink_to_fit();
}

void
MiniMaxH3ChainNormalizeStage::level_(std::uint8_t* px, int channels, int h,
                                     int w)
{
  // The reference's order, kept: match the grade first, then measure the
  // frame the viewer will see, then subtract. Note that the baseline it
  // is compared against was measured BEFORE matching -- that is how the
  // reference does it.
  if (_colour && _have_ref && _ref_channels == channels) {
    double cdf[256];
    std::uint8_t lut[256];
    for (int c = 0; c < channels && c < 3; ++c) {
      h3ctx::channel_cdf(px, channels, h, w, c, cdf);
      h3ctx::match_lut(cdf, _ref_cdf[c], lut);
      h3ctx::apply_lut(px, channels, h, w, c, lut);
    }
  }
  if (!(_baseline > 0.0)) { return; }
  const double cur = h3ctx::norm_laplacian(px, channels, h, w);
  const double target =
      std::max(0.0, cur / std::max(_baseline, 1e-9) - _deadband) * _strength;
  _sigma = _ema * target + (1.0 - _ema) * _sigma;
  const double s = std::min(kMaxSoften, _sigma);
  _peak = std::max(_peak, s);
  if (s > kMinSoften) {
    h3ctx::subtract_structure_band(px, channels, h, w, s, &_scratch);
  }
}

std::vector<std::unique_ptr<BeatPayloadIntf>>
MiniMaxH3ChainNormalizeStage::release_held_()
{
  settle_();
  for (auto& b : _held) {
    int channels = 0, h = 0, w = 0;
    std::uint8_t* px =
        h3ctx::frame_pixels(b.get(), &channels, &h, &w, nullptr);
    if (px != nullptr) { level_(px, channels, h, w); }
  }
  std::vector<std::unique_ptr<BeatPayloadIntf>> out;
  out.swap(_held);
  return out;
}

Job
MiniMaxH3ChainNormalizeStage::process(RuntimeContext& ctx)
{
  if (ctx.eos(0)) {
    ctx.signal_done();
    co_return;
  }
  auto beat = co_await ctx.read(0);
  if (!beat) { co_return; }

  int channels = 0, h = 0, w = 0;
  std::uint8_t* px =
      h3ctx::frame_pixels(beat.get(), &channels, &h, &w, nullptr);
  if (px == nullptr) {
    if (!_odd_said) {
      _odd_said = true;
      session()->warn(fmt(
          "MiniMaxH3ChainNormalizeStage('{}'): {} is not a planar u8 RGB "
          "frame; passing it through", this->id(), beat->describe()));
    }
    // It still has to keep its place. While the baseline window is open
    // there are frames waiting behind it, and writing it now would put
    // it in front of them for good; `release_held_` skips anything it
    // cannot measure, so holding it is safe.
    if (!_settled) {
      _held.push_back(std::move(beat));
      co_return;
    }
    co_await ctx.write(0, std::move(beat));
    co_return;
  }
  if (auto* tb = dynamic_cast<TensorBeatPayload*>(beat.get());
      tb != nullptr && tb->sideband.is_object()) {
    const auto o = tb->sideband.as_object();
    if (o.contains("fps")) {
      const double f = o.at("fps").as_real(0.0);
      if (f > 0.0) { _fps = f; }
    }
  }

  // The reference's window arithmetic, truncating as it does.
  const std::int64_t skip = (std::int64_t)(_skip_s * _fps);
  const std::int64_t take =
      skip + std::max((std::int64_t)kMinBaselineFrames,
                      (std::int64_t)(_base_s * _fps));
  const std::int64_t i = _seen++;

  if (!_settled) {
    // Still inside the window. The frame is held, because the reference
    // corrects it too and cannot know how until the window closes.
    if (_colour && !_have_ref && i == skip + (take - skip) / 2) {
      // The middle frame of the baseline window is the grade everything
      // is held to.
      for (int c = 0; c < channels && c < 3; ++c) {
        h3ctx::channel_cdf(px, channels, h, w, c, _ref_cdf[c]);
      }
      _ref_channels = channels;
      _have_ref     = true;
    }
    if (i >= skip && i < take) {
      // Measured on the frame as it arrived, before any matching -- the
      // reference takes its baseline from the untouched images.
      _baseline_samples.push_back(
          h3ctx::norm_laplacian(px, channels, h, w));
    }
    _held.push_back(std::move(beat));
    if (i + 1 >= take) {
      auto out = release_held_();
      session()->info(fmt(
          "MiniMaxH3ChainNormalizeStage('{}'): baseline {} (median of the "
          "first clip after {} s) | colour {}", this->id(), _baseline,
          _skip_s, _colour ? "matched" : "off"));
      for (auto& b : out) { co_await ctx.write(0, std::move(b)); }
    }
    co_return;
  }
  level_(px, channels, h, w);
  co_await ctx.write(0, std::move(beat));
}

Job
MiniMaxH3ChainNormalizeStage::drain(RuntimeContext& ctx)
{
  // A take that ended inside the baseline window never settled. The
  // reference passes a take shorter than 8 frames through untouched and
  // levels anything longer against whatever window it did get; this does
  // the same, and it is the only place that always runs.
  if (!_held.empty()) {
    if (_seen < kMinTakeFrames) {
      // The reference returns a take this short exactly as it came in --
      // which means the GRADE too, so the colour reference goes with the
      // baseline. Leaving it armed would histogram-match the very frames
      // the contract says are untouched.
      _baseline_samples.clear();
      _settled  = true;
      _baseline = 0.0;
      _have_ref = false;
    }
    auto out = release_held_();
    for (auto& b : out) { co_await ctx.write(0, std::move(b)); }
  }
  if (!_said && _seen > 0) {
    _said = true;
    session()->info(fmt(
        "MiniMaxH3ChainNormalizeStage('{}'): {} frames, baseline {}, peak "
        "correction {}", this->id(), _seen, _baseline, _peak));
  }
  co_return;
}

VPIPE_REGISTER_STAGE(MiniMaxH3ChainNormalizeStage)
VPIPE_REGISTER_SPEC(MiniMaxH3ChainNormalizeStage, kSpec)

}  // namespace vpipe
