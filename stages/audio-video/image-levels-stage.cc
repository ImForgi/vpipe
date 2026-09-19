#include "stages/audio-video/image-levels-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <algorithm>
#include <cmath>
#include <utility>

using namespace std;

namespace vpipe {

ImageLevelsStage::ImageLevelsStage(const SessionContextIntf* session,
                                   string                    id,
                                   vector<InEdge>            iports,
                                   FlexData                  config)
  : TypedStage<ImageLevelsStage>(session, std::move(id), std::move(iports),
                                 std::move(config))
{
  _brightness = attr_real("brightness");
  _contrast   = attr_real("contrast");
  _gamma      = attr_real("gamma");
  _pivot      = attr_real("pivot");

  if (!(_gamma > 0.0)) {
    fail_config(fmt("ImageLevelsStage('{}'): gamma must be > 0, got {}",
                    this->id(), _gamma));
  }
  if (_pivot < 0.0 || _pivot > 1.0) {
    fail_config(fmt("ImageLevelsStage('{}'): pivot is a normalised level, "
                    "so it must be within [0,1]; got {}",
                    this->id(), _pivot));
  }
  _identity = (_brightness == 0.0) && (_contrast == 1.0) && (_gamma == 1.0);

  allocate_oports(1);
}

double
ImageLevelsStage::map_unit(double x) const noexcept
{
  if (_gamma != 1.0) {
    // pow() of a negative base is NaN; an f32 frame may legitimately
    // carry one, and a NaN would then spread over the picture.
    x = x > 0.0 ? std::pow(x, _gamma) : 0.0;
  }
  if (_contrast != 1.0) { x = (x - _pivot) * _contrast + _pivot; }
  x += _brightness;
  return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

namespace {

const PortSpec kIports[] = {
  {.name = "image", .doc = "planar RGB TensorBeat [3,H,W], u8 or f32",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "image", .doc = "the same shape and dtype, tone-adjusted",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};

constexpr ConfigKey kAttrs[] = {
  {.key = "brightness", .type = ConfigType::Real, .required = false,
   .doc = "additive offset in NORMALISED units, applied last: 0.02 is "
          "about +5 on a 0..255 scale. Normalised so one config reads "
          "the same against u8 and f32 frames",
   .def_real = 0.0},
  {.key = "contrast", .type = ConfigType::Real, .required = false,
   .doc = "gain about `pivot`, applied after gamma: 1.0 leaves the "
          "picture alone, >1 expands the range and <1 compresses it. "
          "Pivoting rather than scaling from black is what keeps a "
          "contrast change from also shifting the overall level",
   .def_real = 1.0},
  {.key = "gamma", .type = ConfigType::Real, .required = false,
   .doc = "exponent applied FIRST, on the normalised sample: > 1 darkens "
          "the midtones, < 1 lifts them, and either leaves black and "
          "white where they are. Must be > 0",
   .def_real = 1.0},
  {.key = "pivot", .type = ConfigType::Real, .required = false,
   .doc = "the normalised level `contrast` pivots about; 0.5 is mid "
          "grey. Set it to the level you want held FIXED -- a dark clip "
          "graded about 0.5 loses its shadows before its highlights "
          "move",
   .def_real = 0.5},
};

const StageSpec kSpec = {
  .type_name = "image-levels",
  .doc       = "Tone adjustment for planar RGB frames: gamma, then "
               "contrast about a pivot, then brightness. Geometry is "
               "untouched -- the tonal sibling of image-resample. Every "
               "knob is in normalised [0,1] units whatever the frame's "
               "dtype, and the defaults are the identity.",
  .display_name = "Image Levels",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

const StageSpec&
ImageLevelsStage::spec() const noexcept
{
  return kSpec;
}

Job
ImageLevelsStage::process(RuntimeContext& ctx)
{
  auto in0 = co_await ctx.read(0);
  if (!in0) { ctx.signal_done(); co_return; }

  const auto* tin = dynamic_cast<const TensorBeatPayload*>(in0.get());
  if (tin == nullptr || tin->shape.size() != 3 || tin->shape[0] != 3
      || (tin->dtype != TensorBeat::DType::U8
          && tin->dtype != TensorBeat::DType::F32)) {
    session()->warn(fmt(
        "ImageLevelsStage('{}'): expected planar RGB [3,H,W] u8/f32 "
        "TensorBeat; dropping beat", this->id()));
    co_return;
  }
  // Every knob at its identity: forward the beat untouched rather than
  // pay a pass over the picture to write back what was already there.
  if (_identity) {
    co_await ctx.write(0, std::move(in0));
    co_return;
  }

  const size_t n = tin->element_count();
  TensorBeat tb;
  tb.dtype    = tin->dtype;
  tb.shape    = tin->shape;
  tb.sideband = tin->sideband;
  tb.resize_contiguous(n);

  if (tin->dtype == TensorBeat::DType::U8) {
    // A 256-entry LUT: the curve is per-sample and independent of
    // position, so it is the same eight arithmetic ops 500k times or one
    // table lookup. Build it once per beat rather than once per stage
    // because the knobs are config and the table is 256 bytes.
    std::uint8_t lut[256];
    for (int i = 0; i < 256; ++i) {
      const double y = map_unit((double)i / 255.0);
      lut[i] = (std::uint8_t)std::lround(y * 255.0);
    }
    const std::uint8_t* src = tin->as_u8();
    std::uint8_t*       dst = tb.as_u8();
    for (size_t i = 0; i < n; ++i) { dst[i] = lut[src[i]]; }
  } else {
    const float* src = tin->as_f32();
    float*       dst = tb.as_f32();
    for (size_t i = 0; i < n; ++i) {
      dst[i] = (float)map_unit((double)src[i]);
    }
  }
  co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
}

VPIPE_REGISTER_STAGE(ImageLevelsStage)

}  // namespace vpipe
