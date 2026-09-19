#ifndef VPIPE_STAGES_IMAGE_LEVELS_STAGE_H
#define VPIPE_STAGES_IMAGE_LEVELS_STAGE_H

#include "common/flex-data.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Tone adjustment for planar RGB image beats: gamma, then contrast about
// a pivot, then a brightness offset. Geometry is untouched, so this is
// the tonal sibling of image-resample and sits in the same graphs.
//
//   iport0  planar RGB TensorBeat [3,H,W] (u8 or f32), tag rgb-frames.
//   oport0  the same shape and dtype, tone-adjusted.
//
// EVERYTHING IS IN NORMALISED [0,1] UNITS whatever the dtype, so one
// config reads the same against u8 and f32 frames and a graph that
// switches dtype does not silently change its grading. A u8 frame is
// divided by 255 on the way in and rounded back on the way out.
//
// The order is fixed because it is the one that makes the knobs
// independent: gamma reshapes the response curve, contrast then pivots
// around a chosen grey rather than around whatever gamma left at the
// middle, and brightness finally slides the result. Applying brightness
// first would have contrast scale it too, so the two knobs would fight.
//
//   x = x ^ gamma
//   x = (x - pivot) * contrast + pivot
//   x = x + brightness
//
// Defaults are the identity (gamma 1, contrast 1, brightness 0), so the
// stage in a graph with no config is a copy.
class ImageLevelsStage final : public TypedStage<ImageLevelsStage> {
public:
  static constexpr const char* kTypeName = "image-levels";

  ImageLevelsStage(const SessionContextIntf* session,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test accessors.
  double brightness() const noexcept { return _brightness; }
  double contrast()   const noexcept { return _contrast; }
  double gamma()      const noexcept { return _gamma; }
  double pivot()      const noexcept { return _pivot; }

  // The mapping a single normalised sample takes, exposed so a test can
  // pin the curve without pushing a frame through the pipeline.
  double map_unit(double x) const noexcept;

private:
  double _brightness = 0.0;
  double _contrast   = 1.0;
  double _gamma      = 1.0;
  double _pivot      = 0.5;
  // True when every knob is at its identity, so process() can forward
  // the beat rather than rewrite every sample.
  bool   _identity   = true;
};

}  // namespace vpipe

#endif  // VPIPE_STAGES_IMAGE_LEVELS_STAGE_H
