#ifndef VPIPE_STAGES_LOAD_TENSOR_STAGE_H
#define VPIPE_STAGES_LOAD_TENSOR_STAGE_H

#include "common/flex-data.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Source: reads a tensor written by save-tensor and emits it as one
// TensorBeat, optionally sliced along one axis.
//
//   oport0  the TensorBeat, with the sideband the file carries plus
//           anything `sideband` adds.
//
// The SLICE is here rather than in a stage of its own because the thing
// a caller wants from a saved tensor is usually a window of it -- the
// TAIL of a video latent to condition the next clip on -- and a whole
// beat would otherwise be materialised only to be thrown away. `start`
// counts from the END when negative, the way temporal-slice's does, so
// "the last 5 latent frames" is `axis: 1, start: -5` and needs no
// knowledge of how long the file is.
//
// It refuses rather than clamps a slice that runs off the end: a clip
// conditioned on fewer frames than asked for is a different request,
// and silently shortening it is the kind of thing that shows up later
// as a quality question no one can trace.
class LoadTensorStage final : public TypedStage<LoadTensorStage> {
public:
  static constexpr const char* kTypeName = "load-tensor";

  LoadTensorStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Resolve a configured (start, count) against a concrete extent.
  // `start` < 0 counts from the end; `count` 0 means "to the end".
  // False when the window does not fit, with `err` saying why.
  static bool resolve_window(std::int64_t extent, std::int64_t start,
                             std::int64_t count, std::int64_t* out_start,
                             std::int64_t* out_count, std::string* err);

private:
  std::string  _path;
  std::int64_t _axis  = 0;
  std::int64_t _start = 0;
  std::int64_t _count = 0;
  FlexData     _extra_sideband;
  bool         _emitted = false;
};

}  // namespace vpipe

#endif  // VPIPE_STAGES_LOAD_TENSOR_STAGE_H
