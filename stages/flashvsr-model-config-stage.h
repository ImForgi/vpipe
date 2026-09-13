#ifndef VPIPE_STAGES_FLASHVSR_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_FLASHVSR_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source: FlashVSR's own parameters -- the locality-constrained routing
// and the key/value window the denoiser attends across chunks.
//
// THE WINDOW IS THE MEMORY KNOB. `kv_ratio` is how many temporal windows
// of earlier chunks each block keeps, and the window is the largest
// allocation the model makes above ~512 px: at 1920x1152 it is 9.6 GB at
// kv_ratio 2 and 12.7 GB at the reference's 3, against a 2.9 GB
// checkpoint. A graph that has to fit a box sets it here, and the memory
// plan reads the same beat -- the denoise scratch is declared at the
// window this source asks for, not at the default.
//
// Named for the FAMILY, the tag the consuming stage resolves from the
// checkpoint, like every source beside it; see model-config-source.h for
// the beat contract and the trigger rule.
//
// Configuration (FlexData object): kv_ratio, sparse_ratio, local_range.
// Each is emitted ONLY when set, so an unset key keeps the reference's
// own default rather than a number that merely looks chosen.
class FlashVsrModelConfigStage final
  : public ModelConfigSourceStage<FlashVsrModelConfigStage> {
public:
  static constexpr const char* kTypeName = "flashvsr-model-config";

  FlashVsrModelConfigStage(const SessionContextIntf* session,
                           std::string               id,
                           std::vector<InEdge>       iports,
                           FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;
};

}  // namespace vpipe

#endif
