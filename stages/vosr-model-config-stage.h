#ifndef VPIPE_STAGES_VOSR_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_VOSR_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source: the VOSR-specific parameters of a restoration graph.
//
// Two keys, and they are the restorer's alone: how the one-step DiT is
// SPLIT over a picture bigger than the grid its weights were distilled
// at. Nothing else in this tree tiles -- the generative DiTs run their
// whole latent in one pass -- so these sat on `generate-image` as two
// keys inert on every other family it serves, which is the shape
// stages/model-config-source.h exists to undo.
//
// NAMED FOR THE FAMILY, not the checkpoint generation: "vosr", the tag
// the consuming stage resolves from the weights, so a VOSR 3 would read
// the same source rather than needing a second one. See the note on
// model_config::kFamilyKey.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
//
// Configuration (FlexData object): tile_size, tile_overlap. Each is
// emitted ONLY when set, because UNSET is a real answer here and not a
// missing one -- it means "tile at the resolution this checkpoint was
// distilled at", which is what the reference does and what a graph
// should get without asking.
class VosrModelConfigStage final
  : public ModelConfigSourceStage<VosrModelConfigStage> {
public:
  static constexpr const char* kTypeName = "vosr-model-config";

  VosrModelConfigStage(const SessionContextIntf* session,
                       std::string               id,
                       std::vector<InEdge>       iports,
                       FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;
};

}  // namespace vpipe

#endif
