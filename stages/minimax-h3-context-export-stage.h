#ifndef VPIPE_STAGES_MINIMAX_H3_CONTEXT_EXPORT_STAGE_H
#define VPIPE_STAGES_MINIMAX_H3_CONTEXT_EXPORT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Sink: saves a MiniMax-H3 clip's SAMPLED latents to a context file, so a
// later launch can continue the clip with `minimax-h3-context-import`.
//
// It stores the latents exactly as `generate-video` emitted them -- no
// decode, no re-encode, which is what keeps a chain from losing a little
// picture and sound at every join -- and the WHOLE clip rather than a
// tail, so the context length is chosen when importing and the clip can
// still be decoded later. The format is one safetensors file (see
// stages/minimax-h3-context.h): "video" F32 [24, T, H/16, W/16], "audio"
// F32 [2, 32, A] when wired, and string metadata.
//
//   iport0  latent        generate-video oport0 (video latent)
//   iport1  audio_latent  OPTIONAL generate-video oport1 (audio latent)
//
//   no oports (sink). Fan the generate-video outputs out to the decoders
//   and to this stage.
//
// Config:
//   output_url          (path, required) -- the file to write. When one
//                                           launch emits several clips, the
//                                           later ones get a "-%06u" suffix
//                                           before the extension, as
//                                           save-image does.
//   overwrite_existing  (bool, true)     -- false refuses to replace an
//                                           existing file (the clip is then
//                                           not saved).
class MiniMaxH3ContextExportStage final
  : public TypedStage<MiniMaxH3ContextExportStage> {
public:
  static constexpr const char* kTypeName = "minimax-h3-context-export";

  MiniMaxH3ContextExportStage(const SessionContextIntf* session, std::string id,
                              std::vector<InEdge> iports, FlexData config);

  // Per-launch reset -- see the comment on the definition.
  void reset_run_state() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;


  // The path clip `index` (0-based) of a launch is written to.
  static std::string path_for_clip(const std::string& base,
                                   std::uint64_t index);

private:
  std::string   _output_url;
  bool          _overwrite = true;
  std::uint64_t _clips     = 0;   // per run: the filename index
};

}  // namespace vpipe

#endif
