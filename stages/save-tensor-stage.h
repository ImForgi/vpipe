#ifndef VPIPE_STAGES_SAVE_TENSOR_STAGE_H
#define VPIPE_STAGES_SAVE_TENSOR_STAGE_H

#include "common/flex-data.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Sink: writes TensorBeats to disk verbatim, so a tensor can outlive the
// run that made it.
//
//   iport0  any TensorBeat.
//
// WHY THIS EXISTS. A multi-part video graph hands each clip to the next,
// and doing that through a FILE costs a VAE decode, a codec, a decode
// and a VAE encode -- a round trip that is geometrically exact but not
// tonally (measured -1.7 luma on MiniMax-H3's video VAE), and which a
// latent does not need to take at all. Writing the latent instead lets
// the next run start from the same numbers.
//
// FORMAT: a 32-byte header then the raw payload, little-endian.
//
//   0  : "VPTENSOR"        8 bytes, magic
//   8  : version           u32, 1
//   12 : dtype             u32, TensorBeat::DType
//   16 : rank              u32, <= 8
//   20 : reserved          u32, 0
//   24 : element_count     u64
//   32 : dims              i64 * rank
//   .. : payload           element_count * element_size bytes
//   .. : sideband          the beat's sideband as JSON, when it has one
//
// Deliberately not safetensors: this is one unnamed tensor plus the
// sideband a beat carries, written and read by one pair of stages, and
// a format with a name table and a JSON header would be more to agree
// on rather than less. It IS self-describing enough to refuse a
// mismatched read, which is the property that matters.
class SaveTensorStage final : public TypedStage<SaveTensorStage> {
public:
  static constexpr const char* kTypeName = "save-tensor";

  SaveTensorStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test accessor: how many beats have been written.
  int written() const noexcept { return _written; }

private:
  std::string _path;
  bool        _number = false;   // one file per beat
  int         _written = 0;
};

}  // namespace vpipe

#endif  // VPIPE_STAGES_SAVE_TENSOR_STAGE_H
