#ifndef GENERATIVE_MODELS_VOSR_METAL_DINOV2_ENCODER_H
#define GENERATIVE_MODELS_VOSR_METAL_DINOV2_ENCODER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;   // generative-models/weight-set.h

// DINOv2 ViT-L/14, the VISION-ONLY conditioner VOSR cross-attends to.
//
// A plain pre-norm ViT: LayerNorm (affine), fused-qkv full attention
// (head_dim 64, NO rope and NO qk-norm), a GELU-erf MLP, and a LayerScale
// gamma on each residual. 24 blocks at 1024 wide -- but VOSR reads ONE
// intermediate layer (`out_layer`, 17 in the shipped config) and never
// the final norm, so this runs blocks [0, out_layer] and loads no more
// than that. The tail is dead weight for this use and is not read.
//
// THE INPUT IS SQUASHED TO A SQUARE, which is the reference's own
// preprocessing and not a simplification: it resizes to (size, size)
// with torch's bicubic, ignoring aspect ratio, so a wide picture reaches
// the tower stretched. Reproducing that matters -- the tower is the only
// thing conditioning the restoration, and a different resample is a
// different conditioning signal.
//
// Weight names follow the HF `Dinov2Model` layout (facebook/dinov2-large),
// which is the fetchable form of the same checkpoint torch.hub serves as
// a pickle. The original facebookresearch/dinov2 names are accepted too,
// so a locally converted hub checkpoint loads without a rename pass.
class MetalDinov2Encoder {
 public:
  struct Config {
    int   depth     = 24;
    int   hidden    = 1024;
    int   n_heads   = 16;
    int   head_dim  = 64;
    int   ffn       = 4096;
    int   patch     = 14;
    // sqrt of the position-embedding grid the checkpoint was trained at
    // (518 / 14 = 37). The grid is bicubic-interpolated to whatever grid
    // the request's `size` implies.
    int   pos_grid  = 37;
    float norm_eps  = 1e-6f;
    // DINOv2 interpolates its position grid at scale (out + offset) / M
    // rather than out / M, and the 0.1 is load-bearing: it is what the
    // reference passes, and dropping it shifts every position embedding
    // by a third of a cell.
    float interp_offset = 0.1f;
    // The block whose OUTPUT is the conditioning. Blocks after it are
    // never built. -1 means "all of them, then the final norm".
    int   out_layer = 17;
  };

  // Read `cfg` from an HF Dinov2 config.json in `dir`, leaving any key it
  // does not carry at the struct default. False when there is no readable
  // config (the caller may still load with defaults).
  static bool read_config(const std::string& dir, Config* cfg);

  static std::unique_ptr<MetalDinov2Encoder>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg);

  // Preferred: the manager's shared view of the checkpoint. The returned
  // tower KEEPS the set -- its weights are aliases of buffers it owns.
  static std::unique_ptr<MetalDinov2Encoder>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg);

  ~MetalDinov2Encoder();

  // Encode a planar U8 RGB image [3,H,W] (load-image format) into
  // conditioning tokens: bicubic resize to `size` x `size` the way
  // torch's F.interpolate does it, ImageNet normalisation, patchify,
  // blocks [0, out_layer]. The CLS token is dropped, as the reference
  // drops it. Returns [(size/patch)^2, hidden] bf16 and sets `n_tok`.
  // Empty on failure.
  metal_compute::SharedBuffer
  encode_rgb(const std::uint8_t* rgb, int H, int W, int size, int* n_tok);

  const Config& config() const noexcept { return _cfg; }

  // Bytes this tower holds, for the stage's memory accounting.
  std::uint64_t resident_bytes() const noexcept { return _bytes; }

  // Tokens `encode_rgb` will emit at this input size, so a caller can
  // shape a beat before running the tower.
  int tokens_for(int size) const noexcept
  {
    const int g = size / _cfg.patch;
    return g > 0 ? g * g : 0;
  }

 private:
  MetalDinov2Encoder() = default;

  struct Block {
    metal_compute::SharedBuffer n1_w, n1_b, n2_w, n2_b;      // LayerNorm
    metal_compute::SharedBuffer qkv_w, qkv_b;                // fused [3H,H]
    metal_compute::SharedBuffer proj_w, proj_b;
    metal_compute::SharedBuffer fc1_w, fc1_b, fc2_w, fc2_b;  // GELU MLP
    metal_compute::SharedBuffer ls1, ls2;                    // LayerScale
  };

  bool load_weights_(WeightSet& ws);
  // Build the position embedding for a `g` x `g` patch grid: the
  // checkpoint's 37x37 grid, bicubic-resampled at DINOv2's offset scale,
  // with the CLS row in front. [1 + g*g, hidden] bf16.
  metal_compute::SharedBuffer build_pos_(int g);

  metal_compute::MetalCompute* _mc = nullptr;
  Config                       _cfg;
  std::uint64_t                _bytes = 0;
  int                          _n_blocks = 0;   // blocks actually built

  metal_compute::SharedBuffer _patch_w, _patch_b;   // [H, 3*p*p], [H]
  metal_compute::SharedBuffer _cls;                 // [H]
  std::vector<float>          _pos;                 // f32 [1+M*M, H]
  std::vector<Block>          _blocks;
  // Set only when out_layer selects the last block: the reference then
  // reads the final LayerNorm's output rather than the raw block output.
  metal_compute::SharedBuffer _final_w, _final_b;
  bool                        _use_final_norm = false;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_sdpa;
  metal_compute::ComputeFunction _fn_gemm_bias, _fn_ln, _fn_gelu, _fn_sdpa,
      _fn_hslice, _fn_transpose, _fn_gated_residual, _fn_residual;

  std::shared_ptr<WeightSet> _ws;
};

}  // namespace genai
}  // namespace vpipe

#endif
