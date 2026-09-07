#ifndef GENERATIVE_MODELS_SHARED_WAN_VAE_NAMES_H
#define GENERATIVE_MODELS_SHARED_WAN_VAE_NAMES_H

// Two spellings of ONE VAE.
//
// The Wan 2.1 / Qwen-Image 3D causal-conv VAE ships under two different
// sets of tensor names for the same architecture:
//
//   DIFFUSERS  (AutoencoderKLWan, AutoencoderKLQwenImage) -- what
//     krea/Krea-2-*, Qwen/Qwen-Image and Wan-AI/* publish under `vae/`,
//     and what every loader in this tree reads:
//       decoder.conv_in / decoder.up_blocks.1.resnets.0.conv1 /
//       decoder.mid_block.attentions.0.to_qkv / quant_conv
//
//   NATIVE  -- the upstream research spelling, which is also what
//     ComfyUI's single-file `*_vae.safetensors` carry:
//       decoder.conv1 / decoder.upsamples.4.residual.2 /
//       decoder.middle.1.to_qkv / conv1
//
// The two are a BIJECTION over the whole checkpoint -- same 194 tensors,
// same shapes, same values -- so a native-named file is readable by the
// diffusers loaders once the names are translated. That is all this
// header does.
//
// WHY IT IS STRUCTURAL AND NOT A TABLE. The native form flattens the
// decoder's blocks into one `upsamples` module list, and the diffusers
// form groups them into `up_blocks.B.resnets.R` + `up_blocks.B.
// upsamplers.0`. The grouping is recovered from the list itself -- an
// entry carrying `resample`/`time_conv` is the block's upsampler and
// ENDS it, an entry carrying `residual`/`shortcut` is the next resnet in
// the current block -- rather than from an arithmetic stride over a
// configured block count, so a checkpoint with a different width or
// depth maps by the same rule.
//
// A WRONG INDEX HERE IS SILENT: every tensor at a given level has the
// same shape, so a mis-grouped resnet loads cleanly and decodes into a
// plausible, wrong image. build_name_map() therefore refuses anything it
// cannot map ONE-TO-ONE and says which name defeated it, rather than
// returning a partial map.

#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe {
namespace genai {
namespace wan_vae {

// The layout a checkpoint's tensor names are written in.
enum class NameLayout {
  kUnknown,    // neither spelling's landmark is present
  kDiffusers,  // decoder.conv_in.weight -- read directly, no map
  kNative,     // decoder.conv1.weight
};

NameLayout detect_layout(const std::vector<std::string>& names);

// DIFFUSERS name -> the spelling `names` actually uses.
//
// EMPTY (and true) for a diffusers checkpoint: the caller's names are
// already right, and an identity map of 194 entries would only be a
// slower way to say so. Callers therefore treat "not in the map" as
// "use the name as given" -- see wan_vae::resolve().
//
// False, with `err` set, when the layout is native and some tensor in it
// falls outside the rule. Partial maps are not returned.
bool build_name_map(const std::vector<std::string>&               names,
                    std::unordered_map<std::string, std::string>& out,
                    std::string*                                  err);

// The checkpoint's spelling of `diffusers_name` under `map`.
inline const std::string&
resolve(const std::unordered_map<std::string, std::string>& map,
        const std::string&                                  diffusers_name)
{
  const auto it = map.find(diffusers_name);
  return (it == map.end()) ? diffusers_name : it->second;
}

}  // namespace wan_vae
}  // namespace genai
}  // namespace vpipe

#endif  // GENERATIVE_MODELS_SHARED_WAN_VAE_NAMES_H
