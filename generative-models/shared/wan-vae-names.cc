#include "generative-models/shared/wan-vae-names.h"

#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vpipe {
namespace genai {
namespace wan_vae {

namespace {

bool
starts_with_(const std::string& s, std::string_view p)
{
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// "encoder.downsamples.3.residual.2.weight" -> the parts after the
// module list's index: {"residual", "2", "weight"} joined again as
// "residual.2.weight". Empty when `name` has no such tail.
std::string
tail_after_(const std::string& name, std::size_t field)
{
  std::size_t i = 0;
  for (std::size_t f = 0; f < field; ++f) {
    i = name.find('.', i);
    if (i == std::string::npos) { return {}; }
    ++i;
  }
  return name.substr(i);
}

std::string
field_(const std::string& name, std::size_t field)
{
  const std::string t = tail_after_(name, field);
  if (t.empty()) { return {}; }
  const std::size_t d = t.find('.');
  return (d == std::string::npos) ? t : t.substr(0, d);
}

bool
to_index_(const std::string& s, int& out)
{
  if (s.empty()) { return false; }
  int v = 0;
  for (const char c : s) {
    if (c < '0' || c > '9') { return false; }
    v = v * 10 + (c - '0');
  }
  out = v;
  return true;
}

// A resnet's tail: the native `nn.Sequential` positions against the
// diffusers member names. The gap (1, 4, 5) is the SiLU and dropout the
// checkpoint does not store.
std::string
resnet_tail_(const std::string& tail)
{
  static const std::map<std::string, std::string> kRes = {
    {"0.gamma",  "norm1.gamma"},
    {"2.weight", "conv1.weight"},
    {"2.bias",   "conv1.bias"},
    {"3.gamma",  "norm2.gamma"},
    {"6.weight", "conv2.weight"},
    {"6.bias",   "conv2.bias"},
  };
  if (starts_with_(tail, "residual.")) {
    const auto it = kRes.find(tail.substr(9));
    return (it == kRes.end()) ? std::string() : it->second;
  }
  if (starts_with_(tail, "shortcut.")) {
    return "conv_shortcut." + tail.substr(9);
  }
  return {};
}

// Whether an entry of a `downsamples` / `upsamples` module list is the
// stage's RESAMPLER rather than one of its resnets. This is the whole
// grouping rule: a resampler ends the up_block it terminates.
bool
is_resample_tail_(const std::string& tail)
{
  return starts_with_(tail, "resample.") || starts_with_(tail, "time_conv.");
}

// `middle.N` -> mid_block.{resnets.K | attentions.0}, by what the entry
// HOLDS rather than by its index: a residual is the next resnet, a
// norm/to_qkv/proj is the attention.
struct MiddlePlan {
  std::map<int, std::string> role;   // native index -> diffusers infix
};

// `upsamples.N` -> up_blocks.B.{resnets.R | upsamplers.0}.
struct UpPlan {
  std::map<int, std::string> role;
};

}  // namespace

NameLayout
detect_layout(const std::vector<std::string>& names)
{
  bool diffusers = false;
  bool native    = false;
  for (const std::string& n : names) {
    if (n == "decoder.conv_in.weight") { diffusers = true; }
    if (n == "decoder.conv1.weight")   { native    = true; }
  }
  // Both landmarks at once is not a layout, it is a mixed directory --
  // and mapping it would silently pick one of two decoders.
  if (diffusers && native) { return NameLayout::kUnknown; }
  if (diffusers)           { return NameLayout::kDiffusers; }
  if (native)              { return NameLayout::kNative; }
  return NameLayout::kUnknown;
}

bool
build_name_map(const std::vector<std::string>&               names,
               std::unordered_map<std::string, std::string>& out,
               std::string*                                  err)
{
  out.clear();
  const NameLayout layout = detect_layout(names);
  if (layout != NameLayout::kNative) {
    // Nothing to translate, and nothing to refuse: a diffusers or an
    // unrecognised checkpoint is handed to the loader as it always was.
    return true;
  }

  // ---- pass 1: group the two flattened module lists -------------------
  //
  // Both passes walk the indices in NUMERIC order, which is why they are
  // gathered into a std::map first: the names arrive in whatever order
  // the safetensors header held them, and "upsamples.10" sorts before
  // "upsamples.2" as text.
  std::map<std::string, MiddlePlan> middles;   // "encoder" / "decoder"
  std::map<int, bool> up_is_resample;          // decoder.upsamples
  std::map<std::string, std::map<int, bool>> middle_is_res;

  for (const std::string& n : names) {
    const std::string sec  = field_(n, 0);
    const std::string list = field_(n, 1);
    if (sec != "encoder" && sec != "decoder") { continue; }
    int idx = 0;
    if (!to_index_(field_(n, 2), idx)) { continue; }
    const std::string tail = tail_after_(n, 3);
    if (list == "middle") {
      // OR-accumulated for the same reason the upsamples flag below is:
      // one entry contributes several tensors and only some of them
      // carry the marker.
      bool& res = middle_is_res[sec][idx];
      res = res || starts_with_(tail, "residual.");
    } else if (list == "upsamples" && sec == "decoder") {
      // An index is a resampler if ANY of its tensors says so; the
      // insert-if-absent keeps a `time_conv` from being overwritten by
      // the `resample.1` beside it.
      const bool r = is_resample_tail_(tail);
      auto it = up_is_resample.find(idx);
      if (it == up_is_resample.end()) { up_is_resample[idx] = r; }
      else                            { it->second = it->second || r; }
    }
  }

  // THE GROUPING READS THE WHOLE LIST, so a list with a hole in it is
  // not a checkpoint this can map -- the entries after the hole would
  // shift into the wrong block and load, at the wrong depth, tensors of
  // exactly the right shape. Both positional lists are therefore
  // required to be 0..N-1 with nothing missing. (The encoder's
  // `downsamples` is exempt: diffusers keeps it flat under the SAME
  // indices, so a gap there maps a gap and cannot shift anything.)
  auto contiguous_ = [](const auto& m) {
    int want = 0;
    for (const auto& kv : m) {
      if (kv.first != want++) { return false; }
    }
    return true;
  };
  for (const auto& [sec, flags] : middle_is_res) {
    if (!contiguous_(flags)) {
      if (err != nullptr) {
        *err = "native Wan/Qwen-Image VAE '" + sec +
               ".middle' indices are not 0..N-1 -- the checkpoint is "
               "incomplete and its blocks cannot be grouped";
      }
      return false;
    }
    int resnet = 0;
    for (const auto& [idx, is_res] : flags) {
      middles[sec].role[idx] = is_res
          ? ("resnets." + std::to_string(resnet++))
          : "attentions.0";
    }
  }
  if (!contiguous_(up_is_resample)) {
    if (err != nullptr) {
      *err = "native Wan/Qwen-Image VAE 'decoder.upsamples' indices are "
             "not 0..N-1 -- the checkpoint is incomplete and its "
             "up_blocks cannot be grouped";
    }
    return false;
  }

  UpPlan up;
  {
    int block  = 0;
    int resnet = 0;
    for (const auto& [idx, is_resample] : up_is_resample) {
      if (is_resample) {
        up.role[idx] = "up_blocks." + std::to_string(block) +
                       ".upsamplers.0";
        ++block;
        resnet = 0;
      } else {
        up.role[idx] = "up_blocks." + std::to_string(block) + ".resnets." +
                       std::to_string(resnet++);
      }
    }
  }

  // ---- pass 2: map every name, forward ---------------------------------
  std::unordered_map<std::string, std::string> fwd;   // native -> diffusers
  std::unordered_set<std::string> taken;
  for (const std::string& n : names) {
    std::string d;
    const std::string p0 = field_(n, 0);
    if (p0 == "conv1" || p0 == "conv2") {
      // The latent bottleneck, and the one place the native spelling
      // REUSES a name that means something else one level down
      // (`decoder.conv1` is the decoder's input conv).
      d = (p0 == "conv1" ? "quant_conv." : "post_quant_conv.") +
          tail_after_(n, 1);
    } else if (p0 == "encoder" || p0 == "decoder") {
      const std::string list = field_(n, 1);
      if (list == "conv1") {
        d = p0 + ".conv_in." + tail_after_(n, 2);
      } else if (list == "head") {
        // head.0 = the output RMS norm, head.2 = the output conv; head.1
        // is the SiLU between them and stores nothing.
        const std::string which = field_(n, 2);
        if (which == "0")      { d = p0 + ".norm_out.gamma"; }
        else if (which == "2") { d = p0 + ".conv_out." + tail_after_(n, 3); }
      } else if (list == "middle") {
        int idx = 0;
        const auto sit = middles.find(p0);
        if (to_index_(field_(n, 2), idx) && sit != middles.end()) {
          const auto rit = sit->second.role.find(idx);
          if (rit != sit->second.role.end()) {
            const std::string tail = tail_after_(n, 3);
            const std::string rt   = resnet_tail_(tail);
            // A RESNET entry must have a recognised member; only the
            // attention passes its tail through. Letting an unknown
            // `residual.N` fall through to the tail spelled a diffusers
            // name nothing reads, which is a tensor silently dropped.
            const bool is_res = rit->second.rfind("resnets.", 0) == 0;
            if (!is_res || !rt.empty()) {
              d = p0 + ".mid_block." + rit->second + "." +
                  (rt.empty() ? tail : rt);
            }
          }
        }
      } else if (list == "downsamples" && p0 == "encoder") {
        // The encoder's list is NOT regrouped -- diffusers keeps it flat
        // under the same indices, so only the resnet member names move.
        int idx = 0;
        if (to_index_(field_(n, 2), idx)) {
          const std::string tail = tail_after_(n, 3);
          const std::string rt   = resnet_tail_(tail);
          // Same rule as the decoder's: only a resampler's tail passes
          // through unchanged.
          if (!rt.empty() || is_resample_tail_(tail)) {
            d = p0 + ".down_blocks." + std::to_string(idx) + "." +
                (rt.empty() ? tail : rt);
          }
        }
      } else if (list == "upsamples" && p0 == "decoder") {
        int idx = 0;
        if (to_index_(field_(n, 2), idx)) {
          const auto rit = up.role.find(idx);
          if (rit != up.role.end()) {
            const std::string tail = tail_after_(n, 3);
            const std::string rt   = resnet_tail_(tail);
            const bool is_up = rit->second.find(".upsamplers.") !=
                               std::string::npos;
            if (!rt.empty() || (is_up && is_resample_tail_(tail))) {
              d = p0 + "." + rit->second + "." + (rt.empty() ? tail : rt);
            }
          }
        }
      }
    }
    if (d.empty()) {
      if (err != nullptr) {
        *err = "native Wan/Qwen-Image VAE tensor '" + n +
               "' has no diffusers spelling under this rule";
      }
      out.clear();
      return false;
    }
    if (!taken.insert(d).second) {
      // Two natives onto one diffusers name. Impossible to detect later:
      // the loser is simply never read and its stage silently keeps the
      // winner's weights.
      if (err != nullptr) {
        *err = "native Wan/Qwen-Image VAE tensors collide on '" + d +
               "' (from '" + n + "') -- the name map is not one-to-one";
      }
      out.clear();
      return false;
    }
    fwd.emplace(n, std::move(d));
  }

  out.reserve(fwd.size());
  for (auto& [native, diffusers] : fwd) { out.emplace(diffusers, native); }
  return true;
}

}  // namespace wan_vae
}  // namespace genai
}  // namespace vpipe
