#include "generative-models/vosr/metal-dinov2-encoder.h"

#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// ImageNet statistics, the normalisation DINOv2 was trained with and the
// one VOSR's preprocess_raw_image applies.
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

// C++ mirror of mlx::steel::AttnParams. Same layout as every other copy
// in this tree (metal-wan-transformer.cc, metal-krea2-transformer.cc);
// duplicated rather than shared for the same reason they are -- the
// struct is the kernel's ABI, not a utility.
struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};
constexpr int kSteelBQ = 32, kSteelBK = 16;

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// torch's bicubic convolution weights (A = -0.75), the kernel
// F.interpolate(mode="bicubic") uses. NOT Pillow's: Pillow's A is -0.5
// and its support widens on a downscale (that is its antialias). Torch
// does neither, so an image reaching this tower through a Pillow-exact
// resample is measurably not the reference's input.
inline void
cubic_weights_(float t, float w[4])
{
  constexpr float A = -0.75f;
  const float t2 = t * t;
  const float t3 = t2 * t;
  // c1(x) = ((A+2)x - (A+3))x^2 + 1 on |x| <= 1
  w[1] = (A + 2.0f) * t3 - (A + 3.0f) * t2 + 1.0f;
  const float s = 1.0f - t;
  w[2] = (A + 2.0f) * s * s * s - (A + 3.0f) * s * s + 1.0f;
  // c2(x) = ((Ax - 5A)x + 8A)x - 4A on 1 < |x| < 2
  const float a = t + 1.0f;
  w[0] = ((A * a - 5.0f * A) * a + 8.0f * A) * a - 4.0f * A;
  const float b = 2.0f - t;
  w[3] = ((A * b - 5.0f * A) * b + 8.0f * A) * b - 4.0f * A;
}

// Separable bicubic resample of a PLANAR f32 image [C,sh,sw] -> [C,dh,dw],
// align_corners=false and NO antialias, matching torch.
//
// `scale_y` / `scale_x` are SOURCE PIXELS PER DESTINATION PIXEL and are
// passed rather than derived, because the two callers derive them
// differently: an explicit output size gives in/out, while DINOv2's
// position grid is resampled by a scale_factor whose reciprocal is what
// torch then samples with. Deriving it here would silently drop the
// 0.1 offset.
void
resize_bicubic_(const float* src, int C, int sh, int sw, float* dst, int dh,
                int dw, double scale_y, double scale_x)
{
  std::vector<int>   bx((std::size_t)dw * 4);
  std::vector<float> wx((std::size_t)dw * 4);
  for (int x = 0; x < dw; ++x) {
    const double real = ((double)x + 0.5) * scale_x - 0.5;
    const int i0 = (int)std::floor(real);
    float w[4];
    cubic_weights_((float)(real - (double)i0), w);
    for (int t = 0; t < 4; ++t) {
      bx[(std::size_t)x * 4 + t] = std::min(sw - 1, std::max(0, i0 - 1 + t));
      wx[(std::size_t)x * 4 + t] = w[t];
    }
  }
  std::vector<int>   by((std::size_t)dh * 4);
  std::vector<float> wy((std::size_t)dh * 4);
  for (int y = 0; y < dh; ++y) {
    const double real = ((double)y + 0.5) * scale_y - 0.5;
    const int i0 = (int)std::floor(real);
    float w[4];
    cubic_weights_((float)(real - (double)i0), w);
    for (int t = 0; t < 4; ++t) {
      by[(std::size_t)y * 4 + t] = std::min(sh - 1, std::max(0, i0 - 1 + t));
      wy[(std::size_t)y * 4 + t] = w[t];
    }
  }
  std::vector<float> tmp((std::size_t)sh * dw);
  for (int c = 0; c < C; ++c) {
    const float* s = src + (std::size_t)c * sh * sw;
    for (int y = 0; y < sh; ++y) {
      const float* sr = s + (std::size_t)y * sw;
      for (int x = 0; x < dw; ++x) {
        const int* bi = &bx[(std::size_t)x * 4];
        const float* w = &wx[(std::size_t)x * 4];
        tmp[(std::size_t)y * dw + x] = w[0] * sr[bi[0]] + w[1] * sr[bi[1]]
                                     + w[2] * sr[bi[2]] + w[3] * sr[bi[3]];
      }
    }
    float* d = dst + (std::size_t)c * dh * dw;
    for (int y = 0; y < dh; ++y) {
      const int* bi = &by[(std::size_t)y * 4];
      const float* w = &wy[(std::size_t)y * 4];
      for (int x = 0; x < dw; ++x) {
        d[(std::size_t)y * dw + x] =
            w[0] * tmp[(std::size_t)bi[0] * dw + x]
          + w[1] * tmp[(std::size_t)bi[1] * dw + x]
          + w[2] * tmp[(std::size_t)bi[2] * dw + x]
          + w[3] * tmp[(std::size_t)bi[3] * dw + x];
      }
    }
  }
}

// The two naming schemes for one checkpoint. `hf` is what
// facebook/dinov2-large ships (a Dinov2Model state dict) and is the form
// model-fetch can get; `orig` is facebookresearch/dinov2's own, which is
// what a locally converted torch.hub pickle carries.
struct Names {
  bool hf = true;
  std::string cls() const
  {
    return hf ? "embeddings.cls_token" : "cls_token";
  }
  std::string pos() const
  {
    return hf ? "embeddings.position_embeddings" : "pos_embed";
  }
  std::string patch(const char* leaf) const
  {
    return hf ? std::string("embeddings.patch_embeddings.projection.") + leaf
              : std::string("patch_embed.proj.") + leaf;
  }
  std::string blk(int i, const char* leaf) const
  {
    return (hf ? "encoder.layer." : "blocks.") + std::to_string(i) + "."
           + leaf;
  }
  std::string final_norm(const char* leaf) const
  {
    return hf ? std::string("layernorm.") + leaf
              : std::string("norm.") + leaf;
  }
};

// A tensor as bf16, converted straight out of the checkpoint's bytes.
SharedBuffer
to_bf16_(WeightSet& ws, MetalCompute* mc, const std::string& nm)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  // Uncached: the raw tensor is consumed by the conversion and dropped.
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty()) { return {}; }
  if (info->dtype == "BF16") { return raw; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<std::uint16_t*>(out.contents());
  if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_(s[i]); }
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
  } else {
    return {};
  }
  return out;
}

// A tensor as host f32, whatever it is stored as. Used for the position
// grid, which is resampled on the host at full precision before it is
// rounded once into bf16.
bool
to_f32_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
        std::vector<float>* out)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr || info->shape.empty()) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty()) { return false; }
  out->resize(n);
  if (info->dtype == "F32") {
    std::memcpy(out->data(), raw.contents(), n * 4);
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)s[i] << 16;
      float f; std::memcpy(&f, &u, 4);
      (*out)[i] = f;
    }
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = (float)s[i]; }
  } else {
    return false;
  }
  return true;
}

}  // namespace

MetalDinov2Encoder::~MetalDinov2Encoder() = default;

bool
MetalDinov2Encoder::read_config(const std::string& dir, Config* cfg)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(dir) / "config.json");
  if (!in) { return false; }
  FlexData fd = FlexData::from_json(in);
  if (!fd.is_object()) { return false; }
  auto o = fd.as_object();
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_real((double)d) : d;
  };
  cfg->depth   = geti("num_hidden_layers", cfg->depth);
  cfg->hidden  = geti("hidden_size", cfg->hidden);
  cfg->n_heads = geti("num_attention_heads", cfg->n_heads);
  cfg->patch   = geti("patch_size", cfg->patch);
  const int mlp_ratio = geti("mlp_ratio", 4);
  cfg->ffn = cfg->hidden * mlp_ratio;
  if (cfg->n_heads > 0) { cfg->head_dim = cfg->hidden / cfg->n_heads; }
  const int img = geti("image_size", 518);
  if (cfg->patch > 0) { cfg->pos_grid = img / cfg->patch; }
  if (o.contains("layer_norm_eps")) {
    cfg->norm_eps = (float)o.at("layer_norm_eps").as_real(cfg->norm_eps);
  }
  // A SwiGLU DINOv2 (the giant) has a different MLP and this tower does
  // not build one; refuse rather than load half a model.
  if (o.contains("use_swiglu_ffn") && o.at("use_swiglu_ffn").as_bool(false)) {
    return false;
  }
  return true;
}

std::unique_ptr<MetalDinov2Encoder>
MetalDinov2Encoder::load(const std::string& dir, MetalCompute* mc,
                         const Config& cfg)
{
  auto ws = WeightSet::open(dir, nullptr);
  if (!ws) { return nullptr; }
  return load(std::move(ws), mc, cfg);
}

std::unique_ptr<MetalDinov2Encoder>
MetalDinov2Encoder::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                         const Config& cfg)
{
  if (!ws || mc == nullptr) { return nullptr; }
  std::unique_ptr<MetalDinov2Encoder> m(new MetalDinov2Encoder());
  m->_mc = mc;
  m->_cfg = cfg;
  m->_ws = std::move(ws);

  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_sdpa = mc->load_library("attn_steel");
  m->_fn_gemm_bias = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_ln        = m->_lib_elt.function("layer_norm_affine_f16");
  m->_fn_gelu      = m->_lib_elt.function("gelu_erf_f16");
  m->_fn_hslice    = m->_lib_elt.function("head_slice_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_gated_residual = m->_lib_elt.function("gated_residual_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  if (!m->_fn_gemm_bias.valid() || !m->_fn_ln.valid() ||
      !m->_fn_gelu.valid() || !m->_fn_hslice.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_gated_residual.valid() ||
      !m->_fn_residual.valid() || !m->_lib_sdpa.valid()) {
    return nullptr;
  }
  if (!m->load_weights_(*m->_ws)) { return nullptr; }
  return m;
}

bool
MetalDinov2Encoder::load_weights_(WeightSet& ws)
{
  Names nm;
  nm.hf = ws.has("embeddings.cls_token");
  if (!nm.hf && !ws.has("cls_token")) { return false; }

  const int H = _cfg.hidden;
  const int P = _cfg.patch;

  // The reference reads ONE intermediate layer, so everything past it is
  // never evaluated and is not worth the bytes. -1 asks for the whole
  // stack and the final norm with it.
  _use_final_norm = _cfg.out_layer < 0 || _cfg.out_layer >= _cfg.depth - 1;
  _n_blocks = (_cfg.out_layer < 0 || _cfg.out_layer >= _cfg.depth)
                  ? _cfg.depth
                  : _cfg.out_layer + 1;

  // patch embed: Conv2d [H, 3, P, P] IS a linear over the patch vector in
  // (c, ih, iw) order, which is how encode_rgb packs it.
  _patch_w = to_bf16_(ws, _mc, nm.patch("weight"));
  _patch_b = to_bf16_(ws, _mc, nm.patch("bias"));
  if (_patch_w.empty() || _patch_b.empty()) { return false; }
  if (_patch_w.byte_size() != (std::size_t)H * 3 * P * P * 2) { return false; }

  _cls = to_bf16_(ws, _mc, nm.cls());
  if (_cls.empty()) { return false; }
  if (!to_f32_(ws, _mc, nm.pos(), &_pos)) { return false; }
  const std::size_t want = (std::size_t)(1 + _cfg.pos_grid * _cfg.pos_grid) * H;
  if (_pos.size() != want) { return false; }

  _blocks.resize((std::size_t)_n_blocks);
  for (int i = 0; i < _n_blocks; ++i) {
    Block& b = _blocks[(std::size_t)i];
    b.n1_w = to_bf16_(ws, _mc, nm.blk(i, "norm1.weight"));
    b.n1_b = to_bf16_(ws, _mc, nm.blk(i, "norm1.bias"));
    b.n2_w = to_bf16_(ws, _mc, nm.blk(i, "norm2.weight"));
    b.n2_b = to_bf16_(ws, _mc, nm.blk(i, "norm2.bias"));
    if (nm.hf) {
      // Dinov2Model keeps q/k/v as three linears; the tower runs one
      // fused [3H, H] matmul, so they are concatenated here rather than
      // dispatched three times per block.
      const char* leaf[3] = {"attention.attention.query.",
                             "attention.attention.key.",
                             "attention.attention.value."};
      SharedBuffer w[3], bs[3];
      for (int j = 0; j < 3; ++j) {
        w[j] = to_bf16_(ws, _mc, nm.blk(i, (std::string(leaf[j]) + "weight")
                                              .c_str()));
        bs[j] = to_bf16_(ws, _mc, nm.blk(i, (std::string(leaf[j]) + "bias")
                                               .c_str()));
        if (w[j].empty() || bs[j].empty()) { return false; }
      }
      b.qkv_w = _mc->make_shared_buffer((std::size_t)3 * H * H * 2);
      b.qkv_b = _mc->make_shared_buffer((std::size_t)3 * H * 2);
      if (b.qkv_w.empty() || b.qkv_b.empty()) { return false; }
      auto* dw = static_cast<std::uint8_t*>(b.qkv_w.contents());
      auto* db = static_cast<std::uint8_t*>(b.qkv_b.contents());
      for (int j = 0; j < 3; ++j) {
        std::memcpy(dw + (std::size_t)j * H * H * 2, w[j].contents(),
                    (std::size_t)H * H * 2);
        std::memcpy(db + (std::size_t)j * H * 2, bs[j].contents(),
                    (std::size_t)H * 2);
      }
      b.proj_w = to_bf16_(ws, _mc, nm.blk(i, "attention.output.dense.weight"));
      b.proj_b = to_bf16_(ws, _mc, nm.blk(i, "attention.output.dense.bias"));
      b.ls1 = to_bf16_(ws, _mc, nm.blk(i, "layer_scale1.lambda1"));
      b.ls2 = to_bf16_(ws, _mc, nm.blk(i, "layer_scale2.lambda1"));
    } else {
      b.qkv_w = to_bf16_(ws, _mc, nm.blk(i, "attn.qkv.weight"));
      b.qkv_b = to_bf16_(ws, _mc, nm.blk(i, "attn.qkv.bias"));
      b.proj_w = to_bf16_(ws, _mc, nm.blk(i, "attn.proj.weight"));
      b.proj_b = to_bf16_(ws, _mc, nm.blk(i, "attn.proj.bias"));
      b.ls1 = to_bf16_(ws, _mc, nm.blk(i, "ls1.gamma"));
      b.ls2 = to_bf16_(ws, _mc, nm.blk(i, "ls2.gamma"));
    }
    b.fc1_w = to_bf16_(ws, _mc, nm.blk(i, "mlp.fc1.weight"));
    b.fc1_b = to_bf16_(ws, _mc, nm.blk(i, "mlp.fc1.bias"));
    b.fc2_w = to_bf16_(ws, _mc, nm.blk(i, "mlp.fc2.weight"));
    b.fc2_b = to_bf16_(ws, _mc, nm.blk(i, "mlp.fc2.bias"));
    if (b.n1_w.empty() || b.n1_b.empty() || b.n2_w.empty() ||
        b.n2_b.empty() || b.qkv_w.empty() || b.qkv_b.empty() ||
        b.proj_w.empty() || b.proj_b.empty() || b.fc1_w.empty() ||
        b.fc1_b.empty() || b.fc2_w.empty() || b.fc2_b.empty() ||
        b.ls1.empty() || b.ls2.empty()) {
      return false;
    }
    _bytes += b.n1_w.byte_size() + b.n1_b.byte_size() + b.n2_w.byte_size()
            + b.n2_b.byte_size() + b.qkv_w.byte_size() + b.qkv_b.byte_size()
            + b.proj_w.byte_size() + b.proj_b.byte_size()
            + b.fc1_w.byte_size() + b.fc1_b.byte_size()
            + b.fc2_w.byte_size() + b.fc2_b.byte_size()
            + b.ls1.byte_size() + b.ls2.byte_size();
  }
  if (_use_final_norm) {
    _final_w = to_bf16_(ws, _mc, nm.final_norm("weight"));
    _final_b = to_bf16_(ws, _mc, nm.final_norm("bias"));
    if (_final_w.empty() || _final_b.empty()) { return false; }
  }
  _bytes += _patch_w.byte_size() + _patch_b.byte_size() + _cls.byte_size();
  return true;
}

SharedBuffer
MetalDinov2Encoder::build_pos_(int g)
{
  const int H = _cfg.hidden, M = _cfg.pos_grid;
  const int n = g * g;
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)(1 + n) * H * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<std::uint16_t*>(out.contents());
  // CLS position, unchanged.
  for (int i = 0; i < H; ++i) { d[i] = f32_to_bf16_(_pos[(std::size_t)i]); }
  if (g == M) {
    for (int i = 0; i < n * H; ++i) {
      d[(std::size_t)H + i] = f32_to_bf16_(_pos[(std::size_t)H + i]);
    }
    return out;
  }
  // Channel-planar copy of the MxM grid, resampled to gxg. DINOv2 asks
  // torch for scale_factor (g + 0.1) / M rather than an output size, and
  // torch then samples at its reciprocal -- so the destination grid is
  // NOT g/M-spaced, and using g/M shifts every embedding.
  std::vector<float> src((std::size_t)H * M * M);
  for (int p = 0; p < M * M; ++p) {
    for (int c = 0; c < H; ++c) {
      src[(std::size_t)c * M * M + p] = _pos[(std::size_t)(1 + p) * H + c];
    }
  }
  std::vector<float> dst((std::size_t)H * n);
  const double sc = ((double)g + (double)_cfg.interp_offset) / (double)M;
  resize_bicubic_(src.data(), H, M, M, dst.data(), g, g, 1.0 / sc, 1.0 / sc);
  for (int p = 0; p < n; ++p) {
    for (int c = 0; c < H; ++c) {
      d[(std::size_t)(1 + p) * H + c] =
          f32_to_bf16_(dst[(std::size_t)c * n + p]);
    }
  }
  return out;
}

SharedBuffer
MetalDinov2Encoder::encode_rgb(const std::uint8_t* rgb, int H_in, int W_in,
                               int size, int* n_tok)
{
  if (n_tok != nullptr) { *n_tok = 0; }
  if (rgb == nullptr || H_in <= 0 || W_in <= 0 || size <= 0) { return {}; }
  const int P = _cfg.patch;
  const int g = size / P;
  if (g <= 0) { return {}; }
  const int n = g * g;                 // patch tokens
  const int seq = n + 1;               // + CLS
  const int H = _cfg.hidden, Hd = _cfg.head_dim, NH = _cfg.n_heads;
  const int FF = _cfg.ffn;
  const int sq = g * P;                // the size actually consumed

  // ---- preprocess: u8 -> [0,1] -> bicubic square -> clip -> normalise --
  //
  // The square is the reference's: it interpolates to (size, size) with
  // no regard for aspect ratio, so a 16:9 picture reaches the tower
  // stretched. Matching that is the point.
  std::vector<float> in((std::size_t)3 * H_in * W_in);
  for (std::size_t i = 0; i < in.size(); ++i) {
    in[i] = (float)rgb[i] * (1.0f / 255.0f);
  }
  std::vector<float> img((std::size_t)3 * sq * sq);
  resize_bicubic_(in.data(), 3, H_in, W_in, img.data(), sq, sq,
                  (double)H_in / (double)sq, (double)W_in / (double)sq);
  for (int c = 0; c < 3; ++c) {
    float* p = img.data() + (std::size_t)c * sq * sq;
    const float m = kMean[c], s = 1.0f / kStd[c];
    for (int i = 0; i < sq * sq; ++i) {
      p[i] = (std::min(1.0f, std::max(0.0f, p[i])) - m) * s;
    }
  }

  // ---- patchify into [n, 3*P*P], (c, ih, iw) order ---------------------
  const int PV = 3 * P * P;
  SharedBuffer pix = _mc->make_shared_buffer((std::size_t)n * PV * 2);
  if (pix.empty()) { return {}; }
  {
    auto* d = static_cast<std::uint16_t*>(pix.contents());
    for (int py = 0; py < g; ++py) {
      for (int px = 0; px < g; ++px) {
        std::uint16_t* row = d + (std::size_t)(py * g + px) * PV;
        for (int c = 0; c < 3; ++c) {
          const float* pl = img.data() + (std::size_t)c * sq * sq;
          for (int ih = 0; ih < P; ++ih) {
            const float* sr = pl + (std::size_t)(py * P + ih) * sq + px * P;
            std::uint16_t* dr = row + (std::size_t)c * P * P + ih * P;
            for (int iw = 0; iw < P; ++iw) {
              dr[iw] = f32_to_bf16_(sr[iw]);
            }
          }
        }
      }
    }
  }

  SharedBuffer pos = build_pos_(g);
  if (pos.empty()) { return {}; }

  // ---- steel flash, shaped for this sequence --------------------------
  const float scale = 1.0f / std::sqrt((float)Hd);
  SharedBuffer ap = _mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (ap.empty()) { return {}; }
  {
    auto* p = static_cast<SteelAttnParams*>(ap.contents());
    p->B = 1; p->H = NH; p->D = Hd;
    p->qL = seq; p->kL = seq;
    p->gqa_factor = 1; p->scale = scale;
    p->NQ = (seq + kSteelBQ - 1) / kSteelBQ;
    p->NK = (seq + kSteelBK - 1) / kSteelBK;
    p->NQ_aligned = seq / kSteelBQ;
    p->NK_aligned = seq / kSteelBK;
    p->qL_rem = seq - p->NQ_aligned * kSteelBQ;
    p->kL_rem = seq - p->NK_aligned * kSteelBK;
    p->qL_off = 0;
    p->Q_strides[0] = (std::int64_t)NH * seq * Hd;
    p->Q_strides[1] = (std::int64_t)seq * Hd;
    p->Q_strides[2] = Hd;
    for (int i = 0; i < 3; ++i) {
      p->K_strides[i] = p->Q_strides[i];
      p->V_strides[i] = p->Q_strides[i];
      p->O_strides[i] = p->Q_strides[i];
    }
  }
  metal_compute::FunctionConstants fc;
  fc.set_bool(200, (seq % kSteelBQ) == 0).set_bool(201, (seq % kSteelBK) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false);
  metal_compute::ComputeFunction fn_attn =
      _lib_sdpa.function("attn_steel_h_bd64_bf16", fc);
  if (!fn_attn.valid()) { return {}; }
  const unsigned nqb = (unsigned)((seq + kSteelBQ - 1) / kSteelBQ);

  auto buf = [&](std::size_t elems) {
    return _mc->make_shared_buffer(elems * 2);
  };
  SharedBuffer x = buf((std::size_t)seq * H);
  SharedBuffer nrm = buf((std::size_t)seq * H);
  SharedBuffer qkv = buf((std::size_t)seq * 3 * H);
  SharedBuffer q = buf((std::size_t)seq * H), k = buf((std::size_t)seq * H),
               v = buf((std::size_t)seq * H);
  SharedBuffer qt = buf((std::size_t)seq * H), kt = buf((std::size_t)seq * H),
               vt = buf((std::size_t)seq * H), at = buf((std::size_t)seq * H);
  SharedBuffer att = buf((std::size_t)seq * H), o = buf((std::size_t)seq * H);
  SharedBuffer ff = buf((std::size_t)seq * FF);
  if (x.empty() || qkv.empty() || ff.empty() || att.empty()) { return {}; }
  // CLS occupies row 0; the patch embedding gemm writes rows 1..n.
  std::memcpy(x.contents(), _cls.contents(), (std::size_t)H * 2);

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& xb, const SharedBuffer& w,
                    const SharedBuffer& bs, const SharedBuffer& y,
                    std::size_t yoff, int Mm, int N, int K) {
      enc.set_function(_fn_gemm_bias);
      enc.set_buffer(0, xb); enc.set_buffer(1, w);
      enc.set_buffer(2, bs.empty() ? w : bs);
      enc.set_buffer(3, y, yoff * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, Mm);
      enc.set_constant(7, bs.empty() ? 0 : 1);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((Mm + 63) / 64) * 2), 2}, {32, 2, 2});
    };
    auto ln = [&](const SharedBuffer& xb, const SharedBuffer& w,
                  const SharedBuffer& bs, const SharedBuffer& y) {
      enc.set_function(_fn_ln);
      enc.set_buffer(0, xb); enc.set_buffer(1, w); enc.set_buffer(2, bs);
      enc.set_buffer(3, y);
      enc.set_constant(4, H); enc.set_constant(5, _cfg.norm_eps);
      enc.dispatch({256, (unsigned)seq, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in_b, const SharedBuffer& out_b,
                      int off) {
      enc.set_function(_fn_hslice);
      enc.set_buffer(0, in_b); enc.set_buffer(1, out_b);
      enc.set_constant(2, seq); enc.set_constant(3, 3 * H);
      enc.set_constant(4, H); enc.set_constant(5, off);
      enc.set_constant(6, 0); enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(seq * H), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in_b, const SharedBuffer& out_b,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in_b); enc.set_buffer(1, out_b);
      enc.set_constant(2, A); enc.set_constant(3, Bd);
      enc.set_constant(4, Hd);
      enc.dispatch({(unsigned)Hd, (unsigned)Bd, (unsigned)A},
                   {(unsigned)Hd, 1, 1});
    };
    auto gated = [&](const SharedBuffer& h, const SharedBuffer& gate,
                     const SharedBuffer& sub) {
      enc.set_function(_fn_gated_residual);
      enc.set_buffer(0, h); enc.set_buffer(1, gate); enc.set_buffer(2, sub);
      enc.set_constant(3, H); enc.set_constant(4, seq * H);
      enc.dispatch({(unsigned)(seq * H), 1, 1}, {256, 1, 1});
    };

    gemm(pix, _patch_w, _patch_b, x, (std::size_t)H, n, H, PV);
    // + position embedding (CLS row included).
    enc.set_function(_fn_residual);
    enc.set_buffer(0, x); enc.set_buffer(1, pos); enc.set_buffer(2, x);
    enc.set_constant(3, seq * H);
    enc.dispatch({(unsigned)(seq * H), 1, 1}, {256, 1, 1});

    for (int L = 0; L < _n_blocks; ++L) {
      const Block& b = _blocks[(std::size_t)L];
      ln(x, b.n1_w, b.n1_b, nrm);
      gemm(nrm, b.qkv_w, b.qkv_b, qkv, 0, seq, 3 * H, H);
      hslice(qkv, q, 0); hslice(qkv, k, H); hslice(qkv, v, 2 * H);
      transpose(q, qt, seq, NH);
      transpose(k, kt, seq, NH);
      transpose(v, vt, seq, NH);
      enc.set_function(fn_attn);
      enc.set_buffer(0, qt); enc.set_buffer(1, kt); enc.set_buffer(2, vt);
      enc.set_buffer(3, at); enc.set_buffer(4, ap);
      enc.dispatch({32 * nqb, 4 * (unsigned)NH, 1}, {32, 4, 1});
      transpose(at, att, NH, seq);
      gemm(att, b.proj_w, b.proj_b, o, 0, seq, H, H);
      gated(x, b.ls1, o);
      ln(x, b.n2_w, b.n2_b, nrm);
      gemm(nrm, b.fc1_w, b.fc1_b, ff, 0, seq, FF, H);
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, ff); enc.set_buffer(1, ff);
      enc.set_constant(2, seq * FF);
      enc.dispatch({(unsigned)(seq * FF), 1, 1}, {256, 1, 1});
      gemm(ff, b.fc2_w, b.fc2_b, o, 0, seq, H, FF);
      gated(x, b.ls2, o);
    }
    if (_use_final_norm) {
      ln(x, _final_w, _final_b, nrm);
    }
  }
  std::string err;
  if (!stream.commit().wait_ok(&err)) { return {}; }

  // Drop the CLS row: the reference conditions on the patch tokens only.
  const SharedBuffer& srcb = _use_final_norm ? nrm : x;
  SharedBuffer out = buf((std::size_t)n * H);
  if (out.empty()) { return {}; }
  std::memcpy(out.contents(),
              static_cast<const std::uint8_t*>(srcb.contents())
                  + (std::size_t)H * 2,
              (std::size_t)n * H * 2);
  if (n_tok != nullptr) { *n_tok = n; }
  return out;
}

}  // namespace genai
}  // namespace vpipe
