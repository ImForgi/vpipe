#include "generative-models/wan/metal-wan-vae.h"

#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/wan-vae-names.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <deque>
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

// Namespace for this class's derived-tensor cache keys. A WeightSet is
// shared by everything reading one checkpoint, so a key has to say which
// class's transform produced the bytes, not just which tensor they came
// from. Deliberately DISTINCT from the Qwen-Image VAE's "krea2-vae/": the
// two read the same tensor names off compatible checkpoints but flatten
// them differently (27 taps here, the kt=2 slice there), so sharing a key
// would hand one class the other's bytes.
constexpr const char* kKey = "wan-vae/";

// Read a raw checkpoint tensor as float (F32/F16/BF16 source).
std::vector<float>
read_f32_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm,
          std::size_t& n_out)
{
  const auto* info = wts.info(nm);
  std::vector<float> v;
  if (info == nullptr || info->shape.empty()) { n_out = 0; return v; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { n_out = 0; return v; }
  v.resize(n);
  if (info->dtype == "F32") {
    std::memcpy(v.data(), raw.contents(), n * 4);
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)s[i]; }
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)s[i] << 16;
      float f; std::memcpy(&f, &u, 4); v[i] = f;
    }
  } else {
    n_out = 0; return {};
  }
  n_out = n;
  return v;
}

SharedBuffer
f16_buf_(MetalCompute* mc, const float* src, std::size_t n)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  if (b.empty()) { return {}; }
  auto* d = static_cast<_Float16*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)src[i]; }
  return b;
}

}  // namespace

// Everything one chunk's dispatches share. The buffer pool is the same
// trick the Qwen-Image VAE uses: the net is a serial feed-forward chain, so
// a released buffer is safe to reuse for a later op (serial dispatch orders
// the reuse after the last read), which bounds the live set to the
// concurrent working set rather than the whole chunk.
struct MetalWanVae::Ctx {
  ComputeEncoder* enc = nullptr;
  struct Slot { SharedBuffer buf; std::size_t cap; bool used; };
  std::deque<Slot> pool;
  bool        alloc_ok = true;
  bool        use_pool = true;
  SharedBuffer col;                  // shared im2col band scratch
  std::size_t  col_cap = 0;          // ELEMENTS

  SharedBuffer&
  alloc(MetalCompute* mc, std::size_t elems)
  {
    const std::size_t bytes = elems * 2;
    if (use_pool) {
      for (auto& s : pool) {
        if (!s.used && !s.buf.empty() && s.cap >= bytes) {
          s.used = true;
          return s.buf;
        }
      }
    }
    pool.push_back(Slot{mc->make_shared_buffer(bytes), bytes, true});
    if (pool.back().buf.empty()) { alloc_ok = false; }
    return pool.back().buf;
  }

  void
  release(const SharedBuffer& b)
  {
    if (!use_pool) { return; }
    for (auto& s : pool) {
      if (&s.buf == &b) { s.used = false; return; }
    }
  }

  // The band, sized to the conv that GATHERS rather than to the cap: the
  // temporal (3,1,1) conv needs 3*cin a row, where the cap is set by the
  // 27-tap one, and on the hardware conv it is the only conv that gathers
  // at all. Grows if a later conv needs more; never beyond col_cap.
  bool
  ensure_col(MetalCompute* mc, std::size_t elems)
  {
    elems = std::min(elems, col_cap);
    if (col.byte_size() >= elems * 2) { return true; }
    col = SharedBuffer{};
    col = mc->make_shared_buffer(elems * 2);
    if (col.empty()) { alloc_ok = false; return false; }
    return true;
  }
};

// ---- config ------------------------------------------------------------

bool
MetalWanVae::config_from_json(const std::string& vae_dir, Config& out,
                              std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  fs::path p(vae_dir);
  if (fs::is_directory(p) && !fs::exists(p / "config.json")) {
    p = p / "vae";
  }
  if (fs::is_directory(p)) { p = p / "config.json"; }
  std::ifstream f(p);
  if (!f) { return fail("cannot open " + p.string()); }
  FlexData cfg;
  try {
    cfg = FlexData::from_json(f);
  } catch (...) {
    return fail("cannot parse " + p.string());
  }
  if (!cfg.is_object()) { return fail(p.string() + " is not a JSON object"); }
  auto o = cfg.as_object();
  const std::string cls =
      o.contains("_class_name") ? std::string(o.at("_class_name").as_string(""))
                                : std::string();
  if (cls != "AutoencoderKLWan") {
    return fail("not an AutoencoderKLWan config (_class_name=" + cls + ")");
  }
  auto get_int = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  out.base_dim       = get_int("base_dim", 96);
  out.z_dim          = get_int("z_dim", 16);
  out.num_res_blocks = get_int("num_res_blocks", 2);
  // as_array() returns a VIEW into the owning FlexData, so each one is
  // bound to a named local first -- a temporary would dangle.
  if (o.contains("dim_mult")) {
    const FlexData dm = o.at("dim_mult");
    if (dm.is_array()) {
      auto a = dm.as_array();
      for (std::size_t i = 0; i < 4 && i < a.size(); ++i) {
        out.dim_mult[i] = (int)a.at(i).as_int(out.dim_mult[i]);
      }
    }
  }
  if (o.contains("temperal_downsample")) {
    const FlexData td = o.at("temperal_downsample");
    if (td.is_array()) {
      auto a = td.as_array();
      for (std::size_t i = 0; i < 3 && i < a.size(); ++i) {
        out.temperal_downsample[i] =
            a.at(i).as_bool(out.temperal_downsample[i]);
      }
    }
  }
  auto read_vec = [&](const char* k, std::vector<float>& dst) {
    if (!o.contains(k)) { return; }
    const FlexData v = o.at(k);
    if (!v.is_array()) { return; }
    auto a = v.as_array();
    dst.clear();
    for (std::size_t i = 0; i < a.size(); ++i) {
      dst.push_back((float)a.at(i).as_real(0.0));
    }
  };
  read_vec("latents_mean", out.latents_mean);
  read_vec("latents_std", out.latents_std);
  // A `patch_size` / `is_residual` config is the Wan 2.2 (16x) VAE, whose
  // residual down/up blocks are a different net -- refuse rather than load
  // its weights into this topology and produce noise.
  if (o.contains("is_residual") && o.at("is_residual").as_bool(false)) {
    return fail("is_residual (the Wan2.2 16x VAE) is not supported here");
  }
  return true;
}

bool
MetalWanVae::config_for_native_checkpoint(const std::string& path,
                                          Config& out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  auto w = MetalLlamaWeights::open_model(path);
  if (!w.has_value()) { return fail("no readable checkpoint at " + path); }
  const auto* q  = w->info("conv1.weight");           // the 2*z quant conv
  const auto* e1 = w->info("encoder.conv1.weight");   // [base, 3, 3, 3, 3]
  if (q == nullptr || e1 == nullptr || !w->has("decoder.conv1.weight") ||
      q->shape.size() != 5 || e1->shape.size() != 5) {
    return fail(path + " is not a natively-named Wan VAE");
  }
  if (q->shape[0] != 32 || e1->shape[0] != 96) {
    return fail(fmt("{} is a natively-named Wan VAE, but not 2.1's (z_dim "
                    "{}, base {})", path, q->shape[0] / 2, e1->shape[0])());
  }
  // The geometry is the struct's own default, which IS Wan 2.1's. The
  // statistics are the ones its reference module un-whitens with.
  Config c;
  c.latents_mean = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f,
                    0.9653f,  -0.1517f, 1.5508f,  0.4134f, -0.0715f,
                    0.5517f,  -0.3632f, -0.1922f, -0.9497f, 0.2503f,
                    -0.2921f};
  c.latents_std  = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f,
                    2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f,
                    1.6382f, 1.1253f, 2.8251f, 1.9160f};
  out = std::move(c);
  return true;
}

// ---- weight loading ----------------------------------------------------

// A causal conv3d [Cout,Cin,3,3,3] as a dense-GEMM weight [Cout, 27*Cin],
// flattened (kt,ky,kx,cin) to pair with im2col_hwc_3x3x3_tiled.
// TWO SPELLINGS OF ONE VAE. A natively-named checkpoint (FlashVSR ships
// `Wan2.1_VAE.pth`, ComfyUI ships `*_vae.safetensors`) carries the same
// 194 tensors under the upstream research names rather than the
// diffusers ones every loader here reads. shared/wan-vae-names.h is the
// structural bijection between them; this routes every name through it.
//
// The map is EMPTY for a diffusers checkpoint, so that path is unchanged
// and pays one failed hash lookup. The derived() cache keys deliberately
// stay in the DIFFUSERS spelling: they name a transform, which does not
// change with the file's naming.
const std::string&
MetalWanVae::wname_(const std::string& diffusers_name) const
{
  return wan_vae::resolve(_names, diffusers_name);
}

MetalWanVae::Conv
MetalWanVae::load_conv3d_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(wname_(nm + ".weight"));
  if (info == nullptr || info->shape.size() < 5) { return c; }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  const int kt = (int)sh[2], kh = (int)sh[3], kw = (int)sh[4];
  if (kt != 3 || kh != 3 || kw != 3) {
    // The only 5-D weights that are not 3x3x3 are the 1x1x1 shortcuts and
    // quant convs, which load through load_conv1x1_.
    return c;
  }
  c.cin = Cin; c.cout = Cout; c.kt = 3; c.ks = 9; c.k = 27 * Cin;
  c.w = ws.derived(std::string(kKey) + "c3d|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> w = read_f32_(ws.src(), _mc, wname_(nm + ".weight"), n);
    if (w.empty()) { return {}; }
    std::vector<float> flat((std::size_t)Cout * 27 * Cin, 0.0f);
    for (int o = 0; o < Cout; ++o) {
      for (int t = 0; t < 3; ++t) {
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            for (int i = 0; i < Cin; ++i) {
              const std::size_t si =
                  ((((std::size_t)o * Cin + i) * 3 + t) * 3 + ky) * 3 + kx;
              const std::size_t di =
                  ((std::size_t)o * 27 + ((t * 3 + ky) * 3 + kx)) * Cin + i;
              flat[di] = w[si];
            }
          }
        }
      }
    }
    return f16_buf_(_mc, flat.data(), flat.size());
  }, _part);
  if (c.w.empty()) { return Conv{}; }
  // The hardware conv's twin: one [3,3,Cin,Cout] HWIO block PER TEMPORAL
  // TAP, contiguous in tap order, so tap kt binds the twin at offset
  // kt * 9 * Cin * Cout (see conv3x3_hw_).
  if (_use_hwconv) {
    c.whwio = ws.derived(std::string(kKey) + "c3d-hwio-tap|" + nm,
                         [&]() -> SharedBuffer {
      std::size_t n = 0;
      std::vector<float> w =
          read_f32_(ws.src(), _mc, wname_(nm + ".weight"), n);
      if (w.empty()) { return {}; }
      std::vector<float> hwio((std::size_t)27 * Cin * Cout, 0.0f);
      for (int o = 0; o < Cout; ++o) {
        for (int i = 0; i < Cin; ++i) {
          for (int t = 0; t < 3; ++t) {
            for (int ky = 0; ky < 3; ++ky) {
              for (int kx = 0; kx < 3; ++kx) {
                const std::size_t si =
                    ((((std::size_t)o * Cin + i) * 3 + t) * 3 + ky) * 3 + kx;
                const std::size_t di =
                    ((((std::size_t)t * 3 + ky) * 3 + kx) * Cin + i) *
                        Cout + o;
                hwio[di] = w[si];
              }
            }
          }
        }
      }
      return f16_buf_(_mc, hwio.data(), hwio.size());
    }, _part);
  }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// A plain 2D conv [Cout,Cin,3,3] (the spatial half of a resample) as
// [Cout, 9*Cin], flattened (ky,kx,cin) to pair with im2col_hwc_3x3_tiled.
MetalWanVae::Conv
MetalWanVae::load_conv2d_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(wname_(nm + ".weight"));
  if (info == nullptr || info->shape.size() < 4) { return c; }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  c.cin = Cin; c.cout = Cout; c.kt = 1; c.ks = 9; c.k = 9 * Cin;
  c.w = ws.derived(std::string(kKey) + "c2d|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> w = read_f32_(ws.src(), _mc, wname_(nm + ".weight"), n);
    if (w.empty()) { return {}; }
    std::vector<float> flat((std::size_t)Cout * 9 * Cin, 0.0f);
    for (int o = 0; o < Cout; ++o) {
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          for (int i = 0; i < Cin; ++i) {
            const std::size_t si =
                (((std::size_t)o * Cin + i) * 3 + ky) * 3 + kx;
            const std::size_t di =
                ((std::size_t)o * 9 + (ky * 3 + kx)) * Cin + i;
            flat[di] = w[si];
          }
        }
      }
    }
    return f16_buf_(_mc, flat.data(), flat.size());
  }, _part);
  if (c.w.empty()) { return Conv{}; }
  if (_use_hwconv) {                    // HWIO twin, out-channel fastest
    c.whwio = ws.derived(std::string(kKey) + "c2d-hwio|" + nm,
                         [&]() -> SharedBuffer {
      std::size_t n = 0;
      std::vector<float> w =
          read_f32_(ws.src(), _mc, wname_(nm + ".weight"), n);
      if (w.empty()) { return {}; }
      std::vector<float> hwio((std::size_t)9 * Cin * Cout, 0.0f);
      for (int o = 0; o < Cout; ++o) {
        for (int i = 0; i < Cin; ++i) {
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              const std::size_t si =
                  (((std::size_t)o * Cin + i) * 3 + ky) * 3 + kx;
              const std::size_t di =
                  (((std::size_t)ky * 3 + kx) * Cin + i) * Cout + o;
              hwio[di] = w[si];
            }
          }
        }
      }
      return f16_buf_(_mc, hwio.data(), hwio.size());
    }, _part);
  }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// A temporal conv [Cout,Cin,3,1,1] as [Cout, 3*Cin], flattened (kt,cin) to
// pair with concat3_frames.
MetalWanVae::Conv
MetalWanVae::load_time_conv_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(wname_(nm + ".weight"));
  if (info == nullptr || info->shape.size() < 5) { return c; }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  if ((int)sh[2] != 3) { return c; }
  c.cin = Cin; c.cout = Cout; c.kt = 3; c.ks = 1; c.k = 3 * Cin;
  c.w = ws.derived(std::string(kKey) + "ct|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> w = read_f32_(ws.src(), _mc, wname_(nm + ".weight"), n);
    if (w.empty()) { return {}; }
    std::vector<float> flat((std::size_t)Cout * 3 * Cin, 0.0f);
    for (int o = 0; o < Cout; ++o) {
      for (int t = 0; t < 3; ++t) {
        for (int i = 0; i < Cin; ++i) {
          const std::size_t si = ((std::size_t)o * Cin + i) * 3 + t;
          flat[((std::size_t)o * 3 + t) * Cin + i] = w[si];
        }
      }
    }
    return f16_buf_(_mc, flat.data(), flat.size());
  }, _part);
  if (c.w.empty()) { return Conv{}; }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// A 1x1(x1) conv as a dense-GEMM weight [Cout, Cin] (the trailing singleton
// dims flatten away).
MetalWanVae::Conv
MetalWanVae::load_conv1x1_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(wname_(nm + ".weight"));
  if (info == nullptr || info->shape.size() < 2) { return c; }
  const auto& sh = info->shape;
  c.cout = (int)sh[0]; c.cin = (int)sh[1];
  c.kt = 1; c.ks = 1; c.k = c.cin;
  c.w = load_vec_(ws, nm + ".weight");
  if (c.w.empty()) { return Conv{}; }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// Every scalar/vector/matrix tensor this VAE keeps is stored as f16
// regardless of its on-disk dtype, so "read it as f32 and narrow" IS the
// transform and the result is cached like any other derived tensor.
SharedBuffer
MetalWanVae::load_vec_(WeightSet& ws, const std::string& nm)
{
  return ws.derived(std::string(kKey) + "f16|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> v = read_f32_(ws.src(), _mc, wname_(nm), n);
    if (v.empty()) { return {}; }
    return f16_buf_(_mc, v.data(), n);
  }, _part);
}

bool
MetalWanVae::load_resblock_(WeightSet& ws, const std::string& pre,
                            ResBlock& rb, int cin, int cout)
{
  rb.cin = cin; rb.cout = cout;
  rb.n1g = load_vec_(ws, pre + "norm1.gamma");
  rb.n2g = load_vec_(ws, pre + "norm2.gamma");
  rb.c1 = load_conv3d_(ws, pre + "conv1");
  rb.c2 = load_conv3d_(ws, pre + "conv2");
  rb.has_short = (cin != cout);
  if (rb.has_short) { rb.shortcut = load_conv1x1_(ws, pre + "conv_shortcut"); }
  return !rb.n1g.empty() && !rb.n2g.empty() && !rb.c1.empty() &&
         !rb.c2.empty() && (!rb.has_short || !rb.shortcut.empty());
}

bool
MetalWanVae::load_attn_(WeightSet& ws, const std::string& pre, Attn& a,
                        int dim)
{
  a.dim = dim;
  a.ng = load_vec_(ws, pre + "norm.gamma");
  // to_qkv is one 1x1 conv [3*dim, dim]; split output channels into q/k/v.
  // Each third is its own derived tensor, keyed by which third it is.
  const std::string qbase = pre + "to_qkv";
  std::size_t n = 0, nb = 0;
  std::vector<float> qkv, qkvb;
  auto read_qkv = [&]() {
    if (!qkv.empty()) { return; }
    qkv  = read_f32_(ws.src(), _mc, wname_(qbase + ".weight"), n);
    qkvb = read_f32_(ws.src(), _mc, wname_(qbase + ".bias"), nb);
  };
  const int C = dim;
  auto slice = [&](int off) {
    Conv c; c.cin = C; c.cout = C; c.k = C; c.kt = 1; c.ks = 1;
    const std::string key = std::string(kKey) + "qkv|" + qbase + "|" +
                            std::to_string(off);
    c.w = ws.derived(key + "|w", [&]() -> SharedBuffer {
      read_qkv();
      if (qkv.size() != (std::size_t)3 * C * C) { return {}; }
      return f16_buf_(_mc, qkv.data() + (std::size_t)off * C * C,
                      (std::size_t)C * C);
    }, _part);
    c.b = ws.derived(key + "|b", [&]() -> SharedBuffer {
      read_qkv();
      if (qkvb.size() != (std::size_t)3 * C) { return {}; }
      return f16_buf_(_mc, qkvb.data() + (std::size_t)off * C, (std::size_t)C);
    }, _part);
    return c;
  };
  a.q = slice(0);
  a.k = slice(1);
  a.v = slice(2);
  a.proj = load_conv1x1_(ws, pre + "proj");
  return !a.ng.empty() && !a.q.empty() && !a.k.empty() && !a.v.empty() &&
         !a.proj.empty();
}

std::unique_ptr<MetalWanVae>
MetalWanVae::load(const std::string& model_dir, MetalCompute* mc,
                  const Config& cfg, bool with_encoder)
{
  namespace fs = std::filesystem;
  fs::path p(model_dir);
  // Accept either the pipeline root or the vae/ subdir, as the image VAEs do.
  if (fs::is_directory(p) && fs::exists(p / "vae") &&
      !fs::exists(p / "diffusion_pytorch_model.safetensors")) {
    p = p / "vae";
  }
  return load(WeightSet::open(p.string(), nullptr), mc, cfg, with_encoder);
}

std::unique_ptr<MetalWanVae>
MetalWanVae::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                  const Config& cfg, bool with_encoder)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& wts = *ws_in;

  auto m = std::unique_ptr<MetalWanVae>(new MetalWanVae());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;

  // WHICH SPELLING this checkpoint uses, before anything is read.
  // build_name_map refuses a PARTIAL map rather than returning one: at
  // any given level every tensor has the same shape, so a name the rule
  // cannot place would load some other level's weights and decode a
  // plausible, wrong clip.
  {
    std::string nerr;
    if (!wan_vae::build_name_map(wts.src().tensor_names(), m->_names,
                                 &nerr)) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt("MetalWanVae: {}", nerr));
      }
      return nullptr;
    }
    if (!m->_names.empty() && mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalWanVae: natively-named Wan VAE checkpoint; {} names "
          "translated", m->_names.size()));
    }
  }

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_rms  = mc->load_library("rms_norm");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_fn_gemm_bias   = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms         = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_mul_sigmoid = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_residual    = m->_lib_elt.function("residual_add_f16");
  m->_fn_clamp       = m->_lib_elt.function("clamp_f16");
  m->_fn_copy        = m->_lib_elt.function("copy_f16");
  m->_fn_copy_rect   = m->_lib_elt.function("copy_rect_f16");
  m->_fn_sdpa        = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_sdpa_full_smm = m->_lib_sdpa.function("sdpa_full_mma_f16");
  m->_fn_im2col_tiled = m->_lib_elt.function("im2col_hwc_3x3_tiled_f16");
  m->_fn_im2col_s2_tiled =
      m->_lib_elt.function("im2col_hwc_3x3_s2_tiled_f16");
  m->_fn_im2col3d_tiled =
      m->_lib_elt.function("im2col_hwc_3x3x3_tiled_f16");
  m->_fn_concat3     = m->_lib_elt.function("concat3_frames_f16");
  m->_fn_time_unshuffle = m->_lib_elt.function("wan_time_unshuffle_f16");
  m->_fn_upsample    = m->_lib_elt.function("upsample_nearest2x_hwc_f16");
  if (!m->_fn_gemm_bias.valid() || !m->_fn_rms.valid() ||
      !m->_fn_mul_sigmoid.valid() || !m->_fn_residual.valid() ||
      !m->_fn_clamp.valid() || !m->_fn_copy.valid() ||
      !m->_fn_copy_rect.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_im2col_tiled.valid() || !m->_fn_im2col_s2_tiled.valid() ||
      !m->_fn_im2col3d_tiled.valid() || !m->_fn_concat3.valid() ||
      !m->_fn_time_unshuffle.valid() || !m->_fn_upsample.valid()) {
    return nullptr;
  }
  // Matrix-core dense GEMM (matmul2d) for the conv/1x1 GEMMs. Same guards
  // as the Qwen-Image VAE: bias folds separately, and a tall GEMM splits at
  // _mma_max_m. VPIPE_WAN_NO_MMA2 forces steel (A/B).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_WAN_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid() &&
                   m->_fn_bias_add.valid();
    if (const char* e = std::getenv("VPIPE_WAN_VAE_MMA_MAX_M")) {
      m->_mma_max_m = std::atoi(e);
    }
  }
  // Mid-block attention: the same single-head spatial attention as the
  // Qwen-Image VAE, run per frame, so the same kernel set and the same
  // measured pick. VPIPE_WAN_NO_MMA_ATTN forces scalar.
  if (std::getenv("VPIPE_WAN_NO_MMA_ATTN") == nullptr) {
    const int mid_d = cfg.base_dim * cfg.dim_mult[3];
    const char* fn = (mid_d == 384) ? "sdpa_full_mma2_d384_f16"
                   : (mid_d == 512) ? "sdpa_full_mma2_d512_f16"
                                    : nullptr;
    if (fn != nullptr) {
      m->_lib_sdpa_mma = mc->load_library("sdpa_mma");
      m->_fn_sdpa_full_mma = m->_lib_sdpa_mma.function(fn);
      m->load_wide_attn_(mid_d);
    }
    m->autotune_mid_attn_(mc, mid_d);
  }
  // The gather-free hardware conv for every 3x3 and 3x3x3 conv (see
  // conv3x3_hw_). Decided BEFORE the weights load, because the loaders
  // build the HWIO twins only when it is on. VPIPE_VAE_NO_HWCONV turns it
  // off, the same switch the image VAEs honour.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_VAE_NO_HWCONV") == nullptr) {
    m->_lib_convhw = mc->load_library("conv2d_mma");
    m->_fn_conv_hw64 = m->_lib_convhw.function("conv2d_hw_3x3_s1_f16");
    m->_fn_conv_hw32 = m->_lib_convhw.function("conv2d_hw_3x3_s1_c32_f16");
    m->_fn_conv_hw64_acc =
        m->_lib_convhw.function("conv2d_hw_3x3_s1_acc_f16");
    m->_fn_conv_hw32_acc =
        m->_lib_convhw.function("conv2d_hw_3x3_s1_c32_acc_f16");
    m->_fn_conv_small_cout =
        m->_lib_elt.function("conv3x3_hwc_small_cout_f16");
    m->_fn_conv_small_cout_acc =
        m->_lib_elt.function("conv3x3_hwc_small_cout_acc_f16");
    if (!m->_fn_bias_add.valid()) {
      m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    }
    m->_use_hwconv = m->_fn_conv_hw64.valid() && m->_fn_conv_hw32.valid() &&
                     m->_fn_conv_hw64_acc.valid() &&
                     m->_fn_conv_hw32_acc.valid() &&
                     m->_fn_conv_small_cout.valid() &&
                     m->_fn_conv_small_cout_acc.valid() &&
                     m->_fn_bias_add.valid();
  }

  const int base = cfg.base_dim;                         // 96
  const int dims0 = base * cfg.dim_mult[3];              // 384
  m->_post_quant = m->load_conv1x1_(wts, "post_quant_conv");
  m->_conv_in    = m->load_conv3d_(wts, "decoder.conv_in");

  bool ok = !m->_post_quant.empty() && !m->_conv_in.empty();
  ok = ok && m->load_resblock_(wts, "decoder.mid_block.resnets.0.",
                               m->_mid_res0, dims0, dims0);
  ok = ok && m->load_attn_(wts, "decoder.mid_block.attentions.0.",
                           m->_mid_attn, dims0);
  ok = ok && m->load_resblock_(wts, "decoder.mid_block.resnets.1.",
                               m->_mid_res1, dims0, dims0);

  // Decoder dims = [dim*mult[-1]] + dim*mult[::-1] = [384,384,384,192,96];
  // up i maps dims[i] -> dims[i+1] (for i>0 the resnet in_dim is halved by
  // the previous upsample conv), with an upsampler for i != 3. The
  // temporal flag is temperal_downsample REVERSED.
  const int dims[5] = {dims0, base * cfg.dim_mult[3], base * cfg.dim_mult[2],
                       base * cfg.dim_mult[1], base * cfg.dim_mult[0]};
  const bool tup[3] = {cfg.temperal_downsample[2], cfg.temperal_downsample[1],
                       cfg.temperal_downsample[0]};
  m->_up_blocks.resize(4);
  for (int i = 0; i < 4; ++i) {
    UpBlock& ub = m->_up_blocks[(std::size_t)i];
    int in_dim = (i > 0) ? dims[i] / 2 : dims[i];
    const int out_dim = dims[i + 1];
    ub.resnets.resize((std::size_t)cfg.num_res_blocks + 1);
    int cin = in_dim;
    for (int r = 0; r <= cfg.num_res_blocks; ++r) {
      ok = ok && m->load_resblock_(
          wts, "decoder.up_blocks." + std::to_string(i) + ".resnets." +
                   std::to_string(r) + ".",
          ub.resnets[(std::size_t)r], cin, out_dim);
      cin = out_dim;
    }
    ub.up.present = (i != 3);
    if (ub.up.present) {
      ub.up_dim = out_dim;
      const std::string pre =
          "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.";
      ub.up.space = m->load_conv2d_(wts, pre + "resample.1");
      ub.up.temporal = tup[i];
      if (ub.up.temporal) {
        ub.up.time = m->load_time_conv_(wts, pre + "time_conv");
        ok = ok && !ub.up.time.empty();
      }
      ok = ok && !ub.up.space.empty();
    }
  }

  m->_norm_out_g = m->load_vec_(wts, "decoder.norm_out.gamma");

  m->_conv_out = m->load_conv3d_(wts, "decoder.conv_out");

  ok = ok && !m->_norm_out_g.empty() && !m->_conv_out.empty();

  if (!ok) { return nullptr; }
  if (with_encoder && !m->ensure_encoder()) { return nullptr; }
  return m;
}

bool
MetalWanVae::load_encoder_(WeightSet& ws)
{
  const int base = _cfg.base_dim;
  // Encoder dims = [base*u for u in [1] + dim_mult] = [96,96,192,384,384].
  const int dims[5] = {base, base * _cfg.dim_mult[0], base * _cfg.dim_mult[1],
                       base * _cfg.dim_mult[2], base * _cfg.dim_mult[3]};
  const int dtop = dims[4];

  _enc_conv_in = load_conv3d_(ws, "encoder.conv_in");
  bool ok = !_enc_conv_in.empty();

  // The encoder's down_blocks are a FLAT ModuleList: per stage,
  // num_res_blocks residual blocks then (for all but the last stage) one
  // resample. So the index advances by num_res_blocks + 1 per stage and the
  // names are not grouped the way the decoder's up_blocks are.
  _enc_down.resize(4);
  int idx = 0;
  for (int i = 0; i < 4; ++i) {
    DownStage& ds = _enc_down[(std::size_t)i];
    int cin = dims[i];
    const int cout = dims[i + 1];
    ds.resnets.resize((std::size_t)_cfg.num_res_blocks);
    for (int r = 0; r < _cfg.num_res_blocks; ++r) {
      ok = ok && load_resblock_(ws,
                                "encoder.down_blocks." + std::to_string(idx) +
                                    ".",
                                ds.resnets[(std::size_t)r], cin, cout);
      cin = cout;
      ++idx;
    }
    ds.down.present = (i != 3);
    if (ds.down.present) {
      const std::string pre = "encoder.down_blocks." + std::to_string(idx) + ".";
      ds.down.space = load_conv2d_(ws, pre + "resample.1");
      ds.down.temporal = _cfg.temperal_downsample[i];
      if (ds.down.temporal) {
        ds.down.time = load_time_conv_(ws, pre + "time_conv");
        ok = ok && !ds.down.time.empty();
      }
      ok = ok && !ds.down.space.empty();
      ++idx;
    }
  }

  ok = ok && load_resblock_(ws, "encoder.mid_block.resnets.0.", _enc_mid_res0,
                            dtop, dtop);
  ok = ok && load_attn_(ws, "encoder.mid_block.attentions.0.", _enc_mid_attn,
                        dtop);
  ok = ok && load_resblock_(ws, "encoder.mid_block.resnets.1.", _enc_mid_res1,
                            dtop, dtop);
  _enc_norm_out_g = load_vec_(ws, "encoder.norm_out.gamma");
  _enc_conv_out = load_conv3d_(ws, "encoder.conv_out");
  _quant_conv = load_conv1x1_(ws, "quant_conv");
  ok = ok && !_enc_norm_out_g.empty() && !_enc_conv_out.empty() &&
       !_quant_conv.empty();
  return ok;
}

bool
MetalWanVae::ensure_encoder()
{
  if (_has_encoder) { return true; }
  if (!_ws) { return false; }
  // Through the WeightSet so the encoder half loads ONCE per checkpoint
  // however many VAEs over it need it -- and, when nobody does, never.
  const bool ok = _ws->ensure_part("encoder", [this]() {
    _part = "encoder";
    const bool r = load_encoder_(*_ws);
    _part.clear();
    return r;
  });
  if (!ok) { return false; }
  // A second VAE over a checkpoint whose encoder another one already loaded
  // still has to populate ITS OWN members; ensure_part reports the cached
  // success without re-running the loader, so bind the (now cache-hit)
  // tensors here.
  if (_enc_conv_in.empty()) {
    _part = "encoder";
    const bool r = load_encoder_(*_ws);
    _part.clear();
    if (!r) { return false; }
  }
  _has_encoder = true;
  return true;
}

// ---- mid-block attention (shared kernel set) ---------------------------

void
MetalWanVae::load_wide_attn_(int mid_d)
{
  const std::string base = "sdpa_full_mma2_d" + std::to_string(mid_d) + "_q";
  _fn_sdpa_full_wide16 = _lib_sdpa_mma.function(base + "16_f16");
  _fn_sdpa_full_wide32 = _lib_sdpa_mma.function(base + "32_f16");
  _fn_sdpa_full_wide64 = _lib_sdpa_mma.function(base + "64_f16");
}

bool
MetalWanVae::mid_attn_available_(MidAttn k) const
{
  switch (k) {
    case MidAttn::kScalar: return _fn_sdpa.valid();
    case MidAttn::kSmm:    return _fn_sdpa_full_smm.valid();
    case MidAttn::kMma8:   return _fn_sdpa_full_mma.valid();
    case MidAttn::kWide16: return _fn_sdpa_full_wide16.valid();
    case MidAttn::kWide32: return _fn_sdpa_full_wide32.valid();
    case MidAttn::kWide64: return _fn_sdpa_full_wide64.valid();
    case MidAttn::kMat:    return false;   // no materialized path here
  }
  return false;
}

void
MetalWanVae::encode_mid_attn_(ComputeEncoder& enc, MidAttn kind,
                              const SharedBuffer& q, const SharedBuffer& k,
                              const SharedBuffer& v, const SharedBuffer& att,
                              std::size_t hw, int C, float scale)
{
  enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
  enc.set_buffer(3, att);
  enc.set_constant(4, scale); enc.set_constant(5, (int)hw);
  enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
  enc.set_constant(9, (int)hw); enc.set_constant(10, (int)hw);
  switch (kind) {
    case MidAttn::kWide16:
    case MidAttn::kWide32:
    case MidAttn::kWide64: {
      const int bq = (kind == MidAttn::kWide16) ? 16
                   : (kind == MidAttn::kWide32) ? 32 : 64;
      enc.set_function(kind == MidAttn::kWide16 ? _fn_sdpa_full_wide16
                     : kind == MidAttn::kWide32 ? _fn_sdpa_full_wide32
                                                : _fn_sdpa_full_wide64);
      const unsigned nt = attn_threads_(bq);
      enc.dispatch({nt, 1, (unsigned)(((int)hw + bq - 1) / bq)}, {nt, 1, 1});
      break;
    }
    case MidAttn::kMma8:
      enc.set_function(_fn_sdpa_full_mma);
      enc.dispatch({4 * 32, 1, (unsigned)((hw + 7) / 8)}, {4 * 32, 1, 1});
      break;
    case MidAttn::kSmm: {
      enc.set_function(_fn_sdpa_full_smm);
      const unsigned nt = 4u * (unsigned)(C / 64) * 32u;   // WM*WD*32
      enc.dispatch({nt, 1, (unsigned)((hw + 31) / 32)}, {nt, 1, 1});
      break;
    }
    default:
      enc.set_function(_fn_sdpa);                    // scalar O(N^2)
      enc.dispatch({32, 1, (unsigned)hw}, {32, 1, 1});
      break;
  }
}

void
MetalWanVae::autotune_mid_attn_(MetalCompute* mc, int C)
{
  _attn_pick = MidAttn::kScalar;
  const bool smm_ok = mid_attn_available_(MidAttn::kSmm) &&
                      (C % 64 == 0) && (C <= 512);
  if (smm_ok) { _attn_pick = MidAttn::kSmm; }
  if (mc->supports_matrix_cores()) {
    if (mid_attn_available_(MidAttn::kMma8))   { _attn_pick = MidAttn::kMma8; }
    if (mid_attn_available_(MidAttn::kWide32)) { _attn_pick = MidAttn::kWide32; }
  }
  if (const char* e = std::getenv("VPIPE_WAN_VAE_ATTN_BQ")) {
    const int b = std::atoi(e);
    if (b == 8  && mid_attn_available_(MidAttn::kMma8))   { _attn_pick = MidAttn::kMma8; }
    if (b == 16 && mid_attn_available_(MidAttn::kWide16)) { _attn_pick = MidAttn::kWide16; }
    if (b == 32 && mid_attn_available_(MidAttn::kWide32)) { _attn_pick = MidAttn::kWide32; }
    if (b == 64 && mid_attn_available_(MidAttn::kWide64)) { _attn_pick = MidAttn::kWide64; }
    return;
  }
  std::vector<MidAttn> cands;
  for (MidAttn k : {MidAttn::kSmm, MidAttn::kMma8, MidAttn::kWide16,
                    MidAttn::kWide32, MidAttn::kWide64}) {
    if (k == MidAttn::kSmm && !smm_ok) { continue; }
    if (mid_attn_available_(k)) { cands.push_back(k); }
  }
  std::string detail;
  const auto t0 = std::chrono::steady_clock::now();
  _attn_pick = vae_mid_attn::autotune<MetalCompute, ComputeEncoder>(
      mc, C, cands, _attn_pick,
      [this](ComputeEncoder& enc, MidAttn kind, const SharedBuffer& qq,
             const SharedBuffer& kk, const SharedBuffer& vv,
             const SharedBuffer& oo, std::size_t hw, int c, float sc,
             const vae_mid_attn::Alloc&, const vae_mid_attn::Release&) {
        encode_mid_attn_(enc, kind, qq, kk, vv, oo, hw, c, sc);
      },
      &detail);
  const double tune_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  if (mc->session() != nullptr && !detail.empty()) {
    const auto line = fmt(
        "MetalWanVae: mid-attn autotune (D={}) -> {} [{}] in {} ms",
        C, vae_mid_attn::name(_attn_pick), detail, (long long)tune_ms);
    if (std::getenv("VPIPE_VAE_ATTN_TUNE_LOG") != nullptr) {
      mc->session()->log_normal(line);
    } else {
      mc->session()->log_debug(line);
    }
  }
}

// ---- forward primitives ------------------------------------------------

int
MetalWanVae::mma_row_chunk(int M, int N, int K, int max_m)
{
  int chunk = M > 0 ? M : 1;
  if (max_m > 0 && chunk > max_m) { chunk = max_m; }
  const int band = mma_row_band(N, K);
  if (chunk > band) { chunk = band; }
  return chunk;
}

void
MetalWanVae::gemm_bias_(Ctx& cx, const SharedBuffer& x, const SharedBuffer& w,
                        const SharedBuffer& b, const SharedBuffer& y, int M,
                        int N, int K, int y_row0, std::size_t x_off_rows)
{
  ComputeEncoder& enc = *cx.enc;
  const std::size_t ybase = (std::size_t)y_row0 * N;
  const std::size_t xbase = x_off_rows * (std::size_t)K;
  if (_use_mma2 && M >= _mma_min_m && N >= _mma_min_n) {
    const bool deep = (K >= 6144);
    const int BN = deep ? 256 : 128;
    // Split a tall GEMM: past ~2^19 rows matmul2d corrupts its output,
    // and past 2^31 bytes on ANY operand it stops storing. See
    // mma_row_chunk -- the second limit is the one the row cap alone
    // misses, because it binds on the 27-tap im2col SOURCE.
    const int chunk = mma_row_chunk(M, N, K, _mma_max_m);
    for (int r0 = 0; r0 < M; r0 += chunk) {
      const int mc = (M - r0 < chunk) ? (M - r0) : chunk;
      enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
      enc.set_buffer(0, x, (xbase + (std::size_t)r0 * K) * 2);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);        // bias slot unused (has_bias=0)
      enc.set_buffer(3, y, (ybase + (std::size_t)r0 * N) * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, mc);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                    (unsigned)((mc + 127) / 128), 1}, {256, 1, 1});
    }
    if (!b.empty()) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ybase * 2); enc.set_buffer(1, b);
      enc.set_constant(2, N);
      enc.set_constant(3, (unsigned)((std::size_t)M * N));
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm_bias);
  enc.set_buffer(0, x, xbase * 2); enc.set_buffer(1, w);
  enc.set_buffer(2, b.empty() ? w : b); enc.set_buffer(3, y, ybase * 2);
  enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
  enc.set_constant(7, b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
}

// One output FRAME. `taps[kt]` is the buffer holding temporal tap kt and
// `tap_off[kt]` its ELEMENT offset; a null tap is masked to zero (a frame
// before the start of the sequence). Rows are streamed in bands so the col
// scratch stays bounded -- at 27 taps a full-res [H*W, 27*Cin] would be
// several GB.
void
MetalWanVae::conv_frame_(Ctx& cx, const Conv& c,
                         const SharedBuffer* const taps[3],
                         const std::size_t tap_off[3], const SharedBuffer& out,
                         std::size_t out_row0, int H, int W, int stride)
{
  ComputeEncoder& enc = *cx.enc;
  const int OH = (stride == 2) ? H / 2 : H;
  const int OW = (stride == 2) ? W / 2 : W;
  const std::size_t ohw = (std::size_t)OH * OW;

  // 1x1: no gather at all, the activation IS the GEMM input.
  if (c.ks == 1 && c.kt == 1) {
    gemm_bias_(cx, *taps[0], c.w, c.b, out, (int)ohw, c.cout, c.cin,
               (int)out_row0, tap_off[0] / (std::size_t)std::max(c.cin, 1));
    return;
  }

  // Temporal-only (3,1,1): concat the three frames channel-wise.
  if (c.ks == 1) {
    if (!cx.ensure_col(_mc, ohw * (std::size_t)3 * c.cin)) { return; }
    const std::size_t per_row = (std::size_t)3 * c.cin;
    std::size_t rows = (per_row > 0) ? (cx.col_cap / per_row) : ohw;
    if (_mma_max_m > 0 && rows > (std::size_t)_mma_max_m / 2) {
      rows = (std::size_t)_mma_max_m / 2;
    }
    rows = std::min(std::max<std::size_t>(rows, 1), ohw);
    int mask = 0;
    for (int t = 0; t < 3; ++t) { if (taps[t] != nullptr) { mask |= 1 << t; } }
    const SharedBuffer* any = taps[0] != nullptr ? taps[0]
                            : taps[1] != nullptr ? taps[1] : taps[2];
    for (std::size_t r0 = 0; r0 < ohw; r0 += rows) {
      const int mc = (int)std::min(rows, ohw - r0);
      enc.set_function(_fn_concat3);
      for (int t = 0; t < 3; ++t) {
        const SharedBuffer* b = taps[t] != nullptr ? taps[t] : any;
        const std::size_t off =
            (taps[t] != nullptr) ? (tap_off[t] + r0 * (std::size_t)c.cin) : 0;
        enc.set_buffer((unsigned)t, *b, off * 2);
      }
      enc.set_buffer(3, cx.col);
      enc.set_constant(4, c.cin); enc.set_constant(5, mc);
      enc.set_constant(6, mask);
      enc.dispatch({(unsigned)(3 * c.cin), (unsigned)mc, 1}, {64, 1, 1});
      gemm_bias_(cx, cx.col, c.w, c.b, out, mc, c.cout, c.k,
                 (int)(out_row0 + r0));
    }
    return;
  }

  // Spatial 3x3, either the 27-tap causal conv3d or the plain 2D resample.
  // The hardware conv first; the gather below is the fallback.
  if (stride == 1 && conv3x3_hw_(cx, c, taps, tap_off, out, out_row0, H, W)) {
    return;
  }
  if (!cx.ensure_col(_mc, ohw * (std::size_t)c.k)) { return; }
  const std::size_t per_row = (std::size_t)c.k;
  std::size_t rows = (per_row > 0) ? (cx.col_cap / per_row) : ohw;
  if (_mma_max_m > 0 && rows > (std::size_t)_mma_max_m / 2) {
    rows = (std::size_t)_mma_max_m / 2;
  }
  rows = std::min(std::max<std::size_t>(rows, 1), ohw);
  int mask = 0;
  for (int t = 0; t < 3; ++t) { if (taps[t] != nullptr) { mask |= 1 << t; } }
  const SharedBuffer* any = taps[0] != nullptr ? taps[0]
                          : taps[1] != nullptr ? taps[1] : taps[2];
  for (std::size_t r0 = 0; r0 < ohw; r0 += rows) {
    const int mc = (int)std::min(rows, ohw - r0);
    if (c.kt == 3) {
      enc.set_function(_fn_im2col3d_tiled);
      for (int t = 0; t < 3; ++t) {
        const SharedBuffer* b = taps[t] != nullptr ? taps[t] : any;
        enc.set_buffer((unsigned)t, *b,
                       (taps[t] != nullptr ? tap_off[t] : 0) * 2);
      }
      enc.set_buffer(3, cx.col);
      enc.set_constant(4, H); enc.set_constant(5, W);
      enc.set_constant(6, c.cin);
      enc.set_constant(7, (int)r0); enc.set_constant(8, mc);
      enc.set_constant(9, mask);
      enc.dispatch({(unsigned)c.k, (unsigned)mc, 1}, {64, 1, 1});
    } else {
      // The whole [H*W, cin] frame stays bound: the 3x3 halo spans band
      // edges, so the band applies to OUTPUT rows only.
      enc.set_function(stride == 2 ? _fn_im2col_s2_tiled : _fn_im2col_tiled);
      enc.set_buffer(0, *taps[0], tap_off[0] * 2);
      enc.set_buffer(1, cx.col);
      enc.set_constant(2, H); enc.set_constant(3, W);
      enc.set_constant(4, c.cin);
      enc.set_constant(5, (int)r0); enc.set_constant(6, mc);
      enc.dispatch({(unsigned)(9 * c.cin), (unsigned)mc, 1}, {64, 1, 1});
    }
    gemm_bias_(cx, cx.col, c.w, c.b, out, mc, c.cout, c.k,
               (int)(out_row0 + r0));
  }
}

// One output frame of a 3x3 or 3x3x3 conv on the gather-free hardware conv.
// A causal conv runs as one 2D conv PER TEMPORAL TAP, against that tap's
// [3,3,cin,cout] block of the HWIO twin: the first live tap writes the
// output, every later one accumulates onto it, and a tap before the
// sequence is skipped. No gather -- 27 taps of traffic, 11.5 GB per
// full-resolution conv per frame at 1920x1152 -- and no three-frame concat
// either, which cost 2.2 GB of peak at that size. The head's 3-channel
// output fits neither tile and takes the per-pixel small-cout conv, the
// same way. Verified against the gather per conv
// (conv2d_mma.hw_op_wan_causal_conv3d, rel-L2 ~3e-4) and across the whole
// decoder (flashvsr_vae.hwconv_decode_matches_gather, 6.5e-4).
//
// MEASURED per conv on the M5 Pro against the gather: 5.66x at 1920x1152
// 3x96->96 (40 ms, 27.7 TF/s, against 225), 3.45x at 960x576 and 2.27x at
// 480x288. The concat version it replaced was 3.54x / 2.84x / 2.02x: the
// concat was traffic as well as peak. Decodes end to end: 960x576 859 ->
// 277 ms/frame (3.10x), 1920x1152 ~3400 -> 1149 ms/frame (2.96x) at a
// 13141 MB peak, where the concat version peaked at 15118.
bool
MetalWanVae::conv3x3_hw_(Ctx& cx, const Conv& c,
                         const SharedBuffer* const taps[3],
                         const std::size_t tap_off[3],
                         const SharedBuffer& out, std::size_t out_row0,
                         int H, int W)
{
  if (!_use_hwconv || c.whwio.empty() || c.ks != 9) { return false; }
  const int tile = (c.cout % 64 == 0) ? 64 : (c.cout % 32 == 0) ? 32 : 0;
  if (tile == 0 && c.cout > kSmallCoutMax) { return false; }
  const std::size_t hw = (std::size_t)H * W;
  if (tile != 0) {
    // The op tiles its destination 8x8 and indexes through int32 extents.
    constexpr std::size_t kIdxMax = 0x7fffffffull;
    if ((W % 8) != 0 || (H % 8) != 0 || (std::size_t)c.cin * hw > kIdxMax ||
        (std::size_t)c.cout * hw > kIdxMax) {
      return false;
    }
  }
  const int ntaps = (c.kt == 3) ? 3 : 1;
  bool live = false;
  for (int t = 0; t < ntaps; ++t) { live = live || taps[t] != nullptr; }
  if (!live) { return false; }

  ComputeEncoder& enc = *cx.enc;
  const std::size_t blk = (std::size_t)9 * c.cin * c.cout;      // per tap
  const std::size_t out_off = out_row0 * (std::size_t)c.cout;  // elements
  bool first = true;
  for (int t = 0; t < ntaps; ++t) {
    if (taps[t] == nullptr) { continue; }
    if (tile != 0) {
      const metal_compute::ComputeFunction& fn =
          tile == 64 ? (first ? _fn_conv_hw64 : _fn_conv_hw64_acc)
                     : (first ? _fn_conv_hw32 : _fn_conv_hw32_acc);
      enc.set_function(fn);
      enc.set_buffer(0, *taps[t], tap_off[t] * 2);
      enc.set_buffer(1, c.whwio, (std::size_t)t * blk * 2);
      enc.set_buffer(2, out, out_off * 2);
      enc.set_constant(3, W); enc.set_constant(4, H);
      enc.set_constant(5, c.cin); enc.set_constant(6, c.cout);
      enc.dispatch({(unsigned)((W / 8) * 128), (unsigned)(H / 8),
                    (unsigned)(c.cout / tile)}, {128, 1, 1});
    } else {
      const int has_bias = (first && !c.b.empty()) ? 1 : 0;
      enc.set_function(first ? _fn_conv_small_cout : _fn_conv_small_cout_acc);
      enc.set_buffer(0, *taps[t], tap_off[t] * 2);
      enc.set_buffer(1, c.whwio, (std::size_t)t * blk * 2);
      enc.set_buffer(2, has_bias != 0 ? c.b : c.whwio);
      enc.set_buffer(3, out, out_off * 2);
      enc.set_constant(4, W); enc.set_constant(5, H);
      enc.set_constant(6, c.cin); enc.set_constant(7, c.cout);
      enc.set_constant(8, has_bias);
      enc.dispatch({(unsigned)W, (unsigned)H, 1}, {256, 1, 1});
    }
    first = false;
  }
  if (tile != 0 && !c.b.empty()) {     // the op has no bias slot
    enc.set_function(_fn_bias_add);
    enc.set_buffer(0, out, out_off * 2); enc.set_buffer(1, c.b);
    enc.set_constant(2, c.cout);
    enc.set_constant(3, (unsigned)(hw * (std::size_t)c.cout));
    enc.dispatch({(unsigned)c.cout, (unsigned)hw, 1}, {256, 1, 1});
  }
  return true;
}

// Whole-chunk convolution. Output frame o of a stride-1 causal conv reads
// input frames o-2, o-1, o; anything below 0 comes from `carry` (the
// previous chunk's trailing frames) and anything below -carry->frames is
// before the start of the sequence and masks to zero. `stride` here is the
// SPATIAL stride; the encoder's temporal downsample has its own routine.
SharedBuffer&
MetalWanVae::conv_chunk_(Ctx& cx, const Conv& c, const SharedBuffer& in,
                         int t_in, int H, int W, int stride, Carry* carry)
{
  const int OH = (stride == 2) ? H / 2 : H;
  const int OW = (stride == 2) ? W / 2 : W;
  const std::size_t ihw = (std::size_t)H * W;
  const std::size_t ohw = (std::size_t)OH * OW;
  SharedBuffer& out = cx.alloc(_mc, (std::size_t)t_in * ohw * c.cout);
  if (!cx.alloc_ok) { return out; }

  // A 1x1 with no temporal extent is the same GEMM for every frame, so run
  // the whole chunk as one tall GEMM instead of t of them.
  if (c.kt == 1 && c.ks == 1) {
    gemm_bias_(cx, in, c.w, c.b, out, (int)((std::size_t)t_in * ohw), c.cout,
               c.cin);
    return out;
  }

  for (int o = 0; o < t_in; ++o) {
    const SharedBuffer* taps[3] = {nullptr, nullptr, nullptr};
    std::size_t offs[3] = {0, 0, 0};
    if (c.kt == 1) {
      taps[0] = &in;
      offs[0] = (std::size_t)o * ihw * c.cin;
    } else {
      for (int kt = 0; kt < 3; ++kt) {
        const int fi = o - 2 + kt;         // input frame index in the chunk
        if (fi >= 0) {
          taps[kt] = &in;
          offs[kt] = (std::size_t)fi * ihw * c.cin;
        } else if (carry != nullptr && carry->total + fi >= 0) {
          // Slot j % 2 for sequence frame j; fi is -1 or -2 here, so the
          // absolute index is carry->total + fi.
          taps[kt] = &carry->buf;
          offs[kt] =
              (std::size_t)((carry->total + fi) % 2) * ihw * c.cin;
        }
        // else: before the sequence -- left null, masked to zero.
      }
    }
    conv_frame_(cx, c, taps, offs, out, (std::size_t)o * ohw, H, W, stride);
  }
  if (carry != nullptr && c.kt == 3) {
    save_carry_(cx, *carry, in, t_in, ihw, c.cin);
  }
  return out;
}

void
MetalWanVae::save_carry_(Ctx& cx, Carry& carry, const SharedBuffer& x, int t,
                         std::size_t hw, int cin)
{
  const std::size_t frame_el = hw * (std::size_t)cin;
  if (carry.buf.empty() || carry.hw != hw || carry.cin != cin) {
    carry.buf = _mc->make_shared_buffer(2 * frame_el * 2);
    if (carry.buf.empty()) { cx.alloc_ok = false; return; }
    carry.hw = hw;
    carry.cin = cin;
    carry.total = 0;
  }
  // Only the last two frames of the chunk can still be read, and each goes
  // to the slot its SEQUENCE index selects -- so nothing has to move when
  // a one-frame chunk lands next to a two-frame history. The copies are
  // GPU-side: x exists only on the GPU timeline at this point.
  ComputeEncoder& enc = *cx.enc;
  const int first = std::max(0, t - 2);
  for (int j = first; j < t; ++j) {
    const std::size_t slot = (std::size_t)((carry.total + j) % 2);
    enc.set_function(_fn_copy);
    enc.set_buffer(0, x, (std::size_t)j * frame_el * 2);
    enc.set_buffer(1, carry.buf);
    enc.set_constant(2, (int)(slot * frame_el));
    enc.set_constant(3, (int)frame_el);
    enc.dispatch({(unsigned)frame_el, 1, 1}, {256, 1, 1});
  }
  carry.total += t;
}

SharedBuffer&
MetalWanVae::normc_(Ctx& cx, const SharedBuffer& in, std::size_t rows, int C,
                    const SharedBuffer& g)
{
  // WanRMS_norm over the channel axis: x/||x|| * sqrt(C) * gamma. The rms
  // kernel's x*rsqrt(mean+eps)*w matches with w=gamma and eps ~ 0.
  SharedBuffer& out = cx.alloc(_mc, rows * (std::size_t)C);
  const float eps = 1e-12f / (float)C;
  ComputeEncoder& enc = *cx.enc;
  enc.set_function(_fn_rms);
  enc.set_buffer(0, in); enc.set_buffer(1, g); enc.set_buffer(2, out);
  enc.set_constant(3, C); enc.set_constant(4, eps);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
  return out;
}

void
MetalWanVae::silu_(Ctx& cx, const SharedBuffer& x, std::size_t n)
{
  ComputeEncoder& enc = *cx.enc;
  enc.set_function(_fn_mul_sigmoid);
  enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, x);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

SharedBuffer&
MetalWanVae::resadd_(Ctx& cx, const SharedBuffer& a, const SharedBuffer& b,
                     std::size_t n)
{
  SharedBuffer& out = cx.alloc(_mc, n);
  ComputeEncoder& enc = *cx.enc;
  enc.set_function(_fn_residual);
  enc.set_buffer(0, a); enc.set_buffer(1, b); enc.set_buffer(2, out);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  return out;
}

SharedBuffer&
MetalWanVae::upsample2x_(Ctx& cx, const SharedBuffer& in, int t, int H, int W,
                         int C)
{
  const std::size_t ihw = (std::size_t)H * W;
  SharedBuffer& out = cx.alloc(_mc, (std::size_t)t * 4 * ihw * C);
  ComputeEncoder& enc = *cx.enc;
  for (int f = 0; f < t; ++f) {
    enc.set_function(_fn_upsample);
    enc.set_buffer(0, in, (std::size_t)f * ihw * C * 2);
    enc.set_buffer(1, out, (std::size_t)f * 4 * ihw * C * 2);
    enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, C);
    enc.dispatch({(unsigned)C, (unsigned)(4 * ihw), 1}, {256, 1, 1});
  }
  return out;
}

SharedBuffer&
MetalWanVae::resblock_(Ctx& cx, const ResBlock& rb, const SharedBuffer& x,
                       int t, int H, int W, Carry* c1, Carry* c2)
{
  const std::size_t rows = (std::size_t)t * H * W;
  SharedBuffer& n1 = normc_(cx, x, rows, rb.cin, rb.n1g);
  silu_(cx, n1, rows * (std::size_t)rb.cin);
  SharedBuffer& a = conv_chunk_(cx, rb.c1, n1, t, H, W, 1, c1);
  cx.release(n1);
  SharedBuffer& n2 = normc_(cx, a, rows, rb.cout, rb.n2g);
  cx.release(a);
  silu_(cx, n2, rows * (std::size_t)rb.cout);
  SharedBuffer& b = conv_chunk_(cx, rb.c2, n2, t, H, W, 1, c2);
  cx.release(n2);
  if (rb.has_short) {
    SharedBuffer& h = conv_chunk_(cx, rb.shortcut, x, t, H, W, 1, nullptr);
    SharedBuffer& out = resadd_(cx, b, h, rows * (std::size_t)rb.cout);
    cx.release(b); cx.release(h);
    return out;
  }
  SharedBuffer& out = resadd_(cx, b, x, rows * (std::size_t)rb.cout);
  cx.release(b);
  return out;
}

SharedBuffer&
MetalWanVae::attention_(Ctx& cx, const Attn& a, const SharedBuffer& x, int t,
                        int H, int W)
{
  const std::size_t hw = (std::size_t)H * W;
  const int C = a.dim;
  const std::size_t rows = (std::size_t)t * hw;
  SharedBuffer& n = normc_(cx, x, rows, C, a.ng);
  SharedBuffer& q = cx.alloc(_mc, rows * C);
  SharedBuffer& k = cx.alloc(_mc, rows * C);
  SharedBuffer& v = cx.alloc(_mc, rows * C);
  gemm_bias_(cx, n, a.q.w, a.q.b, q, (int)rows, C, C);
  gemm_bias_(cx, n, a.k.w, a.k.b, k, (int)rows, C, C);
  gemm_bias_(cx, n, a.v.w, a.v.b, v, (int)rows, C, C);
  cx.release(n);
  SharedBuffer& att = cx.alloc(_mc, rows * C);
  const float scale = 1.0f / std::sqrt((float)C);
  // The attention is per FRAME (the reference folds time into the batch),
  // so each frame is its own hw-long sequence.
  for (int f = 0; f < t; ++f) {
    // A frame view of each of q/k/v/att. The attention kernels take whole
    // buffers, so bind the slices through the encoder's byte offset.
    ComputeEncoder& enc = *cx.enc;
    const std::size_t off = (std::size_t)f * hw * C * 2;
    enc.set_buffer(0, q, off); enc.set_buffer(1, k, off);
    enc.set_buffer(2, v, off); enc.set_buffer(3, att, off);
    enc.set_constant(4, scale); enc.set_constant(5, (int)hw);
    enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
    enc.set_constant(9, (int)hw); enc.set_constant(10, (int)hw);
    switch (_attn_pick) {
      case MidAttn::kWide16:
      case MidAttn::kWide32:
      case MidAttn::kWide64: {
        const int bq = (_attn_pick == MidAttn::kWide16) ? 16
                     : (_attn_pick == MidAttn::kWide32) ? 32 : 64;
        enc.set_function(_attn_pick == MidAttn::kWide16 ? _fn_sdpa_full_wide16
                       : _attn_pick == MidAttn::kWide32 ? _fn_sdpa_full_wide32
                                                        : _fn_sdpa_full_wide64);
        const unsigned nt = attn_threads_(bq);
        enc.dispatch({nt, 1, (unsigned)(((int)hw + bq - 1) / bq)}, {nt, 1, 1});
        break;
      }
      case MidAttn::kMma8:
        enc.set_function(_fn_sdpa_full_mma);
        enc.dispatch({4 * 32, 1, (unsigned)((hw + 7) / 8)}, {4 * 32, 1, 1});
        break;
      case MidAttn::kSmm: {
        enc.set_function(_fn_sdpa_full_smm);
        const unsigned nt = 4u * (unsigned)(C / 64) * 32u;
        enc.dispatch({nt, 1, (unsigned)((hw + 31) / 32)}, {nt, 1, 1});
        break;
      }
      default:
        enc.set_function(_fn_sdpa);
        enc.dispatch({32, 1, (unsigned)hw}, {32, 1, 1});
        break;
    }
  }
  cx.release(q); cx.release(k); cx.release(v);
  SharedBuffer& p = cx.alloc(_mc, rows * C);
  gemm_bias_(cx, att, a.proj.w, a.proj.b, p, (int)rows, C, C);
  cx.release(att);
  SharedBuffer& out = resadd_(cx, p, x, rows * (std::size_t)C);
  cx.release(p);
  return out;
}

// The decoder's temporal upsample. The FIRST chunk is passed through
// untouched (the reference's "Rep" sentinel) -- so the sequence the
// time_conv sees starts at chunk 1, and its own causal zero-padding then
// falls at that chunk rather than at the clip's first frame. Every later
// chunk doubles its frame count.
SharedBuffer&
MetalWanVae::time_up_(Ctx& cx, const Resample& rs, const SharedBuffer& x,
                      int& t, std::size_t hw, int C, Carry* carry)
{
  if (carry != nullptr && !carry->seen) {
    carry->seen = true;
    return const_cast<SharedBuffer&>(x);
  }
  // Causal (3,1,1) over the frames, producing 2*C channels.
  SharedBuffer& wide = conv_chunk_(cx, rs.time, x, t, (int)hw, 1, 1, carry);
  SharedBuffer& out = cx.alloc(_mc, (std::size_t)t * 2 * hw * C);
  if (!cx.alloc_ok) { return out; }
  ComputeEncoder& enc = *cx.enc;
  enc.set_function(_fn_time_unshuffle);
  enc.set_buffer(0, wide); enc.set_buffer(1, out);
  enc.set_constant(2, C); enc.set_constant(3, (int)hw); enc.set_constant(4, t);
  enc.dispatch({(unsigned)C, (unsigned)hw, (unsigned)(2 * t)}, {64, 1, 1});
  cx.release(wide);
  t *= 2;
  return out;
}

// The encoder's temporal downsample. The first chunk keeps its frames and
// only seeds the carry; every later chunk prepends the single carried
// frame and strides by two, so a four-frame chunk becomes two frames and a
// two-frame chunk becomes one.
SharedBuffer&
MetalWanVae::time_down_(Ctx& cx, const Resample& rs, const SharedBuffer& x,
                        int& t, std::size_t hw, int C, Carry* carry)
{
  if (carry != nullptr && !carry->seen) {
    carry->seen = true;
    save_carry_(cx, *carry, x, t, hw, C);
    return const_cast<SharedBuffer&>(x);
  }
  // The concatenated sequence is [carry_last, chunk...]; output frame o
  // reads concatenated frames 2o, 2o+1, 2o+2 (kernel 3, stride 2, NO
  // padding), i.e. chunk frames 2o-1, 2o, 2o+1.
  const int t_cat = t + 1;
  const int t_out = (t_cat >= 3) ? ((t_cat - 3) / 2 + 1) : 0;
  SharedBuffer& out = cx.alloc(_mc, (std::size_t)std::max(t_out, 1) * hw * C);
  if (!cx.alloc_ok) { return out; }
  const std::size_t frame_el = hw * (std::size_t)C;
  for (int o = 0; o < t_out; ++o) {
    const SharedBuffer* taps[3] = {nullptr, nullptr, nullptr};
    std::size_t offs[3] = {0, 0, 0};
    for (int kt = 0; kt < 3; ++kt) {
      const int ci = 2 * o + kt - 1;       // chunk-local frame index
      if (ci >= 0) {
        taps[kt] = &x;
        offs[kt] = (std::size_t)ci * frame_el;
      } else if (carry != nullptr && carry->total >= 1) {
        // The single carried frame is the previous chunk's last, at the
        // slot its sequence index selects.
        taps[kt] = &carry->buf;
        offs[kt] = (std::size_t)((carry->total - 1) % 2) * frame_el;
      }
    }
    conv_frame_(cx, rs.time, taps, offs, out, (std::size_t)o * hw, (int)hw, 1,
                1);
  }
  if (carry != nullptr) { save_carry_(cx, *carry, x, t, hw, C); }
  t = t_out;
  return out;
}

// ---- decode ------------------------------------------------------------

std::size_t
MetalWanVae::decode_peak_bytes(int h8, int w8) const noexcept
{
  return decode_peak_bytes(
      _cfg, h8, w8, std::getenv("VPIPE_WAN_VAE_NO_FRAME_SPLIT") == nullptr,
      1.0);
}

// Walks decode()'s topology for a STEADY chunk -- four output frames; the
// first chunk is one, and smaller everywhere. Three terms, in f16 bytes.
//
// CARRIES, exact: two input frames of every causal conv, allocated by the
// first chunk and held to the end of the clip. The estimate this replaced
// had no such term and called itself per-chunk, which is how a 1920x1152
// decode that holds ~10 GB of carries was booked at under 6 GB in total.
//
// The chunk POOL is first-fit and hands nothing back inside a chunk, and
// the decoder only ever widens, so a level keeps the slots it grew while
// the next adds its own: kSlots of each level's largest buffer, where a
// level opens at the nearest-2x buffer that feeds it. kSlots is
// CALIBRATED, not derived (flashvsr_vae.frame_split_is_exact_and_bounded):
// the measured pool is 1.84 slots a level over the whole chunk and 1.74
// split, identically at 256x256 and 512x512 -- the peaks scale by exactly
// 4.0x, so nothing in them is a fixed overhead -- which puts this figure
// 4.3% and 3.3% above the measurement.
//
// The OUTPUT: the chunk's RGB twice, the pool's and the sink's.
std::size_t
MetalWanVae::decode_peak_bytes(const Config& cfg, int h8, int w8,
                               bool frame_split, double tail_frac) noexcept
{
  if (h8 <= 0 || w8 <= 0) { return 0; }
  constexpr std::size_t kSlots = 2;
  const std::size_t hw8 = (std::size_t)h8 * w8;
  const std::size_t base = (std::size_t)cfg.base_dim;
  const std::size_t d[5] = {base * cfg.dim_mult[3], base * cfg.dim_mult[3],
                            base * cfg.dim_mult[2], base * cfg.dim_mult[1],
                            base * cfg.dim_mult[0]};
  const bool tup[3] = {cfg.temperal_downsample[2], cfg.temperal_downsample[1],
                       cfg.temperal_downsample[0]};
  int split_at = 0;
  for (int i = 0; i < 3; ++i) {
    if (tup[i]) { split_at = i; }
  }

  // Below the split the tail may run on TILES, so everything there holds
  // one tile's plane (plus its halo) rather than the whole one. Above it
  // -- conv_in, the mid block with its whole-plane attention, and the
  // blocks up to the split -- nothing tiles and the share is 1.
  const double frac = (tail_frac > 0.0 && tail_frac < 1.0) ? tail_frac : 1.0;
  auto tiled = [&](std::size_t hw) {
    return (std::size_t)((double)hw * frac + 0.5);
  };

  std::size_t carries = 0, pool = 0;
  auto keep = [&](std::size_t hw, std::size_t cin) {
    carries += 2 * hw * cin * 2;
  };
  std::size_t level = hw8 * d[0];            // ELEMENTS, the level's largest
  auto next_level = [&](std::size_t opens) {
    pool += kSlots * level * 2;
    level = opens;
  };

  keep(hw8, (std::size_t)cfg.z_dim);                     // conv_in
  for (int r = 0; r < 4; ++r) { keep(hw8, d[0]); }       // mid, 2 x 2 convs
  // `t` frames in the chunk; `nt` frames in each buffer, which drops to one
  // past the split.
  std::size_t hw = hw8, t = 1, nt = 1;
  for (int i = 0; i < 4; ++i) {
    const std::size_t in = (i > 0) ? d[i] / 2 : d[i];
    const std::size_t out = d[i + 1];
    // The tail begins at the split block's SPATIAL half, so that block's
    // resnets are still whole-plane and everything after them is not.
    const std::size_t rhw = (i > split_at) ? tiled(hw) : hw;
    for (int r = 0; r <= cfg.num_res_blocks; ++r) {
      keep(rhw, r == 0 ? in : out);
      keep(rhw, out);
    }
    level = std::max(level, nt * rhw * std::max(in, out));
    if (i == 3) { break; }                   // the last block has no resample
    if (tup[i]) {
      keep(hw, out);                         // its time_conv
      level = std::max(level, nt * hw * 2 * out);
      t *= 2;
      nt = (frame_split && i >= split_at) ? 1 : t;
    }
    const std::size_t uhw = (i >= split_at) ? tiled(hw) : hw;
    next_level(nt * 4 * uhw * out);
    hw *= 4;
  }
  keep(tiled(hw), d[4]);                     // conv_out
  next_level(0);
  // The chunk's RGB twice: the tile's (pooled) and the whole frame's, which
  // the tiles are assembled into and the sink reads. Untiled the two are
  // the same size, which is the `2 x` this term has always carried.
  const std::size_t rgb_whole = 2 * t * hw * 3;
  const std::size_t rgb_tile =
      (std::size_t)((double)rgb_whole * frac + 0.5);
  return carries + pool + rgb_whole + rgb_tile;
}

// Pixels of real neighbourhood a tile must carry on each side so its
// INTERIOR is what an untiled decode would have produced. One per 3x3
// conv below the split, each counted at its own resolution and brought
// back to the split plane -- a conv at 4x costs a quarter of a
// split-plane pixel, because four of its pixels fit in one of ours.
//
// Exact rather than generous on purpose: the halo is the tiling's only
// overhead, and at 1920x1152 a pixel of it is 480 latent cells.
int
MetalWanVae::tail_halo_() const noexcept
{
  std::size_t split_at = 0;
  for (std::size_t i = 0; i < _up_blocks.size(); ++i) {
    if (_up_blocks[i].up.present && _up_blocks[i].up.temporal) {
      split_at = i;
    }
  }
  double halo = 0.0, scale = 1.0;
  for (std::size_t i = split_at; i < _up_blocks.size(); ++i) {
    const UpBlock& ub = _up_blocks[i];
    if (i != split_at) {
      // Every resblock is two 3x3 convs; the temporal taps add no
      // spatial reach.
      halo += 2.0 * (double)ub.resnets.size() / scale;
    }
    if (ub.up.present) {
      scale *= 2.0;                    // nearest upsample: no reach of its own
      halo += 1.0 / scale;             // the block's spatial conv, after it
    }
  }
  halo += 1.0 / scale;                 // conv_out
  return (int)std::ceil(halo);
}

// The tail's input plane: the latent, doubled by every spatial upsample
// that runs ABOVE the split.
void
MetalWanVae::tail_plane_(int h8, int w8, int* hs, int* ws) const noexcept
{
  std::size_t split_at = 0;
  for (std::size_t i = 0; i < _up_blocks.size(); ++i) {
    if (_up_blocks[i].up.present && _up_blocks[i].up.temporal) {
      split_at = i;
    }
  }
  int h = h8, w = w8;
  for (std::size_t i = 0; i < split_at; ++i) {
    if (_up_blocks[i].up.present) { h *= 2; w *= 2; }
  }
  if (hs != nullptr) { *hs = h; }
  if (ws != nullptr) { *ws = w; }
}

MetalWanVae::TailTiles
MetalWanVae::choose_tail_tiles_(int h8, int w8, std::size_t headroom,
                                bool frame_split, bool* fits) const noexcept
{
  TailTiles t;
  if (fits != nullptr) { *fits = true; }
  if (headroom == 0) { return t; }             // nothing to size against
  if (decode_peak_bytes(_cfg, h8, w8, frame_split, 1.0) <= headroom) {
    return t;                                  // the whole plane fits
  }
  TailTiles finest;                            // the last grid that is legal
  int hs = 0, ws = 0;
  tail_plane_(h8, w8, &hs, &ws);
  const int halo = align8_(tail_halo_());
  // Candidate grids in order of TILE COUNT, because the count is what the
  // tiling costs: every tile re-runs the tail's dispatch chain over a
  // smaller plane. MEASURED at 1920x1152 on the M5 Pro, one chunk:
  // 2x2 is 3% slower than whole and 5x5 is 26%, for 8.1 GB and 3.9 GB of
  // footprint against 8.9. So take the COARSEST grid that fits and stop --
  // a finer one buys memory nobody asked for at a price in time.
  static const int kGrids[][2] = {{1, 2}, {2, 1}, {2, 2}, {2, 3}, {3, 2},
                                  {3, 3}, {3, 4}, {4, 3}, {4, 4}, {4, 5},
                                  {5, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
  for (const auto& g : kGrids) {
    const int bh = align8_((hs + g[0] - 1) / g[0]);
    const int bw = align8_((ws + g[1] - 1) / g[1]);
    // A tile smaller than its own halo is all overhead; skip rather than
    // grind the plane into borders.
    if (bh <= halo || bw <= halo) { continue; }
    const int ty = (hs + bh - 1) / bh, tx = (ws + bw - 1) / bw;
    if (ty <= 1 && tx <= 1) { continue; }
    const double frac = (double)(bh + 2 * halo) * (double)(bw + 2 * halo)
                      / ((double)hs * (double)ws);
    if (frac >= 1.0) { continue; }
    finest.ty = ty; finest.tx = tx; finest.bh = bh; finest.bw = bw;
    finest.halo = halo; finest.frac = frac;
    if (decode_peak_bytes(_cfg, h8, w8, frame_split, frac) <= headroom) {
      return finest;
    }
  }
  // Nothing fit. The finest grid tried is what the caller reports.
  if (fits != nullptr) { *fits = false; }
  return finest;
}

bool
MetalWanVae::decode(const SharedBuffer& z, int T, int h8, int w8,
                    const FrameSink& on_frame, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int Cz = _cfg.z_dim;
  const std::size_t hw0 = (std::size_t)h8 * w8;
  if (T <= 0 || h8 <= 0 || w8 <= 0) { return fail("empty latent"); }
  if (z.byte_size() < (std::size_t)Cz * T * hw0 * 2) {
    return fail("input latent smaller than [z_dim, T, h8, w8]");
  }
  if (!on_frame) { return fail("no frame sink"); }
  MetalCompute* mc = _mc;
  const int Hout = h8 * 8, Wout = w8 * 8;
  const int base = _cfg.base_dim;

  // Preflight: refuse a decode that clearly won't fit rather than
  // allocating into an out-of-memory mid-clip (which corrupts the output).
  //
  // On PHYSICAL memory only. The working set is advisory on UMA -- the
  // same rule vae-decode applies around this call -- and with the carries
  // counted, a 1920x1152 decode that runs needs more than a 24 GB box's
  // working set reports free before anything is allocated.
  //
  // The band's share is bounded by both: whatever this decode leaves of
  // the smaller of the two. Sized from the working set alone it took RAM
  // the box did not have.
  const bool frame_split_on =
      std::getenv("VPIPE_WAN_VAE_NO_FRAME_SPLIT") == nullptr;
  // Spatial tiling of the tail, decided by the preflight below: {1,1}
  // unless the whole plane does not fit.
  TailTiles tiles;
  {
    // Test hook: force a grid so the tiled path can be held against the
    // whole-plane one at a size where both fit. "2x2", or "0" for off.
    const char* e = std::getenv("VPIPE_WAN_VAE_TILE");
    int ty = 0, tx = 0;
    if (e != nullptr && std::sscanf(e, "%dx%d", &ty, &tx) == 2 &&
        ty >= 1 && tx >= 1 && (ty > 1 || tx > 1)) {
      int hs = 0, ws = 0;
      tail_plane_(h8, w8, &hs, &ws);
      const int halo = align8_(tail_halo_());
      const int bh = align8_((hs + ty - 1) / ty);
      const int bw = align8_((ws + tx - 1) / tx);
      if (bh > halo && bw > halo) {
        tiles.ty = (hs + bh - 1) / bh;
        tiles.tx = (ws + bw - 1) / bw;
        tiles.bh = bh; tiles.bw = bw; tiles.halo = halo;
        tiles.frac = (double)(bh + 2 * halo) * (double)(bw + 2 * halo)
                   / ((double)hs * (double)ws);
      }
    }
  }
  std::size_t headroom = 0;
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    headroom = (mb.recommended != 0) ? mb.headroom : 0;
    if (headroom != 0 && mb.available_physical != 0) {
      headroom = std::min(headroom, mb.available_physical);
    }
    std::size_t need = decode_peak_bytes(h8, w8);
    // NO MARGIN on top. fits_physical's default 10% is for a guess, and
    // this is not one: the carries are exact and the pool term is
    // calibrated 3-4% ABOVE a measured peak, so a margin counts the same
    // caution twice. MEASURED at 1920x1152 on a 24 GB box: need 13417 MB
    // against 14153 MB reclaimable -- a real peak of ~13.0 GB with a GB to
    // spare -- refused by the margin alone.
    // ...plus what this process still holds for GPU buffers that no longer
    // exist. The driver returns a freed buffer's pages asynchronously,
    // tens of milliseconds after the free, and a new allocation reuses
    // them meanwhile -- room available_physical does not count yet (see
    // MemoryBudget::self_graphics). A decode that starts right behind a
    // large free is exactly that case: MEASURED on a 24 GB box, a second
    // 1920x1152 decode in one process sampled an 11106 MB footprint over
    // 278 MB of live buffers and was refused against 6945 MB
    // "reclaimable" before this term existed.
    std::size_t live =
        metal_compute::shared_buffer_memory_stats().live_bytes;
    std::size_t reusable =
        mb.self_graphics > live ? mb.self_graphics - live : 0;
    // TILE THE TAIL BEFORE REFUSING. The whole-plane figure above is what
    // an untiled decode holds; below the mid attention everything is local,
    // so the same decode can run on tiles and hold a fraction of it. A
    // geometry that fits whole still runs whole -- tiles are the answer to
    // a box that would otherwise be told no.
    MetalCompute::MemoryBudget cur = mb;
    std::size_t have = (cur.available_physical != 0)
                           ? cur.available_physical + reusable
                           : 0;
    if (tiles.ty > 1 || tiles.tx > 1) {
      need = decode_peak_bytes(_cfg, h8, w8, frame_split_on, tiles.frac);
    }
    // AND WAIT FOR THE DENOISER'S PAGES BEFORE REFUSING. A decode runs
    // right behind the forward that produced its latent, and that forward's
    // buffers are wired: they are neither free, purgeable nor file-backed,
    // so `available_physical` reads a few GB while they are held and tens
    // of GB a moment later. MEASURED on a 24 GB box at 1920x1152: the same
    // clip that was refused against 1.8 GB had 13 GB a second later, and
    // three of 38 clips were lost to that window alone.
    //
    // So a shortfall is a QUESTION about timing, not an answer. Re-sample
    // for a bounded spell, and take the geometry the box can hold when it
    // settles -- tiles included, since a smaller tiling may fit before the
    // whole plane does. Bounded because a box that is genuinely too small
    // must still say so rather than hang the pipeline.
    constexpr int    kWaitSteps = 12;         // x 250 ms = 3 s
    constexpr double kWaitMs    = 250.0;
    int waited = 0;
    for (int i = 0; i < kWaitSteps && have != 0 && need > have; ++i) {
      bool fits = false;
      const TailTiles t =
          choose_tail_tiles_(h8, w8, have, frame_split_on, &fits);
      if (fits && (t.ty > 1 || t.tx > 1)) {
        tiles = t;
        need = decode_peak_bytes(_cfg, h8, w8, frame_split_on, tiles.frac);
        if (need <= have) { break; }
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds((int)kWaitMs));
      ++waited;
      cur = mc->memory_budget();
      live = metal_compute::shared_buffer_memory_stats().live_bytes;
      reusable = cur.self_graphics > live ? cur.self_graphics - live : 0;
      have = (cur.available_physical != 0)
                 ? cur.available_physical + reusable
                 : 0;
      // The whole plane may be affordable again now, and it is both faster
      // and simpler than any tiling.
      if (decode_peak_bytes(_cfg, h8, w8, frame_split_on, 1.0) <= have) {
        tiles = TailTiles{};
        need = decode_peak_bytes(_cfg, h8, w8, frame_split_on, 1.0);
      }
    }
    if (waited > 0 && need <= have && mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalWanVae: waited {} ms for the forward's pages before a {}x{} "
          "decode -- {} MB free now against {} MB needed",
          (int)(waited * kWaitMs), Wout, Hout, have >> 20, need >> 20));
    }
    if (have != 0 && need > have) {
      // Nothing fits even tiled: name the finest grid tried, so a box that
      // is too small for this geometry reads differently from a moment
      // that was.
      const TailTiles finest =
          choose_tail_tiles_(h8, w8, have, frame_split_on);
      if (finest.ty > 1 || finest.tx > 1) {
        tiles = finest;
        need = decode_peak_bytes(_cfg, h8, w8, frame_split_on, finest.frac);
      }
      // Who holds the rest, since "not enough" alone cannot say whether
      // the box is full or this process is. The tile count says whether
      // this is a box too small for the geometry at all, or one that was
      // asked at a bad moment: tiles are already the smallest this decode
      // can be made.
      const std::string how =
          (tiles.ty > 1 || tiles.tx > 1)
              ? fmt(" even with the tail tiled {}x{}", tiles.ty, tiles.tx)()
              : std::string();
      return fail(fmt(
          "insufficient free RAM for a {}x{} video decode{}: need ~{} MB, "
          "~{} MB reclaimable and ~{} MB held for freed GPU buffers; this "
          "process holds ~{} MB, ~{} MB of it in live GPU buffers", Wout,
          Hout, how, need >> 20, mb.available_physical >> 20, reusable >> 20,
          mb.self_footprint >> 20, live >> 20)());
    }
  }

  if ((tiles.ty > 1 || tiles.tx > 1) && mc->session() != nullptr) {
    mc->session()->log_normal(fmt(
        "MetalWanVae: {}x{} decode runs the tail in {}x{} tiles (halo {} px, "
        "~{} MB against ~{} MB whole) -- exact, the mid attention stays "
        "whole-plane",
        Wout, Hout, tiles.ty, tiles.tx, tiles.halo,
        decode_peak_bytes(_cfg, h8, w8, frame_split_on, tiles.frac) >> 20,
        decode_peak_bytes(_cfg, h8, w8, frame_split_on, 1.0) >> 20));
  }

  // The im2col band. At 27 taps the full [H*W, 27*cin] of a top-level conv
  // is multi-GB, so it is always streamed; the cap gets whatever headroom
  // the chunk activations leave, floored at eight output rows.
  const std::size_t widest = (std::size_t)27 * base * _cfg.dim_mult[1];
  const std::size_t floor_band = (std::size_t)Wout * widest * 8;
  const std::size_t full_band = (std::size_t)Hout * Wout * widest;
  std::size_t col_cap = full_band;
  if (headroom > 0) {
    // What the decode itself will hold -- the TILED figure when the
    // preflight chose tiles, or the band is sized against a reserve this
    // decode is not going to take and starves for room that is free.
    const std::size_t reserve =
        decode_peak_bytes(_cfg, h8, w8, frame_split_on, tiles.frac);
    const std::size_t avail = headroom > reserve ? (headroom - reserve) / 2 : 0;
    col_cap = std::min(full_band, std::max(floor_band, avail));
  }
  if (_mma_max_m > 0) {
    col_cap = std::min(col_cap, (std::size_t)_mma_max_m / 2 * widest);
  }
  if (const char* e = std::getenv("VPIPE_WAN_VAE_BAND_ROWS")) {
    const long r = std::atol(e);
    if (r > 0) { col_cap = (std::size_t)r * widest; }
  }
  // What the band came out as, for attributing a slow decode: the cap is a
  // count of GEMM rows (PIXELS) at the widest conv, not of image rows.
  if (std::getenv("VPIPE_WAN_VAE_LOG") != nullptr) {
    std::fprintf(stderr,
                 "[wan-vae] decode %dx%d, T=%d: need %zu MB, headroom %zu MB, "
                 "band %zu MB = %zu px at the widest conv (%.1f image rows)\n",
                 Wout, Hout, T, decode_peak_bytes(h8, w8) >> 20,
                 headroom >> 20, (col_cap * 2) >> 20, col_cap / widest,
                 (double)(col_cap / widest) / (double)Wout);
  }

  // Per-conv carries, in traversal order. The count is fixed by the
  // topology, so index them by a counter reset at each chunk exactly as
  // the reference resets feat_idx.
  std::vector<Carry> carry;
  carry.resize(256);
  std::size_t ci = 0;
  auto next_carry = [&]() -> Carry* {
    if (ci >= carry.size()) { carry.resize(ci + 64); }
    return &carry[ci++];
  };

  for (int f = 0; f < T; ++f) {
    ci = 0;
    CommandStream stream = mc->make_command_stream();
    Ctx cx;
    cx.use_pool = std::getenv("VPIPE_WAN_NO_VAE_POOL") == nullptr;
    // The band is allocated by the first conv that GATHERS (conv_frame_).
    // On the hardware conv none does, and at 1920x1152 that is 2.6 GB this
    // chunk never touches.
    cx.col_cap = col_cap;
    const SharedBuffer* rgb = nullptr;
    int t = 1;                       // frames in this chunk
    int H = h8, W = w8;
    {
      ComputeEncoder enc = stream.begin_compute();
      cx.enc = &enc;

      // The latent frame, channel-first [Cz, T, h8, w8] -> channel-last
      // [hw0, Cz] (host-side; the latent arrives from the sampler).
      SharedBuffer& x0 = cx.alloc(mc, hw0 * Cz);
      if (!cx.alloc_ok) { return fail("chunk allocation failed"); }
      {
        const auto* s = static_cast<const _Float16*>(z.contents());
        auto* d = static_cast<_Float16*>(x0.contents());
        for (int c = 0; c < Cz; ++c) {
          const std::size_t src = ((std::size_t)c * T + f) * hw0;
          for (std::size_t p = 0; p < hw0; ++p) {
            d[p * Cz + c] = s[src + p];
          }
        }
      }
      // post_quant_conv (1x1), then the decoder proper.
      SharedBuffer& pq = cx.alloc(mc, hw0 * Cz);
      gemm_bias_(cx, x0, _post_quant.w, _post_quant.b, pq, (int)hw0, Cz, Cz);
      cx.release(x0);

      const SharedBuffer* x = &conv_chunk_(cx, _conv_in, pq, t, H, W, 1,
                                           next_carry());
      cx.release(pq);
      auto step = [&](SharedBuffer& nx) { cx.release(*x); x = &nx; };

      {
        Carry* a = next_carry(); Carry* b = next_carry();
        step(resblock_(cx, _mid_res0, *x, t, H, W, a, b));
      }
      step(attention_(cx, _mid_attn, *x, t, H, W));
      {
        Carry* a = next_carry(); Carry* b = next_carry();
        step(resblock_(cx, _mid_res1, *x, t, H, W, a, b));
      }

      // The whole chunk runs up to and including the LAST temporal
      // upsample. What follows it -- that block's spatial half, every
      // later block, the head -- mixes frames only through a carry, so it
      // runs one frame at a time: the same arithmetic in the same order,
      // over a quarter of the working set. It is also where the
      // resolution is, so it is where the working set is.
      //
      // MEASURED at 1920x1152 (FlashVSR's 4x output) before this: a 29.0 GB
      // peak footprint on a 24 GB box, ~9.5 GB of swap, and a decode four
      // times longer than the denoise in front of it.
      // VPIPE_WAN_VAE_NO_FRAME_SPLIT runs the tail over the whole chunk.
      std::size_t split_at = 0;
      for (std::size_t i = 0; i < _up_blocks.size(); ++i) {
        if (_up_blocks[i].up.present && _up_blocks[i].up.temporal) {
          split_at = i;
        }
      }
      for (std::size_t i = 0; i <= split_at; ++i) {
        const UpBlock& ub = _up_blocks[i];
        for (const ResBlock& rb : ub.resnets) {
          Carry* a = next_carry(); Carry* b = next_carry();
          step(resblock_(cx, rb, *x, t, H, W, a, b));
        }
        if (ub.up.present && ub.up.temporal) {
          SharedBuffer& up = time_up_(cx, ub.up, *x, t, (std::size_t)H * W,
                                      ub.up_dim, next_carry());
          if (&up != x) { step(up); }
        }
        if (!cx.alloc_ok) { return fail("chunk allocation failed"); }
        if (i == split_at) { break; }
        if (ub.up.present) {
          step(upsample2x_(cx, *x, t, H, W, ub.up_dim));
          H *= 2; W *= 2;
          step(conv_chunk_(cx, ub.up.space, *x, t, H, W, 1, nullptr));
        }
      }

      // Block `split_at`'s spatial half onward, over `nt` frames of `in`
      // at th x tw, which it advances to the output size. Returns the
      // clamped RGB, [nt*th*tw, 3]. Never releases `in`: the split path
      // refills it for every frame.
      auto tail = [&](const SharedBuffer& in, int nt, int& th,
                      int& tw) -> SharedBuffer* {
        const SharedBuffer* y = &in;
        auto adv = [&](SharedBuffer& ny) {
          if (y != &in) { cx.release(*y); }
          y = &ny;
        };
        for (std::size_t i = split_at; i < _up_blocks.size(); ++i) {
          const UpBlock& ub = _up_blocks[i];
          if (i != split_at) {
            for (const ResBlock& rb : ub.resnets) {
              Carry* a = next_carry(); Carry* b = next_carry();
              adv(resblock_(cx, rb, *y, nt, th, tw, a, b));
            }
          }
          if (ub.up.present) {
            adv(upsample2x_(cx, *y, nt, th, tw, ub.up_dim));
            th *= 2; tw *= 2;
            adv(conv_chunk_(cx, ub.up.space, *y, nt, th, tw, 1, nullptr));
          }
          if (!cx.alloc_ok) { return nullptr; }
        }
        const std::size_t rows = (std::size_t)nt * th * tw;
        SharedBuffer& yn = normc_(cx, *y, rows, base, _norm_out_g);
        if (y != &in) { cx.release(*y); }
        silu_(cx, yn, rows * (std::size_t)base);
        SharedBuffer& out = conv_chunk_(cx, _conv_out, yn, nt, th, tw, 1,
                                        next_carry());
        cx.release(yn);
        if (!cx.alloc_ok) { return nullptr; }
        const std::size_t n = rows * 3;
        enc.set_function(_fn_clamp);
        enc.set_buffer(0, out); enc.set_buffer(1, out);
        enc.set_constant(2, (int)n);
        enc.set_constant(3, -1.0f); enc.set_constant(4, 1.0f);
        enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
        return &out;
      };

      // ---- the tail over spatial TILES ---------------------------------
      //
      // Only what runs below this point tiles, and that is the point: the
      // mid-block attention above it is the one layer whose receptive
      // field is the whole plane. Everything here is 3x3 convs, a nearest
      // upsample and a per-pixel RMS over channels, so a tile that carries
      // `halo` pixels of real neighbourhood on each interior side produces
      // an interior identical to the untiled decode's -- the zero padding
      // a tile edge would otherwise invent never reaches it.
      //
      // Each tile keeps its OWN carries: a carry holds the last two input
      // frames at that conv's plane, so tiles sharing one would overwrite
      // each other's temporal history (and `save_carry_` would reallocate
      // on every size change, silently dropping it).
      const int up_mult = [&] {
        int m = 1;
        for (std::size_t i = split_at; i < _up_blocks.size(); ++i) {
          if (_up_blocks[i].up.present) { m *= 2; }
        }
        return m;
      }();
      const std::size_t ci_tail0 = ci;
      std::size_t tail_stride = 0;
      auto tail_tiled_one = [&](const SharedBuffer& in, int hin, int win,
                                int& oh, int& ow) -> SharedBuffer* {
        const int C = _up_blocks[split_at].up_dim;
        oh = hin * up_mult;
        ow = win * up_mult;
        SharedBuffer& out = cx.alloc(mc, (std::size_t)oh * ow * 3);
        if (!cx.alloc_ok) { return nullptr; }
        const int bh = tiles.bh > 0 ? tiles.bh
                                    : (hin + tiles.ty - 1) / tiles.ty;
        const int bw = tiles.bw > 0 ? tiles.bw
                                    : (win + tiles.tx - 1) / tiles.tx;
        int idx = 0;
        for (int gy = 0; gy < tiles.ty; ++gy) {
          for (int gx = 0; gx < tiles.tx; ++gx, ++idx) {
            const int y0 = gy * bh, y1 = std::min(hin, y0 + bh);
            const int x0 = gx * bw, x1 = std::min(win, x0 + bw);
            if (y0 >= y1 || x0 >= x1) { continue; }
            const int ey0 = std::max(0, y0 - tiles.halo);
            const int ey1 = std::min(hin, y1 + tiles.halo);
            const int ex0 = std::max(0, x0 - tiles.halo);
            const int ex1 = std::min(win, x1 + tiles.halo);
            const int th0 = ey1 - ey0, tw0 = ex1 - ex0;
            SharedBuffer& tile =
                cx.alloc(mc, (std::size_t)th0 * tw0 * (std::size_t)C);
            if (!cx.alloc_ok) { return nullptr; }
            enc.set_function(_fn_copy_rect);
            enc.set_buffer(0, in);
            enc.set_buffer(1, tile);
            enc.set_constant(2, (int)(((std::size_t)ey0 * win + ex0) * C));
            enc.set_constant(3, 0);
            enc.set_constant(4, th0);
            enc.set_constant(5, tw0 * C);
            enc.set_constant(6, win * C);
            enc.set_constant(7, tw0 * C);
            enc.dispatch({(unsigned)((std::size_t)th0 * tw0 * C), 1, 1},
                         {256, 1, 1});

            // This tile's carry block. The stride is the tail's own carry
            // count, learned from the first tile and identical for every
            // one after it -- the topology does not vary with the plane.
            ci = ci_tail0 + (std::size_t)idx * tail_stride;
            int th = th0, tw = tw0;
            SharedBuffer* o = tail(tile, 1, th, tw);
            if (o == nullptr) { return nullptr; }
            if (tail_stride == 0) { tail_stride = ci - ci_tail0; }

            const int iy = (y0 - ey0) * up_mult, ix = (x0 - ex0) * up_mult;
            const int rows = (y1 - y0) * up_mult;
            const int cols = (x1 - x0) * up_mult;
            enc.set_function(_fn_copy_rect);
            enc.set_buffer(0, *o);
            enc.set_buffer(1, out);
            enc.set_constant(2, (int)(((std::size_t)iy * tw + ix) * 3));
            enc.set_constant(
                3, (int)(((std::size_t)y0 * up_mult * ow + x0 * up_mult) * 3));
            enc.set_constant(4, rows);
            enc.set_constant(5, cols * 3);
            enc.set_constant(6, tw * 3);
            enc.set_constant(7, ow * 3);
            enc.dispatch({(unsigned)((std::size_t)rows * cols * 3), 1, 1},
                         {256, 1, 1});
            cx.release(*o);
            cx.release(tile);
          }
        }
        return &out;
      };
      const bool tiled = tiles.ty > 1 || tiles.tx > 1;

      const bool split =
          t > 1 && std::getenv("VPIPE_WAN_VAE_NO_FRAME_SPLIT") == nullptr;
      if (!split) {
        if (tiled) {
          // One frame in this chunk (the clip's first), so the same
          // per-frame tiling serves; `t` > 1 always takes the split path.
          int oh = 0, ow = 0;
          rgb = tail_tiled_one(*x, H, W, oh, ow);
          H = oh; W = ow;
        } else {
          rgb = tail(*x, t, H, W);
        }
        cx.release(*x);
      } else {
        // The chunk's frames at the split, and where each one's RGB lands.
        const std::size_t fel =
            (std::size_t)H * W * (std::size_t)_up_blocks[split_at].up_dim;
        int oh = H, ow = W;
        for (std::size_t i = split_at; i < _up_blocks.size(); ++i) {
          if (_up_blocks[i].up.present) { oh *= 2; ow *= 2; }
        }
        const std::size_t ofel = (std::size_t)oh * ow * 3;
        SharedBuffer& all = cx.alloc(mc, (std::size_t)t * ofel);
        SharedBuffer& one = cx.alloc(mc, fel);
        const std::size_t ci0 = ci;
        const int H0 = H, W0 = W;
        bool ok = cx.alloc_ok;
        for (int k = 0; k < t && ok; ++k) {
          ci = ci0;                    // every frame walks the same carries
          enc.set_function(_fn_copy);
          enc.set_buffer(0, *x, (std::size_t)k * fel * 2);
          enc.set_buffer(1, one);
          enc.set_constant(2, 0);
          enc.set_constant(3, (int)fel);
          enc.dispatch({(unsigned)fel, 1, 1}, {256, 1, 1});
          H = H0; W = W0;
          SharedBuffer* o = nullptr;
          if (tiled) {
            int oh = 0, ow = 0;
            o = tail_tiled_one(one, H, W, oh, ow);
            H = oh; W = ow;
          } else {
            o = tail(one, 1, H, W);
          }
          ok = o != nullptr;
          if (!ok) { break; }
          enc.set_function(_fn_copy);
          enc.set_buffer(0, *o);
          enc.set_buffer(1, all);
          enc.set_constant(2, (int)((std::size_t)k * ofel));
          enc.set_constant(3, (int)ofel);
          enc.dispatch({(unsigned)ofel, 1, 1}, {256, 1, 1});
          cx.release(*o);
        }
        cx.release(one);
        cx.release(*x);
        rgb = ok ? &all : nullptr;
      }
      if (rgb == nullptr) { return fail("chunk allocation failed"); }
    }
    if (!cx.alloc_ok) {
      return fail("a decode intermediate allocation failed (out of GPU "
                  "memory)");
    }
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      return fail(gpu_err.empty() ? std::string("GPU video decode failed")
                                  : gpu_err);
    }

    // Channel-last [t*H*W, 3] -> channel-first [3, t, H, W] for the sink.
    const std::size_t hw = (std::size_t)H * W;
    SharedBuffer frames = mc->make_shared_buffer((std::size_t)3 * t * hw * 2);
    if (frames.empty()) { return fail("frame buffer allocation failed"); }
    {
      const auto* s = static_cast<const _Float16*>(rgb->contents());
      auto* d = static_cast<_Float16*>(frames.contents());
      for (int ff = 0; ff < t; ++ff) {
        for (std::size_t p = 0; p < hw; ++p) {
          for (int c = 0; c < 3; ++c) {
            d[((std::size_t)c * t + ff) * hw + p] =
                s[((std::size_t)ff * hw + p) * 3 + c];
          }
        }
      }
    }
    const int frame0 = (f == 0) ? 0 : (1 + 4 * (f - 1));
    if (!on_frame(frames, frame0, t)) { return true; }   // sink stopped
  }
  return true;
}

SharedBuffer
MetalWanVae::decode_frame(const SharedBuffer& z, int h8, int w8,
                          std::string* err)
{
  SharedBuffer out;
  const bool ok = decode(z, 1, h8, w8,
                         [&](const SharedBuffer& rgb, int, int) {
                           out = _mc->make_shared_buffer(rgb.byte_size());
                           if (out.empty()) { return false; }
                           std::memcpy(out.contents(), rgb.contents(),
                                       rgb.byte_size());
                           return true;
                         },
                         err);
  if (!ok) { return {}; }
  return out;
}

// ---- encode ------------------------------------------------------------

SharedBuffer
MetalWanVae::encode(const SharedBuffer& video, int F, int H, int W,
                    std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  if (!_has_encoder && !ensure_encoder()) {
    return fail("the VAE encoder half is not loaded");
  }
  if (F <= 0 || (F - 1) % 4 != 0) {
    return fail("frame count must satisfy F % 4 == 1 (1, 5, ... 81, ...)");
  }
  if ((H % 8) != 0 || (W % 8) != 0) {
    return fail("height and width must be multiples of 8");
  }
  const std::size_t hw = (std::size_t)H * W;
  if (video.byte_size() < (std::size_t)3 * F * hw * 2) {
    return fail("input video smaller than [3, F, H, W]");
  }
  MetalCompute* mc = _mc;
  const int base = _cfg.base_dim;
  const int Cz = _cfg.z_dim;
  const int T = latent_frames(F);
  const int h8 = H / 8, w8 = W / 8;
  const std::size_t lhw = (std::size_t)h8 * w8;

  const std::size_t widest = (std::size_t)27 * base * _cfg.dim_mult[1];
  std::size_t col_cap = (std::size_t)W * widest * 64;
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    if (mb.recommended != 0 && mb.headroom > 0) {
      col_cap = std::min(col_cap, mb.headroom / 4);
    }
  }
  if (_mma_max_m > 0) {
    col_cap = std::min(col_cap, (std::size_t)_mma_max_m / 2 * widest);
  }
  col_cap = std::max(col_cap, (std::size_t)W * widest);

  // The mean half of the posterior, accumulated across chunks.
  SharedBuffer moments =
      mc->make_shared_buffer((std::size_t)2 * Cz * T * lhw * 2);
  if (moments.empty()) { return fail("latent allocation failed"); }

  std::vector<Carry> carry;
  carry.resize(256);
  std::size_t ci = 0;
  auto next_carry = [&]() -> Carry* {
    if (ci >= carry.size()) { carry.resize(ci + 64); }
    return &carry[ci++];
  };

  const int n_chunks = 1 + (F - 1) / 4;
  int lat_out = 0;
  for (int k = 0; k < n_chunks; ++k) {
    ci = 0;
    const int f0 = (k == 0) ? 0 : (1 + 4 * (k - 1));
    int t = (k == 0) ? 1 : 4;
    CommandStream stream = mc->make_command_stream();
    Ctx cx;
    cx.use_pool = std::getenv("VPIPE_WAN_NO_VAE_POOL") == nullptr;
    cx.col = mc->make_shared_buffer(col_cap * 2);
    if (cx.col.empty()) { return fail("im2col band scratch allocation failed"); }
    cx.col_cap = col_cap;
    const SharedBuffer* zout = nullptr;
    int Hc = H, Wc = W;
    int t_final = 0;
    {
      ComputeEncoder enc = stream.begin_compute();
      cx.enc = &enc;
      // Channel-first [3, F, H, W] -> channel-last [t*hw, 3] for this chunk.
      SharedBuffer& x0 = cx.alloc(mc, (std::size_t)t * hw * 3);
      if (!cx.alloc_ok) { return fail("chunk allocation failed"); }
      {
        const auto* s = static_cast<const _Float16*>(video.contents());
        auto* d = static_cast<_Float16*>(x0.contents());
        for (int ff = 0; ff < t; ++ff) {
          for (int c = 0; c < 3; ++c) {
            const std::size_t src = ((std::size_t)c * F + (f0 + ff)) * hw;
            for (std::size_t p = 0; p < hw; ++p) {
              d[((std::size_t)ff * hw + p) * 3 + c] = s[src + p];
            }
          }
        }
      }
      const SharedBuffer* x =
          &conv_chunk_(cx, _enc_conv_in, x0, t, Hc, Wc, 1, next_carry());
      cx.release(x0);
      auto step = [&](SharedBuffer& nx) { cx.release(*x); x = &nx; };

      for (const DownStage& ds : _enc_down) {
        for (const ResBlock& rb : ds.resnets) {
          Carry* a = next_carry(); Carry* b = next_carry();
          step(resblock_(cx, rb, *x, t, Hc, Wc, a, b));
        }
        if (ds.down.present) {
          step(conv_chunk_(cx, ds.down.space, *x, t, Hc, Wc, 2, nullptr));
          Hc /= 2; Wc /= 2;
          if (ds.down.temporal) {
            SharedBuffer& d = time_down_(cx, ds.down, *x, t,
                                         (std::size_t)Hc * Wc,
                                         ds.down.space.cout, next_carry());
            if (&d != x) { step(d); }
          }
        }
        if (!cx.alloc_ok) { return fail("chunk allocation failed"); }
      }
      {
        Carry* a = next_carry(); Carry* b = next_carry();
        step(resblock_(cx, _enc_mid_res0, *x, t, Hc, Wc, a, b));
      }
      step(attention_(cx, _enc_mid_attn, *x, t, Hc, Wc));
      {
        Carry* a = next_carry(); Carry* b = next_carry();
        step(resblock_(cx, _enc_mid_res1, *x, t, Hc, Wc, a, b));
      }
      const std::size_t rows = (std::size_t)t * Hc * Wc;
      SharedBuffer& xn = normc_(cx, *x, rows, _enc_conv_out.cin,
                                _enc_norm_out_g);
      cx.release(*x);
      silu_(cx, xn, rows * (std::size_t)_enc_conv_out.cin);
      SharedBuffer& h = conv_chunk_(cx, _enc_conv_out, xn, t, Hc, Wc, 1,
                                    next_carry());
      cx.release(xn);
      // quant_conv (1x1) over the chunk's latent frames.
      SharedBuffer& q = cx.alloc(mc, rows * (std::size_t)(2 * Cz));
      gemm_bias_(cx, h, _quant_conv.w, _quant_conv.b, q, (int)rows, 2 * Cz,
                 2 * Cz);
      cx.release(h);
      zout = &q;
      t_final = t;
    }
    if (!cx.alloc_ok) { return fail("an encode allocation failed"); }
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      return fail(gpu_err.empty() ? std::string("GPU video encode failed")
                                  : gpu_err);
    }
    // Channel-last [t*lhw, 2*Cz] -> channel-first [2*Cz, T, h8, w8] at the
    // chunk's latent frame offset.
    {
      const auto* s = static_cast<const _Float16*>(zout->contents());
      auto* d = static_cast<_Float16*>(moments.contents());
      for (int ff = 0; ff < t_final && lat_out + ff < T; ++ff) {
        for (int c = 0; c < 2 * Cz; ++c) {
          const std::size_t dst =
              ((std::size_t)c * T + (lat_out + ff)) * lhw;
          for (std::size_t p = 0; p < lhw; ++p) {
            d[dst + p] = s[((std::size_t)ff * lhw + p) * (2 * Cz) + c];
          }
        }
      }
    }
    lat_out += t_final;
  }

  // The posterior MODE (mean) is the first z_dim channels; whiten it.
  SharedBuffer out = mc->make_shared_buffer((std::size_t)Cz * T * lhw * 2);
  if (out.empty()) { return fail("latent allocation failed"); }
  {
    const auto* s = static_cast<const _Float16*>(moments.contents());
    auto* d = static_cast<_Float16*>(out.contents());
    const bool whiten = (int)_cfg.latents_mean.size() == Cz &&
                        (int)_cfg.latents_std.size() == Cz;
    for (int c = 0; c < Cz; ++c) {
      const float mu = whiten ? _cfg.latents_mean[(std::size_t)c] : 0.0f;
      const float sd = whiten ? _cfg.latents_std[(std::size_t)c] : 1.0f;
      for (std::size_t p = 0; p < (std::size_t)T * lhw; ++p) {
        const std::size_t i = (std::size_t)c * T * lhw + p;
        d[i] = (_Float16)(((float)s[i] - mu) / (sd != 0.0f ? sd : 1.0f));
      }
    }
  }
  return out;
}

SharedBuffer
MetalWanVae::unwhiten(const SharedBuffer& z, int T, int h8, int w8)
{
  const int C = _cfg.z_dim;
  const std::size_t n = (std::size_t)T * h8 * w8;
  if ((int)_cfg.latents_mean.size() != C ||
      (int)_cfg.latents_std.size() != C ||
      z.byte_size() < (std::size_t)C * n * 2) {
    return {};
  }
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)C * n * 2);
  if (out.empty()) { return {}; }
  const auto* s = static_cast<const _Float16*>(z.contents());
  auto* d = static_cast<_Float16*>(out.contents());
  for (int c = 0; c < C; ++c) {
    const float mu = _cfg.latents_mean[(std::size_t)c];
    const float sd = _cfg.latents_std[(std::size_t)c];
    for (std::size_t p = 0; p < n; ++p) {
      const std::size_t i = (std::size_t)c * n + p;
      d[i] = (_Float16)((float)s[i] * sd + mu);
    }
  }
  return out;
}

}  // namespace genai
}  // namespace vpipe
