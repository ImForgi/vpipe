// VOSR 2.0 against the reference implementation.
//
// The bar for inference work in this tree is a numerical comparison
// against the model's own code, and for a ONE-STEP restorer that bar is
// unusually load-bearing: there is no sampler loop to average a mistake
// away, so whatever the stack emits on its single evaluation IS the
// picture. A port that is subtly wrong here produces a clean, plausible
// image of something the reference would not have made.
//
// Three tiers, in the order a wrong answer propagates:
//
//   dinov2   the conditioning grid, from the same U8 pixels. Covers the
//            preprocessing as much as the tower -- torch's bicubic is
//            not Pillow's, the resize squashes to a square, and the
//            position grid is resampled at DINOv2's own 0.1 offset.
//   vae      the low-quality latent, which is the other half of the
//            DiT's input and the half a graph gets from `vae-encode`.
//   velocity the whole 36-block stack, from the reference's own latent,
//            tokens and noise -- so a failure here is the DiT and
//            nothing upstream of it.
//
// The noise is the reference's, fed in rather than drawn: two runtimes
// cannot agree on a pseudo-random field, so the comparison has to hold
// it fixed. With one step and dt = 1 the velocity is recoverable from
// the result exactly, z_out = noise - u, which is why no separate entry
// point is needed to get at it.
//
// Env, all three required or the tier skips:
//   VPIPE_VOSR_TEST_MODEL_PATH   the fetched CSWRY/VOSR root
//   VPIPE_DINOV2_TEST_MODEL_PATH facebook/dinov2-large
//   VPIPE_VOSR_GOLDEN            the dir tools/dump_vosr_golden.py wrote

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/weight-set.h"
#include "generative-models/vosr/metal-dinov2-encoder.h"
#include "generative-models/vosr/metal-vosr-transformer.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

namespace fs = std::filesystem;

double
rel_l2_(const float* got, const float* ref, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) { return {}; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  std::vector<float> out((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

std::vector<std::uint8_t>
read_u8_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) { return {}; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  std::vector<std::uint8_t> out((std::size_t)n);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

inline float
bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Everything the three tiers share: the paths, the golden manifest's
// geometry, and a session. Empty `dir` means a tier should skip.
struct Fixture {
  std::string root, dino, golden;
  int H = 0, W = 0;             // the resized picture
  int lh = 0, lw = 0;           // its latent
  // WHICH DINOv2 LAYER this golden was taken at. From the manifest and
  // not from args.json, because the depth sweep that separates
  // accumulated bf16 error from a wrong preprocessing dumps goldens at
  // several layers of the same checkpoint.
  int layer = -1;
  bool ok = false;

  Fixture()
  {
    const char* r = std::getenv("VPIPE_VOSR_TEST_MODEL_PATH");
    const char* d = std::getenv("VPIPE_DINOV2_TEST_MODEL_PATH");
    const char* g = std::getenv("VPIPE_VOSR_GOLDEN");
    if (r == nullptr || d == nullptr || g == nullptr || *r == '\0' ||
        *d == '\0' || *g == '\0') {
      return;
    }
    root = r; dino = d; golden = g;
    std::ifstream in(fs::path(golden) / "manifest.json");
    if (!in) { return; }
    FlexData fd = FlexData::from_json(in);
    if (!fd.is_object()) { return; }
    auto o = fd.as_object();
    auto dims = [&](const char* k, int* a, int* b, int at) {
      if (!o.contains(k)) { return false; }
      FlexData v = o.at(k);
      auto arr = v.as_array();
      if ((int)arr.size() < at + 2) { return false; }
      *a = (int)arr[at].as_real(0.0);
      *b = (int)arr[at + 1].as_real(0.0);
      return true;
    };
    // resized_u8 is [3, H, W]; lq_latent is [16, lh, lw].
    if (!dims("resized_u8", &H, &W, 1)) { return; }
    if (!dims("lq_latent", &lh, &lw, 1)) { return; }
    if (o.contains("dinov2_layer")) {
      layer = (int)o.at("dinov2_layer").as_real(-1.0);
    }
    ok = H > 0 && W > 0 && lh > 0 && lw > 0;
  }
};

}  // namespace

// The conditioning grid, from the reference's own resized pixels so the
// comparison is the tower and its preprocessing rather than the upstream
// resample.
TEST(vosr_reference, dinov2_tokens_match)
{
  Fixture f;
  if (!f.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalVosrTransformer::Config vc;
  std::string why;
  ASSERT_TRUE(MetalVosrTransformer::read_config(f.root, &vc, &why));
  if (!why.empty()) { return; }

  MetalDinov2Encoder::Config dc;
  MetalDinov2Encoder::read_config(f.dino, &dc);
  dc.out_layer = f.layer >= 0 ? f.layer : vc.enc_layer;
  auto tower = MetalDinov2Encoder::load(f.dino, mc, dc);
  ASSERT_TRUE((bool)tower);
  if (!tower) { return; }

  const std::vector<std::uint8_t> px =
      read_u8_((fs::path(f.golden) / "resized_u8.bin").string());
  ASSERT_TRUE(px.size() == (std::size_t)3 * f.H * f.W);
  if (px.size() != (std::size_t)3 * f.H * f.W) { return; }

  int n_tok = 0;
  SharedBuffer tok =
      tower->encode_rgb(px.data(), f.H, f.W, vc.dinov2_size, &n_tok);
  ASSERT_TRUE(!tok.empty());
  if (tok.empty()) { return; }

  const std::vector<float> ref =
      read_f32_((fs::path(f.golden) / "dinov2.bin").string());
  ASSERT_TRUE(ref.size() == (std::size_t)n_tok * dc.hidden);
  if (ref.size() != (std::size_t)n_tok * dc.hidden) { return; }

  std::vector<float> got(ref.size());
  const auto* src = static_cast<const std::uint16_t*>(tok.contents());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = bf16_(src[i]); }
  const double r = rel_l2_(got.data(), ref.data(), got.size());
  std::printf("[vosr] dinov2 rel-L2 %.5f over %d x %d\n", r, n_tok,
              dc.hidden);
  // The tower runs bf16 against the reference's fp32, and this bound is
  // set by ACCUMULATION rather than by the preprocessing. Measured over
  // the depth sweep the dumper's --layer flag exists for, on one 640px
  // picture:
  //
  //   after 1 block   0.0073
  //   after 6         0.0204
  //   after 12        0.0269
  //   after 18        0.0232
  //
  // It grows with depth and then flattens, which is what rounding
  // compounding through a residual stream does. A WRONG resample, a
  // wrong normalisation or a wrong position grid would already be an
  // order of magnitude out after ONE block, where this is 0.0073 -- so
  // the first row is what says the preprocessing is right, and the last
  // is just bf16. What it costs the answer is measured separately, in
  // whole_stack_matches below.
  EXPECT_TRUE(r < 0.04);
}

// The low-quality latent, which is the DiT's other input and the one a
// graph gets from `vae-encode`. Also the check on reading VOSR's
// image-only extraction of the Qwen-Image autoencoder, whose convs
// carry one axis fewer than the checkpoint that code was written for.
TEST(vosr_reference, vae_encode_matches)
{
  Fixture f;
  if (!f.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string vae_dir =
      (fs::path(f.root) / "Qwen-Image-vae-2d").string();
  std::error_code ec;
  if (!fs::is_directory(vae_dir, ec)) { return; }

  MetalKrea2Vae::Config cfg;
  {
    std::ifstream in(fs::path(vae_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        auto arr_f = [&](const char* k, std::vector<float>* out) {
          if (!o.contains(k)) { return; }
          FlexData v = o.at(k);
          auto a = v.as_array();
          for (std::size_t i = 0; i < a.size(); ++i) {
            out->push_back((float)a[i].as_real(0.0));
          }
        };
        arr_f("latents_mean", &cfg.latents_mean);
        arr_f("latents_std", &cfg.latents_std);
        if (o.contains("base_dim")) {
          cfg.base_dim = (int)o.at("base_dim").as_real(96.0);
        }
      }
    }
  }
  auto vae = MetalKrea2Vae::load(vae_dir, mc, cfg, /*with_encoder=*/true);
  ASSERT_TRUE((bool)vae);
  if (!vae) { return; }

  const std::vector<std::uint8_t> px =
      read_u8_((fs::path(f.golden) / "resized_u8.bin").string());
  if (px.size() != (std::size_t)3 * f.H * f.W) { return; }
  SharedBuffer img =
      mc->make_shared_buffer((std::size_t)3 * f.H * f.W * 2);
  ASSERT_TRUE(!img.empty());
  if (img.empty()) { return; }
  {
    auto* d = static_cast<_Float16*>(img.contents());
    for (std::size_t i = 0; i < px.size(); ++i) {
      d[i] = (_Float16)((float)px[i] / 255.0f * 2.0f - 1.0f);
    }
  }
  SharedBuffer lat = vae->encode(img, f.H, f.W);
  ASSERT_TRUE(!lat.empty());
  if (lat.empty()) { return; }

  const std::vector<float> ref =
      read_f32_((fs::path(f.golden) / "lq_latent.bin").string());
  const std::size_t n = (std::size_t)16 * f.lh * f.lw;
  ASSERT_TRUE(ref.size() == n);
  if (ref.size() != n) { return; }
  std::vector<float> got(n);
  const auto* src = static_cast<const _Float16*>(lat.contents());
  for (std::size_t i = 0; i < n; ++i) { got[i] = (float)src[i]; }
  const double r = rel_l2_(got.data(), ref.data(), n);
  std::printf("[vosr] vae encode rel-L2 %.5f over [16, %d, %d]\n", r, f.lh,
              f.lw);
  // f16 against the reference's fp32, through a conv net -- the same
  // bound the other image VAEs in this tree are held to.
  EXPECT_TRUE(r < 0.02);
}

// The whole stack, from the reference's own latent, tokens and noise.
TEST(vosr_reference, dit_velocity_matches)
{
  Fixture f;
  if (!f.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalVosrTransformer::Config vc;
  std::string why;
  ASSERT_TRUE(MetalVosrTransformer::read_config(f.root, &vc, &why));
  if (!why.empty()) { return; }
  const std::string wp = MetalVosrTransformer::weights_path(f.root);
  ASSERT_TRUE(!wp.empty());
  if (wp.empty()) { return; }
  std::string err;
  auto dit = MetalVosrTransformer::load(wp, mc, vc, &err);
  ASSERT_TRUE((bool)dit);
  if (!dit) {
    std::printf("[vosr] load failed: %s\n", err.c_str());
    return;
  }

  const std::vector<float> lq =
      read_f32_((fs::path(f.golden) / "lq_latent.bin").string());
  const std::vector<float> noise =
      read_f32_((fs::path(f.golden) / "noise.bin").string());
  const std::vector<float> vel =
      read_f32_((fs::path(f.golden) / "velocity.bin").string());
  const std::vector<float> cond32 =
      read_f32_((fs::path(f.golden) / "dinov2.bin").string());
  const std::size_t n = (std::size_t)vc.latent_ch * f.lh * f.lw;
  ASSERT_TRUE(lq.size() == n && noise.size() == n && vel.size() == n);
  if (lq.size() != n || noise.size() != n || vel.size() != n) { return; }
  ASSERT_TRUE(!cond32.empty() && cond32.size() % vc.enc_dim == 0);
  if (cond32.empty() || cond32.size() % vc.enc_dim != 0) { return; }

  // The REFERENCE's tokens, rounded to what a conditioning beat carries.
  // Feeding the tower's own output instead would fold its error into
  // this number and stop it being a statement about the DiT.
  std::vector<std::uint16_t> cond(cond32.size());
  for (std::size_t i = 0; i < cond32.size(); ++i) {
    std::uint32_t u;
    std::memcpy(&u, &cond32[i], 4);
    cond[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }

  MetalVosrTransformer::RestoreRequest rr;
  rr.lq        = lq.data();
  rr.lh        = f.lh;
  rr.lw        = f.lw;
  rr.cond      = cond.data();
  rr.cond_rows = (int)(cond32.size() / vc.enc_dim);
  rr.steps     = 1;
  rr.noise     = noise.data();
  std::vector<float> out;
  std::string rerr;
  ASSERT_TRUE(dit->restore(rr, &out, &rerr));
  if (out.size() != n) {
    std::printf("[vosr] restore failed: %s\n", rerr.c_str());
    return;
  }
  // One step over t: 1 -> 0, so dt is 1 and the velocity comes back out
  // of the result exactly.
  std::vector<float> got(n);
  for (std::size_t i = 0; i < n; ++i) { got[i] = noise[i] - out[i]; }
  const double r = rel_l2_(got.data(), vel.data(), n);
  std::printf("[vosr] dit velocity rel-L2 %.5f over [%d, %d, %d]\n", r,
              vc.latent_ch, f.lh, f.lw);
  // 36 blocks of bf16 against fp32, with an interleaved-pair rotary and
  // a per-head qk norm compounding through every one of them. The same
  // bound the other bf16 DiTs here carry.
  EXPECT_TRUE(r < 0.05);
}

// WHAT THE TOWER'S PRECISION COSTS THE ANSWER, which is the question
// its rel-L2 does not answer on its own.
//
// dit_velocity_matches feeds the DiT the REFERENCE's tokens, so it
// measures the DiT alone. This runs the same comparison on the tokens
// vpipe's own tower produced. The gap between the two numbers is the
// entire price of running the conditioner in bf16, and it is the number
// worth having: a 2% error on the conditioning would matter if the DiT
// amplified it and does not if it does not.
TEST(vosr_reference, whole_stack_matches)
{
  Fixture f;
  if (!f.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalVosrTransformer::Config vc;
  std::string why;
  ASSERT_TRUE(MetalVosrTransformer::read_config(f.root, &vc, &why));
  if (!why.empty()) { return; }

  MetalDinov2Encoder::Config dc;
  MetalDinov2Encoder::read_config(f.dino, &dc);
  dc.out_layer = f.layer >= 0 ? f.layer : vc.enc_layer;
  auto tower = MetalDinov2Encoder::load(f.dino, mc, dc);
  ASSERT_TRUE((bool)tower);
  if (!tower) { return; }

  const std::vector<std::uint8_t> px =
      read_u8_((fs::path(f.golden) / "resized_u8.bin").string());
  if (px.size() != (std::size_t)3 * f.H * f.W) { return; }
  int n_tok = 0;
  SharedBuffer tok =
      tower->encode_rgb(px.data(), f.H, f.W, vc.dinov2_size, &n_tok);
  ASSERT_TRUE(!tok.empty());
  if (tok.empty()) { return; }
  tower.reset();

  const std::string wp = MetalVosrTransformer::weights_path(f.root);
  if (wp.empty()) { return; }
  std::string err;
  auto dit = MetalVosrTransformer::load(wp, mc, vc, &err);
  ASSERT_TRUE((bool)dit);
  if (!dit) { return; }

  const std::vector<float> lq =
      read_f32_((fs::path(f.golden) / "lq_latent.bin").string());
  const std::vector<float> noise =
      read_f32_((fs::path(f.golden) / "noise.bin").string());
  const std::vector<float> vel =
      read_f32_((fs::path(f.golden) / "velocity.bin").string());
  const std::size_t n = (std::size_t)vc.latent_ch * f.lh * f.lw;
  if (lq.size() != n || noise.size() != n || vel.size() != n) { return; }

  MetalVosrTransformer::RestoreRequest rr;
  rr.lq        = lq.data();
  rr.lh        = f.lh;
  rr.lw        = f.lw;
  rr.cond      = tok.contents();
  rr.cond_rows = n_tok;
  rr.steps     = 1;
  rr.noise     = noise.data();
  std::vector<float> out;
  std::string rerr;
  ASSERT_TRUE(dit->restore(rr, &out, &rerr));
  if (out.size() != n) { return; }
  std::vector<float> got(n);
  for (std::size_t i = 0; i < n; ++i) { got[i] = noise[i] - out[i]; }
  const double r = rel_l2_(got.data(), vel.data(), n);
  std::printf("[vosr] whole stack rel-L2 %.5f over [%d, %d, %d]\n", r,
              vc.latent_ch, f.lh, f.lw);
  // The same bound the DiT alone carries. If the tower's ~2% moved the
  // answer, this would be visibly worse than dit_velocity_matches and
  // the conditioner would need a wider element type; it is not.
  EXPECT_TRUE(r < 0.05);
}
