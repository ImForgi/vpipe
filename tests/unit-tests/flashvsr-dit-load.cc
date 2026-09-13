// FlashVSR's denoiser: the geometry it derives and the constants it
// folds, against the reference.
//
// BOTH HALVES ARE INVISIBLE AT RUN TIME, which is why they get a test of
// their own rather than being covered by a forward.
//
// The geometry is derived from tensor shapes because the checkpoint
// ships no transformer config -- its config.json names the family and
// stops. Derivation that is subtly wrong loads a model of the wrong
// shape, and the failure surfaces as a bad picture.
//
// The constants are worse. FlashVSR runs at one fixed timestep over one
// fixed conditioning with guidance distilled away, so the timestep
// embedding, the six modulation vectors per block, the head's two, and
// every block's cross-attention key and value are computed ONCE at load
// and never recomputed. Nothing downstream can notice they are wrong;
// there is no second opinion anywhere in the run.
//
// Env, both required or the test skips:
//   VPIPE_FLASHVSR_TEST_MODEL_PATH  the converted FlashVSR-v1.1 dir
//   VPIPE_FLASHVSR_GOLDEN           a dir fvsr/gen_goldens.py wrote

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/weight-set.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

namespace fs = std::filesystem;

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& ref)
{
  if (got.size() != ref.size() || got.empty()) { return 1e9; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::vector<float>
read_f32_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in) { return {}; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  std::vector<float> out((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

const char*
model_dir_()
{
  const char* r = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  return (r != nullptr && *r != '\0') ? r : nullptr;
}

const char*
golden_dir_()
{
  const char* g = std::getenv("VPIPE_FLASHVSR_GOLDEN");
  return (g != nullptr && *g != '\0') ? g : nullptr;
}

// The paths plus the geometry the golden was generated AT, read from its
// own manifest rather than hardcoded -- the same test then runs against
// the dense set and the routed one, which differ in exactly the two
// knobs the routing reads.
struct Fixture {
  std::string root, golden;
  int size = 0, frames = 0, local_range = 11;
  double kv_ratio = 3.0;
  bool ok = false;

  Fixture()
  {
    const char* r = model_dir_();
    const char* g = golden_dir_();
    if (r == nullptr || g == nullptr) { return; }
    root = r;
    golden = g;
    std::ifstream in(fs::path(golden) / "config.json");
    if (!in) { return; }
    FlexData fd = FlexData::from_json(in);
    if (!fd.is_object()) { return; }
    auto o = fd.as_object();
    if (!o.contains("size") || !o.contains("frames")) { return; }
    size = (int)o.at("size").as_real(0.0);
    frames = (int)o.at("frames").as_real(0.0);
    if (o.contains("local_range")) {
      local_range = (int)o.at("local_range").as_real(11.0);
    }
    if (o.contains("kv_ratio")) { kv_ratio = o.at("kv_ratio").as_real(3.0); }
    ok = size > 0 && frames > 0;
  }
};

}  // namespace

TEST(flashvsr_dit, config_is_derived_from_shapes)
{
  const char* root = model_dir_();
  if (root == nullptr) { return; }
  // Either layout: the converted directory's index, or the published
  // denoiser file named directly.
  FlashVsrLayout layout;
  const bool have_layout = resolve_flashvsr_layout(root, &layout);
  ASSERT_TRUE(have_layout);
  if (!have_layout) { return; }

  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  const bool ok = MetalFlashVsrTransformer::config_from_checkpoint(
      layout.denoiser, &cfg, &err);
  ASSERT_TRUE(ok);
  if (!ok) {
    printf("  config_from_checkpoint failed: %s\n", err.c_str());
    return;
  }
  // Wan 2.1-T2V-1.3B, which is what this checkpoint's denoiser is,
  // tensor for tensor.
  EXPECT_TRUE(cfg.hidden == 1536);
  EXPECT_TRUE(cfg.n_heads == 12);
  EXPECT_TRUE(cfg.head_dim == 128);
  EXPECT_TRUE(cfg.ffn == 8960);
  EXPECT_TRUE(cfg.n_layers == 30);
  EXPECT_TRUE(cfg.in_channels == 16);
  EXPECT_TRUE(cfg.out_channels == 16);
  EXPECT_TRUE(cfg.text_dim == 4096);
  EXPECT_TRUE(cfg.text_tokens == 512);
  EXPECT_TRUE(cfg.freq_dim == 256);
  EXPECT_TRUE(cfg.patch_t == 1 && cfg.patch_h == 2 && cfg.patch_w == 2);
  // The 3-axis RoPE split Wan derives: h = w = 2*(head_dim/6), t takes
  // the remainder.
  EXPECT_TRUE(cfg.rope_h() == 42 && cfg.rope_w() == 42);
  EXPECT_TRUE(cfg.rope_t() == 44);
  EXPECT_TRUE(cfg.rope_h() + cfg.rope_w() + cfg.rope_t() == cfg.head_dim);
  printf("  config: %d blocks, %d wide, %dx%d heads, ffn %d\n", cfg.n_layers,
         cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.ffn);
}

TEST(flashvsr_dit, geometry_rules)
{
  // 8k+1 frames, rounded UP, with twenty frames going to a warmup that
  // is never emitted.
  EXPECT_TRUE(MetalFlashVsrTransformer::align_frames(41) == 41);
  EXPECT_TRUE(MetalFlashVsrTransformer::align_frames(42) == 49);
  EXPECT_TRUE(MetalFlashVsrTransformer::align_frames(33) == 33);
  EXPECT_TRUE(MetalFlashVsrTransformer::align_frames(1) == 1);
  // 41 frames is 3 chunks. The OPENING chunk emits six latent frames
  // and the other two emit two each, so the latent is 10 -- not 2 per
  // chunk -- and the VAE expands that to 4*9 + 1 = 37 restored frames,
  // which is the 8n-3 the reference reports.
  EXPECT_TRUE(MetalFlashVsrTransformer::latent_frames(41) == 10);
  EXPECT_TRUE(MetalFlashVsrTransformer::restored_frames(41) == 37);
  EXPECT_TRUE(MetalFlashVsrTransformer::latent_frames(17) == 0);
  EXPECT_TRUE(MetalFlashVsrTransformer::restored_frames(17) == 0);
  EXPECT_TRUE(MetalFlashVsrTransformer::kSizeGrid == 128);
}

TEST(flashvsr_dit, folded_constants_match_reference)
{
  Fixture fx;
  if (!fx.ok) { return; }
  const char* root = fx.root.c_str();
  const char* gold = fx.golden.c_str();

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout));
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  if (!MetalFlashVsrTransformer::config_from_checkpoint(layout.denoiser, &cfg,
                                                        &err)) {
    printf("  config failed: %s\n", err.c_str());
    ASSERT_TRUE(false);
    return;
  }
  auto ws = WeightSet::open(layout.denoiser, nullptr);
  auto ctx_ws = WeightSet::open(layout.context, nullptr);
  ASSERT_TRUE(ws != nullptr && ctx_ws != nullptr);
  if (ws == nullptr || ctx_ws == nullptr) { return; }

  auto dit = MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name,
                                            mc, cfg, &err);
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) {
    printf("  dit load failed: %s\n", err.c_str());
    return;
  }

  const fs::path g(gold);
  // The timestep embedding and its six-way projection. Both are f32
  // here and f32 in the golden, so the bar is tight -- what is left is
  // the reference's float64 sinusoid against this one's, and the bf16
  // the weights were read through.
  const double r_t = rel_l2_(dit->t_vec_f(), read_f32_(g / "t_vec.f32"));
  const double r_m = rel_l2_(dit->t_mod_f(), read_f32_(g / "t_mod.f32"));
  printf("  t_vec rel-L2 %.6f   t_mod rel-L2 %.6f\n", r_t, r_m);
  EXPECT_TRUE(r_t < 0.002);
  EXPECT_TRUE(r_m < 0.002);

  // The cross-attention key and value of block 0, over the fixed
  // conditioning, with the key's RMS norm already applied.
  std::vector<float> k, v;
  ASSERT_TRUE(dit->cross_kv(0, &k, &v));
  const double r_k = rel_l2_(k, read_f32_(g / "cross_k_b0.f32"));
  const double r_v = rel_l2_(v, read_f32_(g / "cross_v_b0.f32"));
  printf("  cross_k rel-L2 %.6f   cross_v rel-L2 %.6f\n", r_k, r_v);
  // These are stored bf16, so the floor is bf16 rounding on a value
  // that went through two projections and a norm.
  EXPECT_TRUE(r_k < 0.01);
  EXPECT_TRUE(r_v < 0.01);

  printf("  resident %llu MB\n",
         (unsigned long long)(dit->resident_bytes() >> 20));

  // ---- the patch embedding, over the first chunk's slice of the very
  // noise the reference drew. The PACK ORDER is what this pins, and it
  // is silent when wrong: a permuted patch is a correctly shaped tensor
  // of a differently arranged picture.
  const auto noise = read_f32_(g / "noise.f32");
  const auto want = read_f32_(g / "chunk0_patch.f32");
  const int C = cfg.in_channels;
  const int size = fx.size;        // from the golden's own manifest
  const int h8 = size / 8, w8 = size / 8;
  const int T_all = (fx.frames - 1) / 4;
  const int T0 = 6;                // chunk 0 takes six latent frames
  if (noise.size() == (std::size_t)C * T_all * h8 * w8 && !want.empty()) {
    std::vector<float> chunk((std::size_t)C * T0 * h8 * w8, 0.0f);
    const std::size_t plane = (std::size_t)h8 * w8;
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < T0; ++t) {
        std::memcpy(chunk.data() + ((std::size_t)c * T0 + t) * plane,
                    noise.data() + ((std::size_t)c * T_all + t) * plane,
                    plane * sizeof(float));
      }
    }
    std::vector<float> got;
    if (!dit->patchify_tokens(chunk.data(), T0, h8, w8, &got, &err)) {
      printf("  patchify failed: %s\n", err.c_str());
      EXPECT_TRUE(false);
    } else {
      const double r = rel_l2_(got, want);
      printf("  patchify rel-L2 %.6f  (%zu tokens)\n", r,
             got.size() / (std::size_t)cfg.hidden);
      EXPECT_TRUE(r < 0.01);
    }
  }

  // ---- block 0's self-attention, where every mechanism this model adds
  // meets: the folded modulation, the three-axis RoPE at an absolute
  // offset, the 3-D window permutation, the routing and the masked
  // flash attention. Its input is the patchified chunk with the source
  // projection's rows added, which is what block 0 actually sees.
  const int D = cfg.hidden;
  const int tf = T0, th = h8 / 2, tw = w8 / 2;
  const std::size_t seq = (std::size_t)tf * th * tw;
  std::vector<float> x = want;              // the reference's own patch
  if (x.size() == seq * (std::size_t)D) {
    bool have_lq = true;
    for (int i = 0; i < tf; ++i) {
      const auto rows =
          read_f32_(g / ("lq_proj" + std::to_string(i) + ".f32"));
      const std::size_t per = (std::size_t)th * tw * D;
      if (rows.size() != per) { have_lq = false; break; }
      for (std::size_t j = 0; j < per; ++j) { x[(std::size_t)i * per + j] += rows[j]; }
    }
    const auto ref_sa = read_f32_(g / "chunk0_b0_selfattn.f32");
    if (have_lq && ref_sa.size() == x.size()) {
      MetalFlashVsrTransformer::Params prm;
      prm.local_range = fx.local_range;       // the golden's own band
      prm.kv_ratio = fx.kv_ratio;
      std::vector<float> sa;
      if (!dit->block_self_attention(0, x.data(), tf, th, tw, 0, prm, &sa,
                                     &err)) {
        printf("  self-attention failed: %s\n", err.c_str());
        EXPECT_TRUE(false);
      } else {
        const double r = rel_l2_(sa, ref_sa);
        printf("  block0 self-attn rel-L2 %.6f\n", r);
        EXPECT_TRUE(r < 0.05);
      }

      // ---- the two routing paths against each other. The GPU one
      // replaced a host one that was verified end to end, so what needs
      // measuring is how many blocks the numerical difference between
      // them actually flips -- not whether it works.
      std::size_t n_flags = 0, n_diff = 0;
      if (dit->route_compare(x.data(), tf, th, tw, 0, prm, &n_flags, &n_diff,
                             &err)) {
        printf("  routing gpu vs host: %zu/%zu flags differ (%.4f%%)\n",
               n_diff, n_flags,
               n_flags ? 100.0 * (double)n_diff / (double)n_flags : 0.0);
        // A HANDFUL is expected and benign: the host summed the proxy in
        // double where the GPU and the reference both use float, so a
        // near-tie at the top-k boundary can land either way. Percent-
        // level disagreement would mean a different rule, not a
        // different rounding.
        EXPECT_TRUE(n_flags > 0);
        EXPECT_TRUE((double)n_diff < 0.01 * (double)n_flags);
      }
      // ---- the WHOLE of block 0: self-attention, then cross-attention
      // into the folded constant, then the ungated feed-forward and both
      // gated residuals. The input is the patch alone, because the
      // source rows are added inside the block, not before it.
      const auto ref_b0 = read_f32_(g / "chunk0_block0.f32");
      std::vector<float> lq_rows(x.size(), 0.0f);
      for (int i = 0; i < tf; ++i) {
        const auto rows =
            read_f32_(g / ("lq_proj" + std::to_string(i) + ".f32"));
        const std::size_t per = (std::size_t)th * tw * D;
        std::copy(rows.begin(), rows.end(),
                  lq_rows.begin() + (std::ptrdiff_t)((std::size_t)i * per));
      }
      if (ref_b0.size() == want.size()) {
        std::vector<float> b0;
        if (!dit->run_chunk_blocks(1, want.data(), lq_rows.data(), tf, th, tw,
                                   0, prm, &b0, &err)) {
          printf("  block0 failed: %s\n", err.c_str());
          EXPECT_TRUE(false);
        } else {
          const double rb = rel_l2_(b0, ref_b0);
          printf("  block0 full rel-L2 %.6f\n", rb);
          EXPECT_TRUE(rb < 0.05);
        }

      }
    }
  }
}

// The whole denoise: every chunk, the key/value window between them, and
// the single step at a fixed timestep. This is what generate-video would
// emit, so it is the number that decides whether the port is a port.
//
// The noise is the REFERENCE'S, fed in rather than drawn -- two runtimes
// cannot agree on a pseudo-random field, and with one step the latent is
// exactly noise minus the prediction, so holding it fixed compares the
// model and nothing else.
TEST(flashvsr_dit, clip_latent_matches_reference)
{
  Fixture fx;
  if (!fx.ok) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  FlashVsrLayout layout;
  if (!resolve_flashvsr_layout(fx.root, &layout) ||
      !MetalFlashVsrTransformer::config_from_checkpoint(layout.denoiser, &cfg,
                                                        &err)) {
    return;
  }
  auto ws = WeightSet::open(layout.denoiser, nullptr);
  auto ctx_ws = WeightSet::open(layout.context, nullptr);
  if (ws == nullptr || ctx_ws == nullptr) { return; }
  auto dit = MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name,
                                            mc, cfg, &err);
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) { return; }

  const fs::path g(fx.golden);
  const auto want = read_f32_(g / "latent.f32");
  const auto noise = read_f32_(g / "noise.f32");
  if (want.empty() || noise.empty()) { return; }

  // The REFERENCE'S OWN projected rows, concatenated. Taking them from
  // the golden rather than computing them isolates the denoiser from
  // the projection completely: a failure here is the 30 blocks, the
  // key/value window or the chunk loop, and cannot be the projection --
  // which has its own test.
  const int T = MetalFlashVsrTransformer::latent_frames(fx.frames);
  const std::size_t tok = (std::size_t)(fx.size / 16) * (fx.size / 16);
  std::vector<std::uint16_t> rows(tok * (std::size_t)T * cfg.hidden);
  for (int i = 0; i < T; ++i) {
    const auto rf =
        read_f32_(g / ("lq_proj" + std::to_string(i) + ".f32"));
    if (rf.size() != tok * (std::size_t)cfg.hidden) { return; }
    for (std::size_t k = 0; k < rf.size(); ++k) {
      std::uint32_t u;
      std::memcpy(&u, &rf[k], 4);
      rows[(std::size_t)i * rf.size() + k] =
          (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
  }

  MetalFlashVsrTransformer::Request req;
  req.rows = rows.data();
  req.row_frames = T;
  req.frames = fx.frames;
  req.height = fx.size;
  req.width = fx.size;
  req.seed = 0;
  req.init_noise = noise.data();
  req.params.local_range = fx.local_range;
  req.params.kv_ratio = fx.kv_ratio;

  std::vector<float> got;
  std::vector<int> shape;
  // Timed separately from the load, which dominates this test: the
  // checkpoint is 5.7 GB of f32 converted to bf16 on the way in.
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = dit->generate(req, &got, &shape, &err);
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ok);
  if (!ok) {
    printf("  generate failed: %s\n", err.c_str());
    return;
  }
  printf("  denoise %.0f ms for %d chunks at %dx%d\n", ms,
         (fx.frames - 1) / 8 - 2, fx.size, fx.size);
  printf("  latent shape [%d,%d,%d,%d]\n", shape[0], shape[1], shape[2],
         shape[3]);
  EXPECT_TRUE(shape[1] ==
              MetalFlashVsrTransformer::latent_frames(fx.frames));
  const double r = rel_l2_(got, want);
  printf("  clip latent rel-L2 %.6f\n", r);
  // Thirty blocks of bf16 accumulation over three chunks, against an f32
  // reference. A structural error lands far above this.
  EXPECT_TRUE(r < 0.05);
}
