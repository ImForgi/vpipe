// FlashVSR's key/value window, held to a latent the forward produced
// before the window's layout changed.
//
// THE REFERENCE IS THIS TREE'S OWN EARLIER OUTPUT, not the model's. The
// question is whether a re-laid-out cache and re-used scratch compute the
// same function, and a saved latent from the layout it replaced answers
// exactly that. It is not bit-exact by construction -- a ring puts the
// keys in a different order, so the attention's float accumulation
// reassociates -- which is why the bound is a rel-L2 and not equality.
//
// FIVE CHUNKS, so the window FILLS and then WRAPS: at kv_ratio 3 the
// fourth window lands in the slot the first trim freed, and at kv_ratio 2
// that happens one chunk earlier. Two runs, one per ratio.
//
// Env, all required or the test skips:
//   VPIPE_FLASHVSR_TEST_MODEL_PATH  a FlashVSR-v1.1 directory (either layout)
//   VPIPE_FVSR_KV_REF               a directory holding kv3.f32 / kv2.f32
//   VPIPE_FVSR_KV_SAVE              set: WRITE the references instead

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/weight-set.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

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

}  // namespace

TEST(flashvsr_kv, the_window_matches_a_saved_latent)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  const char* ref = std::getenv("VPIPE_FVSR_KV_REF");
  if (root == nullptr || ref == nullptr || *root == '\0' || *ref == '\0') {
    return;
  }
  const bool save = std::getenv("VPIPE_FVSR_KV_SAVE") != nullptr;
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout));
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalFlashVsrTransformer::config_from_checkpoint(
      layout.denoiser, &cfg, &err));
  auto ws = WeightSet::open(layout.denoiser, nullptr);
  auto ctx_ws = WeightSet::open(layout.context, nullptr);
  ASSERT_TRUE(ws != nullptr && ctx_ws != nullptr);
  if (ws == nullptr || ctx_ws == nullptr) { return; }
  auto dit = MetalFlashVsrTransformer::load(ws, ctx_ws, layout.context_name,
                                            mc, cfg, &err);
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) {
    std::printf("  load: %s\n", err.c_str());
    return;
  }

  constexpr int kSize = 384;
  // Five chunks by default; VPIPE_FVSR_KV_FRAMES=33 is the two-chunk clip
  // whose window never trims before its last chunk, so the old layout and
  // the ring put every key in the same place and must agree EXACTLY.
  const char* fe = std::getenv("VPIPE_FVSR_KV_FRAMES");
  const int kFrames = (fe != nullptr && std::atoi(fe) > 0) ? std::atoi(fe)
                                                           : 57;
  const int T = MetalFlashVsrTransformer::latent_frames(kFrames);
  const int tok = (kSize / 16) * (kSize / 16);
  const int h8 = kSize / 8;
  std::mt19937_64 rng(0x6b76u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> noise((std::size_t)16 * T * h8 * h8);
  for (auto& v : noise) { v = nd(rng); }
  std::vector<std::uint16_t> rows((std::size_t)T * tok * cfg.hidden);
  for (auto& v : rows) { v = to_bf16_(0.5f * nd(rng)); }

  for (int kv : {3, 2}) {
    MetalFlashVsrTransformer::Request req;
    req.rows = rows.data();
    req.row_frames = T;
    req.frames = kFrames;
    req.height = kSize;
    req.width = kSize;
    req.init_noise = noise.data();
    req.params.kv_ratio = kv;
    std::vector<float> lat;
    std::vector<int> shape;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = dit->generate(req, &lat, &shape, &err);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(ok);
    if (!ok) {
      std::printf("  kv_ratio %d: %s\n", kv, err.c_str());
      continue;
    }
    const fs::path f = fs::path(ref) / ("kv" + std::to_string(kv) + ".f32");
    if (save) {
      std::ofstream o(f, std::ios::binary);
      o.write(reinterpret_cast<const char*>(lat.data()),
              (std::streamsize)(lat.size() * sizeof(float)));
      std::printf("  kv_ratio %d: saved %zu values to %s (%.0f ms)\n", kv,
                  lat.size(), f.c_str(), ms);
      continue;
    }
    std::ifstream in(f, std::ios::binary | std::ios::ate);
    ASSERT_TRUE((bool)in);
    if (!in) { continue; }
    std::vector<float> want((std::size_t)in.tellg() / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(want.data()),
            (std::streamsize)(want.size() * sizeof(float)));
    const double r = rel_l2_(lat, want);
    std::printf("  kv_ratio %d: rel-L2 %.3e against the saved latent "
                "(%.0f ms)\n", kv, r, ms);
    // TWO CHUNKS AT kv_ratio 3 never trim before the last chunk, so the
    // shifting layout and the ring hold every key in the same slot and
    // must agree EXACTLY -- which is the proof the re-used scratch is
    // right. Past that the ring reorders the keys and the attention
    // reassociates. MEASURED at five chunks: 8.8e-3 (kv 3) and 1.0e-2
    // (kv 2), against 1.4e-2 for this same run with ALU attention in
    // place of NAX -- float noise grown through thirty blocks and the
    // cached chunks, not a different function.
    if (kFrames <= 33 && kv >= 3) {
      EXPECT_TRUE(r == 0.0);
    } else {
      EXPECT_TRUE(r < 2e-2);
    }
  }
}
