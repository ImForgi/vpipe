// FlashVSR's source projection against the reference implementation.
//
// This module is the whole of how the low-quality clip reaches the
// denoiser: no VAE encode, no cross-attention, just these rows added
// into the residual stream at block 0. So a mistake here is not a
// degraded restoration, it is a restoration of something else, and one
// that still looks like video.
//
// THE TEST IS THE CHUNK SEQUENCE, not one call. The projection is
// causal and carries its last two input frames across every boundary,
// so a carry that is written before it is read -- which is what the
// reference's OTHER, buffered spelling of this module does -- gives
// right-shaped rows that are quietly conditioned on the wrong frames.
// Only replaying the reference's own irregular call pattern catches it:
// a one-frame opening call, six more of four frames, then two per
// denoise chunk.
//
// The opening call PRODUCES NOTHING. It seeds both carries and returns
// zero row frames, so the count below is 10 row sets from 11 calls.
//
// Env, both required or the test skips:
//   VPIPE_FLASHVSR_TEST_MODEL_PATH  the converted FlashVSR-v1.1 dir
//   VPIPE_FLASHVSR_GOLDEN           a dir fvsr/gen_goldens.py wrote

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flashvsr/flashvsr-lq-proj.h"
#include "generative-models/weight-set.h"

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

std::vector<std::uint8_t>
read_u8_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary | std::ios::ate);
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

struct Fixture {
  std::string root, golden;
  int size = 0, frames = 0;
  bool ok = false;

  Fixture()
  {
    const char* r = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
    const char* g = std::getenv("VPIPE_FLASHVSR_GOLDEN");
    if (r == nullptr || g == nullptr || *r == '\0' || *g == '\0') { return; }
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
    ok = size > 0 && frames > 0;
  }
};

// The reference feeds the projection in an irregular pattern: one frame,
// then six slices of four to fill the warmup, then two slices of four
// per denoise chunk. Reproduced here rather than simplified, because the
// carry makes the SEQUENCE part of the function.
std::vector<std::pair<int, int>>
call_plan_(int frames)
{
  std::vector<std::pair<int, int>> v;    // [lo, hi)
  for (int i = 0; i < 7; ++i) {
    const int lo = std::max(0, i * 4 - 3);
    const int hi = (i + 1) * 4 - 3;
    if (hi > lo && hi <= frames) { v.emplace_back(lo, hi); }
  }
  const int chunks = (frames - 1) / 8 - 2;
  for (int c = 1; c < chunks; ++c) {
    for (int i = 0; i < 2; ++i) {
      const int lo = c * 8 + 17 + i * 4;
      const int hi = c * 8 + 21 + i * 4;
      if (hi <= frames) { v.emplace_back(lo, hi); }
    }
  }
  return v;
}

}  // namespace

TEST(flashvsr_lq_proj, matches_reference)
{
  Fixture fx;
  if (!fx.ok) { return; }   // env-gated; skips vacuously without goldens

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(fx.root, &layout));
  auto ws = WeightSet::open(layout.source, nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (ws == nullptr) { return; }

  std::string err;
  auto proj = FlashVsrLqProj::load(ws, mc, &err, layout.source_prefix);
  ASSERT_TRUE(proj != nullptr);
  if (proj == nullptr) {
    printf("  lq-proj load failed: %s\n", err.c_str());
    return;
  }
  // v1.1 ships ONE output layer, so the source reaches block 0 only. A
  // checkpoint that shipped thirty would condition the whole stack, and
  // silently: the rows would be right and there would be more of them.
  EXPECT_TRUE(proj->config().n_out_layers == 1);

  const auto u8 = read_u8_(fs::path(fx.golden) / "lq_video_u8.bin");
  const std::size_t px = (std::size_t)fx.size * fx.size;
  ASSERT_TRUE(u8.size() == 3 * px * (std::size_t)fx.frames);
  if (u8.size() != 3 * px * (std::size_t)fx.frames) { return; }

  const int OH = fx.size / 16, OW = fx.size / 16;
  const std::size_t tokens = (std::size_t)OH * OW;
  const int dim = proj->config().out_dim;

  proj->reset();
  int produced = 0;
  double worst = 0.0;
  bool all_ok = true;

  for (const auto& [lo, hi] : call_plan_(fx.frames)) {
    const int n = hi - lo;
    // The golden clip is [3, F, H, W]; this API takes frames of planar
    // [3, H, W], so the slice is a transpose, not a memcpy.
    std::vector<std::uint8_t> chunk((std::size_t)n * 3 * px);
    for (int f = 0; f < n; ++f) {
      for (int c = 0; c < 3; ++c) {
        const std::size_t src =
            ((std::size_t)c * fx.frames + (lo + f)) * px;
        const std::size_t dst = ((std::size_t)f * 3 + c) * px;
        std::memcpy(chunk.data() + dst, u8.data() + src, px);
      }
    }

    std::vector<SharedBuffer> rows;
    int row_frames = 0;
    const bool ok = proj->stream_forward(chunk.data(), n, fx.size, fx.size,
                                         &rows, &row_frames, &err);
    ASSERT_TRUE(ok);
    if (!ok) {
      printf("  stream_forward failed at [%d,%d): %s\n", lo, hi, err.c_str());
      return;
    }
    if (row_frames == 0) {
      // The opening call, which seeds the carries and emits nothing.
      EXPECT_TRUE(rows.empty());
      continue;
    }
    ASSERT_TRUE(rows.size() == 1);
    if (rows.size() != 1) { return; }

    const auto ref =
        read_f32_(fs::path(fx.golden) / ("lq_proj" + std::to_string(produced) +
                                        ".f32"));
    const std::size_t n_el = tokens * (std::size_t)row_frames *
                             (std::size_t)dim;
    ASSERT_TRUE(ref.size() == n_el);
    if (ref.size() != n_el) { return; }

    std::vector<float> got(n_el);
    const auto* src = (const std::uint16_t*)rows[0].contents();
    for (std::size_t i = 0; i < n_el; ++i) { got[i] = bf16_(src[i]); }

    const double r = rel_l2_(got.data(), ref.data(), n_el);
    if (r > worst) { worst = r; }
    // The golden is the reference's arithmetic at f32; this runs bf16,
    // so the residual IS bf16 rounding through two convolutions, two
    // norms and a projection. A structural error -- a wrong unshuffle
    // order, a carry read after it was written -- lands orders above
    // this, not just outside it.
    const bool pass = r < 0.02;
    if (!pass) {
      printf("  lq_proj%d rel-L2 %.5f\n", produced, r);
      all_ok = false;
    }
    EXPECT_TRUE(pass);
    ++produced;
  }

  printf("  lq-proj: %d row sets, worst rel-L2 %.5f\n", produced, worst);
  EXPECT_TRUE(produced == 10);
  EXPECT_TRUE(all_ok);
}
