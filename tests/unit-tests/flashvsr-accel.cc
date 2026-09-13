// FlashVSR on the matrix cores: the NAX flash entry over the routed
// spans, the matmul2d GEMMs, and the two opt-in tiers, each held to this
// tree's own ALU path.
//
// NEEDS NO CHECKPOINT. The denoiser and the source projection are driven
// over SYNTHETIC weights written to a temporary directory at the real
// widths -- 1536 wide, 12 x 128 heads, an 8960 feed-forward, a 512-token
// context -- because what is checked here is that every accelerated arm
// computes the SAME FUNCTION as the ALU arm, and that question does not
// need trained weights. Whether the port matches the reference is
// flashvsr_dit's question, and that one needs the golden.
//
// THE ARM THAT MATTERS MOST IS `nax-mask`. The NAX entry tiles queries by
// 64 and keys by 32 where the ALU entry tiles 32 and 16, and the span
// list's unit is the kernel's key block -- so a CSR built for the wrong
// kernel visits the wrong keys and still produces a plausible frame. The
// per-key mask path cannot make that mistake (its unit is one key), so
// spans against mask on the SAME kernel is the check that the tiles
// followed the kernel.
//
// Routing is made to actually DECIDE: at 512 px the reference's defaults
// keep every block, which would make spans and mask agree vacuously. The
// default knobs below drop about a quarter of the key blocks in the chunk
// that attends over the cache.
//
// NOT HARSHER here, because harsher routing sends some query tiles to NO
// key block at all, and that case has a test of its own below
// (an_unrouted_query_is_zero_not_nan).
//
// M5 only: without matrix cores there is only the ALU arm.
//
// Env:
//   VPIPE_FVSR_ACCEL_LAYERS  denoiser depth for the synthetic stack (2)
//   VPIPE_FVSR_ACCEL_SIZE    output size, a multiple of 128 (512)
//   VPIPE_FVSR_ACCEL_SPARSE  the routing's sparse_ratio (0.8)
//   VPIPE_FVSR_ACCEL_LOCAL   the routing's local_range (11)
//   VPIPE_FVSR_ACCEL_BIG     the source projection at its real widths
//                            (2048 / 3072, ~550 MB of weights) instead of
//                            256 / 512

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/flashvsr/flashvsr-lq-proj.h"
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/quantize/safetensors-writer.h"
#include "generative-models/weight-set.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

namespace fs = std::filesystem;

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int
env_int_(const char* name, int dflt)
{
  const char* e = std::getenv(name);
  const int v = (e != nullptr) ? std::atoi(e) : 0;
  return v > 0 ? v : dflt;
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

bool
finite_(const std::vector<float>& v)
{
  for (const float x : v) {
    if (!std::isfinite(x)) { return false; }
  }
  return !v.empty();
}

// Writes BF16 tensors of gaussian noise at a chosen spread.
struct SynthWriter {
  SafetensorsWriter wr;
  std::mt19937_64 rng;
  bool ok = true;

  SynthWriter(const fs::path& dir, std::uint64_t seed)
      : wr(dir.string()), rng(seed) {}

  void
  add(const std::string& name, const std::vector<std::int64_t>& shape,
      float sd, float mean = 0.0f)
  {
    std::size_t n = 1;
    for (const auto d : shape) { n *= (std::size_t)d; }
    std::normal_distribution<float> nd(mean, sd);
    std::vector<std::uint16_t> buf(n);
    for (auto& v : buf) { v = to_bf16_(nd(rng)); }
    ok = ok && wr.add(name, "BF16", shape, buf.data(), n * 2);
  }
  // A linear, scaled so an activation keeps its spread through it.
  void
  linear(const std::string& stem, std::int64_t out, std::int64_t in)
  {
    add(stem + ".weight", {out, in}, 1.0f / std::sqrt((float)in));
    add(stem + ".bias", {out}, 0.02f);
  }
  bool close() { return wr.close() && ok; }
};

// Wan 2.1's tensor names at FlashVSR's geometry, over `layers` blocks.
bool
write_dit_(const fs::path& dir, int layers)
{
  constexpr std::int64_t D = 1536, FF = 8960, TD = 256, TT = 512, FQ = 256;
  SynthWriter w(dir, 0xf1a5u);
  w.add("patch_embedding.weight", {D, 16, 1, 2, 2}, 1.0f / 8.0f);
  w.add("patch_embedding.bias", {D}, 0.02f);
  w.linear("text_embedding.0", D, TD);
  w.linear("text_embedding.2", D, D);
  w.linear("time_embedding.0", D, FQ);
  w.linear("time_embedding.2", D, D);
  w.linear("time_projection.1", 6 * D, D);
  w.linear("head.head", 64, D);
  w.add("head.modulation", {1, 2, D}, 0.1f);
  w.add("posi_context", {TT, TD}, 1.0f);
  for (int i = 0; i < layers; ++i) {
    const std::string b = "blocks." + std::to_string(i) + ".";
    for (const char* a : {"self_attn", "cross_attn"}) {
      for (const char* p : {"q", "k", "v", "o"}) {
        w.linear(b + a + "." + p, D, D);
      }
      w.add(b + a + ".norm_q.weight", {D}, 0.02f, 1.0f);
      w.add(b + a + ".norm_k.weight", {D}, 0.02f, 1.0f);
    }
    w.add(b + "norm3.weight", {D}, 0.02f, 1.0f);
    w.add(b + "norm3.bias", {D}, 0.02f);
    w.linear(b + "ffn.0", FF, D);
    w.linear(b + "ffn.2", D, FF);
    w.add(b + "modulation", {1, 6, D}, 0.1f);
  }
  return w.close();
}

// The source projection's tensor names. conv1 reads the 16x16 pixel
// unshuffle of RGB, so its input is always 768 channels.
bool
write_lq_(const fs::path& dir, int c1, int c2)
{
  SynthWriter w(dir, 0x19b0u);
  w.add("lq_proj.conv1.weight", {c1, 768, 4, 3, 3},
        1.0f / std::sqrt(768.0f * 36.0f));
  w.add("lq_proj.conv1.bias", {c1}, 0.02f);
  w.add("lq_proj.norm1.gamma", {c1}, 0.02f, 1.0f);
  w.add("lq_proj.conv2.weight", {c2, c1, 4, 3, 3},
        1.0f / std::sqrt((float)c1 * 36.0f));
  w.add("lq_proj.conv2.bias", {c2}, 0.02f);
  w.add("lq_proj.norm2.gamma", {c2}, 0.02f, 1.0f);
  w.linear("lq_proj.linear_layers.0", 1536, c2);
  return w.close();
}

// Env for one arm, restored on the way out. The switches are read at
// load and at the first forward, so the scope has to span both.
struct EnvScope {
  std::vector<std::string> names;
  explicit EnvScope(std::vector<std::string> n) : names(std::move(n))
  {
    for (const auto& s : names) { ::setenv(s.c_str(), "1", 1); }
  }
  ~EnvScope()
  {
    for (const auto& s : names) { ::unsetenv(s.c_str()); }
  }
};

struct TmpDir {
  fs::path p;
  explicit TmpDir(const char* stem)
      : p(fs::temp_directory_path() /
          (std::string(stem) + "-" + std::to_string((long)::getpid())))
  {
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p, ec);
  }
  ~TmpDir()
  {
    std::error_code ec;
    fs::remove_all(p, ec);
  }
};

}  // namespace

TEST(flashvsr_accel, denoiser_arms_agree_with_the_alu_path)
{
  vpipe::Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("  no matrix cores -- only the ALU arm exists, SKIPPED\n");
    return;
  }

  const int layers = env_int_("VPIPE_FVSR_ACCEL_LAYERS", 2);
  const int size = env_int_("VPIPE_FVSR_ACCEL_SIZE", 512);
  if (size % MetalFlashVsrTransformer::kSizeGrid != 0) { return; }
  TmpDir tmp("vpipe-flashvsr-accel-dit");
  const bool wrote = write_dit_(tmp.p, layers);
  ASSERT_TRUE(wrote);
  if (!wrote) { return; }

  MetalFlashVsrTransformer::Config base;
  std::string err;
  const bool cfg_ok = MetalFlashVsrTransformer::config_from_checkpoint(
      tmp.p.string(), &base, &err);
  ASSERT_TRUE(cfg_ok);
  if (!cfg_ok) {
    std::printf("  config: %s\n", err.c_str());
    return;
  }
  EXPECT_TRUE(base.n_layers == layers && base.hidden == 1536);
  // One WeightSet for every arm, so the bf16 conversions are paid once.
  auto ws = WeightSet::open(tmp.p.string(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (ws == nullptr) { return; }

  // Two chunks: the opening one (6 latent frames, no cache) and one that
  // attends over it -- which is where spans, the cache stride and the
  // growing key length all meet.
  constexpr int kFrames = 33;
  const int T = MetalFlashVsrTransformer::latent_frames(kFrames);
  const int h8 = size / 8, tok = (size / 16) * (size / 16);
  std::mt19937_64 rng(0xacce1u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> noise((std::size_t)16 * T * h8 * h8);
  for (auto& v : noise) { v = nd(rng); }
  std::vector<std::uint16_t> rows((std::size_t)T * tok * base.hidden);
  for (auto& v : rows) { v = to_bf16_(0.5f * nd(rng)); }

  MetalFlashVsrTransformer::Request req;
  req.rows = rows.data();
  req.row_frames = T;
  req.frames = kFrames;
  req.height = size;
  req.width = size;
  req.init_noise = noise.data();
  if (const char* e = std::getenv("VPIPE_FVSR_ACCEL_SPARSE")) {
    req.params.sparse_ratio = std::atof(e);
  } else {
    req.params.sparse_ratio = 0.8;
  }
  req.params.local_range = env_int_("VPIPE_FVSR_ACCEL_LOCAL", 11);
  std::printf("  %dx%d, %d layers, sparse_ratio %.2f, local_range %d\n",
              size, size, layers, req.params.sparse_ratio,
              req.params.local_range);

  struct Arm {
    const char* name;
    std::vector<std::string> env;
    bool i8 = false, sage = false;
    const char* ref;      // the arm this one is held to
    double bound;
  };
  const std::vector<Arm> arms = {
    {"alu", {"VPIPE_FVSR_NO_MMA2", "VPIPE_FVSR_NO_ATTN_NAX"}, false, false,
     nullptr, 0.0},
    {"alu-mask", {"VPIPE_FVSR_NO_MMA2", "VPIPE_FVSR_NO_ATTN_NAX",
                  "VPIPE_FVSR_MASK_ATTN"}, false, false, "alu", 0.01},
    {"gemm-mma", {"VPIPE_FVSR_NO_ATTN_NAX"}, false, false, "alu", 0.05},
    {"attn-nax", {"VPIPE_FVSR_NO_MMA2"}, false, false, "alu", 0.05},
    {"nax", {}, false, false, "alu", 0.05},
    {"nax-mask", {"VPIPE_FVSR_MASK_ATTN"}, false, false, "nax", 0.01},
    {"nax+i8", {}, true, false, "nax", 0.15},
    {"nax+sage", {}, false, true, "nax", 0.10},
  };

  std::vector<std::pair<std::string, std::vector<float>>> got;
  auto find = [&](const char* name) -> const std::vector<float>* {
    for (const auto& g : got) {
      if (g.first == name) { return &g.second; }
    }
    return nullptr;
  };
  double alu_ms = 0.0;
  for (const Arm& a : arms) {
    EnvScope env(a.env);
    MetalFlashVsrTransformer::Config cfg = base;
    cfg.i8_gemm = a.i8;
    cfg.sage.enabled = a.sage;
    auto m = MetalFlashVsrTransformer::load(ws, mc, cfg, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) {
      std::printf("  %-9s load: %s\n", a.name, err.c_str());
      continue;
    }
    // The PATH, before the numbers: every one of these falls back to a
    // slower kernel that computes the same answer, which is exactly what
    // an agreement bound cannot see.
    const bool want_mma = std::string(a.name) != "alu" &&
                          std::string(a.name) != "alu-mask" &&
                          std::string(a.name) != "attn-nax";
    const bool want_nax = std::string(a.name).rfind("nax", 0) == 0 ||
                          std::string(a.name) == "attn-nax";
    EXPECT_TRUE(m->uses_matrix_cores() == want_mma);
    EXPECT_TRUE(m->uses_attn_nax() == want_nax);
    EXPECT_TRUE(m->uses_i8_gemm() == a.i8);
    EXPECT_TRUE(m->uses_sage() == a.sage);

    std::vector<float> lat;
    std::vector<int> shape;
    bool ok = true;
    // WARM TWICE: the first builds pipelines and scratch, the second is
    // where the int8 context tunes its split for the shapes the first
    // recorded. Timing either would charge an arm a one-time cost.
    for (int i = 0; i < 2 && ok; ++i) {
      ok = m->generate(req, &lat, &shape, &err);
    }
    const auto t0 = std::chrono::steady_clock::now();
    ok = ok && m->generate(req, &lat, &shape, &err);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(ok);
    if (!ok) {
      std::printf("  %-9s generate: %s\n", a.name, err.c_str());
      continue;
    }
    const bool fin = finite_(lat);
    EXPECT_TRUE(fin);
    if (a.ref == nullptr) {
      alu_ms = ms;
      std::printf("  %-9s %8.1f ms  (baseline)  %s\n", a.name, ms,
                  m->accel_summary().c_str());
    } else {
      const std::vector<float>* ref = find(a.ref);
      const std::vector<float>* alu = find("alu");
      const double r = ref != nullptr ? rel_l2_(lat, *ref) : 1e9;
      const double ra = alu != nullptr ? rel_l2_(lat, *alu) : 1e9;
      std::printf("  %-9s %8.1f ms  %.2fx  vs %-4s %.2e  vs alu %.2e%s\n",
                  a.name, ms, alu_ms > 0.0 ? alu_ms / ms : 0.0, a.ref, r,
                  ra, fin ? "" : "  NON-FINITE");
      EXPECT_TRUE(r < a.bound);
    }
    got.emplace_back(a.name, std::move(lat));
  }
}

TEST(flashvsr_accel, source_projection_arms_agree_with_the_alu_path)
{
  vpipe::Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("  no matrix cores -- only the ALU arm exists, SKIPPED\n");
    return;
  }
  const bool big = std::getenv("VPIPE_FVSR_ACCEL_BIG") != nullptr;
  const int c1 = big ? 2048 : 256, c2 = big ? 3072 : 512;
  const int size = env_int_("VPIPE_FVSR_ACCEL_SIZE", 512);
  if (size % 16 != 0) { return; }
  TmpDir tmp("vpipe-flashvsr-accel-lq");
  const bool wrote = write_lq_(tmp.p, c1, c2);
  ASSERT_TRUE(wrote);
  if (!wrote) { return; }
  auto ws = WeightSet::open(tmp.p.string(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (ws == nullptr) { return; }

  // Nine frames: the seeding call and two that produce rows, which is
  // enough for both carries to be live.
  constexpr int kFrames = 9;
  const std::size_t frame_px = (std::size_t)3 * size * size;
  std::vector<std::uint8_t> clip(frame_px * kFrames);
  {
    std::mt19937_64 rng(0x5eedu);
    for (auto& v : clip) { v = (std::uint8_t)(rng() & 0xff); }
  }

  struct Arm {
    const char* name;
    std::vector<std::string> env;
    const char* ref;
    double bound;
  };
  const std::vector<Arm> arms = {
    {"alu", {"VPIPE_FVSR_NO_MMA2"}, nullptr, 0.0},
    {"mma-1op", {"VPIPE_LM_NO_SPLITK"}, "alu", 0.02},
    {"mma", {}, "alu", 0.02},
  };
  std::vector<std::pair<std::string, std::vector<float>>> got;
  double alu_ms = 0.0;
  for (const Arm& a : arms) {
    EnvScope env(a.env);
    std::string err;
    auto p = FlashVsrLqProj::load(ws, mc, &err);
    ASSERT_TRUE(p != nullptr);
    if (p == nullptr) {
      std::printf("  %-8s load: %s\n", a.name, err.c_str());
      continue;
    }
    EXPECT_TRUE(p->uses_matrix_cores() == (a.ref != nullptr));

    std::vector<SharedBuffer> out;
    auto drive = [&]() {
      out.clear();
      p->reset();
      bool ok = true;
      const int cuts[][2] = {{0, 1}, {1, 5}, {5, 9}};
      for (const auto& c : cuts) {
        int rf = 0;
        ok = ok && p->stream_forward(clip.data() + frame_px * c[0],
                                     c[1] - c[0], size, size, &out, &rf,
                                     &err);
      }
      return ok;
    };
    bool ok = drive();
    const auto t0 = std::chrono::steady_clock::now();
    ok = ok && drive();
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(ok && !out.empty());
    if (!ok || out.empty()) {
      std::printf("  %-8s forward: %s\n", a.name, err.c_str());
      continue;
    }
    std::vector<float> rows;
    for (const auto& b : out) {
      const std::size_t n = b.byte_size() / 2;
      const auto* s = (const std::uint16_t*)b.contents();
      for (std::size_t i = 0; i < n; ++i) { rows.push_back(from_bf16_(s[i])); }
    }
    const bool fin = finite_(rows);
    EXPECT_TRUE(fin);
    if (a.ref == nullptr) {
      alu_ms = ms;
      std::printf("  %-8s %8.1f ms  (baseline, %zu row sets, K %d / %d)\n",
                  a.name, ms, out.size(), 36 * 768, 36 * c1);
    } else {
      const std::vector<float>* ref = nullptr;
      for (const auto& g : got) {
        if (g.first == a.ref) { ref = &g.second; }
      }
      const double r = ref != nullptr ? rel_l2_(rows, *ref) : 1e9;
      std::printf("  %-8s %8.1f ms  %.2fx  vs %s %.2e%s\n", a.name, ms,
                  alu_ms > 0.0 ? alu_ms / ms : 0.0, a.ref, r,
                  fin ? "" : "  NON-FINITE");
      EXPECT_TRUE(r < a.bound);
    }
    got.emplace_back(a.name, std::move(rows));
  }
}

// A QUERY THE ROUTING SENDS NOWHERE attends to nothing, and its output is
// ZERO -- the FlashAttention answer for a row with no keys -- on the span
// path and the per-key mask path alike, on both flash entries.
//
// It used to be 0/0 on every span arm and a uniform average of the
// excluded keys on the mask arms. On real weights at 1920x1152 about
// eighteen such tiles in the first block went NaN, poisoned every later
// block's routing, and decoded to a clip of black frames. The harsh
// routing here produces such tiles on purpose, and the test checks that
// it did before believing anything else it measures.
TEST(flashvsr_accel, an_unrouted_query_is_zero_not_nan)
{
  vpipe::Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  TmpDir tmp("vpipe-flashvsr-accel-unrouted");
  const bool wrote = write_dit_(tmp.p, 2);
  ASSERT_TRUE(wrote);
  if (!wrote) { return; }
  MetalFlashVsrTransformer::Config cfg;
  std::string err;
  const bool cfg_ok = MetalFlashVsrTransformer::config_from_checkpoint(
      tmp.p.string(), &cfg, &err);
  ASSERT_TRUE(cfg_ok);
  if (!cfg_ok) { return; }
  auto ws = WeightSet::open(tmp.p.string(), nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (ws == nullptr) { return; }

  constexpr int kSize = 512, kFrames = 33;
  const int T = MetalFlashVsrTransformer::latent_frames(kFrames);
  const int h8 = kSize / 8, tok = (kSize / 16) * (kSize / 16);
  std::mt19937_64 rng(0xe3b7u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> noise((std::size_t)16 * T * h8 * h8);
  for (auto& v : noise) { v = nd(rng); }
  std::vector<std::uint16_t> rows((std::size_t)T * tok * cfg.hidden);
  for (auto& v : rows) { v = to_bf16_(0.5f * nd(rng)); }

  MetalFlashVsrTransformer::Request req;
  req.rows = rows.data();
  req.row_frames = T;
  req.frames = kFrames;
  req.height = kSize;
  req.width = kSize;
  req.init_noise = noise.data();
  req.params.sparse_ratio = 0.3;
  req.params.local_range = 5;

  struct Arm {
    const char* name;
    std::vector<std::string> env;
    bool mask;
    bool nax;
  };
  const std::vector<Arm> arms = {
    {"alu", {"VPIPE_FVSR_NO_MMA2", "VPIPE_FVSR_NO_ATTN_NAX"}, false, false},
    {"alu-mask", {"VPIPE_FVSR_NO_MMA2", "VPIPE_FVSR_NO_ATTN_NAX",
                  "VPIPE_FVSR_MASK_ATTN"}, true, false},
    {"nax", {}, false, true},
    {"nax-mask", {"VPIPE_FVSR_MASK_ATTN"}, true, true},
  };
  std::vector<float> alu;
  for (const Arm& a : arms) {
    if (a.nax && !mc->supports_matrix_cores()) { continue; }
    EnvScope env(a.env);
    auto m = MetalFlashVsrTransformer::load(ws, mc, cfg, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { continue; }
    std::vector<float> lat;
    std::vector<int> shape;
    const bool ok = m->generate(req, &lat, &shape, &err);
    std::printf("  %-8s %s, %llu query tiles routed nowhere\n", a.name,
                ok ? "finite" : err.c_str(),
                (unsigned long long)m->empty_query_tiles());
    // Refused as non-finite is exactly the failure this test is for.
    ASSERT_TRUE(ok);
    if (!ok) { continue; }
    // The case has to have HAPPENED for a finite answer to mean anything.
    // Only the span path counts: the mask path has no list to be empty.
    if (!a.mask) { EXPECT_TRUE(m->empty_query_tiles() > 0); }
    if (alu.empty()) {
      alu = std::move(lat);
      continue;
    }
    const double r = rel_l2_(lat, alu);
    std::printf("           vs alu %.2e\n", r);
    EXPECT_TRUE(r < 0.05);
  }
}
