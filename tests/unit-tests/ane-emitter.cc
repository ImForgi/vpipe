// AneEmitter: writing a compiled CoreML model without coremltools.
//
// These need NO model files and NO Python -- the emitter builds its own
// input -- so unlike the rest of the ANE tests they run unconditionally
// and guard the thing most likely to break silently: an undocumented
// format drifting under a macOS update.

#include "minitest.h"

#include "common/session.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/coreml/ane-emitter.h"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "interfaces/session-services-intf.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

#ifdef VPIPE_BUILD_APPLE_SILICON
std::string
scratch_()
{
  return std::filesystem::temp_directory_path().string();
}
#endif

}  // namespace

// The guard the whole tier leans on. If this fails on some future
// macOS, the emitted format has moved and every consumer must fall back
// to its GPU path -- so a failure here is informative, not flaky.
TEST(ane_emitter, self_test_passes)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  using namespace vpipe;
  Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }
  std::string err;
  const bool ok = AneEmitter::self_test(*mgr, scratch_(), &err);
  if (!ok) {
    std::printf("[ane_emitter] self-test FAILED: %s\n", err.c_str());
  }
  EXPECT_TRUE(ok);
#endif
}

// A shape the emitter has to get right for real use: tiled, so the
// weights arrive as many slabs and every BLOBFILE offset has to land on
// its own entry header. A wrong offset still loads and still runs.
TEST(ane_emitter, tiled_matmul_matches_reference)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  using namespace vpipe;
  Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }

  AneGraphSpec s;
  s.kind = AneGraphSpec::Kind::Matmul;
  s.M = 128; s.K = 256; s.N = 256;
  s.bk = 128; s.bn = 128;                 // 2 x 2 tiles
  ASSERT_TRUE(s.valid());

  const std::vector<std::size_t> want = AneEmitter::weight_slabs(s);
  ASSERT_TRUE(want.size() == 4u);
  if (want.size() != 4u) { return; }

  // One logical [K,N] weight, gathered into tiles in the emitter's
  // order: n-tile major, k-tile minor.
  std::vector<float> Wf((std::size_t)s.K * s.N);
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  for (auto& v : Wf) { v = (float)(_Float16)(d(rng) * 0.25f); }

  std::vector<std::vector<std::uint16_t>> tiles;
  for (int j = 0; j < s.N / s.bn; ++j) {
    for (int t = 0; t < s.K / s.bk; ++t) {
      std::vector<std::uint16_t> tile((std::size_t)s.bk * s.bn);
      for (int k = 0; k < s.bk; ++k) {
        for (int n = 0; n < s.bn; ++n) {
          const _Float16 h =
              (_Float16)Wf[(std::size_t)(t * s.bk + k) * s.N + j * s.bn + n];
          std::memcpy(&tile[(std::size_t)k * s.bn + n], &h, sizeof(h));
        }
      }
      tiles.push_back(std::move(tile));
    }
  }
  std::vector<AneBlobSlab> slabs;
  for (auto& t : tiles) { slabs.push_back({t.data(), t.size() * 2}); }

  const std::string dir = scratch_() + "/ane-emitter-tiled.mlmodelc";
  std::string err;
  const bool emitted = AneEmitter::emit(s, slabs, dir, &err);
  if (!emitted) { std::printf("[ane_emitter] emit: %s\n", err.c_str()); }
  ASSERT_TRUE(emitted);
  if (!emitted) { return; }

  auto model = mgr->load(dir, 3);
  ASSERT_TRUE(model != nullptr && model->valid());
  if (model == nullptr || !model->valid()) { return; }

  std::vector<float> xf((std::size_t)s.M * s.K);
  std::vector<std::uint16_t> x(xf.size());
  for (std::size_t i = 0; i < xf.size(); ++i) {
    xf[i] = (float)(_Float16)(d(rng) * 0.5f);
    const _Float16 h = (_Float16)xf[i];
    std::memcpy(&x[i], &h, sizeof(h));
  }
  std::vector<std::uint16_t> y((std::size_t)s.M * s.N, 0);

  CoreMLPredictInput in;
  in.name = AneEmitter::input_name();
  in.data = x.data();
  in.dtype = CoreMLDType::F16;
  in.shape = {s.M, s.K};
  CoreMLPredictOutput out;
  out.name = AneEmitter::output_name(s);
  out.want = CoreMLDType::F16;
  out.backing = y.data();
  out.backing_elems = y.size();
  const CoreMLPredictInput ins[1] = {in};
  CoreMLPredictOutput outs[1] = {out};
  const bool ran = model->predict(ins, outs);
  model.reset();
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  ASSERT_TRUE(ran);
  if (!ran) { return; }

  double num = 0.0, den = 0.0;
  for (int m = 0; m < s.M; ++m) {
    for (int n = 0; n < s.N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < s.K; ++k) {
        acc += (double)xf[(std::size_t)m * s.K + k] *
               (double)Wf[(std::size_t)k * s.N + n];
      }
      _Float16 h;
      std::memcpy(&h, &y[(std::size_t)m * s.N + n], sizeof(h));
      num += ((double)h - acc) * ((double)h - acc);
      den += acc * acc;
    }
  }
  const double rel = (den > 0.0) ? std::sqrt(num / den) : 1.0;
  std::printf("[ane_emitter] tiled %dx%dx%d (%zu slabs): rel-L2 %.3e\n",
              s.M, s.K, s.N, want.size(), rel);
  EXPECT_TRUE(rel < 5e-2);
#endif
}

// The graph a DiT block actually uses: gate/up/silu/mul/down, tiled,
// nine weight slabs in three groups. Exercises the op vocabulary beyond
// matmul and the slab ORDER across three separate matrices -- gate's
// tiles, then up's, then down's.
TEST(ane_emitter, swiglu_ffn_matches_reference)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  using namespace vpipe;
  Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }

  AneGraphSpec s;
  s.kind = AneGraphSpec::Kind::SwiGluFfn;
  s.M = 64; s.K = 128; s.N = 256;      // M rows, D=128 hidden, F=256 ffn
  s.bk = 64; s.bn = 64;
  ASSERT_TRUE(s.valid());

  std::mt19937 rng(23);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  auto mat = [&](int R, int C) {
    std::vector<float> v((std::size_t)R * C);
    for (auto& e : v) { e = (float)(_Float16)(d(rng) * 0.3f); }
    return v;
  };
  const std::vector<float> Wg = mat(s.K, s.N);
  const std::vector<float> Wu = mat(s.K, s.N);
  const std::vector<float> Wd = mat(s.N, s.K);

  // Gather each matrix into tiles, n-major / k-minor, in the order the
  // emitter consumes them: gate, then up, then down.
  std::vector<std::vector<std::uint16_t>> tiles;
  auto gather = [&](const std::vector<float>& W, int K, int N) {
    for (int j = 0; j < N / s.bn; ++j) {
      for (int t = 0; t < K / s.bk; ++t) {
        std::vector<std::uint16_t> tl((std::size_t)s.bk * s.bn);
        for (int k = 0; k < s.bk; ++k) {
          for (int n = 0; n < s.bn; ++n) {
            const _Float16 h =
                (_Float16)W[(std::size_t)(t * s.bk + k) * N + j * s.bn + n];
            std::memcpy(&tl[(std::size_t)k * s.bn + n], &h, sizeof(h));
          }
        }
        tiles.push_back(std::move(tl));
      }
    }
  };
  gather(Wg, s.K, s.N);
  gather(Wu, s.K, s.N);
  gather(Wd, s.N, s.K);
  ASSERT_TRUE(tiles.size() == AneEmitter::weight_slabs(s).size());
  if (tiles.size() != AneEmitter::weight_slabs(s).size()) { return; }

  std::vector<AneBlobSlab> slabs;
  for (auto& t : tiles) { slabs.push_back({t.data(), t.size() * 2}); }
  const std::string dir = scratch_() + "/ane-emitter-swiglu.mlmodelc";
  std::string err;
  const bool emitted = AneEmitter::emit(s, slabs, dir, &err);
  if (!emitted) { std::printf("[ane_emitter] emit: %s\n", err.c_str()); }
  ASSERT_TRUE(emitted);
  if (!emitted) { return; }

  auto model = mgr->load(dir, 3);
  ASSERT_TRUE(model != nullptr && model->valid());
  if (model == nullptr || !model->valid()) { return; }

  std::vector<float> xf((std::size_t)s.M * s.K);
  std::vector<std::uint16_t> x(xf.size());
  for (std::size_t i = 0; i < xf.size(); ++i) {
    xf[i] = (float)(_Float16)(d(rng) * 0.5f);
    const _Float16 h = (_Float16)xf[i];
    std::memcpy(&x[i], &h, sizeof(h));
  }
  std::vector<std::uint16_t> y((std::size_t)s.M * s.K, 0);
  CoreMLPredictInput in;
  in.name = AneEmitter::input_name(); in.data = x.data();
  in.dtype = CoreMLDType::F16; in.shape = {s.M, s.K};
  CoreMLPredictOutput out;
  out.name = AneEmitter::output_name(s); out.want = CoreMLDType::F16;
  out.backing = y.data(); out.backing_elems = y.size();
  const CoreMLPredictInput ins[1] = {in};
  CoreMLPredictOutput outs[1] = {out};
  const bool ran = model->predict(ins, outs);
  model.reset();
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  ASSERT_TRUE(ran);
  if (!ran) { return; }

  double num = 0.0, den = 0.0;
  std::vector<double> h(s.N);
  for (int m = 0; m < s.M; ++m) {
    for (int f = 0; f < s.N; ++f) {
      double g = 0.0, u = 0.0;
      for (int k = 0; k < s.K; ++k) {
        g += (double)xf[(std::size_t)m * s.K + k] * Wg[(std::size_t)k * s.N + f];
        u += (double)xf[(std::size_t)m * s.K + k] * Wu[(std::size_t)k * s.N + f];
      }
      h[f] = (g / (1.0 + std::exp(-g))) * u;      // silu(g) * u
    }
    for (int c = 0; c < s.K; ++c) {
      double acc = 0.0;
      for (int f = 0; f < s.N; ++f) {
        acc += h[f] * Wd[(std::size_t)f * s.K + c];
      }
      _Float16 hv;
      std::memcpy(&hv, &y[(std::size_t)m * s.K + c], sizeof(hv));
      num += ((double)hv - acc) * ((double)hv - acc);
      den += acc * acc;
    }
  }
  const double rel = (den > 0.0) ? std::sqrt(num / den) : 1.0;
  std::printf("[ane_emitter] swiglu M=%d D=%d F=%d (%zu slabs): rel-L2 "
              "%.3e\n", s.M, s.K, s.N, tiles.size(), rel);
  EXPECT_TRUE(rel < 5e-2);
#endif
}

// The slab contract is the one a caller can get wrong silently, so it
// is refused rather than truncated.
TEST(ane_emitter, wrong_slab_sizes_are_refused)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  using namespace vpipe;
  AneGraphSpec s;
  s.kind = AneGraphSpec::Kind::Matmul;
  s.M = 64; s.K = 64; s.N = 64;
  std::vector<std::uint16_t> junk(16);
  const AneBlobSlab bad[1] = {{junk.data(), junk.size() * 2}};
  std::string err;
  EXPECT_FALSE(AneEmitter::emit(s, bad, scratch_() + "/ane-bad.mlmodelc",
                                &err));
  EXPECT_TRUE(!err.empty());

  // ...and a spec whose tiling does not divide is refused up front,
  // rather than emitting a graph with a ragged last tile.
  AneGraphSpec r = s;
  r.bk = 48;
  EXPECT_FALSE(r.valid());
#endif
}
