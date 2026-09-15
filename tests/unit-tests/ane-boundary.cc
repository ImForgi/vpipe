// GPU <-> ANE handoff cost.
//
// The ANE reaches ~14-16 TOPS on fp16 matmul at tile shapes the GPU
// path measures 6.6-7.3 TFLOP/s on, so moving a DiT's GEMMs there is
// worth roughly 2x on an M4 Pro and more on a base M4 (same 16-core
// ANE, about half the GPU). What decides whether any of that survives
// is the BOUNDARY: CoreML exposes no Metal-shared-event interop, so a
// GPU -> ANE -> GPU crossing drains through the CPU and costs a
// pipeline bubble that no amount of ANE throughput pays back.
//
// This measures one crossing, on buffers the ANE reads and writes
// ZERO-COPY out of Metal-shared memory (CoreMLPredictInput::data /
// CoreMLPredictOutput::backing), which is the arrangement a real
// integration would use.
//
// Three arms, so the bubble can be separated from the work:
//
//   predict-only   steady-state predict(), no GPU work at all
//   gpu-only       the same elementwise kernel, commit + wait
//   round-trip     kernel -> commit/wait -> predict -> kernel ->
//                  commit/wait
//
// handoff = round-trip - predict-only - 2 * gpu-only, i.e. what the
// crossing costs beyond the work on either side.
//
// Gated on a compiled model whose single input and single output are
// both fixed-shape fp16 (tools generate one; any ANE-resident matmul
// module will do):
//
//     VPIPE_ANE_TEST_MODEL=/path/to/module.mlmodelc
//
// Unset -> the test passes trivially, so default suites stay green.

#include "minitest.h"

#include "apple-silicon/coreml/coreml-model-manager.h"
#include "common/session.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char*
env_or_null_(const char* name)
{
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

double
ms_since_(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

// Median of a sample, which is what to report when the ANE shares a
// power budget with the GPU and an occasional outlier is the norm.
double
median_(std::vector<double> v)
{
  if (v.empty()) { return 0.0; }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

std::size_t
elems_of_(const std::vector<std::int64_t>& shape)
{
  std::size_t n = 1;
  for (std::int64_t d : shape) { n *= (std::size_t)std::max<std::int64_t>(d, 1); }
  return n;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace

TEST(ane_boundary, gpu_ane_handoff_cost)
{
  const char* model_path = env_or_null_("VPIPE_ANE_TEST_MODEL");
  if (model_path == nullptr) { return; }

#ifndef VPIPE_BUILD_APPLE_SILICON
  (void)model_path;
#else
  using namespace vpipe;
  using namespace vpipe::metal_compute;

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto* mgr = sess.coreml_model_manager();
  if (mgr == nullptr) { return; }

  // 3 == CPU+NeuralEngine: forces the ANE-or-CPU choice, so a model
  // that cannot run on the ANE shows up as a huge time rather than
  // quietly succeeding on the GPU.
  auto model = mgr->load(model_path, 3);
  ASSERT_TRUE(model != nullptr);
  if (model == nullptr) { return; }
  ASSERT_TRUE(model->valid());
  if (!model->valid()) { return; }

  ASSERT_TRUE(model->input_names().size() == 1u);
  ASSERT_TRUE(model->output_names().size() == 1u);
  if (model->input_names().size() != 1u ||
      model->output_names().size() != 1u) {
    return;
  }
  const std::string in_name  = model->input_names()[0];
  const std::string out_name = model->output_names()[0];

  const auto in_it  = model->input_descs().find(in_name);
  const auto out_it = model->output_descs().find(out_name);
  ASSERT_TRUE(in_it != model->input_descs().end());
  ASSERT_TRUE(out_it != model->output_descs().end());
  if (in_it == model->input_descs().end() ||
      out_it == model->output_descs().end()) {
    return;
  }

  // Zero-copy on the output needs a fixed shape whose native dtype is
  // what we ask for; anything else silently becomes a conversion into
  // `owned` and the number stops describing the boundary.
  ASSERT_TRUE(out_it->second.fixed);
  ASSERT_TRUE(in_it->second.dtype == CoreMLDType::F16);
  ASSERT_TRUE(out_it->second.dtype == CoreMLDType::F16);
  if (!out_it->second.fixed ||
      in_it->second.dtype != CoreMLDType::F16 ||
      out_it->second.dtype != CoreMLDType::F16) {
    std::printf("[ane_boundary] model is not fixed-shape fp16 on both "
                "ends; the zero-copy path would not be exercised\n");
    return;
  }

  const std::vector<std::int64_t> in_shape  = in_it->second.shape;
  const std::vector<std::int64_t> out_shape = out_it->second.shape;
  const std::size_t in_elems  = elems_of_(in_shape);
  const std::size_t out_elems = elems_of_(out_shape);

  // in / scratch feed the elementwise kernel; out receives the ANE
  // result in place. All three are UMA, so nothing is copied at the
  // boundary -- only synchronised.
  SharedBuffer a   = mc->make_shared_buffer(in_elems * 2);
  SharedBuffer b   = mc->make_shared_buffer(in_elems * 2);
  SharedBuffer in  = mc->make_shared_buffer(in_elems * 2);
  SharedBuffer out = mc->make_shared_buffer(out_elems * 2);
  ASSERT_TRUE(!a.empty() && !b.empty() && !in.empty() && !out.empty());
  if (a.empty() || b.empty() || in.empty() || out.empty()) { return; }
  std::memset(a.contents(), 0, in_elems * 2);
  std::memset(b.contents(), 0, in_elems * 2);

  ComputeLibrary lib = mc->load_library("llm_elementwise");
  ASSERT_TRUE(lib.valid());
  if (!lib.valid()) { return; }
  ComputeFunction add = lib.function("residual_add_f16");
  ASSERT_TRUE(add.valid());
  if (!add.valid()) { return; }

  const int n = (int)in_elems;
  auto encode_add = [&](ComputeEncoder& enc, const SharedBuffer& dst) {
    enc.set_function(add);
    enc.set_buffer(0, a);
    enc.set_buffer(1, b);
    enc.set_buffer(2, dst);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)((n + 255) / 256 * 256), 1, 1}, {256, 1, 1});
  };
  auto gpu_pass = [&](const SharedBuffer& dst) {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder enc = st.begin_compute();
      encode_add(enc, dst);
    }
    st.commit().wait();
  };

  auto do_predict = [&]() {
    CoreMLPredictInput pi;
    pi.name  = in_name;
    pi.data  = in.contents();
    pi.dtype = CoreMLDType::F16;
    pi.shape = in_shape;
    CoreMLPredictOutput po;
    po.name          = out_name;
    po.want          = CoreMLDType::F16;
    po.backing       = out.contents();
    po.backing_elems = out_elems;
    const CoreMLPredictInput ins[1] = {pi};
    CoreMLPredictOutput      outs[1] = {po};
    return model->predict(ins, outs);
  };

  const int kWarm = 3;
  const int kIters = 12;
  for (int i = 0; i < kWarm; ++i) {
    gpu_pass(in);
    ASSERT_TRUE(do_predict());
  }

  // The control has the SAME number of command-stream commits as the
  // arm under test, so the only difference is the ANE sitting between
  // them. Subtracting a standalone gpu_pass instead would double-count
  // its commit/wait overhead -- which dominates this kernel -- and can
  // drive the result negative.
  std::vector<double> t_predict, t_pair, t_round;
  for (int i = 0; i < kIters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    do_predict();
    t_predict.push_back(ms_since_(t0));

    // control: two GPU passes, two commits, no crossing
    t0 = std::chrono::steady_clock::now();
    gpu_pass(in);
    gpu_pass(out);
    t_pair.push_back(ms_since_(t0));

    // under test: GPU produces `in`, the ANE consumes it and writes
    // `out`, the GPU consumes `out`. Each commit().wait() is the drain
    // CoreML gives us no way to avoid.
    t0 = std::chrono::steady_clock::now();
    gpu_pass(in);
    do_predict();
    gpu_pass(out);
    t_round.push_back(ms_since_(t0));
  }

  const double p  = median_(t_predict);
  const double c  = median_(t_pair);
  const double r  = median_(t_round);
  const double hd = r - c - p;

  std::printf("[ane_boundary] in %zu elems, out %zu elems (fp16)\n",
              in_elems, out_elems);
  std::printf("[ane_boundary] predict alone       %8.3f ms\n", p);
  std::printf("[ane_boundary] 2 GPU passes        %8.3f ms  (control, "
              "same commit count)\n", c);
  std::printf("[ane_boundary] GPU -> ANE -> GPU   %8.3f ms\n", r);
  // A NEGATIVE figure is the expected healthy result, not an error: it
  // says the mixed sequence beats the GPU-only control plus a
  // standalone predict, i.e. the ANE phase overlaps the GPU's
  // commit/wait rather than serialising behind it. Only a clearly
  // POSITIVE number would mean the CPU drain is costing a bubble, and
  // that is the thing this test exists to catch.
  std::printf("[ane_boundary] CROSSING COST       %8.3f ms  (%.1f%% of "
              "the round trip)%s\n", hd,
              r > 0.0 ? 100.0 * hd / r : 0.0,
              hd <= 0.0 ? "   <= 0: free / overlapped" : "");
  if (hd > 0.0) {
    std::printf("[ane_boundary] a 28-block x 2-crossing step would "
                "spend %.1f ms crossing\n", hd * 28 * 2);
  }

  // The arms must at least be ordered sanely; the handoff itself is a
  // measurement, not a pass/fail bar.
  EXPECT_TRUE(r > 0.0);
  EXPECT_TRUE(p > 0.0);
#endif
}
