#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/metal-compute/texture.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "generative-models/weight-set.h"

#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

MetalCompute*
get_mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  return mc;
}

}  // namespace

TEST(metal_compute_residency, support_probe_returns_bool) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Just check it doesn't crash; older hosts return false.
  const bool supported = mc->residency_set_supported();
  (void)supported;
  EXPECT_TRUE(true);
}

TEST(metal_compute_residency, add_buffer_increments_counter) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  SharedBuffer b = mc->make_shared_buffer(4096);
  if (b.empty()) {
    return;
  }
  const auto before = mc->residency_stats();
  EXPECT_TRUE(mc->residency_add(b));
  EXPECT_TRUE(mc->residency_commit());
  const auto after = mc->residency_stats();
  EXPECT_TRUE(after.add_calls == before.add_calls + 1);
  EXPECT_TRUE(after.current >= before.current + 1);
}

TEST(metal_compute_residency, remove_buffer_decrements) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  SharedBuffer b = mc->make_shared_buffer(4096);
  if (b.empty()) {
    return;
  }
  mc->residency_add(b);
  mc->residency_commit();
  const auto mid = mc->residency_stats();

  EXPECT_TRUE(mc->residency_remove(b));
  EXPECT_TRUE(mc->residency_commit());
  const auto after = mc->residency_stats();
  EXPECT_TRUE(after.remove_calls == mid.remove_calls + 1);
  EXPECT_TRUE(after.current <= mid.current);
}

TEST(metal_compute_residency, add_texture_works) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  TextureDesc d{};
  d.format = PixelFormat::RGBA8Unorm;
  d.width  = 16;
  d.height = 16;
  Texture t = mc->make_texture(d);
  if (!t.valid()) {
    return;
  }
  EXPECT_TRUE(mc->residency_add(t));
  EXPECT_TRUE(mc->residency_commit());
  mc->residency_remove(t);
  mc->residency_commit();
}

TEST(metal_compute_residency, request_and_end_are_no_op_safe) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  // Even with no allocations these should not crash.
  EXPECT_TRUE(mc->residency_request());
  EXPECT_TRUE(mc->residency_end());
}

TEST(metal_compute_residency, add_empty_buffer_is_rejected) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer empty;
  EXPECT_FALSE(mc->residency_add(empty));
}

// Is a buffer we are holding still IN RAM?
//
// The question a streamed DiT's residency policy has to answer, and the
// only one that distinguishes a pin that is paying for its memory from
// one that is being compressed behind our back. Free-memory arithmetic
// cannot answer it -- on the box this fixes, `available_physical` read
// 18.5 GB while the machine held 28.7 GB of swap, because the figure
// counts file cache and a streaming model's cache is its own re-reads.
//
// VERIFIED against real pressure while this was written: a 2 GB
// anonymous buffer on a 16 GB box went from 2048/2048 sampled pages
// resident to 1636/2048 with 412 PAGED_OUT the moment 8 GB of ballast
// engaged the compressor -- before any swap, which is what makes it an
// early enough signal to act on.
TEST(metal_compute_residency, page_residency_sees_a_live_buffer)
{
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  SharedBuffer b = mc->make_shared_buffer(64ull << 20);
  if (b.empty()) { EXPECT_TRUE(false); return; }
  // Touch it, so the pages exist rather than being lazily unbacked.
  std::memset(b.contents(), 1, b.byte_size());

  const auto r = b.page_residency();
  EXPECT_TRUE(r.valid);
  EXPECT_TRUE(r.examined > 0);
  EXPECT_TRUE(r.fully_resident());
  EXPECT_TRUE(r.paged_out == 0);
  EXPECT_TRUE(r.resident_fraction() > 0.999);

  // Sampling looks at fewer pages and must agree about the verdict.
  const auto s = b.page_residency(64);
  EXPECT_TRUE(s.valid);
  EXPECT_TRUE(s.examined > 0);
  EXPECT_TRUE(s.examined < r.examined);
  EXPECT_TRUE(s.fully_resident());

  // An empty handle answers without claiming anything.
  SharedBuffer none;
  const auto e = none.page_residency();
  EXPECT_TRUE(!e.valid);
  EXPECT_TRUE(e.fully_resident());     // vacuous: nothing has left RAM
}

// The process-scoped figures the policy polls each forward. Unlike the
// system-wide compressor count these say whose memory it is, which is
// what makes "our own weights are being squeezed" a statement we can
// make at all.
TEST(metal_compute_residency, budget_reports_this_process)
{
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const auto mb = mc->memory_budget();
  EXPECT_TRUE(mb.total_physical >= (4ull << 30));
  EXPECT_TRUE(mb.self_footprint > 0);
  EXPECT_TRUE(mb.self_footprint <= mb.total_physical);
  // Idle memory is a subset of what the cache-inclusive figure reports.
  EXPECT_TRUE(mb.free_physical <= mb.available_physical);
}

// ---------------------------------------------------------------------
// Does a block read 100% in RAM the moment it is kept?
//
// The residency policy sheds a block whenever ANY of the resident set
// has left RAM, so the whole thing rests on a freshly-kept block
// measuring as fully resident. If it does not -- if a buffer reads
// partly out-of-core the instant it is written -- then the signal is
// not eviction at all, and the policy would shed on its own arrival and
// converge to keeping nothing. That is a silent failure: the run still
// produces correct output, just at streaming speed forever.
TEST(metal_compute_residency, fresh_buffer_is_fully_incore)
{
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  // Block-sized, so this asks the question at the shape that matters --
  // a few pages could hide behind any allocator rounding.
  SharedBuffer b = mc->make_shared_buffer(256ull << 20);
  if (b.empty()) { return; }
  std::memset(b.contents(), 0xA5, b.byte_size());
  const auto r = b.page_residency(64);
  EXPECT_TRUE(r.valid);
  EXPECT_TRUE(r.examined > 0);
  EXPECT_TRUE(r.fully_resident());
  EXPECT_TRUE(r.paged_out == 0);
}

// The same question one layer up, on the path a promoted block actually
// takes: WeightSet::stream_tensor(Copied) -- allocate, memcpy from the
// mapped shard, hand the bytes to the caller. Copied is the point;
// Mapped would leave the tensor on clean file-backed pages, which the
// OS reclaims freely and which would therefore read out-of-core under
// no memory pressure at all.
TEST(metal_compute_residency, streamed_copy_is_fully_incore)
{
  const char* dir = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (dir == nullptr || *dir == '\0') { return; }
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto ws = genai::WeightSet::open(dir, nullptr);
  if (!ws) { return; }
  // The largest tensor in the checkpoint: big enough that the sampled
  // walk examines a real number of pages.
  std::string best;
  std::size_t best_bytes = 0;
  for (const std::string& nm : ws->src().tensor_names()) {
    const auto* i = ws->src().info(nm);
    if (i == nullptr) { continue; }
    std::size_t n = 1;
    for (long d : i->shape) { n *= (std::size_t)(d > 0 ? d : 0); }
    if (n > best_bytes) { best_bytes = n; best = nm; }
  }
  if (best.empty()) { return; }
  SharedBuffer b =
      ws->stream_tensor(best, mc, genai::WeightSet::Residency::Copied);
  if (b.empty()) { return; }
  const auto r = b.page_residency(64);
  EXPECT_TRUE(r.valid);
  EXPECT_TRUE(r.examined > 0);
  EXPECT_TRUE(r.fully_resident());
}

// WHERE DOES THE FILE CACHE COME FROM ON A CHECKPOINT READ?
//
// MEASURED on the M4 Pro: loading the 48 GB text encoder took file-backed
// pages 24 -> 56 GB, which is memory competing with everything the new
// management is trying to hold. WeightSet::read(Copied) is supposed to
// take an uncached pread, so either it does not or something else does
// -- and a standalone F_NOCACHE pread of the same file (with and without
// a live mmap) grows the cache by ZERO, so the answer is in this path
// rather than in the syscall.
//
// Reads a bounded slice of a real checkpoint through the REAL loader and
// reports the delta. Env-gated: it needs a model and moves gigabytes.
TEST(metal_compute_residency, a_copied_read_does_not_grow_the_file_cache)
{
  const char* dir = std::getenv("VPIPE_CACHE_PROBE_MODEL");
  if (dir == nullptr || *dir == '\0') { return; }
  auto session = std::make_shared<Session>();
  MetalCompute* mc = get_mc_(*session);
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(dir, session.get());
  if (!ws) { return; }

  auto file_mb = [] {
    FILE* p = popen("vm_stat | awk '/File-backed/{print $3}'", "r");
    if (p == nullptr) { return (long)-1; }
    long pages = 0;
    if (std::fscanf(p, "%ld", &pages) != 1) { pages = 0; }
    pclose(p);
    return pages / 64;                    // 16 KB pages -> MB
  };

  // Biggest-first, so a bounded budget covers real tensors rather than
  // a long tail of norms.
  std::vector<std::pair<std::size_t, std::string>> by_size;
  for (const std::string& nm : ws->src().tensor_names()) {
    const auto* ti = ws->src().info(nm);
    if (ti != nullptr) { by_size.emplace_back((std::size_t)ti->nbytes, nm); }
  }
  std::sort(by_size.begin(), by_size.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  const long before = file_mb();
  std::size_t read = 0;
  const std::size_t budget = 6ull << 30;
  for (const auto& [nb, nm] : by_size) {
    if (read >= budget) { break; }
    SharedBuffer b = ws->read(nm, mc, genai::WeightSet::Residency::Copied);
    if (b.empty()) { continue; }
    read += nb;
  }
  const long after = file_mb();
  session->log_normal(fmt(
      "copied-read probe: {} MB read, file-backed {} -> {} MB (delta {} MB)",
      read >> 20, before, after, after - before));
  // The read is uncached, so the cache must not grow by anything like
  // what was read. A little movement is other processes.
  EXPECT_TRUE(after - before < (long)((read >> 20) / 4));
}

// THE LIFETIME THE SET IMPOSES, which is the part a caller gets wrong.
//
// MTLResidencySet RETAINS every allocation added to it. So a buffer that
// is added and then dropped is NOT freed -- the set is holding it, and
// on this tree the set lives on MetalCompute (a session service), so it
// outlives every model. This is not a detail: it is the difference
// between a DiT that gives its scratch back when it is destroyed and one
// that does not.
TEST(metal_compute_residency, added_allocation_survives_its_handle) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr || !mc->residency_set_supported()) {
    return;
  }
  constexpr std::size_t kBytes = 256ull << 20;
  const std::size_t base = mc->memory_budget().allocated;
  {
    SharedBuffer b = mc->make_shared_buffer(kBytes);
    if (b.empty()) { return; }
    mc->residency_add(b);
    mc->residency_commit();
  }
  // The handle is gone. If the set did not retain, this would be back at
  // `base`; it is not, and the gap is the whole allocation.
  const std::size_t held = mc->memory_budget().allocated;
  EXPECT_TRUE(held >= base + kBytes / 2);

  // ...and the only thing that gives it back is a matching remove. There
  // is no handle left to pass, which is exactly why a caller has to
  // remove BEFORE it drops -- or keep the means to.
  (void)held;
}

// The same buffer, removed while the handle is still alive: the bytes
// come back. This is the shape every residency_add caller must have.
TEST(metal_compute_residency, remove_before_drop_returns_the_bytes) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr || !mc->residency_set_supported()) {
    return;
  }
  constexpr std::size_t kBytes = 256ull << 20;
  const std::size_t base = mc->memory_budget().allocated;
  {
    SharedBuffer b = mc->make_shared_buffer(kBytes);
    if (b.empty()) { return; }
    mc->residency_add(b);
    mc->residency_commit();
    mc->residency_remove(b);
    mc->residency_commit();
  }
  const std::size_t after = mc->memory_budget().allocated;
  EXPECT_TRUE(after < base + kBytes / 2);
}

// A SUBVIEW ADDS ITS PARENT, and this is the amplification that turns a
// few hundred megabytes of borrowed scratch into gigabytes.
//
// subview() shares the parent's MTL::Buffer -- that is what makes it
// free -- so residency_add(subview) hands the set the WHOLE parent. A
// caller that carves its scratch out of a buffer somebody LENT it, and
// then adds the carvings, has made the lender's allocation immortal.
TEST(metal_compute_residency, a_subview_adds_the_whole_parent) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr || !mc->residency_set_supported()) {
    return;
  }
  constexpr std::size_t kBytes = 256ull << 20;
  const std::size_t base = mc->memory_budget().allocated;
  {
    SharedBuffer parent = mc->make_shared_buffer(kBytes);
    if (parent.empty()) { return; }
    SharedBuffer win = parent.subview(0, 4096);
    EXPECT_TRUE(win.mtl_buffer() == parent.mtl_buffer());
    mc->residency_add(win);      // 4 KB asked for, 256 MB pinned
    mc->residency_commit();
  }
  const std::size_t held = mc->memory_budget().allocated;
  EXPECT_TRUE(held >= base + kBytes / 2);
}

// WHERE A FREED GPU BUFFER'S PAGES GO. A probe, not a check.
//
// After a 1920x1152 Wan decode the process footprint stayed at 11 GB with
// 278 MB of live SharedBuffers, and the graphics ledger said it was GPU
// memory. Nothing in this tree retains a freed buffer -- the residency set
// is empty, command buffers are released in local pools -- so this asks
// the driver directly, one release strategy at a time: device
// currentAllocatedSize, the process graphics ledger, the footprint and the
// live-handle count after each step. VPIPE_FREED_MEMORY_PROBE to run.
TEST(metal_compute_residency, freed_buffer_pages_probe) {
  if (std::getenv("VPIPE_FREED_MEMORY_PROBE") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("llm_elementwise");
  ComputeFunction clamp = lib.function("clamp_f16");
  ASSERT_TRUE(clamp.valid());
  if (!clamp.valid()) { return; }

  constexpr std::size_t kBytes = 1ull << 30;
  auto snap = [&](const char* what) {
    const auto mb = mc->memory_budget();
    std::printf("  %-32s allocated %6zu  graphics %6zu  footprint %6zu  "
                "live %6zu  (MB)\n", what, mb.allocated >> 20,
                mb.self_graphics >> 20, mb.self_footprint >> 20,
                shared_buffer_memory_stats().live_bytes >> 20);
  };
  // Every element read and written by a GPU dispatch, in place.
  auto gpu_touch = [&](SharedBuffer& b) {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder enc = st.begin_compute();
      const int n = (int)(b.byte_size() / 2);
      enc.set_function(clamp);
      enc.set_buffer(0, b); enc.set_buffer(1, b);
      enc.set_constant(2, n);
      enc.set_constant(3, -1.0e4f); enc.set_constant(4, 1.0e4f);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    }
    st.commit().wait();
  };
  auto settle = [](int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  };

  snap("start");
  {
    SharedBuffer b = mc->make_shared_buffer(kBytes);
    ASSERT_TRUE(!b.empty());
    if (b.empty()) { return; }
    std::memset(b.contents(), 0, kBytes);
    snap("A cpu-touched, live");
  }
  snap("A freed");
  {
    SharedBuffer b = mc->make_shared_buffer(kBytes);
    if (b.empty()) { return; }
    gpu_touch(b);
    snap("B gpu-touched, live");
  }
  snap("B freed");
  settle(2000);
  snap("B freed + 2 s");
  {
    SharedBuffer b = mc->make_shared_buffer(kBytes);
    if (b.empty()) { return; }
    gpu_touch(b);
    b.mtl_buffer()->setPurgeableState(MTL::PurgeableStateEmpty);
    snap("C purgeable-empty, live");
  }
  snap("C freed");
  {
    SharedBuffer b = mc->make_shared_buffer(kBytes);
    if (b.empty()) { return; }
    gpu_touch(b);
    snap("D same size again, live");
  }
  snap("D freed");
  {
    SharedBuffer b = mc->make_shared_buffer(2 * kBytes);
    if (b.empty()) { return; }
    gpu_touch(b);
    snap("E twice the size, live");
  }
  snap("E freed");
  settle(2000);
  snap("E freed + 2 s");

  // HOW LONG, and does GPU work flush it. Free a GPU-touched buffer and
  // poll the graphics ledger every 20 ms until it is back within 64 MB of
  // where it started; with `flush`, a tiny unrelated command buffer is
  // submitted and waited on right after the free.
  SharedBuffer tiny = mc->make_shared_buffer(1 << 20);
  if (tiny.empty()) { return; }
  auto time_release = [&](std::size_t bytes, bool flush) {
    const std::size_t base = mc->memory_budget().self_graphics;
    {
      SharedBuffer b = mc->make_shared_buffer(bytes);
      if (b.empty()) { return; }
      gpu_touch(b);
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (flush) { gpu_touch(tiny); }
    const std::size_t held = mc->memory_budget().self_graphics;
    double ms = -1.0;
    for (int i = 0; i < 500; ++i) {
      if (mc->memory_budget().self_graphics <= base + (64ull << 20)) {
        ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0).count();
        break;
      }
      settle(20);
    }
    std::printf("  release %zu MB%s: held %zu MB just after, back in %s\n",
                bytes >> 20, flush ? " + flush" : "        ",
                held > base ? (held - base) >> 20 : 0,
                ms < 0 ? "> 10 s" : fmt("{:.0f} ms", ms)().c_str());
    settle(500);
  };
  for (std::size_t gb : {1, 2, 4}) {
    time_release(gb << 30, false);
    time_release(gb << 30, true);
  }
}
