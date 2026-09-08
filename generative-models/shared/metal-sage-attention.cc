#include "generative-models/shared/metal-sage-attention.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cstdlib>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The steel NAX flash kernel's tiles at head_dim 128. Not this file's
// choice: they come from attn_steel_nax.metal, and the scale arrays are
// indexed by the kernel with them.
constexpr int kNaxBQ = 64;
constexpr int kNaxBK = 32;

// What sage_quant_block's threadgroup staging tile bounds. Asserted
// rather than assumed, because exceeding either is a write past a
// threadgroup array -- which is not a crash, it is a wrong answer.
constexpr int kMaxBlock = 64;
constexpr int kMaxD     = 128;

// Page-aligned carving, for the reason MetalSolAttention's is: a
// private allocation is page-rounded, so each buffer used to have slack
// behind it, and a kernel writing a padded tile past its destination's
// exact size landed there rather than in a neighbour.
constexpr std::size_t kCarveAlign = 16384;

}  // namespace

int MetalSageAttention::nax_query_block() { return kNaxBQ; }
int MetalSageAttention::nax_key_block() { return kNaxBK; }

bool
MetalSageAttention::available(const MetalCompute* mc)
{
  if (mc == nullptr || !mc->valid()) { return false; }
  // MATRIX CORES ARE THE WHOLE CONDITION. The int8 fragment MMA is the
  // instruction being bought; on a box without one there is no int8
  // attention to fall back to -- the ALU steel kernel has no such
  // branch -- so this is a hardware question, not a preference.
  if (!mc->supports_matrix_cores()) { return false; }
  return std::getenv("VPIPE_NO_SAGE_ATTN") == nullptr;
}

std::unique_ptr<MetalSageAttention>
MetalSageAttention::load(MetalCompute* mc, bool bf16, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<MetalSageAttention>{};
  };
  if (mc == nullptr) { return fail("no metal-compute backend"); }
  if (!available(mc)) {
    return fail("sage_attn needs matrix cores (M5 or later)");
  }
  auto s = std::unique_ptr<MetalSageAttention>(new MetalSageAttention());
  s->_mc = mc;
  s->_bf16 = bf16;
  s->_lib = mc->load_library(bf16 ? "sage_quant_bf16" : "sage_quant");
  if (!s->_lib.valid()) { return fail("sage_quant library unavailable"); }
  s->_fn_sum    = s->_lib.function("sage_k_sum_chunk");
  s->_fn_reduce = s->_lib.function("sage_k_mean_reduce");
  s->_fn_quant  = s->_lib.function("sage_quant_block");
  // REFUSED, not warned. An unvalidated ComputeFunction dispatches as a
  // no-op, so a missing quantize kernel here is an attention reading
  // whatever the int8 buffers happened to hold -- which is not a
  // degraded picture, it is noise, and the kernel would report nothing.
  if (!s->_fn_sum.valid() || !s->_fn_reduce.valid() ||
      !s->_fn_quant.valid()) {
    return fail("sage_quant kernels did not validate");
  }
  return s;
}

std::unique_ptr<MetalSageAttention>
MetalSageAttention::load_for_model(MetalCompute* mc, bool bf16,
                                   const sage::Config& cfg, const char* who,
                                   bool* fatal)
{
  if (fatal != nullptr) { *fatal = false; }
  if (!cfg.enabled) { return {}; }
  auto* sess = mc != nullptr ? mc->session() : nullptr;
  if (!available(mc)) {
    if (sess != nullptr) {
      sess->log_normal(fmt(
          "{}: sage_attn asked for but this GPU has no matrix cores -- "
          "attention stays dense", who));
    }
    return {};
  }
  std::string err;
  std::unique_ptr<MetalSageAttention> s = load(mc, bf16, &err);
  if (!s) {
    if (fatal != nullptr) { *fatal = true; }
    if (sess != nullptr) {
      sess->log_normal(fmt("{}: sage_attn requested but {}", who, err));
    }
    return {};
  }
  if (sess != nullptr) {
    sess->log_normal(fmt(
        "{}: SageAttention ON -- INT8 QK^T, layers {}+, key smoothing {}",
        who, cfg.dense_layers, cfg.smooth_k ? "on" : "OFF (A/B only)"));
  }
  return s;
}

MetalSageAttention::~MetalSageAttention()
{
  drop_residency_();
}

void
MetalSageAttention::drop_residency_()
{
  if (!_res_added || _mc == nullptr) { return; }
  _res_added = false;
  for (SharedBuffer* b : scratch_buffers_()) {
    if (!b->empty()) { _mc->residency_remove(*b); }
  }
  _mc->residency_commit();
}

std::vector<SharedBuffer*>
MetalSageAttention::scratch_buffers_()
{
  // Everything ensure_scratch_ hands out, in ONE list, so counting them,
  // announcing them and giving them back cannot disagree about which
  // they are.
  return {&_km, &_kmp, &_q8, &_k8, &_qs, &_ks};
}

std::size_t
MetalSageAttention::scratch_bytes(int heads, int kv_heads, int q_tokens,
                                  int k_tokens, int d, int bq, int bk)
{
  if (heads <= 0 || kv_heads <= 0 || q_tokens <= 0 || k_tokens <= 0 ||
      d <= 0 || bq <= 0 || bk <= 0) {
    return 0;
  }
  const std::size_t H = (std::size_t)heads, D = (std::size_t)d;
  const std::size_t KH = (std::size_t)kv_heads;
  const std::size_t nq = (std::size_t)sage::query_blocks(q_tokens, bq);
  const std::size_t nk = (std::size_t)sage::key_blocks(k_tokens, bk);
  // Summed with the SAME alignment the carve applies, so a caller that
  // lends exactly this much finds every buffer fits. A bare sum is
  // short by up to one alignment per buffer, which strands the tail in
  // private allocations and reads as the lend not working.
  auto pad = [](std::size_t n) {
    return (n + kCarveAlign - 1) & ~(kCarveAlign - 1);
  };
  return pad(KH * D * 4)                                  // km
       + pad(KH * (std::size_t)sage::kMeanChunks * D * 4) // kmp
       + pad(H * (std::size_t)q_tokens * D)               // q8
       + pad(KH * (std::size_t)k_tokens * D)              // k8
       + pad(H * nq * 4)                                  // qs
       + pad(KH * nk * 4);                                // ks
}

void
MetalSageAttention::set_arena(const SharedBuffer& a, const SharedBuffer& b)
{
  const bool same = a.contents() == _arena_a.contents() &&
                    a.byte_size() == _arena_a.byte_size() &&
                    b.contents() == _arena_b.contents() &&
                    b.byte_size() == _arena_b.byte_size();
  if (same) { return; }
  auto hold = [](const SharedBuffer& x) {
    return x.byte_size() > 0 ? x.subview(0, x.byte_size()) : SharedBuffer{};
  };
  _arena_a = hold(a);
  _arena_b = hold(b);
  // BEFORE the windows go, because removing needs the handles, and what
  // the set is holding is the OLD arena.
  drop_residency_();
  for (SharedBuffer* p : scratch_buffers_()) { *p = SharedBuffer{}; }
  _heads = _qt = _kt = _d = 0;
  _ready = false;
  _arena_used[0] = 0;
  _arena_used[1] = 0;
}

std::size_t
MetalSageAttention::resident_bytes() const
{
  std::size_t n = 0;
  for (const SharedBuffer* b : {&_km, &_kmp, &_q8, &_k8, &_qs, &_ks}) {
    n += b->byte_size();
  }
  return n;
}

bool
MetalSageAttention::ensure_scratch_(int heads, int kv_heads, int q_tokens,
                                    int k_tokens, int d, int bq, int bk,
                                    std::string* err)
{
  if (_heads == heads && _kvh == kv_heads && _qt == q_tokens &&
      _kt == k_tokens && _d == d && _bq == bq && _bk == bk &&
      !_q8.empty()) {
    return true;
  }
  // A REBUILD: the previous geometry's buffers are about to be
  // overwritten, and the assignments below are the last references to
  // them.
  drop_residency_();
  _ready = false;

  const std::size_t H = (std::size_t)heads, D = (std::size_t)d;
  const std::size_t KH = (std::size_t)kv_heads;
  const int nq = sage::query_blocks(q_tokens, bq);
  const int nk = sage::key_blocks(k_tokens, bk);

  _arena_used[0] = 0;
  _arena_used[1] = 0;
  auto mk = [&](std::size_t n) -> SharedBuffer {
    SharedBuffer* reg[2] = {&_arena_a, &_arena_b};
    for (int i = 0; i < 2; ++i) {
      if (reg[i]->empty()) { continue; }
      const std::size_t off =
          (_arena_used[i] + kCarveAlign - 1) & ~(kCarveAlign - 1);
      if (off < reg[i]->byte_size() && n <= reg[i]->byte_size() - off) {
        _arena_used[i] = off + n;
        return reg[i]->subview(off, n);
      }
    }
    return _mc->make_shared_buffer(n);
  };

  _km  = mk(KH * D * 4);
  _kmp = mk(KH * (std::size_t)sage::kMeanChunks * D * 4);
  _q8  = mk(H * (std::size_t)q_tokens * D);
  _k8  = mk(KH * (std::size_t)k_tokens * D);
  _qs  = mk(H * (std::size_t)nq * 4);
  _ks  = mk(KH * (std::size_t)nk * 4);
  if (_km.empty() || _kmp.empty() || _q8.empty() || _k8.empty() ||
      _qs.empty() || _ks.empty()) {
    for (SharedBuffer* p : scratch_buffers_()) { *p = SharedBuffer{}; }
    if (err != nullptr) { *err = "sage_attn scratch allocation failed"; }
    return false;
  }

  for (const SharedBuffer* b : scratch_buffers_()) {
    if (!b->empty()) { _mc->residency_add(*b); }
  }
  _res_added = true;
  _mc->residency_commit();

  _heads = heads; _kvh = kv_heads; _qt = q_tokens; _kt = k_tokens; _d = d;
  _bq = bq; _bk = bk;
  return true;
}

bool
MetalSageAttention::prepare(ComputeEncoder& e, const Operand& q,
                            const Operand& k, int heads, int kv_heads,
                            int q_tokens, int k_tokens, int d, int bq, int bk,
                            const sage::Config& cfg, std::string* err)
{
  _ready = false;
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (_mc == nullptr) { return fail("sage_attn not loaded"); }
  if (heads <= 0 || kv_heads <= 0 || q_tokens <= 0 || k_tokens <= 0 ||
      d <= 0) {
    return fail("sage_attn: degenerate geometry");
  }
  // The kernel derives the KV head from gqa_factor = heads / kv_heads,
  // so a ratio that is not whole would put it on a head this never
  // quantized. Refused rather than rounded.
  if (kv_heads > heads || (heads % kv_heads) != 0) {
    return fail(fmt("sage_attn: {} query heads is not a whole multiple of "
                    "{} kv heads", heads, kv_heads)());
  }
  // REFUSED rather than clamped. Both bounds are threadgroup array
  // sizes inside sage_quant_block, so exceeding one writes past a
  // threadgroup allocation -- which produces a plausible wrong answer
  // rather than a fault, and would be found downstream as a bad frame.
  if (d > kMaxD) {
    return fail(fmt("sage_attn: head_dim {} exceeds {}", d, kMaxD)());
  }
  if (bq > kMaxBlock || bk > kMaxBlock || bq <= 0 || bk <= 0) {
    return fail(fmt("sage_attn: block ({}, {}) outside 1..{}", bq, bk,
                    kMaxBlock)());
  }
  if (q.buf == nullptr || k.buf == nullptr || q.buf->empty() ||
      k.buf->empty()) {
    return fail("sage_attn: no q/k");
  }
  if (q.row <= 0 || k.row <= 0 || q.head <= 0 || k.head <= 0) {
    return fail("sage_attn: q/k strides must be positive");
  }
  if (!ensure_scratch_(heads, kv_heads, q_tokens, k_tokens, d, bq, bk,
                       err)) {
    return false;
  }

  const unsigned H = (unsigned)heads, D = (unsigned)d;
  const unsigned KH = (unsigned)kv_heads;
  const int chunks = sage::kMeanChunks;
  const int nq = sage::query_blocks(q_tokens, bq);
  const int nk = sage::key_blocks(k_tokens, bk);

  // 1. The key mean, in two stages. See sage::kMeanChunks for why the
  //    obvious one-pass shape is not used.
  if (cfg.smooth_k) {
    e.set_function(_fn_sum);
    e.set_buffer(0, *k.buf);
    e.set_buffer(1, _kmp);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, k.row);
    e.set_constant(5, (unsigned)k.off);
    e.set_constant(6, chunks);
    e.set_constant(7, k.head);
    e.dispatch({D, KH, (unsigned)chunks}, {D, 1, 1});

    e.set_function(_fn_reduce);
    e.set_buffer(0, _kmp);
    e.set_buffer(1, _km);
    e.set_constant(2, k_tokens);
    e.set_constant(3, d);
    e.set_constant(4, chunks);
    e.dispatch({D, KH, 1}, {D, 1, 1});
  }

  // 2. Q, unsmoothed, at the QUERY block. See sage-attention.h for why
  //    only the key side carries a mean.
  e.set_function(_fn_quant);
  e.set_buffer(0, *q.buf);
  e.set_buffer(1, _km);
  e.set_buffer(2, _q8);
  e.set_buffer(3, _qs);
  e.set_constant(4, q_tokens);
  e.set_constant(5, d);
  e.set_constant(6, q.row);
  e.set_constant(7, (unsigned)q.off);
  e.set_constant(8, bq);
  e.set_constant(9, 0);
  e.set_constant(10, q.head);
  e.dispatch({256u, H, (unsigned)nq}, {256, 1, 1});

  // 3. K, smoothed, at the KEY block.
  e.set_function(_fn_quant);
  e.set_buffer(0, *k.buf);
  e.set_buffer(1, _km);
  e.set_buffer(2, _k8);
  e.set_buffer(3, _ks);
  e.set_constant(4, k_tokens);
  e.set_constant(5, d);
  e.set_constant(6, k.row);
  e.set_constant(7, (unsigned)k.off);
  e.set_constant(8, bk);
  e.set_constant(9, cfg.smooth_k ? 1 : 0);
  e.set_constant(10, k.head);
  e.dispatch({256u, KH, (unsigned)nk}, {256, 1, 1});

  _ready = true;
  return true;
}

void
MetalSageAttention::bind(ComputeEncoder& e) const
{
  if (!_ready) { return; }
  e.set_buffer(15, _q8);
  e.set_buffer(16, _k8);
  e.set_buffer(17, _qs);
  e.set_buffer(18, _ks);
}

}  // namespace genai
}  // namespace vpipe
