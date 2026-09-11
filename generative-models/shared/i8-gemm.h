#ifndef VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H
#define VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-autotune.h"

#include <cstddef>
#include <cstdlib>
#include <vector>

namespace vpipe {
namespace genai {

// Dynamic-int8 accelerated GEMM ("accelerated mode" for large matmuls --
// DiT blocks, LLM prefill): the f16/bf16 activation is quantized ON THE FLY
// to i8 with per-(row, 512-group) scales (quant_f16_i8_row_g512), the
// f16/bf16 weight (dense, or a dequant scratch) is quantized per-(out-
// channel, 512-group) into a reusable i8 scratch, and the product runs on
// the matrix units' int8 pipe -- ~2x the matmul2d rate at qualifying shapes
// -- accumulating each 512-deep group in f32 with its own scales
// (gemm_i8i8_sc_f16_n64_g512) and storing the element type back.
//
// The activation/weight/scale/output element type is chosen by `bf16`: the
// f16 caller loads the `dense_gemm_mma`/`affine_dequant` metallibs, the bf16
// caller (e.g. the FLUX.2 klein DiT, whose residual stream overflows f16) the
// `_bf16` twins -- the same kernels compiled with VPIPE_ELT=bfloat. The int8
// scratch is format-independent and the scale scratch is 2 bytes either way,
// so only the library selection changes. Reading a bf16 weight/activation
// through the f16 kernels (or vice versa) reinterprets the bits -> garbage;
// the caller MUST match `bf16` to its buffers' element type.
//
// LOSSY: int8 quantization, rel-L2 ~1e-2 per GEMM. Strictly OPT-IN and
// never part of a token-exact path. The per-call weight re-quant costs
// one extra pass over the weight (~1-2% of a qualifying GEMM), so the
// mode has NO persistent memory cost and composes with every checkpoint
// format that already produces an f16/bf16 weight for the matmul.
//
// The stage/model switch arrives via `want`; env VPIPE_I8_GEMM=0|1
// overrides either way (A/B), VPIPE_I8_GEMM_MIN_M tunes the M gate.
class I8GemmContext {
 public:
  I8GemmContext(metal_compute::MetalCompute* mc, bool want, bool bf16 = false);

  bool enabled() const { return _on; }

  // Shape gate: the win regime is big-M compute-bound GEMMs (measured
  // crossover ~1k rows on M5). K must split into whole 512-groups, OR the
  // padding quantizer must be available to make it so -- see kpad_() -- and
  // the padding must be CHEAP, which is the last clause.
  //
  // Padding is exact but not free: it is (KP-K)/K extra int8 MACs on every
  // call, and the int8 rate has to beat bf16 by more than that for the mode
  // to still pay. Capped at _max_pad_pct (10 by default).
  //
  // Where the cap bites: the pad is at most 511, so any K >= 10*512 = 5120
  // passes whatever its remainder (511/5120 = 9.98%). Below that it depends
  // entirely on K % 512 -- K=2816 pads 256 (9.1%, taken), K=1600 pads 448
  // (28%, refused). So the rule reads as "shallow contractions must be
  // nearly aligned already; deep ones need not care".
  bool accepts(int M, int N, int K) const
  {
    if (!_on || M < _min_m || K < 1024 || N < 16) { return false; }
    const int KP = kpad_(K);
    if (KP == K) { return true; }
    if (!_fn_quant_pad.valid()) { return false; }
    return (KP - K) * 100 <= K * _max_pad_pct;
  }

  // Encode act-quant + weight-quant + the i8 GEMM (element type = the ctor's
  // `bf16`; x/w/y must all be that type):
  //   y[M,N] (elem offset ye) = x[M,K] (elem offset xe) @ w[N,K]^T (dense)
  // Returns false -- with nothing encoded -- when the shape does not
  // qualify or a scratch allocation fails (caller keeps its dense path).
  bool gemm(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t xe,
            const metal_compute::SharedBuffer& w,
            const metal_compute::SharedBuffer& y, std::size_t ye,
            int M, int N, int K)
  {
    return gemm(enc, x, xe, w, metal_compute::SharedBuffer{}, y, ye, M, N,
                K);
  }

  // ...and the twin that adds a per-column bias. A model whose every
  // projection carries one (VOSR) could otherwise reach none of this:
  // the int8 path computes x @ w^T and had nowhere to put the b. An
  // empty `b` is the overload above, and is byte-identical to what
  // shipped -- the term is not computed, not added as zero.
  bool gemm(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t xe,
            const metal_compute::SharedBuffer& w,
            const metal_compute::SharedBuffer& b,
            const metal_compute::SharedBuffer& y, std::size_t ye,
            int M, int N, int K);

  // Drop the grow-only act/weight scratches (they re-grow on demand at the
  // next gemm()). Call between generations on a memory-bounded box so the
  // idle scratch doesn't crowd out a large downstream allocation.
  void release_scratch();

  // ---- split-K over the 512-groups ---------------------------------
  //
  // int8 and split-K answer DIFFERENT deficits, and a deep-K GEMM has
  // both. int8 doubles the matrix pipe's rate; a split fixes OCCUPANCY --
  // a single-op reduction over a deep K leaves only (M/64)*(N/64)
  // threadgroups each walking one long serial contraction, and the units
  // stall with nothing else in flight. The int8 pipe is not exempt: the
  // prototype measured the ff-down at K=12288 running 18.5 TOPS against
  // 22.5-23.6 at K=4096, the same cliff the f16 path has.
  //
  // It is nearly free to add here because the g512 kernel ALREADY walks K
  // as independent 512-groups, each scaled by its own pair before it joins
  // a float accumulator -- so a plane is a contiguous range of that loop
  // and the fold is the rest of the sum. Splitting by GROUP INDEX (not by
  // K) keeps every plane boundary on a 512 multiple for free.
  //
  // MEASURED on an M5 at H3's fc2 shape, N=7168 K=14336 (gemm_i8.splitk),
  // against this class's own single-op arm:
  //
  //     M     single   best split      gain
  //    256    9.15     15.22 (S=7)    1.66x
  //   1024   12.86     16.02 (S=2)    1.25x
  //   2560   12.99     15.96 (S=2)    1.23x
  //
  // NUMERICS: a reassociation and a weaker one than the dense f32-plane
  // split-K's, because the per-group summand is bit-IDENTICAL whichever
  // plane owns it -- only the order of the G float adds moves. Measured
  // 0.02-0.03% of outputs differing by at most ~1 f16 ulp, against the
  // dense path's own 0.48-0.60%.

  // Pick the split for this shape by MEASURING, and cache it per
  // (N, K, row bucket).
  //
  // CALL WITH NO ENCODER OPEN: it runs its own command streams. `run(enc)`
  // must encode the CALLER'S normal GEMM -- the same call the forward
  // makes -- so what is compared is by construction what will run; the
  // flags below steer it. Same contract as MmaSplitK::tune, deliberately:
  // a caller that already tunes one can tune the other beside it.
  //
  // NOTE the S that wins is NOT the one the dense chooser's kc heuristic
  // would pick, and it moves with M: at K=14336 that rule says S=7, where
  // S=2 wins once M >= 1024 and S=14 LOSES (0.95x). The base grid is
  // (N/64)*(M/64) threadgroups, so a wide N and a large M already fill the
  // machine; and the fold reads S*M*N*4 bytes, 411 MB at S=14/M=1024. So
  // this is measured rather than derived.
  template <class Run>
  int tune(metal_compute::MetalCompute* mc, int K, int N, int M, Run&& run)
  {
    if (!_on || mc == nullptr || !split_available()) { return 0; }
    const int key = m_key_(M);
    for (const Tuned& t : _tuned) {
      if (t.N == N && t.K == K && t.M == key) { return t.splits; }
    }
    std::vector<int> cands{0};
    for (int S : split_candidates(M, N, K)) { cands.push_back(S); }
    if (cands.size() < 2) {
      _tuned.push_back(Tuned{N, K, key, 0});
      return 0;
    }
    const int w = autotune_vote((int)cands.size(), /*rounds=*/3,
        /*reps_for_us=*/1,
        [&](int i) {
          bypass_split = (cands[(std::size_t)i] == 0);
          force_splits = bypass_split ? 0 : cands[(std::size_t)i];
          const double t = autotune_time(mc, 1, run);
          bypass_split = false;
          force_splits = 0;
          return t;
        });
    const int splits = cands[(std::size_t)w];
    _tuned.push_back(Tuned{N, K, key, splits});
    return splits;
  }

  // DEFERRED TUNING, for a caller that has no convenient closure.
  //
  // tune() above wants the caller's own GEMM as a thunk, which suits a
  // family that already tunes something (H3 hands it the same
  // dispatch_row_bands_ its forward runs). The others do not have one to
  // hand: their weights may still be streaming when the shape is first
  // known, and the dense operand a quantized checkpoint contracts is a
  // dequant scratch that only exists INSIDE an encoder.
  //
  // So gemm() records a shape it has not tuned, and this replays it later
  // -- at any point with no encoder open -- using the quantized scratches
  // that call already left behind. Nothing is allocated but a destination,
  // nothing is read from the caller, and what is measured is exactly the
  // part that differs: the candidates share the two quantize passes, so
  // excluding them removes variance from the vote rather than hiding
  // anything.
  //
  // The scratch contents are the LAST shape's by then, which does not
  // matter -- an int8 GEMM's time does not depend on its data -- and the
  // buffers are grow-only, so a recorded shape is one they were already
  // sized for.
  //
  // Cheap and idempotent: returns immediately when nothing is pending.
  // Call it between forwards, or at the end of one.
  void tune_pending(metal_compute::MetalCompute* mc);

  // Whether the split kernels loaded at all (the fold lives in a different
  // metallib, so it can be absent independently).
  bool split_available() const
  {
    return _fn_gemm_sk.valid() && _fn_fold.valid() && _max_splits > 0;
  }

  // Every split this shape admits: G divisible for balanced planes, and a
  // row block that still fills the 64-row tile.
  std::vector<int> split_candidates(int M, int N, int K) const;

  // Set by tune() around each candidate; see MmaSplitK for the pattern.
  bool bypass_split = false;
  int  force_splits = 0;

  // The width this shape would use right now. Test-only: a split is not
  // observable from the OUTPUT (it is a reassociation), so a test that
  // wants to know the tuner's answer took effect has to ask.
  int plan_for_test(int M, int N, int K) const { return plan_(M, N, K); }

 private:
  metal_compute::MetalCompute* _mc = nullptr;
  bool _on = false;
  int _min_m = 1024;
  int _max_pad_pct = 10;   // VPIPE_I8_MAX_PAD_PCT
  metal_compute::ComputeLibrary _lib_q, _lib_g;
  metal_compute::ComputeFunction _fn_quant, _fn_quant_pad, _fn_gemm;
  metal_compute::ComputeLibrary   _lib_elt;
  metal_compute::ComputeFunction  _fn_gemm_sk, _fn_fold;
  metal_compute::ComputeFunction  _fn_fold_bias;
  // Max f32 plane scratch, bytes (VPIPE_I8_SPLITK_MAX_MB, default 512).
  // A constraint on the SEARCH, not a veto after it: past the budget the
  // next-narrower split still runs, and past every one the single-op GEMM
  // does -- slower, allocation-free.
  std::size_t _plane_budget = 0;
  int _max_splits = 16;              // VPIPE_I8_SPLITK_MAX_S; 0 disables
  metal_compute::SharedBuffer _planes;

  struct Tuned { int N, K, M, splits; };
  std::vector<Tuned> _tuned;
  // Shapes gemm() has run but nobody has tuned. Bounded, because a caller
  // sweeping sequence lengths must not grow it without limit.
  std::vector<Tuned> _pending;
  metal_compute::SharedBuffer _tune_y;
  // Rows the deferred tuner MEASURES at, whatever the shape's real M. A
  // destination is M*N*2 bytes and a video forward's M is six figures --
  // 1.5 GB at H3's fc2 -- where the answer is already flat above one row
  // block (S=2 won at both 1024 and 2560 rows there). So it measures a
  // bounded stand-in and keys the answer to the real row bucket.
  static constexpr int kTuneRows = 2048;
  // Encode ONE candidate's GEMM (+ fold) over the existing scratches.
  void encode_tuned_(metal_compute::ComputeEncoder& enc, int M, int N,
                     int KP, int S);
  // Row buckets, so a forward whose row count wobbles does not re-tune --
  // and so the answer is attached to the count it was measured at.
  static int m_key_(int M)
  {
    if (M <= 512) { return 512; }
    if (M <= 1024) { return 1024; }
    if (M <= 2048) { return 2048; }
    return 4096;
  }
  // Rows whose S planes fit the budget, floored at the 64-row tile.
  int rows_per_block_(int S, int N) const
  {
    const std::size_t per_row = (std::size_t)S * (std::size_t)N * 4u;
    if (per_row == 0) { return 0; }
    const std::size_t r = _plane_budget / per_row;
    return (int)((r / 64u) * 64u);
  }
  // The split this call will use: the tuner's answer when there is one,
  // else the measured-safe default. See the note on `tune`.
  int plan_(int M, int N, int K) const;

  // K rounded up to a whole number of 512-groups. The int8 GEMM contracts
  // in 512-wide chunks, so a K that is not a multiple of 512 has no chunk
  // to sit in; the quantizer zero-fills up to here instead, which is exact
  // (zeros add nothing to the dot product and cannot move an absmax scale)
  // and costs (kpad-K)/K extra int8 MACs -- 4.8% at H3's hidden 5376.
  static int kpad_(int K) { return ((K + 511) / 512) * 512; }
  // Grow-only scratches: xq[M,K] i8 + as[M,G] scales (activations),
  // wq[N,K] i8 + ws[N,G] scales (per-call weight re-quant). Scales are the
  // element type (2 bytes either way), so the byte sizes are format-agnostic.
  metal_compute::SharedBuffer _xq, _as, _wq, _ws;
};

}  // namespace genai
}  // namespace vpipe

#endif
