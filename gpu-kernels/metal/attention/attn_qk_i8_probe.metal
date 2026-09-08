// attn_qk_i8_probe.metal -- can the NAX fragment MMA run int8, and at
// what rate, in the shape a flash attention's QK^T actually has?
//
// THE QUESTION THIS SETTLES. The chunk-depth sweep in
// gemm_i8.k512_chunked says int8 holds 20.8 TFLOP/s against f16's 11.0
// when 128-deep products are issued back to back into a live
// accumulator. But attn_steel_nax does not issue a 128-deep matmul2d
// per key block -- it builds that product out of EIGHT 16-deep fragment
// MMAs (NAXFrag_t::mma, a matmul2d<16,32,16>) chained into the same
// register fragments. Whether int8's advantage survives at 16-deep
// fragments is a different question from whether it survives at 128,
// and it is the one that decides whether an int8 QK is worth wiring
// into the kernel.
//
// So this is the QK inner loop and nothing else: a [16, D] Q fragment
// against a [32, D] K fragment pair, TD = D/16 fragment MMAs deep,
// repeated over many key blocks into accumulators that never leave
// registers. No softmax, no P*V, no store until the end -- what is
// being timed is the fragment chain.
//
//   0:out (device, one value per threadgroup so nothing is dead)
//   1:iters (int, key blocks to walk)
// Dispatch: {32 * SG * tg_count, 1, 1} threads, threadgroup {32*SG,1,1}.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

using namespace metal;

template <typename U>
struct Limits {
  static constexpr constant U max = metal::numeric_limits<U>::max();
  static constexpr constant U min = metal::numeric_limits<U>::min();
  static constexpr constant U finite_max = metal::numeric_limits<U>::max();
  static constexpr constant U finite_min = metal::numeric_limits<U>::min();
};

#if defined(__HAVE_TENSOR__)

#include "mlx/backend/metal/kernels/steel/attn/nax.h"

using namespace mlx::steel;

// head_dim 128 -> TD = 8 fragment MMAs per (query frag, key frag pair),
// which is exactly what attn_steel_nax's QK loop runs.
constant constexpr int kTD = 8;

template <typename AccT, typename OpT>
METAL_FUNC void qk_frag_chain_(device float* out, int iters, uint tgid,
                               uint sgid)
{
  using Afrag = NAXTile<OpT, 1, 1>;
  using Cfrag = NAXTile<AccT, 1, 2>;
  Cfrag C;
  C.clear();

  // Operands live in registers for the whole loop: what is being timed
  // is the matrix pipe, not the loads.
  //
  // ONE SET, AND TWO ELEMENTS REWRITTEN PER ITERATION.
  //
  // Loop-invariant operands accumulating into one destination are
  // something a compiler may fold: with INT8 the accumulation is exact
  // integer arithmetic, so `iters` repetitions of a fixed product is
  // legally a multiply -- and it could not do the same to the f16 arm,
  // whose f32 accumulation is not associative. That asymmetry would
  // manufacture precisely the result this probe is looking for.
  //
  // So one element of each operand carries the iteration counter, which
  // is enough that no product is loop-invariant. NOT several rotating
  // SETS, which was the first attempt: 4 x 8 x 2 fragments spills, and
  // it spills the f16 arm harder because a half fragment is twice a
  // char one -- which manufactures the same wrong answer from the other
  // direction. Both arms here hold identical fragment COUNTS and the
  // loop writes two registers.
  Afrag Q[kTD], K0[kTD], K1[kTD];
  for (short d = 0; d < kTD; ++d) {
    for (short i = 0; i < Afrag::NAXFrag_t::kElemsPerFrag; ++i) {
      const short v = (short)((int(tgid) + int(sgid) + d + i) & 7) - 3;
      Q[d].frag_at(0, 0)[i] = (OpT)v;
      K0[d].frag_at(0, 0)[i] = (OpT)(v + 1);
      K1[d].frag_at(0, 0)[i] = (OpT)(v - 1);
    }
  }

  for (int it = 0; it < iters; ++it) {
    Q[0].frag_at(0, 0)[0] = (OpT)((short)(it & 7) - 3);
    K0[0].frag_at(0, 0)[0] = (OpT)((short)((it >> 3) & 7) - 3);
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < kTD; ++d) {
      Cfrag::NAXFrag_t::template mma<AccT, OpT, OpT>(
          C.frag_at(0, 0), C.frag_at(0, 1), Q[d].frag_at(0, 0),
          metal::false_type{}, K0[d].frag_at(0, 0), K1[d].frag_at(0, 0),
          metal::true_type{});
    }
  }

  // One store, for the whole loop -- the flash kernel's shape.
  float acc = 0.0f;
  for (short i = 0; i < Cfrag::kElemsPerTile; ++i) {
    acc += (float)C.elems()[i];
  }
  if (sgid == 0) { out[tgid] = acc; }
}

kernel void qk_frag_rate_f16(
    device float*       out   [[buffer(0)]],
    constant int&       iters [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
  qk_frag_chain_<float, half>(out, iters, tgid, sgid);
}

kernel void qk_frag_rate_i8(
    device float*       out   [[buffer(0)]],
    constant int&       iters [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
  qk_frag_chain_<int32_t, int8_t>(out, iters, tgid, sgid);
}

#else
kernel void qk_frag_rate_f16(device float* out [[buffer(0)]],
                             uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = 0.0f; } }
kernel void qk_frag_rate_i8(device float* out [[buffer(0)]],
                            uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = 0.0f; } }
#endif
