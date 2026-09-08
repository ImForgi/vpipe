// sol_proxy_mma.metal -- Sol-Attn's ROUTING PROXY on the matrix cores
// (M5+), via Metal 4 MetalPerformancePrimitives matmul2d.
//
// The proxy is one number per (query block, key block):
//
//     proxy[h][qb][n] = <qc[h][qb], kc[h][n]>
//
// which is a [NQ, D] x [D, NK] product per head -- an ordinary GEMM
// that the routing kernel was computing as NQ * NK separate simdgroup
// dot products, each one re-reading a key centroid no other query block
// in flight could share. At 56 heads and 313 blocks either way that is
// 1.4 GB of centroid traffic for 100 MFLOP of arithmetic; a tile reads
// each centroid once and lands the arithmetic on the matrix units on
// the way.
//
// WHY THIS PASS AND NOT THE OTHERS. Once both halves of the attention
// are on the flash kernel, what is left of Sol's own cost is the
// summaries, this, the CSR emit and the merge -- and the summaries and
// the merge each read their inputs exactly once, so they are already at
// the bandwidth roofline and have nothing to gain. This is the only
// remaining pass whose cost is arithmetic-shaped.
//
// FP32 OUT, from an f32 accumulation. The scores are compared against a
// threshold in units of their own standard deviation, so a decision
// near the threshold is decided by the last bits -- and rounding them
// to the operand dtype would move blocks in and out of the exact set
// for no saving worth having on a 22 MB destination.
//
//   0:qc (VPIPE_ELT [H,NQ,D]) 1:kc (VPIPE_ELT [H,NK,D])
//   2:proxy (float [H,NQ,NK]) 3:NQ 4:NK 5:D
// Dispatch: grid {SG*32 * n_tiles, m_tiles, H} THREADS, threadgroup
// {SG*32, 1, 1} -- so tgid is (n tile, m tile, head).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// 64x64 over 4 simdgroups. Deliberately not the 128-wide tile the
// projection GEMMs use: NQ and NK are the BLOCK counts, a few hundred at
// video geometry and a few dozen in the tests, so a 128 tile would spend
// most of a dispatch on rows that do not exist.
#ifndef SOL_PX_BM
#define SOL_PX_BM 64
#endif
#ifndef SOL_PX_BN
#define SOL_PX_BN 64
#endif
#ifndef SOL_PX_SG
#define SOL_PX_SG 4
#endif

#if defined(__HAVE_TENSOR__)

kernel void sol_proxy_mma(
    const device VPIPE_ELT* qc    [[buffer(0)]],
    const device VPIPE_ELT* kc    [[buffer(1)]],
    device float*           proxy [[buffer(2)]],
    constant int&           NQ    [[buffer(3)]],
    constant int&           NK    [[buffer(4)]],
    constant int&           D     [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const int h  = (int)tgid.z;
  const int m0 = (int)tgid.y * SOL_PX_BM;
  const int n0 = (int)tgid.x * SOL_PX_BN;
  // A tile that STARTS past the extent is not a ragged tail, it is a
  // slice from an out-of-contract origin -- the matmul2d tensors clamp
  // the first and not the second, and the store then lands on whatever
  // follows. The bounds are uniform across the threadgroup, so this
  // costs no divergence.
  if (m0 >= NQ || n0 >= NK) { return; }

  using TE = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  using TF = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
  // (contiguous, outer): both centroid arrays are [blocks, D] per head,
  // so the CONTRACTION is the contiguous axis for each -- the left
  // operand is already [M, K] and the right one is [N, K], which is the
  // transpose_right form the dense GEMM uses for the same reason.
  const int64_t qbase = (int64_t)h * (int64_t)NQ * (int64_t)D;
  const int64_t kbase = (int64_t)h * (int64_t)NK * (int64_t)D;
  const int64_t pbase = (int64_t)h * (int64_t)NQ * (int64_t)NK;
  TE tQ(const_cast<device VPIPE_ELT*>(qc) + qbase,
        dextents<int32_t, 2>(D, NQ));
  TE tK(const_cast<device VPIPE_ELT*>(kc) + kbase,
        dextents<int32_t, 2>(D, NK));
  TF tP(proxy + pbase, dextents<int32_t, 2>(NK, NQ));

  constexpr auto desc = matmul2d_descriptor(
      SOL_PX_BM, SOL_PX_BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false);
  matmul2d<desc, execution_simdgroups<SOL_PX_SG>> op;

  auto mQ = tQ.slice(0, m0);
  auto mK = tK.slice(0, n0);
  auto cT =
      op.template get_destination_cooperative_tensor<decltype(mQ),
                                                     decltype(mK), float>();
  op.run(mQ, mK, cT);
  auto mP = tP.slice(n0, m0);
  cT.store(mP);
}

#else
// No tensor ops for this target: a stub so the metallib still builds.
// The loader never binds this on a pre-M5 GPU -- MetalSolAttention asks
// supports_matrix_cores() first and keeps the fused routing kernel
// otherwise.
kernel void sol_proxy_mma(device float* proxy [[buffer(2)]],
                          uint t [[thread_position_in_grid]])
{
  if (t == 0) { proxy[0] = 0.0f; }
}
#endif
