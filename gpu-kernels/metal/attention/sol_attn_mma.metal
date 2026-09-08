// sol_attn_mma.metal -- Sol-Attn's routing and its APPROXIMATE half, on
// simdgroup matrices. The exact half is not here: it is the tree's own
// steel flash kernel, run block-sparse over the list this file emits.
//
// The split is the whole point. Sol's cost is dominated by the blocks it
// KEEPS, and those are ordinary attention over a subset -- which
// `attn_steel`'s has_spans path already does at full MMA throughput. So
// this file does the two things steel cannot: decide the subset, and
// summarise everything outside it.
//
//   sol_summaries_mma   per-block centroids and value MEANS
//   sol_kc_stats_mma    the key centroids' spread, for the threshold
//   sol_route_mma       proxy scores -> threshold -> CSR block list
//   sol_approx_mma      a flash attention over the SUMMARY sequence
//   sol_merge_mma       the two partial softmaxes into one output
//
// TWO CONVENTIONS THAT MAKE THE APPROXIMATE HALF A PLAIN FLASH LOOP, and
// both are worth stating because the published form has neither:
//
//   * the value summary is the block MEAN, not its sum, and the score
//     carries +log2(64) instead. exp2(s + 6) * (sum_j v_j / 64) is the
//     same numerator as exp2(s) * sum_j v_j, and the denominator term
//     falls out as a plain row sum rather than a length-weighted one --
//     so no per-column weight ever enters the MMA fragments.
//   * THE TAIL BLOCK IS ALWAYS EXACT. It is the one block whose length
//     is not 64, and forcing it exact is what makes that +log2(64)
//     uniform. It costs one block of ~314 and it is the block whose
//     centroid summarises fewest keys, so it is the least worth
//     approximating anyway.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define SOL_LOG2E 1.44269504088896340736f
#define SOL_D 128            // Sol-Attn is specified at head_dim 128
#define SOL_FRAG 8

// Per-block summaries, at a routing block size the host chooses.
//
// `qc`/`kc` are centroids and `vc` is the value MEAN (see the header).
// The element type is the model's, so the MMA below consumes them
// directly; the threshold statistics that need fp32 are separate and
// small.
//
//   0:q 1:k 2:v (VPIPE_ELT [H,T,D])  3:qc 4:kc 5:vc (VPIPE_ELT [H,N,D])
//   6:T 7:D 8:N 9:BLK (int)
// grid (D, H, N) THREADS; threadgroup (D,1,1).
kernel void sol_summaries_mma(
    const device VPIPE_ELT* q  [[buffer(0)]],
    const device VPIPE_ELT* k  [[buffer(1)]],
    const device VPIPE_ELT* v  [[buffer(2)]],
    device VPIPE_ELT*       qc [[buffer(3)]],
    device VPIPE_ELT*       kc [[buffer(4)]],
    device VPIPE_ELT*       vc [[buffer(5)]],
    constant int&           T  [[buffer(6)]],
    constant int&           D  [[buffer(7)]],
    constant int&           N  [[buffer(8)]],
    constant int&           BLK [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  d   [[thread_index_in_threadgroup]])
{
  const int h  = (int)tid.y;
  const int b  = (int)tid.z;
  const int t0 = b * BLK;
  if ((int)d >= D || t0 >= T) { return; }
  const int len = min(BLK, T - t0);
  const device VPIPE_ELT* qh = q + ((uint)h * T + t0) * D;
  const device VPIPE_ELT* kh = k + ((uint)h * T + t0) * D;
  const device VPIPE_ELT* vh = v + ((uint)h * T + t0) * D;
  float qs = 0.0f, ks = 0.0f, vs = 0.0f;
  for (int t = 0; t < len; ++t) {
    qs += float(qh[(uint)t * D + d]);
    ks += float(kh[(uint)t * D + d]);
    vs += float(vh[(uint)t * D + d]);
  }
  const uint o = ((uint)h * N + b) * D + d;
  const float inv = 1.0f / float(len);
  qc[o] = VPIPE_ELT(qs * inv);
  kc[o] = VPIPE_ELT(ks * inv);
  vc[o] = VPIPE_ELT(vs * inv);
}

//   0:kc (VPIPE_ELT [H,N,D]) 1:mean 2:var (float [H,D]) 3:D 4:N (int)
// grid (D, H, 1) THREADS; threadgroup (D,1,1).
kernel void sol_kc_stats_mma(
    const device VPIPE_ELT* kc   [[buffer(0)]],
    device float*           mean [[buffer(1)]],
    device float*           var  [[buffer(2)]],
    constant int&           D    [[buffer(3)]],
    constant int&           N    [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  d   [[thread_index_in_threadgroup]])
{
  const int h = (int)tid.y;
  if ((int)d >= D || N <= 0) { return; }
  const device VPIPE_ELT* kh = kc + (uint)h * N * D;
  float s = 0.0f, s2 = 0.0f;
  for (int b = 0; b < N; ++b) {
    const float x = float(kh[(uint)b * D + d]);
    s += x; s2 += x * x;
  }
  const float inv = 1.0f / float(N);
  const float m = s * inv;
  mean[(uint)h * D + d] = m;
  var[(uint)h * D + d] = max(s2 * inv - m * m, 0.0f);
}

// Routing, in three passes, because the block list is a CSR and a CSR
// needs a prefix sum.
//
//   sol_route_mma  per (head, query block): the threshold, the decision,
//                  the flag row, and how many blocks it kept
//   sol_scan_mma   the kept counts into qb_off, per head
//   sol_emit_mma   the flag rows into qb_blocks at those offsets
//
// A one-pass version would have to write qb_off[i] and qb_off[i+1] from
// query block i, and qb_off[i+1] also belongs to block i+1 -- so the two
// would race, or the offsets would have to be a fixed stride, which is
// not a CSR at all. The scan costs one threadgroup over H * NQ counts.
//
// THE LIST IS IN STEEL'S KEY BLOCKS, not Sol's. Sol routes BLK keys at a
// time and steel loads BK, so one routing block expands to BLK/BK
// entries. BLK is a multiple of BK by construction, so nothing is
// rounded and no edge mask is needed -- which is why span_params can be
// handed tokens_per_frame = 0 and steel's edge predicate switches itself
// off.
//
// THE QUERY BLOCK IS STEEL'S BQ, not BLK. Steel indexes qb_off by its
// own query block, so the routing decision has to be taken at that
// granularity or two steel blocks would share one CSR entry. Query and
// key block sizes are independent here; only the KEY size is what the
// method calls its block.
//
//   0:qc (VPIPE_ELT [H,NQ,D]) 1:kc (VPIPE_ELT [H,NK,D])
//   2:mean 3:var (float [H,D])  4:flags (uchar [H,NQ,NK])
//   5:kept (uint [H,NQ])  6:counts (atomic_uint[2])
//   7:scale 8:tau (float) 9:D 10:NK 11:NQ 12:BLK 13:BQ
//   14:radius 15:sink_lo 16:sink_hi (int; radius/sink in KEY blocks)
// grid (32, H, NQ) THREADS; threadgroup (32,1,1).
kernel void sol_route_mma(
    const device VPIPE_ELT* qc      [[buffer(0)]],
    const device VPIPE_ELT* kc      [[buffer(1)]],
    const device float*     mean    [[buffer(2)]],
    const device float*     var     [[buffer(3)]],
    device uchar*           flags   [[buffer(4)]],
    device uint*            kept    [[buffer(5)]],
    device atomic_uint*     counts  [[buffer(6)]],
    constant float&         scale   [[buffer(7)]],
    constant float&         tau     [[buffer(8)]],
    constant int&           D       [[buffer(9)]],
    constant int&           NK      [[buffer(10)]],
    constant int&           NQ      [[buffer(11)]],
    constant int&           BLK     [[buffer(12)]],
    constant int&           BQ      [[buffer(13)]],
    constant int&           radius  [[buffer(14)]],
    constant int&           sink_lo [[buffer(15)]],
    constant int&           sink_hi [[buffer(16)]],
    constant int&           per     [[buffer(17)]],
    constant int&           NKS     [[buffer(18)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int h  = (int)tid.y;
  const int qb = (int)tid.z;
  const float ls = scale * SOL_LOG2E;
  const device VPIPE_ELT* qbar = qc + ((uint)h * NQ + qb) * D;
  const device float* mh = mean + (uint)h * D;
  const device float* vh = var  + (uint)h * D;

  float rm = 0.0f, rv = 0.0f;
  for (int i = (int)lane; i < D; i += 32) {
    const float qi = float(qbar[i]);
    rm += qi * mh[i];
    rv += qi * qi * vh[i];
  }
  rm = simd_sum(rm) * ls;
  rv = max(simd_sum(rv), 0.0f) * ls * ls;
  const float thr = rm + tau * sqrt(rv + 1.0e-6f);

  // The local band is in KEY blocks around the key block this QUERY
  // block sits in, which is where the two granularities meet.
  const int qk = (qb * BQ) / BLK;
  device uchar* fl = flags + ((uint)h * NQ + qb) * NK;
  uint n_keep = 0;
  for (int n = 0; n < NK; ++n) {
    const device VPIPE_ELT* kr = kc + ((uint)h * NK + n) * D;
    float dot = 0.0f;
    for (int i = (int)lane; i < D; i += 32) {
      dot += float(qbar[i]) * float(kr[i]);
    }
    dot = simd_sum(dot) * ls;
    const bool tail  = (n == NK - 1);   // see the file header
    const bool local = abs(qk - n) <= radius;
    const bool sink  = (n >= sink_lo && n < sink_hi);
    const bool keep  = (dot > thr) || local || sink || tail;
    if (lane == 0) { fl[n] = keep ? (uchar)1 : (uchar)0; }
    // COUNT STEEL BLOCKS, NOT ROUTING BLOCKS, and count only the ones
    // that exist. A routing block at the end of a sequence that is not a
    // multiple of BLK covers fewer than `per` steel blocks -- emitting
    // the missing ones sends steel's loader past the end of K/V, which
    // is a jump into whatever follows rather than a fault.
    const int lo = n * per;
    const int hi = min(lo + per, NKS);
    n_keep += keep ? (uint)max(0, hi - lo) : 0u;
  }
  if (lane == 0) {
    kept[(uint)h * NQ + qb] = n_keep;
    if (counts != nullptr) {
      atomic_fetch_add_explicit(&counts[0], n_keep, memory_order_relaxed);
    }
  }
}

// The same routing, over a PRECOMPUTED proxy matrix.
//
// Identical arithmetic to sol_route_mma with one substitution: the
// <qbar, kc[n]> dot it computed per key block, re-reading every key
// centroid once per query block, is read from `proxy` instead -- which
// sol_proxy_mma produced as one batched GEMM on the matrix units. The
// THRESHOLD is still computed here, being a per-query-block reduction
// over D against the centroids' mean and variance rather than anything
// a GEMM produces.
//
// The decisions are the same decisions: the proxy accumulates in f32
// and is stored f32, exactly as simd_sum did, so the only difference is
// summation order.
//
//   0:proxy (float [H,NQ,NK]) 1:qc (VPIPE_ELT [H,NQ,D])
//   2:mean 3:var (float [H,D])  4:flags (uchar [H,NQ,NK])
//   5:kept (uint [H,NQ])  6:counts (atomic_uint[2])
//   7:scale 8:tau (float) 9:D 10:NK 11:NQ 12:BLK 13:BQ
//   14:radius 15:sink_lo 16:sink_hi 17:per 18:NKS (int)
// grid (32, H, NQ) THREADS; threadgroup (32,1,1).
kernel void sol_route_p_mma(
    const device float*     proxy   [[buffer(0)]],
    const device VPIPE_ELT* qc      [[buffer(1)]],
    const device float*     mean    [[buffer(2)]],
    const device float*     var     [[buffer(3)]],
    device uchar*           flags   [[buffer(4)]],
    device uint*            kept    [[buffer(5)]],
    device atomic_uint*     counts  [[buffer(6)]],
    constant float&         scale   [[buffer(7)]],
    constant float&         tau     [[buffer(8)]],
    constant int&           D       [[buffer(9)]],
    constant int&           NK      [[buffer(10)]],
    constant int&           NQ      [[buffer(11)]],
    constant int&           BLK     [[buffer(12)]],
    constant int&           BQ      [[buffer(13)]],
    constant int&           radius  [[buffer(14)]],
    constant int&           sink_lo [[buffer(15)]],
    constant int&           sink_hi [[buffer(16)]],
    constant int&           per     [[buffer(17)]],
    constant int&           NKS     [[buffer(18)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int h  = (int)tid.y;
  const int qb = (int)tid.z;
  const float ls = scale * SOL_LOG2E;
  const device VPIPE_ELT* qbar = qc + ((uint)h * NQ + qb) * D;
  const device float* mh = mean + (uint)h * D;
  const device float* vh = var  + (uint)h * D;

  float rm = 0.0f, rv = 0.0f;
  for (int i = (int)lane; i < D; i += 32) {
    const float qi = float(qbar[i]);
    rm += qi * mh[i];
    rv += qi * qi * vh[i];
  }
  rm = simd_sum(rm) * ls;
  rv = max(simd_sum(rv), 0.0f) * ls * ls;
  const float thr = rm + tau * sqrt(rv + 1.0e-6f);

  const int qk = (qb * BQ) / BLK;
  const device float* pr = proxy + ((uint)h * NQ + qb) * NK;
  device uchar* fl = flags + ((uint)h * NQ + qb) * NK;
  // ONE LANE PER KEY BLOCK now that the dot is a load: the row is
  // walked 32 blocks at a time and the kept count is a simd sum, where
  // the fused kernel had all 32 lanes cooperate on one block's dot and
  // step one block at a time.
  uint n_keep = 0;
  for (int n0 = 0; n0 < NK; n0 += 32) {
    const int n = n0 + (int)lane;
    uint mine = 0;
    if (n < NK) {
      const float dot = pr[n] * ls;
      const bool tail  = (n == NK - 1);   // see the file header
      const bool local = abs(qk - n) <= radius;
      const bool sink  = (n >= sink_lo && n < sink_hi);
      const bool keep  = (dot > thr) || local || sink || tail;
      fl[n] = keep ? (uchar)1 : (uchar)0;
      const int lo = n * per;
      const int hi = min(lo + per, NKS);
      mine = keep ? (uint)max(0, hi - lo) : 0u;
    }
    n_keep += simd_sum(mine);
  }
  if (lane == 0) {
    kept[(uint)h * NQ + qb] = n_keep;
    if (counts != nullptr) {
      atomic_fetch_add_explicit(&counts[0], n_keep, memory_order_relaxed);
    }
  }
}

// The prefix sum, per head, into steel's [H][NQ + 1] qb_off.
//
//   0:kept (uint [H,NQ]) 1:qb_off (int [H*(NQ+1)])
//   2:NQ 3:H 4:per (int, unused -- kept is already in steel blocks)
// grid (1, 1, 1) THREADS; threadgroup (1,1,1) -- one thread, H*NQ adds.
kernel void sol_scan_mma(
    const device uint* kept    [[buffer(0)]],
    device int*        qb_off  [[buffer(1)]],
    constant int&      NQ      [[buffer(2)]],
    constant int&      H       [[buffer(3)]],
    constant int&      per     [[buffer(4)]])
{
  // Bound and unread: `kept` already counts steel blocks, so the factor
  // has no use here. The SLOT stays because the emit kernel beside it
  // does need `per` at the same index, and a caller that had to
  // remember which of the two skips it is a caller that will get it
  // wrong.
  (void)per;
  int acc = 0;
  for (int h = 0; h < H; ++h) {
    for (int q = 0; q < NQ; ++q) {
      qb_off[h * (NQ + 1) + q] = acc;
      // `kept` is already in STEEL blocks, so no per factor here.
      acc += (int)kept[h * NQ + q];
    }
    qb_off[h * (NQ + 1) + NQ] = acc;
  }
}

//   0:flags (uchar [H,NQ,NK]) 1:qb_off (int) 2:qb_blocks (int)
//   3:NQ 4:NK 5:per (int)
// grid (32, H, NQ) THREADS; threadgroup (32,1,1).
kernel void sol_emit_mma(
    const device uchar* flags  [[buffer(0)]],
    const device int*   qb_off [[buffer(1)]],
    device int*         qb_blk [[buffer(2)]],
    constant int&       NQ     [[buffer(3)]],
    constant int&       NK     [[buffer(4)]],
    constant int&       per    [[buffer(5)]],
    constant int&       NKS    [[buffer(6)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  if (lane != 0) { return; }
  const int h  = (int)tid.y;
  const int qb = (int)tid.z;
  const device uchar* fl = flags + ((uint)h * NQ + qb) * NK;
  int w = qb_off[h * (NQ + 1) + qb];
  // ASCENDING and duplicate-free: steel's loaders only ever move
  // forward, so a repeat would read the wrong bytes for the rest of the
  // row rather than merely double-counting.
  for (int n = 0; n < NK; ++n) {
    if (fl[n] == (uchar)0) { continue; }
    const int lo = n * per;
    const int hi = min(lo + per, NKS);   // see sol_route_mma
    for (int j = lo; j < hi; ++j) { qb_blk[w++] = j; }
  }
}

// The APPROXIMATE half: a flash attention over the SUMMARY sequence.
//
// N keys instead of T, so this is 1/BLK of the dense work -- but only if
// it runs on the matrix units, which is what the tiling below is for.
// One threadgroup per (head, 32 query rows); four simdgroups, eight rows
// each; summary chunks of BN staged in threadgroup memory.
//
// THE PER-ROW RESCALE STAYS IN REGISTERS, as a DIAGONAL MMA. The online
// softmax has to multiply the [8, D] accumulator by a per-row correction
// every chunk, and simdgroup_matrix has no row operator -- but
// Diag(corr) x O is exactly that, and an 8x8 multiply per output
// fragment is far cheaper than staging 4 KB of accumulator through
// threadgroup memory and back. It is also what keeps this under the
// 32 KB threadgroup budget the M4 Pro actually has.
//
// Outputs the UNNORMALISED numerator and the two softmax statistics, so
// the merge can combine it with steel's exact partial. Dividing here and
// multiplying back there would lose the rows where the approximate half
// carries almost all of the mass.
//
//   0:q (VPIPE_ELT [H,T,D]) 1:kc 2:vc (VPIPE_ELT [H,N,D])
//   3:flags (uchar [H,NQ,N])
//   4:o_a (float [H,T,D]) 5:m_a 6:l_a (float [H,T])
//   7:scale(float) 8:T 9:D 10:N 11:NQ 12:BLK 13:TPAD 14:BQR (int)
// TPAD is T rounded up to a whole 32-row tile: o_a is allocated at that
// height so the last tile's store needs no per-element guard.
//
// BQR IS THE ROUTING QUERY BLOCK AND IS NOT THIS KERNEL'S TILE. The
// routing is decided at the exact half's own query block -- 32 on the
// ALU flash kernel and 64 on the matrix-core one -- while this kernel
// tiles 32 rows either way, so two tiles share one flag row when the
// two differ. Keeping this kernel's tile fixed is what lets the exact
// half's tile change without a second entry point.
// grid (32, 4*H, ceil(T/32)) THREADS -- y is 4*H because the
// threadgroup is 2-D and tid.y must come out as the head; threadgroup
// (32,4,1).
kernel void sol_approx_mma(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* kc    [[buffer(1)]],
    const device VPIPE_ELT* vc    [[buffer(2)]],
    const device uchar*     flags [[buffer(3)]],
    device float*           o_a   [[buffer(4)]],
    device float*           m_a   [[buffer(5)]],
    device float*           l_a   [[buffer(6)]],
    constant float&         scale [[buffer(7)]],
    constant int&           T     [[buffer(8)]],
    constant int&           D     [[buffer(9)]],
    constant int&           N     [[buffer(10)]],
    constant int&           NQ    [[buffer(11)]],
    constant int&           BLK   [[buffer(12)]],
    constant int&           TPAD  [[buffer(13)]],
    constant int&           BQR   [[buffer(14)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint3 ltid [[thread_position_in_threadgroup]])
{
  constexpr int BQ = 32;          // query rows per threadgroup
  constexpr int BN = 16;          // summary blocks per chunk
  constexpr int LD = SOL_D + 8;   // padded row stride

  threadgroup VPIPE_ELT Qs[BQ * LD];
  threadgroup VPIPE_ELT KCs[BN * LD];
  threadgroup VPIPE_ELT VCs[BN * LD];
  threadgroup float     Sm[4 * 8 * BN];
  threadgroup VPIPE_ELT Pm[4 * 8 * BN];
  threadgroup float     Dm[4 * 64];

  const int h  = (int)tid.y;
  const int q0 = (int)tid.z * BQ;
  if (q0 >= T || D != SOL_D) { return; }

  const device VPIPE_ELT* qh = q + (uint)h * T * D;
  // WHICH FLAG ROW THESE 32 ROWS OBEY. Routing is taken at the exact
  // half's query block so that qb_off is one entry per block of its
  // own, and that block is 32 rows on the ALU kernel and 64 on the
  // matrix-core one -- so this is q0 / BQR and not tid.z. Two tiles
  // sharing a flag row is exactly right: both halves of a 64-row
  // routing block were routed together.
  const device uchar* fl = flags + ((uint)h * NQ + (q0 / BQR)) * N;

  const uint tflat = ltid.y * 32 + ltid.x;
  for (int i = (int)tflat; i < BQ * SOL_D; i += 128) {
    const int r = i / SOL_D, c = i % SOL_D;
    const int row = q0 + r;
    Qs[r * LD + c] = (row < T) ? qh[(uint)row * D + c] : VPIPE_ELT(0);
  }

  simdgroup_matrix<float, 8, 8> Sf[BN / 8];
  simdgroup_matrix<float, 8, 8> Of[SOL_D / 8];
  for (int i = 0; i < SOL_D / 8; ++i) {
    Of[i] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  float m = -INFINITY, l = 0.0f;
  // +log2(BLK): the value summary is a MEAN, so the score carries the
  // block's length. Uniform because the tail block is never approximate.
  const float bonus = log2((float)BLK);
  const float ls = scale * SOL_LOG2E;

  for (int n0 = 0; n0 < N; n0 += BN) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int)tflat; i < BN * SOL_D; i += 128) {
      const int r = i / SOL_D, c = i % SOL_D;
      const int n = n0 + r;
      const bool use = (n < N) && (fl[n] == (uchar)0);
      KCs[r * LD + c] = use ? kc[((uint)h * N + n) * D + c] : VPIPE_ELT(0);
      VCs[r * LD + c] = use ? vc[((uint)h * N + n) * D + c] : VPIPE_ELT(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j = 0; j < BN / 8; ++j) {
      Sf[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (int dk = 0; dk < SOL_D; dk += 8) {
      simdgroup_matrix<VPIPE_ELT, 8, 8> A;
      simdgroup_load(A, Qs + (int)sg * 8 * LD + dk, LD);
      for (int j = 0; j < BN / 8; ++j) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> B;
        // Transposed: KCs is [BN, D] and the product wants [D, BN].
        simdgroup_load(B, KCs + j * 8 * LD + dk, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Sf[j], A, B, Sf[j]);
      }
    }
    for (int j = 0; j < BN / 8; ++j) {
      simdgroup_store(Sf[j], Sm + (int)sg * (8 * BN) + j * 8, BN);
    }
    // mem_threadgroup, not mem_none: what is being published here is
    // THREADGROUP memory, between lanes of one simdgroup. mem_none
    // orders nothing and the softmax below then reads whatever was in
    // Sm -- which is a finite, normalised, wrong answer.
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // The row softmax: eight rows per simdgroup, one lane each. A
    // masked column carries a ZERO key, so its raw score is 0 rather
    // than -inf -- it is excluded by the predicate, never by arithmetic.
    if (lane < 8) {
      const int r = (int)lane;
      threadgroup float* sr = Sm + (int)sg * (8 * BN) + r * BN;
      float mm = m;
      for (int j = 0; j < BN; ++j) {
        const int n = n0 + j;
        const bool use = (n < N) && (fl[n] == (uchar)0);
        const float s = use ? (sr[j] * ls + bonus) : -INFINITY;
        sr[j] = s;
        mm = max(mm, s);
      }
      const float corr = (mm == -INFINITY || m == -INFINITY)
                             ? ((m == -INFINITY) ? 0.0f : 1.0f)
                             : exp2(m - mm);
      float ll = 0.0f;
      for (int j = 0; j < BN; ++j) {
        const float p = (sr[j] == -INFINITY || mm == -INFINITY)
                            ? 0.0f
                            : exp2(sr[j] - mm);
        Pm[(int)sg * (8 * BN) + r * BN + j] = VPIPE_ELT(p);
        ll += p;
      }
      l = l * corr + ll;
      if (mm != -INFINITY) { m = mm; }
      // Diag(corr), for the register rescale below.
      threadgroup float* dg = Dm + (int)sg * 64;
      for (int j = 0; j < 8; ++j) { dg[r * 8 + j] = (j == r) ? corr : 0.0f; }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> Dg;
    simdgroup_load(Dg, Dm + (int)sg * 64, 8);
    for (int c = 0; c < SOL_D / 8; ++c) {
      simdgroup_matrix<float, 8, 8> tmp;
      simdgroup_multiply(tmp, Dg, Of[c]);
      Of[c] = tmp;
    }
    for (int j = 0; j < BN / 8; ++j) {
      simdgroup_matrix<VPIPE_ELT, 8, 8> Pf;
      simdgroup_load(Pf, Pm + (int)sg * (8 * BN) + j * 8, BN);
      for (int c = 0; c < SOL_D / 8; ++c) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> Bv;
        simdgroup_load(Bv, VCs + j * 8 * LD + c * 8, LD);
        simdgroup_multiply_accumulate(Of[c], Pf, Bv, Of[c]);
      }
    }
  }

  // STRAIGHT TO DEVICE MEMORY, no staging. The eight rows a simdgroup
  // owns are contiguous in o_a, so simdgroup_store writes them where
  // they belong -- and staging them would have wanted 16 KB of
  // threadgroup memory this kernel does not have to spare.
  //
  // o_a is allocated with its row count ROUNDED UP to a whole tile,
  // which is what makes the store safe on the last tile: the rows past
  // T are written and then never read, rather than guarded per element.
  for (int c = 0; c < SOL_D / 8; ++c) {
    simdgroup_store(
        Of[c], o_a + ((uint)h * TPAD + q0 + (int)sg * 8) * SOL_D + c * 8,
        SOL_D);
  }
  if (lane < 8) {
    const int row = q0 + (int)sg * 8 + (int)lane;
    if (row < T) {
      m_a[(uint)h * T + row] = m;
      l_a[(uint)h * T + row] = l;
    }
  }
}

// Merge the exact partial (steel's O, already divided by its own
// denominator, plus its max and sum) with the approximate one.
//
//   0:o_e (VPIPE_ELT [H,T,D]) 1:m_e 2:l_e (float [H,T])
//   3:o_a (float [H,TPAD,D]) 4:m_a 5:l_a (float [H,T])
//   6:out (VPIPE_ELT [H,T,D])  7:T 8:D 9:TPAD (int)
// grid (D, H, T) THREADS; threadgroup (D,1,1).
kernel void sol_merge_mma(
    const device VPIPE_ELT* o_e [[buffer(0)]],
    const device float*     m_e [[buffer(1)]],
    const device float*     l_e [[buffer(2)]],
    const device float*     o_a [[buffer(3)]],
    const device float*     m_a [[buffer(4)]],
    const device float*     l_a [[buffer(5)]],
    device VPIPE_ELT*       out [[buffer(6)]],
    constant int&           T   [[buffer(7)]],
    constant int&           D   [[buffer(8)]],
    constant int&           TPAD [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  d   [[thread_index_in_threadgroup]])
{
  const int h = (int)tid.y;
  const int t = (int)tid.z;
  if ((int)d >= D || t >= T) { return; }
  const uint r = (uint)h * T + t;
  const float me = m_e[r], le = l_e[r];
  const float ma = m_a[r], la = l_a[r];
  // Either half can be empty: a query block whose routing kept nothing
  // has le == 0, and one that kept everything has la == 0. Both are
  // legitimate and neither may produce a 0/0.
  const float m = max(me, ma);
  const float we = (le > 0.0f && me > -INFINITY) ? le * exp2(me - m) : 0.0f;
  const float wa = (la > 0.0f && ma > -INFINITY) ? la * exp2(ma - m) : 0.0f;
  const float den = we + wa;
  // o_e is already divided by le, so multiplying by we recovers its
  // numerator at the merged scale; o_a was never divided.
  const float ne = (we > 0.0f) ? float(o_e[r * D + d]) * we : 0.0f;
  const float na = (wa > 0.0f)
                       ? o_a[((uint)h * TPAD + t) * D + d] *
                             (la > 0.0f ? wa / la : 0.0f)
                       : 0.0f;
  out[r * D + d] = VPIPE_ELT(den > 0.0f ? (ne + na) / den : 0.0f);
}

// The same merge, for an approximate half produced by the FLASH kernel
// rather than by sol_approx_mma.
//
// Three differences, all of them consequences of that half now being an
// ordinary attention over the summary sequence:
//
//   * o_a is ALREADY DIVIDED by its own denominator, exactly as o_e is,
//     so the numerator is recovered by multiplying by `wa` and not by
//     `wa / la`;
//   * it is VPIPE_ELT [H, T, D] rather than fp32 [H, TPAD, D], the
//     flash kernel storing in the tensor dtype and needing no padded
//     tail;
//   * the +log2(BLK) the value summary's MEAN requires is added HERE.
//     Every approximate score carries the same constant, so it shifts
//     the maximum and leaves both the denominator and the normalised
//     output untouched -- which is what makes it a merge-time term
//     rather than something the flash kernel has to know.
//
// AND THE EMPTY ROW IS A SENTINEL, not a zero. A query block that kept
// every block exact has no approximate keys at all; the flash kernel's
// mask writes finite_min, which is also its initial maximum, so that
// row comes back with m_a == finite_min, l_a == the masked column count
// and o_a the mean of the excluded value summaries. `-1e30` separates
// that from any real score -- they are bounded by |q||kc| * scale --
// and it has to, because the arithmetic below would otherwise weight
// pure garbage by exp2(finite_min - m), which is 0 only if m is finite.
//
//   0:o_e (VPIPE_ELT [H,T,D]) 1:m_e 2:l_e (float [H,T])
//   3:o_a (VPIPE_ELT [H,T,D]) 4:m_a 5:l_a (float [H,T])
//   6:out (VPIPE_ELT [H,T,D])  7:T 8:D (int) 9:bonus (float)
// grid (D, H, T) THREADS; threadgroup (D,1,1).
kernel void sol_merge_ml_mma(
    const device VPIPE_ELT* o_e   [[buffer(0)]],
    const device float*     m_e   [[buffer(1)]],
    const device float*     l_e   [[buffer(2)]],
    const device VPIPE_ELT* o_a   [[buffer(3)]],
    const device float*     m_a   [[buffer(4)]],
    const device float*     l_a   [[buffer(5)]],
    device VPIPE_ELT*       out   [[buffer(6)]],
    constant int&           T     [[buffer(7)]],
    constant int&           D     [[buffer(8)]],
    constant float&         bonus [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint  d   [[thread_index_in_threadgroup]])
{
  const int h = (int)tid.y;
  const int t = (int)tid.z;
  if ((int)d >= D || t >= T) { return; }
  const uint r = (uint)h * T + t;
  const float me = m_e[r], le = l_e[r];
  const float ma_raw = m_a[r], la = l_a[r];
  const bool  a_live = (ma_raw > -1.0e30f) && (la > 0.0f);
  const float ma = a_live ? ma_raw + bonus : -INFINITY;
  const bool  e_live = (le > 0.0f) && (me > -1.0e30f);
  const float m = max(e_live ? me : -INFINITY, ma);
  const float we = e_live ? le * exp2(me - m) : 0.0f;
  const float wa = a_live ? la * exp2(ma - m) : 0.0f;
  const float den = we + wa;
  const float ne = (we > 0.0f) ? float(o_e[r * D + d]) * we : 0.0f;
  const float na = (wa > 0.0f) ? float(o_a[r * D + d]) * wa : 0.0f;
  out[r * D + d] = VPIPE_ELT(den > 0.0f ? (ne + na) / den : 0.0f);
}
