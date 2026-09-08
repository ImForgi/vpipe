#ifndef GENERATIVE_MODELS_SHARED_METAL_SOL_ATTENTION_H
#define GENERATIVE_MODELS_SHARED_METAL_SOL_ATTENTION_H

// Sol-Attn on the metal-compute backend: seven dispatches and the
// scratch they share. The method, and the CPU oracle these kernels are
// checked against, are in sol-attention.h.
//
// THE ATTENTION GOES TO THE FLASH KERNEL. That is the whole design. A
// first port ran the entire method in one plain simdgroup kernel and
// measured 281 GF/s against steel's 6656, so the routing's saving was
// buried under a 24x kernel deficit. Here BOTH halves are the tree's
// own flash attention -- the exact blocks through its has_spans path,
// the approximate ones over the summary sequence with the routing's
// flags as a mask -- and only the summaries, the routing and the merge
// are Sol's own code.
//
//   sol_summaries_mma  x2   block centroids; q at the QUERY block size,
//                           k/v at the KEY block size
//   sol_kc_stats_mma        the centroids' spread, for the threshold
//   sol_proxy_mma           the proxy scores, one batched GEMM (M5)
//   sol_route_p_mma         threshold, decision, flag rows, kept counts
//     (or sol_route_mma, which fuses the proxy into the decision)
//   sol_scan_mma            the counts into a CSR offset array
//   sol_emit_mma            the flag rows into the CSR block list
//   attn_steel[_nax] (has_block_mask + export_ml)   the approximate half
//   attn_steel[_nax] (has_spans + export_ml)        the exact half
//   sol_merge_ml_mma        the two partial softmaxes into one output
//     (or sol_merge_mma, beside sol_approx_mma on the ALU arm)
//
// WHICH FLASH KERNEL DECIDES THE TILES, and they are not the same: the
// ALU steel entry routes at a 32-row query block and walks a CSR in
// 16-key units, the matrix-core one at 64 and 32. A coarser query block
// keeps slightly more (15.1% against 13.8% at tau 1.0) for the same
// accuracy, and is far faster; VPIPE_SOL_NO_NAX is the A/B.
//
// MEASURED, 56 heads x 20036 rows x 128, bf16, each arm against THE
// SAME kernel run dense:
//
//   arm    dense       sol    speedup   kept
//   ALU    3271 ms   640 ms    5.11x    13.8%
//   NAX    1149 ms   247 ms    4.65x    15.0%
//
// THE NUMBER THAT MATTERS ON AN M5 IS NEITHER COLUMN'S RATIO. The model
// runs its dense attention on the NAX kernel, so before this port Sol
// was 1149 -> 640 ms, i.e. 1.80x, not the 5.11x the ALU column reads:
// the exact half was giving up the faster kernel that the baseline it
// replaces was already using. It is now 4.65x, and 13.2x against the
// ALU kernel run dense.
//
// The key block knee moved with the port and then moved back. On the
// ALU arm 64 wins (5.11x against 4.27 / 4.36 / 2.89 at 32 / 128 / 256);
// with only the exact half on the matrix cores 128 briefly won, the
// simdgroup approximate half having become a third of the call; with
// both halves there it is 64 again -- which is also the more accurate
// block, so the fast choice and the faithful one are the same one.
//
// AND IT ALLOCATES ALMOST NOTHING. Every buffer here lives for ONE
// call -- summaries, routing, CSR and both partial softmaxes are all
// consumed by the merge that ends it -- so it takes memory the caller
// is not using rather than memory of its own. The DiT has more than
// enough: with Sol on, the fused qkv projection has already been
// transposed out and is dead until the feed-forward, and the attention
// arena's last window is dead until the transpose that follows the
// call. MEASURED at 56 heads x 14861 rows: 861 MB held became 0.
//
// Two of those 861 were never needed at all -- 427 MB of it was the ALU
// arm's fp32 padded partial, allocated on the matrix-core arm too,
// where nothing reads it.
//
// FAMILY-AGNOSTIC ON PURPOSE. Nothing here knows about a DiT: it takes
// head-major [H, T, D] q/k/v and returns the same, which is the shape
// every unfused attention path in this tree already speaks. A model
// adopts it by routing one dispatch through `encode` -- see the
// MiniMax-H3 transformer, which is the first caller.

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/sol-attention.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalSolAttention {
 public:
  // `bf16` selects the bfloat twin; false is f16. Null (with `err` set)
  // when a ComputeFunction does not validate -- never an object that
  // dispatches no-ops, which is the failure this backend makes silent.
  //
  // The steel function is specialised per SEQUENCE LENGTH (align_Q /
  // align_K are function constants), so it is built on first use and
  // rebuilt when the geometry moves, not here.
  static std::unique_ptr<MetalSolAttention>
  load(metal_compute::MetalCompute* mc, bool bf16, std::string* err);

  // Scratch bytes for one call at this geometry, so a caller can size
  // the box before encoding.
  //
  // WHICH IT HAD BETTER DO. This is spent at the first encode, inside
  // the first forward and therefore after every planning decision --
  // and at 56 heads x 14861 rows it was 861 MB. Until now nothing
  // called it at all, so the DiT kept streamed blocks resident against
  // a budget that had never heard of Sol.
  //
  // `nax` and `key_block` change the answer and are not this class's to
  // guess from a static: the matrix-core arm allocates a bf16 partial
  // where the ALU one allocates an fp32 padded one, and the key block
  // sets the summary sequence. uses_matrix_cores() answers the first
  // once an object exists; a planner with none asks
  // MetalCompute::supports_matrix_cores().
  static std::size_t scratch_bytes(int heads, int tokens, int d,
                                   int key_block, bool nax);

  // ...and what is left AFTER a lend of `lend_a` / `lend_b` bytes: the
  // same greedy carve, alignment and all, rather than a subtraction.
  // This is the number a planner adds to the model's own scratch.
  static std::size_t private_bytes(int heads, int tokens, int d,
                                   int key_block, bool nax,
                                   std::size_t lend_a, std::size_t lend_b);

  // Encode summaries -> stats -> threshold -> forward into `enc`.
  //
  // The four are ordered by data dependence and Metal's serial dispatch
  // type orders consecutive dispatches, so no commit is needed between
  // them -- the whole call is one encoder, like every other attention
  // path here.
  bool encode(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& q,
              const metal_compute::SharedBuffer& k,
              const metal_compute::SharedBuffer& v,
              metal_compute::SharedBuffer& out, int heads, int tokens,
              int d, float scale, const sol::Config& cfg,
              std::string* err);

  // Blocks kept exact by the LAST completed encode, and the total the
  // routing chose from. Valid only after the command buffer has run.
  // Realized sparsity is a property of the data, so it is read back
  // rather than predicted.
  long long exact_blocks() const;
  long long total_blocks() const { return _total_blocks; }
  void      reset_counts();

  std::size_t resident_bytes() const;

  // LEND THIS MEMORY THAT IS DEAD FOR THE LENGTH OF ONE CALL, and the
  // scratch above stops being an allocation.
  //
  // Every buffer this class holds bar a few hundred bytes is written
  // and read inside ONE encode: the summaries, the routing decision,
  // the CSR and the two partial softmaxes are all consumed by the merge
  // that ends it. So it does not need memory of its own, it needs
  // memory nobody else is using between the projection and the output.
  //
  // THE CALLER HAS EXACTLY THAT, AND MORE THAN ENOUGH OF IT. Sol reads
  // head-major q/k/v, which means the fused [seq, 3*inner] projection
  // has already been transposed out and is dead until the feed-forward
  // writes over it -- three attention windows' worth -- and the
  // attention arena's LAST window is dead too, being written only by
  // the transpose that follows the call. MEASURED at 56 heads x 14861
  // rows: 853 MB dead against 489 MB wanted.
  //
  // Two regions rather than one because those are two allocations.
  // Anything that fits neither is allocated privately, per buffer, so a
  // caller that lends nothing (the tests, the isolated benches) is
  // unchanged.
  //
  // Set it BEFORE the first encode, and lend nothing again before
  // replacing what was lent: the carved buffers are windows that hold
  // the lender's allocation alive, so a caller that reallocates without
  // taking the loan back pays for both.
  void set_arena(const metal_compute::SharedBuffer& a,
                 const metal_compute::SharedBuffer& b);

  // What this object allocates for itself however much is lent: the
  // AttnParams pair, the span params and their bounds, and the routing
  // counter. All host-written or host-read, so none of them can live in
  // memory the caller reuses between calls.
  static std::size_t pinned_bytes();

  // Whether the exact half is on the matrix-core flash kernel, and the
  // query/key block the routing is therefore decided at. A bench that
  // compares against dense has to run the SAME kernel dense or it is
  // measuring the port and calling it the approximation.
  bool uses_matrix_cores() const { return _nax; }
  int  query_block() const { return _bq; }
  int  key_block_unit() const { return _bk; }

  // Removes this object's scratch from the process-wide residency set.
  // Public because the DESTRUCTOR is not the only moment it matters: a
  // lender that is about to take its arena back wants the set to stop
  // holding it, and set_arena() does that on the way through.
  ~MetalSolAttention();

 private:
  MetalSolAttention() = default;
  std::vector<metal_compute::SharedBuffer*> scratch_buffers_();
  // THE OTHER HALF OF residency_add. MTLResidencySet RETAINS what is
  // added to it and the set lives on MetalCompute, which outlives every
  // model -- so an add without a matching remove is not a hint the
  // kernel may ignore, it is an allocation that never comes back.
  //
  // It is worse than its own size, because most of these buffers are
  // WINDOWS carved out of an arena the caller lent (set_arena): a
  // subview shares its parent's MTL::Buffer, so adding one hands the set
  // the whole parent. Sol's few hundred megabytes of scratch that way
  // pinned the DiT's multi-gigabyte forward arena, which then survived
  // `unload_when_idle: destroy` and left the VAE decode with nothing.
  // See metal_compute_residency.a_subview_adds_the_whole_parent.
  void drop_residency_();
  bool ensure_scratch_(int heads, int tokens, int d, std::string* err);
  bool ensure_steel_(int tokens, std::string* err);

  metal_compute::MetalCompute* _mc = nullptr;
  bool _bf16 = false;
  metal_compute::ComputeLibrary  _lib;
  metal_compute::ComputeLibrary  _lib_steel;
  metal_compute::ComputeLibrary  _lib_nax;
  metal_compute::ComputeFunction _fn_sum, _fn_stats, _fn_route, _fn_scan;
  metal_compute::ComputeFunction _fn_emit, _fn_approx, _fn_merge;
  metal_compute::ComputeFunction _fn_merge_ml;
  metal_compute::ComputeFunction _fn_steel;   // has_spans + export_ml
  // THE APPROXIMATE HALF ON THE SAME FLASH KERNEL, where there is one
  // worth using. It is an attention over the summary sequence with the
  // routing's flag array as a per-(query block, key) mask, so it needs
  // no kernel of its own -- and it was 34% of a routed call once the
  // exact half moved to the matrix cores, being the last piece still
  // running on plain simdgroup matrices at 1/64 of the work but nothing
  // like 1/64 of the time.
  metal_compute::ComputeFunction _fn_approx_nax;
  // The routing proxy as one batched GEMM, and the routing kernel that
  // reads it. The fused sol_route_mma stays for the ALU arm: it is the
  // reference the two are checked against, and on a GPU with no matrix
  // units a separate proxy matrix would be pure extra traffic.
  metal_compute::ComputeLibrary  _lib_proxy;
  metal_compute::ComputeFunction _fn_proxy, _fn_route_p;
  int _steel_seq = -1, _steel_nk = -1;
  // THE EXACT HALF'S KERNEL, AND ITS TILE. On a matrix-core GPU the
  // blocks the routing keeps go to attn_steel_nax, whose query block is
  // 64 and key block 32 against the ALU kernel's 32 and 16 -- and those
  // tiles are not this class's to choose: the CSR is in the exact
  // kernel's own units and the routing is decided at its query block,
  // so picking the kernel picks both.
  //
  // A COARSER ROUTING BLOCK IS A REAL DIFFERENCE, not just a faster
  // one: one threshold then serves 64 rows rather than 32, so the two
  // kernels keep slightly different block sets and the outputs differ
  // by more than precision. That is the same trade `key_block` makes on
  // the other axis, and it is why VPIPE_SOL_NO_NAX exists.
  bool _nax = false;
  int  _bq = 0, _bk = 0;

  metal_compute::SharedBuffer _qc, _kc, _vc, _mean, _var;
  metal_compute::SharedBuffer _flags, _kept, _qb_off, _qb_blk;
  metal_compute::SharedBuffer _o_a, _m_a, _l_a, _o_e, _m_e, _l_e;
  // The flash approximate half's output: the tensor dtype and the plain
  // [H, T, D] shape, against sol_approx_mma's fp32 [H, TPAD, D].
  metal_compute::SharedBuffer _o_an, _params_a;
  // [H, NQ, NK] fp32 -- the proxy scores, when they are computed as a
  // GEMM rather than inside the routing kernel.
  metal_compute::SharedBuffer _proxy;
  // Up to two regions the caller has lent (set_arena), each with its own
  // bump cursor. Empty is the ordinary standalone-allocation case.
  metal_compute::SharedBuffer _arena_a, _arena_b;
  std::size_t                 _arena_used[2] = {0, 0};
  // Whether the buffers below are currently in the residency set, so the
  // remove walks the same list the add did and only when there is
  // something to walk.
  bool                        _res_added = false;
  metal_compute::SharedBuffer _counts, _params, _sp_params, _sp_bounds;
  int _heads = 0, _tokens = 0, _d = 0, _nq = 0, _nk = 0, _tpad = 0;
  int _nqa = 0;      // approximate-half tiles: ceil(T / 32), always
  int _blk = 0;
  long long _total_blocks = 0;
};

}  // namespace genai
}  // namespace vpipe

#endif  // GENERATIVE_MODELS_SHARED_METAL_SOL_ATTENTION_H
