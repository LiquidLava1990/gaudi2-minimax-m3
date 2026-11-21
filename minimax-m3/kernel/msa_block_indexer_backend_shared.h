/*
 * msa_block_indexer_backend_shared.h -- the SHARED, frontend-agnostic backend
 * lowering for the MiniMax-M3 MSA (block-sparse) *block indexer* top-k as a
 * GC-captured `habana::OpBackend` op
 * (`torch.ops.hpu_msa_ext::msa_block_indexer_topk`).
 *
 * This is an ADDITIVE, SELF-CONTAINED extension of the already-built + on-card-
 * validated DeepSeek/GLM DSA indexer kernel
 * ($HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/dsa/dsa_indexer_backend_shared.h),
 * copied here as the base and made config-driven (a `mode` + `block_size` schema
 * arg) so the SAME op serves BOTH:
 *
 *   mode=0 / block_size==0  -> DSA token-sparse (BYTE-IDENTICAL to the DSA chain:
 *                              ReLU + per-head weight + head-sum + topk TOKEN ids).
 *                              Default. Kept so this package is a strict superset
 *                              and the M3 kernel is self-contained.
 *   mode=1 / block_size>0    -> MSA GQA block-sparse (the NEW branch): per-(query,
 *                              GQA-group) block-max-pool + topk BLOCK ids. No ReLU,
 *                              no head-sum (M3 selects per index head independently).
 *
 * It registers its OWN op namespace (`hpu_msa_ext`) and builds its OWN .so
 * (libmsa_block_indexer{,_lazy}.so) so it NEVER collides with the installed/serving
 * `hpu_dsa_ext` ops. The running vllm-active serve (the installed hpu-dsa-ext /
 * hpu-moe-ext .so) is untouched by this package.
 *
 * Modeled VERBATIM on the proven MoE + DSA OpBackend pattern: a `struct : OpBackend`
 * whose AddNode() chains `OpBackend::BuildNode({guid, {syn_inputs}, {output_attrs},
 * &params, size})` calls in PyTorch ROW-MAJOR order (last dim == FCD). The factoring
 * into a *_shared.h header (backend only, frontend-free) lets the eager TU
 * (msa_block_indexer_backend.cpp, EagerOp) and the lazy TU
 * (msa_block_indexer_backend_lazy.cpp, LazyOp) reuse this byte-for-byte.
 *
 * ============================ THE ORACLE MATH MATCHED ========================
 * From $HOME/AI/minimax-m3/ref/msa_reference.py (the numerically-authoritative
 * MSA spec/oracle, pinned to the official MSA paper Eq. 5-7 + the MiniMax-AI/MSA
 * kernel semantics). The MSA INDEX BRANCH (msa_index_scores :203-248 +
 * msa_block_select :254-316) is, per (query i, GQA group r, KV block b):
 *
 *     S_idx[r]_{i,j} = (q_idx[i,r] . k_idx[j]) / sqrt(d_idx)   # raw scaled dot,
 *                                                              #   NO ReLU/exp (PAPER
 *                                                              #   Eq.6, "exp-free" 4)
 *     S_idx         = masked_fill(key_pos > abs_pos, -inf)     # causal j<=i (Eq.6)
 *     M_idx[r]_{i,b}= max_{j in block_b}  S_idx[r]_{i,j}       # BlockMaxPool (score_type=max)
 *     bid[i,r]      = topk(M_idx[r]_{i,:}, k=topk_blocks)      # top-k BLOCK ids (Eq.7)
 *
 * KEY DIFFERENCES vs DSA (dsa_indexer_backend_shared.h :21-37), all config-gated:
 *   (1) NO ReLU on the per-head logits (DSA does .clamp_(0); MSA is raw dot).
 *   (2) NO per-head weighting + NO head-sum (DSA sums w_h*logit over heads to a
 *       single score[Sq,Sk]; MSA keeps the H_kv index heads SEPARATE -- each GQA
 *       group selects its own block set). So the head axis H is PRESERVED through to
 *       the topk; the MSA output is [H_kv, Sq, topk_blocks] BLOCK ids, not [Sq, kk].
 *   (3) a block-max-pool reduce (reshape -> reduce_max_multi_dim_fwd) is INSERTED
 *       between the masked score and the topk; topk runs over n_blocks, not Sk.
 *
 * WHAT STAYS UPSTREAM (Python, msa_reference.py -- NOT in this op, exactly as DSA's
 * -1 right-pad/sort/force-local/block->token expansion are consume-side):
 *   - force-local / force-init block injection (msa_block_select :289-301): M3
 *     force-includes the block containing the query (sparse_local_block=1). Done by
 *     the runner (e.g. OR the local block id into the returned set, or +inf its score
 *     before this op -- the op returns the raw scored top-k; the runner reconciles).
 *   - ascending sort + -1 padding of the block ids (msa_block_select :309-315).
 *   - block->token expansion bid -> [.. , topk_blocks*block_size] token positions
 *     (msa_expand_blocks_to_tokens :322-340), which then feeds the consume op.
 * This op returns the raw top-k BLOCK ids per (query, group) -- the analog of the DSA
 * op returning raw top-k TOKEN ids; the SET-match parity gate (Jaccard/recall vs
 * msa_reference_block_topk) is order-insensitive, so leaving sort/pad upstream is
 * faithful (mirrors test_dsa_indexer_op.py's set comparison).
 *
 * The order-preserving scale fold: `1/sqrt(d_idx)` (PAPER Eq.6) is a single positive
 * scalar applied to EVERY S_idx entry; it does NOT change the per-block argmax nor
 * the topk ordering. So like DSA folds q_scale*softmax_scale*H^-0.5 into `weights`
 * and the test still SET-matches, MSA passes the index scale broadcast in `weights`
 * (=1/sqrt(d_idx), all heads/rows) and the block-max + topk SET is exact. (Passing
 * weights=ones is equally valid for the SET gate; the runner picks.)
 *
 * NOTHING is hardcoded to a model: H/D/Sq/Sk come from the input tensor shapes;
 * block_size / topk_blocks are scalar args. GLM/DeepSeek DSA (H=32/64) and M3 MSA
 * (H_kv=4, block_size=128, topk_blocks=16) all "just work" -- the §5 generality the
 * MoE + DSA ops proved.
 *
 * ============================ NODE-CHAIN (GUIDs + params) ====================
 * Inputs (order == schema; PyTorch row-major, last dim == FCD). The head axis H is
 * the leading BATCH dim so per-head Q.K^T is ONE batch_gemm (the single shared MQA/
 * index key is expanded to H index heads upstream):
 *   q       [H, Sq, D]  bf16   index queries, per index head (NO rope; PAPER Eq.5)
 *   k       [H, Sk, D]  bf16   index key, single shared head expanded to H
 *   weights [H, Sq, 1]  bf16   DSA: folded per-head weight; MSA: index scale broadcast
 *   cmask   [Sq, Sk]   bool    causal mask, TRUE where key must be DROPPED (key_pos >
 *                              abs_pos); broadcasts over H in MSA (MaskedFillMeta uses
 *                              at::infer_size, masked_fill_gen.cpp:27).
 *   index_topk         int     DSA: top-k TOKENS (clamp to Sk); MSA: top-k BLOCKS
 *                              (clamp to n_blocks).
 *   mode               int     0 = DSA token-sparse (default), 1 = MSA block-sparse.
 *   block_size         int     MSA: tokens per KV block (e.g. 128). Ignored if mode==0.
 *                              Sk MUST be a multiple of block_size (pad upstream).
 *
 *  --- shared NODE 1 (both modes) ---
 *  1. logits = batch_gemm(q[H,Sq,D], k[H,Sk,D], {transpose_a=0, transpose_b=1})
 *                                                            -> [H, Sq, Sk] bf16
 *        guid "batch_gemm", synGEMMParams{false,true}. NON-PERSISTENT.
 *
 *  --- DSA-only NODES 2,4 are GATED OUT in MSA mode ---
 *  2. (DSA) relu = relu_fwd(logits)                          -> [H, Sq, Sk]
 *        MSA: SKIPPED (raw dot, no activation; PAPER §4).
 *  3. wl   = mult_fwd(<relu|logits>, weights[H,Sq,1])        -> [H, Sq, Sk]
 *        DSA: per-head weight of the head-sum; MSA: order-preserving index scale
 *        (weights broadcast over the Sk/FCD axis). Present in BOTH modes (uniform
 *        chain); MSA weights = 1/sqrt(d_idx) (or ones).
 *  4. (DSA) score = reduce_sum_multi_dim_fwd(wl, dim=0(H), keepDim=false)
 *                                                            -> [Sq, Sk]
 *        MSA: SKIPPED (heads stay separate -> score keeps [H,Sq,Sk]).
 *  --- shared NODE 5 (mask) ---
 *  5. masked = masked_fill_fwd(<score|wl>, cmask, value.f=-inf)
 *        DSA: [Sq,Sk]; MSA: [H,Sq,Sk] (cmask[Sq,Sk] broadcasts over H).
 *
 *  --- MSA-only NODES 6a,6b (the block-max pool) ---
 *  6a.(MSA) scoreB = ReshapeHelper(masked, [H, Sq, n_blocks, block_size])  (rank 4)
 *        n_blocks = Sk / block_size (Sk a multiple of block_size). op_backend.h:302.
 *  6b.(MSA) blkmax = reduce_max_multi_dim_fwd(scoreB, dim=3, keepDim=false)
 *                                                            -> [H, Sq, n_blocks]
 *        guid get_guid_with_precision("reduce_max_multi_dim_fwd", bf16),
 *        ns_Reduction::ParamsV2 via FillReductionParams(rank=4,{3},keepdim=false):
 *        reductionDimensionMask = 1<<(4-3-1) = 1<<0 = 1 (reduction_template.cpp:33).
 *        Fully-padded blocks reduce-max to -inf -> never selected (PAPER :230-231).
 *
 *  --- FINAL NODE 6/7 (topk; same GUID, different axis/width by mode) ---
 *  6/7.(vals,idx) = topk(<masked|blkmax>, k=kk, axis, largest)
 *        DSA: over Sk  -> idx[Sq, kk]            int32 (TOKEN ids), axis=tpc(1,2)=0.
 *        MSA: over n_blocks -> idx[H,Sq,kk]      int32 (BLOCK ids), axis=tpc(2,3)=0.
 *        guid "topk", ns_TopkNodeV2::ParamsV4{bsw=kk, axis=get_dim_in_tpc_order(last,
 *        rank), bottomK=false, kType=K_TENSOR_NONE}. Indices declared Int directly
 *        (no cast node -- the synapse topk CGUID emits device i32; declaring Long then
 *        casting lowers to the degenerate cast_i32_to_i32 the LAZY GC refuses, exactly
 *        the DSA fix dsa_indexer_backend_shared.h:282-299).
 *
 * Build CPU-only (vllm-gaudi:next, host g++):  bash build_msa_block_indexer.sh
 * RUN ON THE HPU BOX ONLY via tests/test_msa_block_indexer_op.py (on-card numeric
 * validation is a later GATED step; the cards are busy serving).
 */
#pragma once

// fmt (torch bundles fmtlib v12) must be header-only here: the compiled fmt::v12
// symbols our TU pulls via torch's fmt headers are NOT exported by any shipped
// torch/habana .so (hidden visibility), so they must be inlined. Guard to avoid a
// -Wmacro-redefined warning. VERBATIM rationale from the DSA shared header.
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- bridge headers (the SAME cloned bridge the DSA op uses, tag v1.24.0-1007) --
#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr, BuildNode, ReshapeHelper
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor, get_dim_in_tpc_order
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair{pt_t, syn_t}
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND
#include "backend/synapse_helpers/graph.h"

// --- torch + synapse params --------------------------------------------------
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"   // syn_type_* enums, HABANA_DIM_MAX, TSize
#include "synapse_common_types.hpp" // synGEMMParams
#include "perf_lib_layer_params.h"  // ns_TopkNodeV2::ParamsV4, ns_Reduction::ParamsV2, ns_MaskedFill::ParamsV2

#include <limits>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta. Shape/dtype depend on `mode` (stack arg 5):
//   mode==0 (DSA): top-k TOKEN indices [Sq, kk] int32, kk = min(index_topk, Sk).
//   mode==1 (MSA): top-k BLOCK indices [H, Sq, kk] int32, kk = min(index_topk,
//                  n_blocks), n_blocks = Sk / block_size.
// Used by the frontend (size the result) AND the backend (final topk output attr).
//
// Stack order == schema: (q, k, weights, cmask, index_topk, mode, block_size).
//   q          = stack[0]  [H, Sq, D]
//   k          = stack[1]  [H, Sk, D]
//   index_topk = stack[4]  int
//   mode       = stack[5]  int  (0 DSA / 1 MSA)
//   block_size = stack[6]  int
// =============================================================================
inline OutputMetaDataVector MsaBlockIndexerTopkMeta(const at::Stack& stack) {
  const at::Tensor& q = stack_tensor(stack, 0);  // [H, Sq, D]
  const at::Tensor& k = stack_tensor(stack, 1);  // [H, Sk, D]
  TORCH_CHECK(q.dim() == 3, "[msa_block_indexer] q must be 3D [H,Sq,D], got ", q.dim());
  TORCH_CHECK(k.dim() == 3, "[msa_block_indexer] k must be 3D [H,Sk,D], got ", k.dim());
  const int64_t H  = q.size(0);
  const int64_t Sq = q.size(1);
  const int64_t Sk = k.size(1);
  const int64_t topk = stack.at(4).toInt();
  const int64_t mode = stack.at(5).toInt();
  const int64_t block_size = stack.at(6).toInt();

  OutputMetaDataVector meta(1);
  if (mode == 0) {
    // DSA token-sparse (unchanged behavior).
    const int64_t kk = std::min(topk, Sk);
    meta[0].shape = {Sq, kk};            // top-k key (token) indices per query
  } else {
    // MSA block-sparse: per (GQA-group head, query) top-k BLOCK ids.
    TORCH_CHECK(block_size > 0, "[msa_block_indexer] MSA mode needs block_size>0");
    TORCH_CHECK(Sk % block_size == 0,
                "[msa_block_indexer] MSA needs Sk(", Sk, ") a multiple of block_size(",
                block_size, "); pad the key axis (and cmask) upstream");
    const int64_t n_blocks = Sk / block_size;
    const int64_t kk = std::min(topk, n_blocks);
    meta[0].shape = {H, Sq, kk};         // top-k BLOCK ids per (group, query)
  }
  meta[0].dtype = c10::ScalarType::Int;  // int32 (consume-side contract)
  return meta;
}

// =============================================================================
// The OpBackend: emits the indexer node-chain (see the file header). Each
// non-final BuildNode output is a graph-internal (non-persistent)
// synapse_helpers::tensor; its .get() synTensor feeds the next node -- that is
// the SRAM-resident edge (the proven MoE/DSA lever). Modeled VERBATIM on
// DsaIndexerTopk (dsa_indexer_backend_shared.h:160-318).
// =============================================================================
struct MsaBlockIndexerTopk : OpBackend {
  MsaBlockIndexerTopk(int device_id, c10::ScalarType scalar_type)
      // guid "batch_gemm" is a harmless placeholder (real guids set per-BuildNode
      // below); res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "batch_gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MsaBlockIndexerTopkMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MsaBlockIndexerTopk::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  StackGetter sg(this, stack, "MsaBlockIndexerTopk::AddNode");

  // ---- typed inputs (order == schema) ----
  auto q       = sg.getNextInput<TensorsPair>();    // bf16  [H, Sq, D]
  auto k       = sg.getNextInput<TensorsPair>();    // bf16  [H, Sk, D]
  auto weights = sg.getNextInput<TensorsPair>();    // bf16  [H, Sq, 1]
  auto cmask   = sg.getNextInput<TensorsPair>();    // bool  [Sq, Sk]  TRUE => drop
  const int64_t topk       = sg.getNextInput<int64_t>();  // index_topk
  const int64_t mode       = sg.getNextInput<int64_t>();  // 0 DSA / 1 MSA
  const int64_t block_size = sg.getNextInput<int64_t>();  // tokens/block (MSA)

  // ---- derive shapes (PyTorch row-major; last dim == FCD) ----
  const auto q_sizes = q.pt_t.sizes().vec();
  const auto k_sizes = k.pt_t.sizes().vec();
  TORCH_CHECK(q.pt_t.dim() == 3, "[msa_block_indexer] q must be 3D [H,Sq,D]");
  TORCH_CHECK(k.pt_t.dim() == 3, "[msa_block_indexer] k must be 3D [H,Sk,D]");
  const int64_t H  = q_sizes[0];
  const int64_t Sq = q_sizes[1];
  const int64_t D  = q_sizes[2];
  const int64_t Hk = k_sizes[0];
  const int64_t Sk = k_sizes[1];
  const int64_t Dk = k_sizes[2];
  TORCH_CHECK(Hk == H, "[msa_block_indexer] k head dim ", Hk, " != q head dim ", H,
              " (expand the shared index key to H index heads upstream)");
  TORCH_CHECK(Dk == D, "[msa_block_indexer] k head_dim ", Dk, " != q head_dim ", D);
  TORCH_CHECK(weights.pt_t.dim() == 3 && weights.pt_t.size(0) == H &&
                  weights.pt_t.size(1) == Sq && weights.pt_t.size(2) == 1,
              "[msa_block_indexer] weights must be [H,Sq,1]");
  TORCH_CHECK(cmask.pt_t.dim() == 2 && cmask.pt_t.size(0) == Sq &&
                  cmask.pt_t.size(1) == Sk,
              "[msa_block_indexer] cmask must be [Sq,Sk] bool");
  TORCH_CHECK(mode == 0 || mode == 1,
              "[msa_block_indexer] mode must be 0 (DSA) or 1 (MSA), got ", mode);

  const at::ScalarType bf16 = c10::ScalarType::BFloat16;

  // ===========================================================================
  // NODE 1 (BOTH modes) -- per-head Q.K^T: logits[H,Sq,Sk] = batch_gemm(q,k,b^T).
  // synGEMMParams{transpose_a=false, transpose_b=true} transposes k's last two dims
  // (k[H,Sk,D] -> k^T[H,D,Sk]) so out = q @ k^T = [H,Sq,Sk]. NodeOutputAttr.sizes is
  // a NON-OWNING IntArrayRef -> every shape MUST be a NAMED vector that outlives the
  // BuildNode call (the bridge's discipline).
  // ===========================================================================
  synGEMMParams qkt_params{/*transpose_a=*/false, /*transpose_b=*/true};
  std::vector<int64_t> logits_shape{H, Sq, Sk};
  auto logits = OpBackend::BuildNode(
      this, graph,
      {"batch_gemm",
       {q.syn_t, k.syn_t},
       {{logits_shape, bf16, std::nullopt}},  // NON-PERSISTENT
       &qkt_params, sizeof(qkt_params)});

  // The current "live" per-(head,query,key) score edge; the chain mutates it.
  synTensor cur = logits[0].get();
  // We must keep every intermediate synapse_helpers::tensor ALIVE until BuildNode
  // has consumed its .get() (the edge owns the device tensor). Hold them here.
  std::vector<sh::tensor> keepalive;
  keepalive.reserve(4);

  // ===========================================================================
  // NODE 2 (DSA ONLY) -- ReLU: relu[H,Sq,Sk] = relu_fwd(logits). (oracle DSA :66
  // clamp_(0).) MSA SKIPS this: the MSA index score is the RAW scaled dot product,
  // no activation (PAPER Eq.6 / §4 "exp-free selection", msa_reference.py:66-73).
  // ===========================================================================
  if (mode == 0) {
    auto relu = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("relu_fwd"sv, bf16),
         {cur},
         {{logits_shape, bf16, std::nullopt}}});
    cur = relu[0].get();
    keepalive.push_back(std::move(relu[0]));
  }

  // ===========================================================================
  // NODE 3 (BOTH modes) -- weight/scale: wl[H,Sq,Sk] = mult_fwd(cur, weights[H,Sq,1])
  // (broadcast over the Sk/FCD axis -- the PROVEN [H,Sq,1] mult broadcast). DSA: the
  // per-head weight of the weighted head-sum (oracle :67). MSA: the order-preserving
  // index scale 1/sqrt(d_idx) (a single positive scalar per (head,query); does NOT
  // change the per-block argmax / topk order; weights=ones is equally valid).
  // ===========================================================================
  {
    auto wl = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("mult_fwd"sv, bf16),
         {cur, weights.syn_t},
         {{logits_shape, bf16, std::nullopt}}});
    cur = wl[0].get();
    keepalive.push_back(std::move(wl[0]));
  }

  // ===========================================================================
  // NODE 4 (DSA ONLY) -- head-sum: score[Sq,Sk] = reduce_sum(wl, dim=0(H), keepDim=0).
  // ns_Reduction::ParamsV2{mask = 1<<(rank-d-1) = 1<<(3-0-1) = 4, keepDim=false}.
  // (oracle DSA :67.) MSA SKIPS this: the H_kv index heads stay SEPARATE (each GQA
  // group selects independently, PAPER Eq.7) -> the score keeps its [H,Sq,Sk] shape.
  // ===========================================================================
  // The shape the mask/blockpool/topk operate on next: DSA collapses H, MSA keeps it.
  std::vector<int64_t> dsa_score_shape{Sq, Sk};        // DSA post-head-sum
  if (mode == 0) {
    ns_Reduction::ParamsV2 red_params{};
    red_params.reductionDimensionMask = 1u << static_cast<unsigned>(3 - 0 - 1);  // dim 0 of rank 3
    red_params.keepDim = false;
    auto score = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("reduce_sum_multi_dim_fwd"sv, bf16),
         {cur},
         {{dsa_score_shape, bf16, std::nullopt}},
         &red_params, sizeof(red_params)});
    cur = score[0].get();
    keepalive.push_back(std::move(score[0]));
  }

  // ===========================================================================
  // NODE 5 (BOTH modes) -- causal mask: masked = masked_fill(cur, cmask, -inf).
  // ns_MaskedFill::ParamsV2{value.f = -inf} (the scalar path). cmask is bool [Sq,Sk],
  // TRUE where the key is non-causal (key_pos>abs_pos) -> -inf so top-k/block-max
  // never selects it (oracle DSA :71 / MSA msa_index_scores :237-240).
  //   DSA: self is [Sq,Sk] -> masked [Sq,Sk].
  //   MSA: self is [H,Sq,Sk]; cmask[Sq,Sk] broadcasts over the leading H axis
  //        (MaskedFillMeta uses at::infer_size(self, mask), masked_fill_gen.cpp:27 --
  //        [Sq,Sk] aligns to the trailing dims of [H,Sq,Sk]).
  // ===========================================================================
  ns_MaskedFill::ParamsV2 mf_params{};
  mf_params.value.f = -std::numeric_limits<float>::infinity();
  std::vector<int64_t> masked_shape = (mode == 0)
      ? std::vector<int64_t>{Sq, Sk}
      : std::vector<int64_t>{H, Sq, Sk};
  {
    auto masked = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("masked_fill_fwd"sv, bf16),
         {cur, cmask.syn_t},
         {{masked_shape, bf16, std::nullopt}},
         &mf_params, sizeof(mf_params)});
    cur = masked[0].get();
    keepalive.push_back(std::move(masked[0]));
  }

  if (mode == 0) {
    // =========================================================================
    // DSA FINAL -- top-k over keys: (vals,idx) = topk(masked[Sq,Sk], k=kk, axis,
    // largest). axis for the last dim (1) of a rank-2 tensor is tpc(1,2)=0. Indices
    // declared Int directly (no cast; the LAZY-GC-safe form, see header). (oracle :72.)
    // =========================================================================
    const int64_t kk = std::min(topk, Sk);
    TORCH_CHECK(kk >= 1, "[msa_block_indexer] DSA index_topk clamps to ", kk,
                " (<1); Sk=", Sk);
    ns_TopkNodeV2::ParamsV4 topk_params{};
    topk_params.bsw = static_cast<unsigned int>(kk);
    topk_params.axis = get_dim_in_tpc_order(/*dim=*/1, /*max_dims=*/2);  // last dim -> 0
    topk_params.bottomK = false;        // largest
    topk_params.isStable = false;
    topk_params.isVcData = false;
    topk_params.kType = K_TENSOR_NONE;  // static k
    std::vector<int64_t> topk_shape{Sq, kk};
    auto topk_out = OpBackend::BuildNode(
        this, graph,
        {"topk",
         {cur},
         {{topk_shape, bf16, std::nullopt},                 // values (discarded)
          {topk_shape, c10::ScalarType::Int, 0}},           // indices int32 == RESULT
         &topk_params, sizeof(topk_params)});
    syn_out(0) = std::move(topk_out[1]);
    return;
  }

  // ===========================================================================
  // MSA branch. masked is [H,Sq,Sk]. Block-max-pool then topk over n_blocks.
  // ===========================================================================
  TORCH_CHECK(block_size > 0, "[msa_block_indexer] MSA mode needs block_size>0");
  TORCH_CHECK(Sk % block_size == 0,
              "[msa_block_indexer] MSA needs Sk(", Sk, ") a multiple of block_size(",
              block_size, "); pad the key axis (and cmask) upstream so fully-padded "
              "blocks reduce-max to -inf");
  const int64_t n_blocks = Sk / block_size;

  // ---------------------------------------------------------------------------
  // NODE 6a (MSA) -- reshape [H,Sq,Sk] -> [H,Sq,n_blocks,block_size] (rank 4).
  // ReshapeHelper (op_backend.h:302). The Sk (FCD) axis splits into (n_blocks,
  // block_size); since Sk is contiguous row-major, block b owns the contiguous span
  // [b*block_size, (b+1)*block_size). NON-PERSISTENT.
  // ---------------------------------------------------------------------------
  std::vector<int64_t> scoreB_shape{H, Sq, n_blocks, block_size};
  auto scoreB = ReshapeHelper(graph, cur, scoreB_shape, bf16, std::nullopt);
  synTensor scoreB_t = scoreB.get();
  keepalive.push_back(std::move(scoreB));

  // ---------------------------------------------------------------------------
  // NODE 6b (MSA) -- block-max-pool: blkmax[H,Sq,n_blocks] =
  //   reduce_max_multi_dim_fwd(scoreB, dim=3 (block_size axis), keepDim=false).
  // guid reduce_max_multi_dim_fwd (amax/AminAmax lowering, amax_amin_gen.cpp:118 /
  // max_min_gen.cpp:125). ns_Reduction::ParamsV2 via FillReductionParams(rank=4,{3},
  // keepdim=false): reductionDimensionMask = 1<<(4-3-1) = 1<<0 = 1
  // (reduction_template.cpp:30-35). A fully -inf block (all members non-causal/pad)
  // reduce-maxes to -inf -> never wins the topk (PAPER msa_paper :230-231); a partial
  // last block scores by its REAL tokens only (pad cols are -inf). This is the MSA
  // analog of DSA's kk=min(topk,Sk) clamp / the -inf padding contract.
  // ---------------------------------------------------------------------------
  ns_Reduction::ParamsV2 bmax_params{};
  bmax_params.reductionDimensionMask = 1u << static_cast<unsigned>(4 - 3 - 1);  // dim 3 of rank 4 -> bit 0
  bmax_params.keepDim = false;
  std::vector<int64_t> blkmax_shape{H, Sq, n_blocks};
  auto blkmax = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("reduce_max_multi_dim_fwd"sv, bf16),
       {scoreB_t},
       {{blkmax_shape, bf16, std::nullopt}},
       &bmax_params, sizeof(bmax_params)});
  cur = blkmax[0].get();
  keepalive.push_back(std::move(blkmax[0]));

  // ---------------------------------------------------------------------------
  // NODE 7 (MSA FINAL) -- top-k over BLOCKS: (vals,bid) = topk(blkmax[H,Sq,n_blocks],
  // k=kk, axis, largest). kk = min(topk_blocks, n_blocks). axis for the last dim (2)
  // of a rank-3 tensor is tpc(2,3)=0. Output bid[H,Sq,kk] int32 == selected BLOCK ids
  // per (GQA-group head, query). Indices declared Int directly (LAZY-GC-safe; header).
  // (oracle msa_block_select :303-304 .topk(k).indices over the block axis.)
  // ---------------------------------------------------------------------------
  const int64_t kk = std::min(topk, n_blocks);
  TORCH_CHECK(kk >= 1, "[msa_block_indexer] MSA topk_blocks clamps to ", kk,
              " (<1); n_blocks=", n_blocks);
  ns_TopkNodeV2::ParamsV4 topk_params{};
  topk_params.bsw = static_cast<unsigned int>(kk);
  topk_params.axis = get_dim_in_tpc_order(/*dim=*/2, /*max_dims=*/3);  // last dim of rank3 -> 0
  topk_params.bottomK = false;        // largest
  topk_params.isStable = false;
  topk_params.isVcData = false;
  topk_params.kType = K_TENSOR_NONE;  // static k
  std::vector<int64_t> topk_shape{H, Sq, kk};
  auto topk_out = OpBackend::BuildNode(
      this, graph,
      {"topk",
       {cur},
       {{topk_shape, bf16, std::nullopt},                 // values (discarded)
        {topk_shape, c10::ScalarType::Int, 0}},           // BLOCK ids int32 == RESULT
       &topk_params, sizeof(topk_params)});
  syn_out(0) = std::move(topk_out[1]);
}

} // namespace habana
