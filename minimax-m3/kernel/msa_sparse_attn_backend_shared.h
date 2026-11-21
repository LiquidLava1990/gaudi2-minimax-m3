/*
 * msa_sparse_attn_backend_shared.h -- the SHARED, frontend-agnostic backend lowering
 * for the MiniMax-M3 MSA GATHER-CONSUME sparse attention as a GC-captured
 * `habana::OpBackend` op (`torch.ops.hpu_msa_ext::sparse_attn`).
 *
 * THIS GRAPH IS BYTE-IDENTICAL to the validated DSA consume op
 * ($HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/dsa/dsa_sparse_attn_backend_shared.h):
 * gather K/V at the selected token positions -> SDPA over the gathered width with a
 * pad-only additive mask. It is copied here UNCHANGED (only the op namespace and the
 * struct/meta NAMES are MSA-prefixed, so it registers in its OWN `hpu_msa_ext`
 * KernelRegistry slot and NEVER collides with the installed/serving
 * `hpu_dsa_ext::sparse_attn`). Copying it makes the M3 kernel self-contained.
 *
 * WHY NO GRAPH CHANGE FOR MSA (vs DSA): the MSA block->token expansion is done
 * UPSTREAM in Python (ref/msa_reference.py::msa_expand_blocks_to_tokens: each
 * selected block id b -> its block_size consecutive token positions [b*B,(b+1)*B),
 * -1 for pad/out-of-range). The op is then fed `token_idx[T, K]` with
 * K = topk_blocks*block_size and gather_width=K, and real GQA K/V already expanded
 * to H query heads upstream (head h reads kv-head h//G -- the SAME upstream
 * expand DSA does for its single MQA key). Real (uncompressed) K/V means NO MLA
 * latent absorption -- strictly simpler than DSA on the consume side; the op already
 * consumes per-head real K/V, nothing to add. So MSA consume == this op AS-IS.
 *
 * NOTE on per-(query,GQA-group) selection: MSA selects blocks per GQA group (H_kv
 * sets), and all G heads in a group share that block set. The runner expands the
 * per-group token_idx to per-(query-head) rows OR, equivalently, runs this op per
 * group; either way the op sees `token_idx[T,K]` aligned to the H query heads it is
 * given (q[T,H,1,Dh]). The token_idx contract (>=0 real / -1 pad, causal already in
 * the selection) is identical to DSA's.
 *
 * ============================ THE ORACLE MATH MATCHED ========================
 * ref/msa_reference.py::msa_gqa_gather_attention :346-409 (the PRODUCT gather form,
 * the consume oracle the kernel diffs). PAPER Eq.8 (scaled-dot GQA softmax over the
 * gathered tokens, causal on ABSOLUTE positions, SENTINEL row for -1/OOB, NaN guard).
 * Identical discipline to DSA's sparse_mla_gather_attention.
 *
 *     valid[t,j]   = token_idx[t,j] >= 0  (& < S)                  # pad/OOB detector
 *     gid[t,j]     = valid ? token_idx[t,j] : S    (sentinel row S)# never clamp(0)
 *     k_g[t,j]     = k_ext[gid[t,j]]   (k_ext = [k ; zero_row])    # gather
 *     v_g[t,j]     = v_ext[gid[t,j]]
 *     logits[h,t,j]= (q[t,h,:] . k_g[t,j,h,:]) * scale
 *     logits      += where(valid, 0, -inf)                         # pad-only additive bias
 *     attn         = softmax(logits) ; attn = nan_to_num(attn,0)
 *     out[t,h,:]   = sum_j attn[h,t,j] * v_g[t,j,h,:]
 *
 * Causality is ALREADY in the selected token set (the block indexer pre-masked the
 * non-causal keys before its top-k, AND msa_expand_blocks_to_tokens -1's any token
 * position >= seqlen / from a pad block), so we mask ONLY the pad/sentinel slots here
 * -- exactly DSA's "causality already in the gather geometry; mask only the pad slots".
 *
 * `scale` is the MSA main softmax scale (head_dim**-0.5, PAPER Eq.8). NOTHING is
 * hardcoded: T/H/Dh/Dv/K come from the input tensor shapes.
 *
 * The full per-node GUID/param documentation (greater_equal_fwd -> where_fwd ->
 * where_fwd+reshape -> concat -> gather_fwd -> permute -> sdpa_fwd -> reshape) is the
 * SAME as the DSA consume header; see dsa_sparse_attn_backend_shared.h:61-118 for the
 * node-by-node citations. The code below is that chain, MSA-named.
 *
 * Build CPU-only (vllm-gaudi:next, host g++):  bash build_msa_sparse_attn.sh
 * RUN ON THE HPU BOX ONLY via tests/test_msa_sparse_attn_op.py (later, gated).
 */
#pragma once

// fmt header-only (see the indexer/MoE shared headers for the full rationale).
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- bridge headers (the SAME cloned bridge the DSA op uses, tag v1.24.0-1007) --
#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr, BuildNode, ConstantHelper, *Helper
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor, get_dim_in_tpc_order
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair{pt_t, syn_t}
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND
#include "backend/synapse_helpers/graph.h"

// --- torch + synapse params --------------------------------------------------
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"   // syn_type_* enums, HABANA_DIM_MAX, TSize
#include "synapse_common_types.hpp" // synGEMMParams, synConcatenateParams
#include "perf_lib_layer_params.h"  // ns_GatherKernel::Params, ns_Sdpa::ParamsV7, SdpaSoftmaxMode_t

#include <limits>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta: the op returns the sparse attention output, shape [T, H, Dv] bf16.
// Stack order == schema: (q, k, v, token_idx, scale, gather_width).
//   q = stack[0]  [T, H, 1, Dh] ; v = stack[2]  [S, H, Dv]
// =============================================================================
inline OutputMetaDataVector MsaSparseAttnMeta(const at::Stack& stack) {
  const at::Tensor& q = stack_tensor(stack, 0);  // [T, H, 1, Dh]
  const at::Tensor& v = stack_tensor(stack, 2);  // [S, H, Dv]
  TORCH_CHECK(q.dim() == 4, "[msa_sparse_attn] q must be 4D [T,H,1,Dh], got ", q.dim());
  TORCH_CHECK(v.dim() == 3, "[msa_sparse_attn] v must be 3D [S,H,Dv], got ", v.dim());
  const int64_t T = q.size(0);
  const int64_t H = q.size(1);
  const int64_t Dv = v.size(2);

  OutputMetaDataVector meta(1);
  meta[0].shape = {T, H, Dv};                 // sparse attention output, per head
  meta[0].dtype = q.scalar_type();            // bf16 (== q dtype, the consume contract)
  return meta;
}

// =============================================================================
// The OpBackend: emits the gather-consume node-chain (byte-identical to DsaSparseAttn,
// MSA-named so it owns its own KernelRegistry slot).
// =============================================================================
struct MsaSparseAttn : OpBackend {
  MsaSparseAttn(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "sdpa_fwd", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MsaSparseAttnMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MsaSparseAttn::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  StackGetter sg(this, stack, "MsaSparseAttn::AddNode");

  // ---- typed inputs (order == schema) ----
  auto q        = sg.getNextInput<TensorsPair>();  // bf16  [T, H, 1, Dh]
  auto k        = sg.getNextInput<TensorsPair>();  // bf16  [S, H, Dh]   (GQA-expanded to H upstream)
  auto v        = sg.getNextInput<TensorsPair>();  // bf16  [S, H, Dv]
  auto topk_idx = sg.getNextInput<TensorsPair>();  // int32 [T, K]       token ids (-1 padded)
  const double scale = sg.getNextInput<double>();  // MSA main softmax scale (head_dim**-0.5)
  const int64_t gather_width = sg.getNextInput<int64_t>();  // == K (topk_blocks*block_size)

  // ---- derive shapes (PyTorch row-major; last dim == FCD) ----
  const auto q_sizes = q.pt_t.sizes().vec();
  const auto k_sizes = k.pt_t.sizes().vec();
  const auto v_sizes = v.pt_t.sizes().vec();
  const auto ti_sizes = topk_idx.pt_t.sizes().vec();
  TORCH_CHECK(q.pt_t.dim() == 4, "[msa_sparse_attn] q must be 4D [T,H,1,Dh]");
  TORCH_CHECK(k.pt_t.dim() == 3, "[msa_sparse_attn] k must be 3D [S,H,Dh]");
  TORCH_CHECK(v.pt_t.dim() == 3, "[msa_sparse_attn] v must be 3D [S,H,Dv]");
  TORCH_CHECK(topk_idx.pt_t.dim() == 2, "[msa_sparse_attn] topk_idx must be 2D [T,K]");
  const int64_t T  = q_sizes[0];
  const int64_t H  = q_sizes[1];
  const int64_t L  = q_sizes[2];   // target-seq per query == 1
  const int64_t Dh = q_sizes[3];
  const int64_t S  = k_sizes[0];
  const int64_t Hk = k_sizes[1];
  const int64_t Dhk = k_sizes[2];
  const int64_t Sv = v_sizes[0];
  const int64_t Hv = v_sizes[1];
  const int64_t Dv = v_sizes[2];
  const int64_t Tt = ti_sizes[0];
  const int64_t K  = ti_sizes[1];
  TORCH_CHECK(L == 1, "[msa_sparse_attn] q target-seq dim 2 must be 1 (per-query), got ", L);
  TORCH_CHECK(Hk == H && Hv == H, "[msa_sparse_attn] head dim mismatch: qH=", H,
              " kH=", Hk, " vH=", Hv, " (expand GQA KV to H query heads upstream)");
  TORCH_CHECK(Dhk == Dh, "[msa_sparse_attn] k head_dim ", Dhk, " != q head_dim ", Dh);
  TORCH_CHECK(Sv == S, "[msa_sparse_attn] v seq ", Sv, " != k seq ", S);
  TORCH_CHECK(Tt == T, "[msa_sparse_attn] topk_idx T ", Tt, " != q T ", T);
  TORCH_CHECK(gather_width == K, "[msa_sparse_attn] gather_width ", gather_width,
              " must equal the token-idx width K=", K,
              " (K == topk_blocks*block_size; the buffer is already the bucket)");
  TORCH_CHECK(K >= 1, "[msa_sparse_attn] K (token_idx width) must be >= 1, got ", K);

  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType i32  = c10::ScalarType::Int;
  // Additive pad-mask "-inf": most-negative FINITE bf16 (NOT true -inf), same as the
  // DSA consume (bridge nantonum bf16 neg-inf / SDPA test LNEG). softmax(fp32) of
  // (logit + bf16::lowest) -> 0 exactly == the oracle's true-(-inf) result.
  const at::Scalar NEG_BIG = std::numeric_limits<c10::BFloat16>::lowest();

  // NODE 1 -- pad detector: valid[T,K] bool = (token_idx >= 0).
  std::vector<int64_t> tk_shape{T, K};
  auto zero_i32 = ConstantHelper(graph, 0, i32, tk_shape);   // [T,K] i32 zeros
  auto valid = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("greater_equal_fwd"sv, i32),
       {topk_idx.syn_t, zero_i32.get()},
       {{tk_shape, c10::ScalarType::Bool, std::nullopt}}});

  // NODE 2 -- gather id: gid[T,K] i32 = where(valid, token_idx, S) (sentinel row S).
  auto const_S = ConstantHelper(graph, static_cast<int>(S), i32, tk_shape);
  auto gid = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("where_fwd"sv, i32),
       {valid[0].get(), topk_idx.syn_t, const_S.get()},
       {{tk_shape, i32, std::nullopt}}});

  // NODE 3 -- pad bias: pad_bias[T,K] bf16 = where(valid,0,-inf) -> reshape [T,1,1,K].
  auto zero_bf16 = ConstantHelper(graph, 0.0, bf16, tk_shape);
  auto neg_bf16  = ConstantHelper(graph, NEG_BIG, bf16, tk_shape);
  auto pad_bias = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("where_fwd"sv, bf16),
       {valid[0].get(), zero_bf16.get(), neg_bf16.get()},
       {{tk_shape, bf16, std::nullopt}}});
  std::vector<int64_t> mask_shape{T, 1, 1, K};
  auto mask4 = ReshapeHelper(graph, pad_bias[0].get(), mask_shape, bf16);

  // NODE 4 -- append SENTINEL trash row: k_ext[S+1,H,Dh], v_ext[S+1,H,Dv].
  synConcatenateParams cat_params{};
  cat_params.axis = static_cast<unsigned>(get_dim_in_tpc_order(/*dim=*/0, /*max_dims=*/3));
  std::vector<int64_t> zrowk_shape{1, H, Dh};
  std::vector<int64_t> zrowv_shape{1, H, Dv};
  auto zrow_k = ConstantHelper(graph, 0.0, bf16, zrowk_shape);
  auto zrow_v = ConstantHelper(graph, 0.0, bf16, zrowv_shape);
  std::vector<int64_t> kext_shape{S + 1, H, Dh};
  std::vector<int64_t> vext_shape{S + 1, H, Dv};
  auto k_ext = OpBackend::BuildNode(
      this, graph,
      {"concat", {k.syn_t, zrow_k.get()}, {{kext_shape, bf16, std::nullopt}},
       &cat_params, sizeof(cat_params)});
  auto v_ext = OpBackend::BuildNode(
      this, graph,
      {"concat", {v.syn_t, zrow_v.get()}, {{vext_shape, bf16, std::nullopt}},
       &cat_params, sizeof(cat_params)});

  // NODE 5 -- row gather (index_select on the flattened key axis).
  ns_GatherKernel::Params gather_params{};
  gather_params.axis = get_dim_in_tpc_order(/*dim=*/0, /*max_dims=*/2);  // 2-0-1 = 1
  std::vector<int64_t> gidflat_shape{T * K};
  auto gid_flat = ReshapeHelper(graph, gid[0].get(), gidflat_shape, i32);

  std::vector<int64_t> kflat_shape{S + 1, H * Dh};
  auto k_flat = ReshapeHelper(graph, k_ext[0].get(), kflat_shape, bf16);
  std::vector<int64_t> kgflat_shape{T * K, H * Dh};
  auto k_g_flat = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("gather_fwd"sv, bf16),
       {k_flat.get(), gid_flat.get()},
       {{kgflat_shape, bf16, std::nullopt}},
       &gather_params, sizeof(gather_params)});
  std::vector<int64_t> kg_shape{T, K, H, Dh};
  auto k_g = ReshapeHelper(graph, k_g_flat[0].get(), kg_shape, bf16);

  std::vector<int64_t> vflat_shape{S + 1, H * Dv};
  auto v_flat = ReshapeHelper(graph, v_ext[0].get(), vflat_shape, bf16);
  std::vector<int64_t> vgflat_shape{T * K, H * Dv};
  auto v_g_flat = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("gather_fwd"sv, bf16),
       {v_flat.get(), gid_flat.get()},
       {{vgflat_shape, bf16, std::nullopt}},
       &gather_params, sizeof(gather_params)});
  std::vector<int64_t> vg_shape{T, K, H, Dv};
  auto v_g = ReshapeHelper(graph, v_g_flat[0].get(), vg_shape, bf16);

  // NODE 6 -- permute K/V to SDPA [B=T,H,K,D] layout (perm (0,2,1,3); pass INPUT shape).
  std::vector<int64_t> kg_in_shape{T, K, H, Dh};
  std::vector<int64_t> vg_in_shape{T, K, H, Dv};
  std::vector<int64_t> perm0213{0, 2, 1, 3};
  auto k4 = PermuteHelper(graph, k_g.get(), kg_in_shape, perm0213, bf16);
  auto v4 = PermuteHelper(graph, v_g.get(), vg_in_shape, perm0213, bf16);

  // NODE 7 -- fused sparse SDPA: out4[T,H,1,Dv] = sdpa_fwd(q4,k4,v4,mask4).
  ns_Sdpa::ParamsV7 sdpa_params{};
  sdpa_params.scale = static_cast<float>(scale);
  sdpa_params.dropout.ratio = 0.0f;
  sdpa_params.is_causal = false;        // causality already in the selected token set
  sdpa_params.is_inference = true;
  sdpa_params.softmax_mode = SdpaSoftmaxMode_t::SDPA_FP32_SOFTMAX;
  sdpa_params.flags = 0;
  sdpa_params.wl = -1;
  sdpa_params.wr = -1;

  std::vector<int64_t> out4_shape{T, H, L, Dv};   // [T,H,1,Dv]
  std::vector<int64_t> qkt_shape{T, H, L, K};     // [T,H,1,K] (softmax stats, discarded)

  std::vector<synTensor> sdpa_inputs = {
      q.syn_t, k4.get(), v4.get(),
      mask4.get(),     // attention_mask (pad-only additive bias)
      nullptr};        // seed (no dropout)
  sdpa_inputs.insert(sdpa_inputs.end(), 7, nullptr);  // 7 trailing optionals

  auto out4 = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("sdpa_fwd"sv, bf16),
       std::move(sdpa_inputs),
       {{out4_shape, bf16, std::nullopt},
        {qkt_shape, bf16, std::nullopt}},
       &sdpa_params, sizeof(sdpa_params)});

  // NODE 8 -- drop L=1: out[T,H,Dv] = reshape(out4[T,H,1,Dv]) == RESULT.
  std::vector<int64_t> out_shape{T, H, Dv};
  auto out = ReshapeHelper(graph, out4[0].get(), out_shape, bf16, /*final_result_index=*/0);
  syn_out(0) = std::move(out);
}

} // namespace habana
