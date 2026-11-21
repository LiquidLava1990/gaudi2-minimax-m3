/*
 * kv_rotate_quant_backend_shared.h -- the SHARED, frontend-agnostic backend lowering
 * for the MiniMax-M3 sub-fp8 KV-cache TurboQuant *KV-WRITE* op as a GC-captured
 * `habana::OpBackend` op (`torch.ops.hpu_kvq_ext::kv_rotate_quant`).
 *
 * SEPARATE PACKAGE, OWN NAMESPACE, OWN .so. This registers `hpu_kvq_ext` and builds
 * libkv_rotate_quant{,_lazy}.so. It does NOT collide with -- and never rebuilds --
 * the installed/serving hpu_dsa_ext / hpu_moe_ext / hpu_msa_ext ops. Nothing in /opt
 * or vllm-active is touched. Modeled VERBATIM on the proven MoE/DSA/MSA OpBackend
 * pattern (a `struct : OpBackend` whose AddNode() chains `OpBackend::BuildNode({guid,
 * {syn_inputs}, {output_attrs}, &params, size})` calls in PyTorch ROW-MAJOR order,
 * last dim == FCD), so the eager TU (kv_rotate_quant_backend.cpp, EagerOp) and the lazy
 * TU (kv_rotate_quant_backend_lazy.cpp, LazyOp) reuse this byte-for-byte.
 *
 * ============================ THE METHOD (TurboQuant) ========================
 * TurboQuant (Google, arXiv 2504.19874; vLLM PR #38479): apply a calibration-free
 * random rotation (Walsh-Hadamard, the structured stand-in for the paper's Haar
 * rotation) PER HEAD to K and V to spread per-coordinate energy, THEN scalar low-bit
 * quantize. The WHT is its own inverse (H_n = WHT/sqrt(n), H_n @ H_n == I), so the
 * read-side inverse-rotation is the SAME matrix (cheap; see the read op). The matrix is
 * data-OBLIVIOUS -- one [D,D] constant shared by K, V and all heads -- so NO calibration
 * pass and NO per-model weights. Primary config k8v4 (K fp8 / V int4 = the upstream-
 * blessed safe knee); a 2-bit "lightest" V mode (k8v2) is also supported for the 1M-fit
 * case (review quant-kv-turboquant.md rejects 2-bit values for quality, so it is built +
 * measured, never default).
 *
 * The numerically-authoritative spec this chain mirrors STEP-FOR-STEP is
 * ref/turboquant_kv_ref.py (kv_rotate_quant). The on-card test diffs the device op vs
 * that reference; the CPU test diffs the reference vs fp8-KV.
 *
 * ============================ NODE-CHAIN (GUIDs + params) ====================
 * Inputs (order == schema; PyTorch row-major, last dim == FCD == head_dim D):
 *   k       [N, D]   bf16   K vectors, ROW == one (token,head) pair, D == head_dim
 *                           (caller flattens [T,H,D] -> [T*H, D]; nothing model-specific)
 *   v       [N, D]   bf16   V vectors, same flattening
 *   rot     [D, D]   bf16   the shared Walsh-Hadamard rotation R (R@R^T==I), a CONSTANT
 *                           the caller builds once (ref.make_rotation) and reuses; passed
 *                           as a tensor so the op is rotation-agnostic (any orthogonal R)
 *   v_bits           int    4 (k8v4) or 2 (k8v2) -- selects the V quant width + pack
 *
 * Outputs (4 tensors; res_ids order):
 *   0 k_fp8   [N, D]      float8_e4m3fn   rotated K, fp8
 *   1 k_scale [N, 1]      float32         per-row K scale (absmax/FP8_E4M3_MAX)
 *   2 v_packed[N, D/8] (b=4) | [N, D/16] (b=2)  int32   rotated V, signed-int packed
 *   3 v_scale [N, 1]      float32         per-row V scale (absmax/qmax)
 *
 *  --- ROTATE (both K and V) ---
 *  1. k_rot = gemm(k[N,D], rot[D,D], {transpose_a=0, transpose_b=0})  -> [N,D] bf16
 *  2. v_rot = gemm(v[N,D], rot[D,D], {transpose_a=0, transpose_b=0})  -> [N,D] bf16
 *        guid "gemm", synGEMMParams{false,false}. NON-PERSISTENT. (PyTorch-order
 *        self[M,K]@other[K,N]; here [N,D]@[D,D]=[N,D] -- x @ R == ref's rotate.)
 *
 *  --- K QUANT (-> fp8 e4m3, per-row scale) ---
 *  3. k_abs   = abs_fwd(k_rot)                                        -> [N,D]
 *  4. k_amax  = reduce_max_multi_dim_fwd(k_abs, dim=1, keepDim=true)  -> [N,1]
 *        ns_Reduction::ParamsV2 mask=1<<(2-1-1)=1<<0=1, keepDim=true.
 *  5. k_scale = mult_fwd(k_amax, const(1/FP8_E4M3_MAX))               -> [N,1] (== RESULT 1)
 *        (scale = amax / FP8_E4M3_MAX ; clamp_min handled by the eps add below.)
 *  6. k_rscale= reciprocal_fwd(k_scale)                              -> [N,1]
 *  7. k_fp8   = convert_to_fp8(k_rot, k_rscale)  [e4m3, ROUND_HALF_NE] -> [N,D] fp8 (== RESULT 0)
 *        guid get_guid_with_precision("convert_to_fp8", bf16); the bridge convert_to_fp8
 *        multiplies by the (reciprocal) scale internally and clips to e4m3 range
 *        (fp8_ops.cpp:153 / CAST_CLIP). Per-row scale broadcasts over D.
 *
 *  --- V QUANT (-> signed int b-bit, per-row scale, packed) ---
 *  8.  v_abs   = abs_fwd(v_rot)                                       -> [N,D]
 *  9.  v_amax  = reduce_max_multi_dim_fwd(v_abs, dim=1, keepDim=true) -> [N,1]
 *  10. v_scale = mult_fwd(v_amax, const(1/qmax))                      -> [N,1] (== RESULT 3)
 *  11. v_rscale= reciprocal_fwd(v_scale)                             -> [N,1]
 *  12. v_div   = mult_fwd(v_rot, v_rscale (bcast D))                  -> [N,D]
 *  13. v_round = round_fwd(v_div)  [ROUND_HALF_NEAREST_EVEN]          -> [N,D]
 *  14. v_clip  = clamp(v_round, -qmax, +qmax)                        -> [N,D]
 *        ns_ClampKernel::Params{lowerBound=-qmax, upperBound=+qmax}. Symmetric range
 *        (the -2^(b-1) code is unused) so the dequant is exactly symmetric.
 *  15. v_i8    = cast_<bf16>_to_i8(v_clip)                            -> [N,D] int8
 *  16. v_packed= PACK(v_i8): pack 8 (b=4) / 16 (b=2) signed codes per int32 lane via
 *        bitwise_and(0xF|0x3) -> bitshift -> reshape-reduce-or. (== RESULT 2)
 *        (See PackInt4/PackInt2 below -- the bridge has bitwise_and_fwd / bitshift_fwd /
 *        bitwise_or_fwd / reshape; this composes them, no ISA kernel.)
 *
 * qmax = (1<<(v_bits-1))-1  (4b->7, 2b->1). FP8_E4M3_MAX = 448.
 *
 * NOTHING is hardcoded to a model: N/D come from the input shapes; v_bits is a scalar
 * arg; rot is a tensor. M3 GQA (H_kv=4, head_dim=128 -> rot is 128x128) and any other
 * dims "just work".
 *
 * Build CPU-only (vllm-gaudi:next, host g++):  bash build_kv_rotate_quant.sh
 * RUN ON THE HPU BOX ONLY via tests/test_turboquant_kv_op.py (on-card numeric
 * validation is a later GATED step; the cards are busy serving).
 */
#pragma once

// fmt header-only (same rationale as the MoE/DSA/MSA shared headers: torch's bundled
// fmt::v12 symbols are hidden-visibility in every shipped .so, so inline them).
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- bridge headers (the SAME cloned bridge the MoE/DSA/MSA ops use, tag v1.24.0-1007) --
#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr, BuildNode, ReshapeHelper, ConstantHelper
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor, get_dim_in_tpc_order
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair{pt_t, syn_t}
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND
#include "backend/synapse_helpers/graph.h"

// --- torch + synapse params --------------------------------------------------
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"   // syn_type_* enums, TSize
#include "synapse_common_types.hpp" // synGEMMParams
#include "perf_lib_layer_params.h"  // ns_ConvertToFp8::ParamsV2, ns_Reduction::ParamsV2,
                                    // ns_RoundKernel::Params, ns_ClampKernel::Params,
                                    // ns_BitShiftKernel (bitshift), RoundMode_t, CAST_ROUND_*

#include <limits>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

inline constexpr float KVQ_FP8_E4M3_MAX = 448.0f;

// =============================================================================
// Output meta. Stack order == schema: (k, v, rot, v_bits).
//   k = stack[0] [N,D] ; v_bits = stack[3].
// Four outputs: k_fp8 [N,D] e4m3, k_scale [N,1] f32, v_packed [N,D/L] i32, v_scale [N,1] f32
//   where L = 8 (v_bits==4) or 16 (v_bits==2).
// =============================================================================
inline OutputMetaDataVector KvRotateQuantMeta(const at::Stack& stack) {
  const at::Tensor& k = stack_tensor(stack, 0);  // [N,D]
  TORCH_CHECK(k.dim() == 2, "[kv_rotate_quant] k must be 2D [N,D], got ", k.dim());
  const int64_t N = k.size(0);
  const int64_t D = k.size(1);
  const int64_t v_bits = stack.at(3).toInt();
  TORCH_CHECK(v_bits == 4 || v_bits == 2,
              "[kv_rotate_quant] v_bits must be 4 (k8v4) or 2 (k8v2), got ", v_bits);
  const int64_t per_lane = (v_bits == 4) ? 8 : 16;
  TORCH_CHECK(D % per_lane == 0,
              "[kv_rotate_quant] head_dim D=", D, " must be a multiple of ", per_lane,
              " for ", v_bits, "-bit packing");
  const int64_t Lv = D / per_lane;

  OutputMetaDataVector meta(4);
  meta[0].shape = {N, D};   meta[0].dtype = c10::ScalarType::Float8_e4m3fn;  // k_fp8
  meta[1].shape = {N, 1};   meta[1].dtype = c10::ScalarType::Float;          // k_scale
  meta[2].shape = {N, Lv};  meta[2].dtype = c10::ScalarType::Int;            // v_packed (int32 lanes)
  meta[3].shape = {N, 1};   meta[3].dtype = c10::ScalarType::Float;          // v_scale
  return meta;
}

// =============================================================================
// The OpBackend: emits the rotate->quant->pack node-chain. Each non-final BuildNode
// output is a graph-internal (non-persistent) synapse_helpers::tensor whose .get()
// feeds the next node -- the SRAM-resident edge (the proven MoE/DSA/MSA lever).
// =============================================================================
struct KvRotateQuant : OpBackend {
  KvRotateQuant(int device_id, c10::ScalarType scalar_type)
      // guid "gemm" is a harmless placeholder (real guids set per-BuildNode); res_ids
      // marks all 4 outputs as results; not out/inplace.
      : OpBackend(device_id, "gemm", scalar_type, {0, 1, 2, 3}, {}, {}, false) {
    SetOutputMetaFn(KvRotateQuantMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;

 private:
  // Pack signed b-bit codes (held as bf16 in [-qmax,qmax]) into int32 lanes. Composes
  // cast->and->shift->reshape->reduce-or from bridge GUIDs; returns the packed [N,Lv] i32
  // synapse tensor and pushes its keepalive intermediates.
  sh::tensor PackSubByte(sh::graph& graph, synTensor codes_bf16, int64_t N, int64_t D,
                         int64_t v_bits, std::vector<sh::tensor>& keepalive,
                         int final_result_index);
};

inline void KvRotateQuant::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  StackGetter sg(this, stack, "KvRotateQuant::AddNode");

  auto k   = sg.getNextInput<TensorsPair>();          // bf16  [N,D]
  auto v   = sg.getNextInput<TensorsPair>();          // bf16  [N,D]
  auto rot = sg.getNextInput<TensorsPair>();          // bf16  [D,D]
  const int64_t v_bits = sg.getNextInput<int64_t>();  // 4 or 2

  const auto k_sizes = k.pt_t.sizes().vec();
  const auto r_sizes = rot.pt_t.sizes().vec();
  TORCH_CHECK(k.pt_t.dim() == 2, "[kv_rotate_quant] k must be 2D [N,D]");
  TORCH_CHECK(v.pt_t.dim() == 2, "[kv_rotate_quant] v must be 2D [N,D]");
  TORCH_CHECK(rot.pt_t.dim() == 2, "[kv_rotate_quant] rot must be 2D [D,D]");
  const int64_t N = k_sizes[0];
  const int64_t D = k_sizes[1];
  TORCH_CHECK(r_sizes[0] == D && r_sizes[1] == D,
              "[kv_rotate_quant] rot must be [D,D]=[", D, ",", D, "], got [",
              r_sizes[0], ",", r_sizes[1], "]");
  TORCH_CHECK(v.pt_t.size(0) == N && v.pt_t.size(1) == D,
              "[kv_rotate_quant] v must match k shape [N,D]");
  TORCH_CHECK(v_bits == 4 || v_bits == 2,
              "[kv_rotate_quant] v_bits must be 4 or 2, got ", v_bits);

  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType f32  = c10::ScalarType::Float;
  const float qmax = static_cast<float>((1 << (v_bits - 1)) - 1);  // 4b->7, 2b->1

  std::vector<sh::tensor> keepalive;
  keepalive.reserve(24);

  // ===========================================================================
  // ROTATE -- k_rot = k @ rot, v_rot = v @ rot. gemm(self[N,D], other[D,D]) with
  // synGEMMParams{false,false} == PyTorch [N,D]@[D,D] = [N,D] (the ref's x @ R).
  // ===========================================================================
  synGEMMParams rot_params{/*transpose_a=*/false, /*transpose_b=*/false};
  std::vector<int64_t> nd_shape{N, D};
  auto k_rot = OpBackend::BuildNode(
      this, graph, {"gemm", {k.syn_t, rot.syn_t}, {{nd_shape, bf16, std::nullopt}},
                    &rot_params, sizeof(rot_params)});
  synTensor k_rot_t = k_rot[0].get();
  keepalive.push_back(std::move(k_rot[0]));
  auto v_rot = OpBackend::BuildNode(
      this, graph, {"gemm", {v.syn_t, rot.syn_t}, {{nd_shape, bf16, std::nullopt}},
                    &rot_params, sizeof(rot_params)});
  synTensor v_rot_t = v_rot[0].get();
  keepalive.push_back(std::move(v_rot[0]));

  // reduce_max over the FCD (dim 1 of rank 2), keepDim -> [N,1].
  ns_Reduction::ParamsV2 amax_params{};
  amax_params.reductionDimensionMask = 1u << static_cast<unsigned>(2 - 1 - 1);  // dim1/rank2 -> bit0
  amax_params.keepDim = true;
  std::vector<int64_t> n1_shape{N, 1};

  // ===========================================================================
  // K QUANT -> fp8 e4m3. scale = absmax(k_rot)/MAX ; k_fp8 = convert_to_fp8(k_rot, 1/scale).
  // ===========================================================================
  auto k_abs = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("abs_fwd"sv, bf16), {k_rot_t},
                    {{nd_shape, bf16, std::nullopt}}});
  keepalive.push_back(std::move(k_abs[0]));
  auto k_amax = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("reduce_max_multi_dim_fwd"sv, bf16),
                    {keepalive.back().get()},
                    {{n1_shape, bf16, std::nullopt}}, &amax_params, sizeof(amax_params)});
  keepalive.push_back(std::move(k_amax[0]));
  // scale (f32) = amax * (1/MAX). Cast amax bf16->f32 implicitly via the f32 const mult.
  auto inv_fp8max = ConstantHelper(graph, 1.0f / KVQ_FP8_E4M3_MAX, f32, n1_shape);
  // RESULT 1 (k_scale, f32 [N,1]).
  auto k_scale = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("mult_fwd"sv, f32),
                    {keepalive.back().get(), inv_fp8max.get()},
                    {{n1_shape, f32, /*final_result_index=*/1}}});
  synTensor k_scale_t = k_scale[0].get();
  syn_out(1) = std::move(k_scale[0]);
  auto k_rscale = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("reciprocal_fwd"sv, f32), {k_scale_t},
                    {{n1_shape, f32, std::nullopt}}});
  keepalive.push_back(std::move(k_rscale[0]));
  // convert_to_fp8(k_rot, reciprocal-scale) -> e4m3 [N,D]. The bridge op multiplies by
  // the per-row scale (broadcast over D) and clips to e4m3. RESULT 0.
  ns_ConvertToFp8::ParamsV2 castp{};
  castp.round_mode = CAST_ROUND_HALF_NE;
  auto k_fp8 = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("convert_to_fp8"sv, bf16),
                    {k_rot_t, keepalive.back().get()},
                    {{nd_shape, c10::ScalarType::Float8_e4m3fn, /*final_result_index=*/0}},
                    &castp, sizeof(castp)});
  syn_out(0) = std::move(k_fp8[0]);

  // ===========================================================================
  // V QUANT -> signed int b-bit. scale = absmax(v_rot)/qmax ; codes = clamp(round(v_rot/
  // scale), +/-qmax) ; pack.
  // ===========================================================================
  auto v_abs = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("abs_fwd"sv, bf16), {v_rot_t},
                    {{nd_shape, bf16, std::nullopt}}});
  keepalive.push_back(std::move(v_abs[0]));
  auto v_amax = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("reduce_max_multi_dim_fwd"sv, bf16),
                    {keepalive.back().get()},
                    {{n1_shape, bf16, std::nullopt}}, &amax_params, sizeof(amax_params)});
  keepalive.push_back(std::move(v_amax[0]));
  auto inv_qmax = ConstantHelper(graph, 1.0f / qmax, f32, n1_shape);
  // RESULT 3 (v_scale, f32 [N,1]).
  auto v_scale = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("mult_fwd"sv, f32),
                    {keepalive.back().get(), inv_qmax.get()},
                    {{n1_shape, f32, /*final_result_index=*/3}}});
  synTensor v_scale_t = v_scale[0].get();
  syn_out(3) = std::move(v_scale[0]);
  auto v_rscale = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("reciprocal_fwd"sv, f32), {v_scale_t},
                    {{n1_shape, f32, std::nullopt}}});
  keepalive.push_back(std::move(v_rscale[0]));
  // v_div = v_rot * (1/scale)   (broadcast [N,1] over D). Compute in bf16.
  auto v_div = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
                    {v_rot_t, keepalive.back().get()},
                    {{nd_shape, bf16, std::nullopt}}});
  keepalive.push_back(std::move(v_div[0]));
  ns_RoundKernel::Params roundp{};
  roundp.roundMode = RoundMode_t::ROUND_HALF_NEAREST_EVEN;
  auto v_round = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("round_fwd"sv, bf16), {keepalive.back().get()},
                    {{nd_shape, bf16, std::nullopt}}, &roundp, sizeof(roundp)});
  keepalive.push_back(std::move(v_round[0]));
  ns_ClampKernel::Params clampp{};
  clampp.lowerBound.f = -qmax;
  clampp.upperBound.f =  qmax;
  auto v_clip = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("clamp_fwd"sv, bf16), {keepalive.back().get()},
                    {{nd_shape, bf16, std::nullopt}}, &clampp, sizeof(clampp)});
  synTensor v_clip_t = v_clip[0].get();
  keepalive.push_back(std::move(v_clip[0]));

  // PACK (RESULT 2). Composed from cast + bitwise + shift + reshape-or.
  auto v_packed = PackSubByte(graph, v_clip_t, N, D, v_bits, keepalive,
                              /*final_result_index=*/2);
  syn_out(2) = std::move(v_packed);
}

// -----------------------------------------------------------------------------
// PackSubByte: pack signed b-bit codes (bf16 in [-qmax,qmax]) into int32 lanes, L codes
// per lane (L=8 for 4-bit, 16 for 2-bit). Strategy (all bridge GUIDs):
//   codes_i32  = cast_<bf16>_to_i32(round-already-applied codes)            [N,D]
//   masked     = bitwise_and(codes_i32, MASK)  (MASK=0xF | 0x3)             [N,D] (two's-comp low bits)
//   reshaped   = reshape [N,D] -> [N, D/L, L]
//   shifted    = bitshift_left(reshaped, shifts[L])   (shifts = 0,b,2b,... broadcast over [N,D/L])
//   lane       = reduce_sum over the L axis (== bitwise-or of disjoint bitfields)        [N,D/L]
// The disjoint-field sum equals the OR (no overlap), so reduce_sum is exact and cheaper
// than an L-1 fold of bitwise_or. shifts is a small constant tensor.
// -----------------------------------------------------------------------------
inline sh::tensor KvRotateQuant::PackSubByte(
    sh::graph& graph, synTensor codes_bf16, int64_t N, int64_t D, int64_t v_bits,
    std::vector<sh::tensor>& keepalive, int final_result_index) {
  using namespace std::literals;
  const at::ScalarType i32 = c10::ScalarType::Int;
  const int64_t L  = (v_bits == 4) ? 8 : 16;     // codes per int32 lane
  const int64_t Lv = D / L;
  const int mask   = (v_bits == 4) ? 0xF : 0x3;

  std::vector<int64_t> nd_shape{N, D};
  // bf16 codes -> i32 (codes are integral-valued floats in [-qmax,qmax]).
  auto codes_i32 = OpBackend::BuildNode(
      this, graph, {"cast_bf16_to_i32", {codes_bf16}, {{nd_shape, i32, std::nullopt}}});
  keepalive.push_back(std::move(codes_i32[0]));
  // low-bit two's-complement: AND with mask.
  auto mask_c = ConstantHelper(graph, mask, i32, nd_shape);
  auto low = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("bitwise_and_fwd"sv, i32),
                    {keepalive.back().get(), mask_c.get()},
                    {{nd_shape, i32, std::nullopt}}});
  keepalive.push_back(std::move(low[0]));
  // reshape [N,D] -> [N, Lv, L].
  std::vector<int64_t> grp_shape{N, Lv, L};
  auto grp = ReshapeHelper(graph, keepalive.back().get(), grp_shape, i32);
  synTensor grp_t = grp.get();
  keepalive.push_back(std::move(grp));
  // per-position shift amounts [1,1,L] = {0, b, 2b, ...}; broadcast over [N,Lv].
  std::vector<int64_t> shift_shape{1, 1, L};
  std::vector<float> shift_vals(static_cast<size_t>(L));
  for (int64_t i = 0; i < L; ++i) shift_vals[static_cast<size_t>(i)] =
      static_cast<float>(i * v_bits);
  // build the shift constant as i32 via a small host-filled constant tensor.
  auto shift_c = AllocateConstantSynapseTensor(graph, p_context_->device_id_,
      std::vector<double>(shift_vals.begin(), shift_vals.end()), shift_shape);
  keepalive.push_back(std::move(shift_c));
  auto shifted = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("bitshift_fwd"sv, i32),
                    {grp_t, keepalive.back().get()},
                    {{grp_shape, i32, std::nullopt}}});
  keepalive.push_back(std::move(shifted[0]));
  // reduce_sum over the L axis (dim 2 of rank 3) -> [N,Lv]. Disjoint fields => sum==or.
  ns_Reduction::ParamsV2 sum_params{};
  sum_params.reductionDimensionMask = 1u << static_cast<unsigned>(3 - 2 - 1);  // dim2/rank3 -> bit0
  sum_params.keepDim = false;
  std::vector<int64_t> lane_shape{N, Lv};
  auto lane = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("reduce_sum_multi_dim_fwd"sv, i32),
                    {keepalive.back().get()},
                    {{lane_shape, i32, final_result_index}},
                    &sum_params, sizeof(sum_params)});
  return std::move(lane[0]);
}

} // namespace habana
