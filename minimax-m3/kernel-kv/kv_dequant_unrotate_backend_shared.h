/*
 * kv_dequant_unrotate_backend_shared.h -- the SHARED, frontend-agnostic backend lowering
 * for the MiniMax-M3 sub-fp8 KV-cache TurboQuant *KV-READ* op as a GC-captured
 * `habana::OpBackend` op (`torch.ops.hpu_kvq_ext::kv_dequant_unrotate`). This is the
 * read side folded into the attention path: it un-quantizes the stored codes and
 * INVERSE-rotates K,V so the SDPA sees ordinary K,V.
 *
 * SEPARATE PACKAGE, OWN NAMESPACE (`hpu_kvq_ext`), OWN .so (libkv_dequant_unrotate
 * {,_lazy}.so). Does NOT collide with / rebuild the installed hpu_dsa_ext / hpu_moe_ext
 * / hpu_msa_ext ops or anything in /opt. Modeled VERBATIM on the proven OpBackend
 * pattern; eager (kv_dequant_unrotate_backend.cpp) + lazy
 * (kv_dequant_unrotate_backend_lazy.cpp) share this byte-for-byte.
 *
 * Mirrors ref/turboquant_kv_ref.py::kv_dequant_unrotate STEP-FOR-STEP. The rotation is
 * orthogonal (R @ R^T == I) so the INVERSE rotation is just R^T -- a transpose, the
 * caller passes `rot_t` ([D,D]); for the default symmetric Walsh-Hadamard R, rot_t==R.
 * "inverse = transpose -> cheap" (one more [D,D] gemm per K and V).
 *
 * ============================ NODE-CHAIN (GUIDs + params) ====================
 * Inputs (order == schema; PyTorch row-major, last dim == FCD):
 *   k_fp8    [N, D]      float8_e4m3fn   rotated K codes (from the write op, RESULT 0)
 *   k_scale  [N, 1]      float32         per-row K scale            (write op RESULT 1)
 *   v_packed [N, D/8|D/16] int32         rotated V codes, packed    (write op RESULT 2)
 *   v_scale  [N, 1]      float32         per-row V scale            (write op RESULT 3)
 *   rot_t    [D, D]      bf16            inverse rotation R^T (== R for symmetric WHT)
 *   v_bits             int    4 or 2 (selects unpack)
 *   out_fp8            int    0 -> K,V out bf16 (bf16 SDPA); 1 -> K out fp8 e4m3 (fp8 SDPA)
 *
 * Outputs (2 tensors):
 *   0 k_out  [N, D]   bf16 (out_fp8==0) | float8_e4m3fn (out_fp8==1)   un-rotated K
 *   1 v_out  [N, D]   bf16                                             un-rotated V
 *
 *  --- K DEQUANT + UNROTATE ---
 *  1. k_deq  = convert_from_fp8(k_fp8, k_scale) -> bf16 [N,D]   (still rotated; *scale, bcast D)
 *        guid get_guid_with_precision("convert_from_fp8", bf16); ns_ConvertFromFp8::Params.
 *  2. k_un   = gemm(k_deq[N,D], rot_t[D,D], {false,false})      -> [N,D] bf16   (== k_deq @ R^T)
 *  3. (out_fp8==1) k_out = convert_to_fp8(k_un, scale=1.0) -> e4m3 [N,D]   (RESULT 0, fp8 SDPA)
 *     (out_fp8==0) k_out = k_un (bf16)                                     (RESULT 0, bf16 SDPA)
 *
 *  --- V DEQUANT + UNROTATE ---
 *  4. v_i8   = UNPACK(v_packed): int32 lanes -> signed b-bit codes [N,D] (shift+and+sign-extend)
 *  5. v_codes= cast_i8_to_bf16(v_i8)                            -> [N,D] bf16
 *  6. v_deq  = mult_fwd(v_codes, v_scale (bcast D))             -> [N,D] bf16   (still rotated)
 *  7. v_un   = gemm(v_deq[N,D], rot_t[D,D], {false,false})      -> [N,D] bf16   (RESULT 1)   (== v_deq @ R^T)
 *
 * The V SDPA path stays bf16 (values aren't re-quantized to fp8 here; the fp8-SDPA on
 * Gaudi multiplies P@V in bf16/fp32 accumulate regardless -- the K side is what gets the
 * fp8 Q.K^T speedup). out_fp8 controls only K's output dtype.
 *
 * NOTHING is hardcoded: N/D from shapes, v_bits + out_fp8 scalar args, rot_t a tensor.
 *
 * Build CPU-only (vllm-gaudi:next, host g++):  bash build_kv_dequant_unrotate.sh
 * RUN ON THE HPU BOX ONLY via tests/test_turboquant_kv_op.py (GATED).
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/stack_getter.h"
#include "backend/habana_operator.h"
#include "backend/synapse_helpers/graph.h"

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"
#include "synapse_common_types.hpp"   // synGEMMParams
#include "perf_lib_layer_params.h"    // ns_ConvertToFp8::ParamsV2, ns_ConvertFromFp8::Params,
                                      // ns_BitShiftKernel, RoundMode_t, CAST_ROUND_*

#include <limits>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta. Stack order == schema:
//   (k_fp8, k_scale, v_packed, v_scale, rot_t, v_bits, out_fp8).
//   k_fp8 = stack[0] [N,D]; out_fp8 = stack[6].
// Two outputs: k_out [N,D] (bf16 or e4m3 by out_fp8), v_out [N,D] bf16.
// =============================================================================
inline OutputMetaDataVector KvDequantUnrotateMeta(const at::Stack& stack) {
  const at::Tensor& k_fp8 = stack_tensor(stack, 0);  // [N,D]
  TORCH_CHECK(k_fp8.dim() == 2, "[kv_dequant_unrotate] k_fp8 must be 2D [N,D]");
  const int64_t N = k_fp8.size(0);
  const int64_t D = k_fp8.size(1);
  const int64_t out_fp8 = stack.at(6).toInt();

  OutputMetaDataVector meta(2);
  meta[0].shape = {N, D};
  meta[0].dtype = (out_fp8 == 1) ? c10::ScalarType::Float8_e4m3fn
                                 : c10::ScalarType::BFloat16;   // k_out
  meta[1].shape = {N, D};
  meta[1].dtype = c10::ScalarType::BFloat16;                   // v_out
  return meta;
}

struct KvDequantUnrotate : OpBackend {
  KvDequantUnrotate(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "convert_from_fp8", scalar_type, {0, 1}, {}, {}, false) {
    SetOutputMetaFn(KvDequantUnrotateMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;

 private:
  // Unpack int32 lanes -> signed b-bit codes as bf16 [N,D]. Inverse of PackSubByte.
  sh::tensor UnpackSubByte(sh::graph& graph, synTensor packed_i32, int64_t N, int64_t D,
                           int64_t v_bits, std::vector<sh::tensor>& keepalive);
};

inline void KvDequantUnrotate::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  StackGetter sg(this, stack, "KvDequantUnrotate::AddNode");

  auto k_fp8    = sg.getNextInput<TensorsPair>();   // e4m3 [N,D]
  auto k_scale  = sg.getNextInput<TensorsPair>();   // f32  [N,1]
  auto v_packed = sg.getNextInput<TensorsPair>();   // i32  [N,D/L]
  auto v_scale  = sg.getNextInput<TensorsPair>();   // f32  [N,1]
  auto rot_t    = sg.getNextInput<TensorsPair>();   // bf16 [D,D]
  const int64_t v_bits  = sg.getNextInput<int64_t>();
  const int64_t out_fp8 = sg.getNextInput<int64_t>();

  const auto kf_sizes = k_fp8.pt_t.sizes().vec();
  const auto rt_sizes = rot_t.pt_t.sizes().vec();
  TORCH_CHECK(k_fp8.pt_t.dim() == 2, "[kv_dequant_unrotate] k_fp8 must be 2D [N,D]");
  TORCH_CHECK(rot_t.pt_t.dim() == 2, "[kv_dequant_unrotate] rot_t must be 2D [D,D]");
  const int64_t N = kf_sizes[0];
  const int64_t D = kf_sizes[1];
  TORCH_CHECK(rt_sizes[0] == D && rt_sizes[1] == D,
              "[kv_dequant_unrotate] rot_t must be [D,D]=[", D, ",", D, "]");
  TORCH_CHECK(v_bits == 4 || v_bits == 2,
              "[kv_dequant_unrotate] v_bits must be 4 or 2, got ", v_bits);
  const int64_t L = (v_bits == 4) ? 8 : 16;
  TORCH_CHECK(v_packed.pt_t.size(0) == N && v_packed.pt_t.size(1) == D / L,
              "[kv_dequant_unrotate] v_packed must be [N,D/", L, "]");

  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  std::vector<int64_t> nd_shape{N, D};
  synGEMMParams rot_params{/*transpose_a=*/false, /*transpose_b=*/false};

  std::vector<sh::tensor> keepalive;
  keepalive.reserve(20);

  // ===========================================================================
  // K: convert_from_fp8(k_fp8, k_scale) -> bf16 (rotated) ; then @ rot_t (unrotate).
  // ===========================================================================
  ns_ConvertFromFp8::Params fromp{};
  auto k_deq = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("convert_from_fp8"sv, bf16),
                    {k_fp8.syn_t, k_scale.syn_t},
                    {{nd_shape, bf16, std::nullopt}}, &fromp, sizeof(fromp)});
  keepalive.push_back(std::move(k_deq[0]));
  auto k_un = OpBackend::BuildNode(
      this, graph, {"gemm", {keepalive.back().get(), rot_t.syn_t},
                    {{nd_shape, bf16, std::nullopt}}, &rot_params, sizeof(rot_params)});
  synTensor k_un_t = k_un[0].get();
  keepalive.push_back(std::move(k_un[0]));
  if (out_fp8 == 1) {
    // re-cast un-rotated K to fp8 e4m3 (scale=1.0) for the fp8 SDPA. RESULT 0.
    ns_ConvertToFp8::ParamsV2 castp{};
    castp.round_mode = CAST_ROUND_HALF_NE;
    auto one = OpBackend::BuildConstantTensor(this, graph, 1.0);
    auto k_out = OpBackend::BuildNode(
        this, graph, {get_guid_with_precision("convert_to_fp8"sv, bf16),
                      {k_un_t, one.get()},
                      {{nd_shape, c10::ScalarType::Float8_e4m3fn, /*final_result_index=*/0}},
                      &castp, sizeof(castp)});
    syn_out(0) = std::move(k_out[0]);
  } else {
    // bf16 K out: re-emit the gemm output as the result (identity reshape to tag result).
    auto k_out = ReshapeHelper(graph, k_un_t, nd_shape, bf16, /*final_result_index=*/0);
    syn_out(0) = std::move(k_out);
  }

  // ===========================================================================
  // V: unpack int lanes -> codes(bf16) ; * v_scale ; @ rot_t (unrotate). RESULT 1.
  // ===========================================================================
  auto v_codes = UnpackSubByte(graph, v_packed.syn_t, N, D, v_bits, keepalive);  // bf16 [N,D]
  synTensor v_codes_t = v_codes.get();
  keepalive.push_back(std::move(v_codes));
  auto v_deq = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
                    {v_codes_t, v_scale.syn_t},
                    {{nd_shape, bf16, std::nullopt}}});
  keepalive.push_back(std::move(v_deq[0]));
  auto v_un = OpBackend::BuildNode(
      this, graph, {"gemm", {keepalive.back().get(), rot_t.syn_t},
                    {{nd_shape, bf16, /*final_result_index=*/1}}, &rot_params, sizeof(rot_params)});
  syn_out(1) = std::move(v_un[0]);
}

// -----------------------------------------------------------------------------
// UnpackSubByte: int32 lanes [N,Lv] -> signed b-bit codes as bf16 [N,D]. Strategy:
//   broadcast lane to [N,Lv,L] ; shift right by {0,b,2b,...} ; and with field-mask ;
//   sign-extend (where code >= 2^(b-1): code -= 2^b) ; reshape [N,D] ; cast i32->bf16.
// All bridge GUIDs (bitshift_fwd, bitwise_and_fwd, greater_equal/where via sub, reshape,
// cast). Exact inverse of the write-side pack; the device round-trips two's complement.
// -----------------------------------------------------------------------------
inline sh::tensor KvDequantUnrotate::UnpackSubByte(
    sh::graph& graph, synTensor packed_i32, int64_t N, int64_t D, int64_t v_bits,
    std::vector<sh::tensor>& keepalive) {
  using namespace std::literals;
  const at::ScalarType i32 = c10::ScalarType::Int;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const int64_t L  = (v_bits == 4) ? 8 : 16;
  const int64_t Lv = D / L;
  const int mask     = (v_bits == 4) ? 0xF : 0x3;     // field mask
  const int signbit  = 1 << (v_bits - 1);             // 8 (4b) | 2 (2b)
  const int modulus  = 1 << v_bits;                   // 16 (4b) | 4 (2b)

  // reshape lane [N,Lv] -> [N,Lv,1] then broadcast to [N,Lv,L] is implicit via the shift
  // const [1,1,L]; bitshift broadcasts. First reshape to [N,Lv,1].
  std::vector<int64_t> lane3_shape{N, Lv, 1};
  auto lane3 = ReshapeHelper(graph, packed_i32, lane3_shape, i32);
  synTensor lane3_t = lane3.get();
  keepalive.push_back(std::move(lane3));
  // right-shift by per-position amounts -> [N,Lv,L].
  std::vector<int64_t> grp_shape{N, Lv, L};
  std::vector<int64_t> shift_shape{1, 1, L};
  std::vector<double> rshift(static_cast<size_t>(L));
  for (int64_t i = 0; i < L; ++i) rshift[static_cast<size_t>(i)] =
      -static_cast<double>(i * v_bits);   // negative == right shift for bitshift_fwd
  auto rshift_c = AllocateConstantSynapseTensor(graph, p_context_->device_id_, rshift, shift_shape);
  keepalive.push_back(std::move(rshift_c));
  auto shifted = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("bitshift_fwd"sv, i32),
                    {lane3_t, keepalive.back().get()},
                    {{grp_shape, i32, std::nullopt}}});
  keepalive.push_back(std::move(shifted[0]));
  // and with field mask.
  auto mask_c = ConstantHelper(graph, mask, i32, grp_shape);
  auto field = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("bitwise_and_fwd"sv, i32),
                    {keepalive.back().get(), mask_c.get()},
                    {{grp_shape, i32, std::nullopt}}});
  synTensor field_t = field[0].get();
  keepalive.push_back(std::move(field[0]));
  // sign-extend: signed = field - modulus * (field >= signbit). Implement as
  //   ge = greater_equal(field, signbit) (bool->i32 via where) ; signed = field - mod*ge.
  auto signbit_c = ConstantHelper(graph, signbit, i32, grp_shape);
  auto ge = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("greater_equal_fwd"sv, i32),
                    {field_t, signbit_c.get()},
                    {{grp_shape, c10::ScalarType::Bool, std::nullopt}}});
  // ge(bool) -> i32 {0,1} via where(ge, mod, 0).
  auto mod_c  = ConstantHelper(graph, modulus, i32, grp_shape);
  auto zero_c = ConstantHelper(graph, 0, i32, grp_shape);
  auto modsel = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("where_fwd"sv, i32),
                    {ge[0].get(), mod_c.get(), zero_c.get()},
                    {{grp_shape, i32, std::nullopt}}});
  keepalive.push_back(std::move(modsel[0]));
  auto signed_codes = OpBackend::BuildNode(
      this, graph, {get_guid_with_precision("sub_fwd"sv, i32),
                    {field_t, keepalive.back().get()},
                    {{grp_shape, i32, std::nullopt}}});
  keepalive.push_back(std::move(signed_codes[0]));
  // reshape [N,Lv,L] -> [N,D] and cast i32 -> bf16.
  std::vector<int64_t> nd_shape{N, D};
  auto codes_nd = ReshapeHelper(graph, keepalive.back().get(), nd_shape, i32);
  synTensor codes_nd_t = codes_nd.get();
  keepalive.push_back(std::move(codes_nd));
  auto codes_bf16 = OpBackend::BuildNode(
      this, graph, {"cast_i32_to_bf16", {codes_nd_t}, {{nd_shape, bf16, std::nullopt}}});
  return std::move(codes_bf16[0]);
}

} // namespace habana
