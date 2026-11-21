/*
 * kv_rotate_quant_backend.cpp -- MiniMax-M3 TurboQuant KV-WRITE: the EAGER-frontend TU
 * for the rotate->quant->pack OpBackend op
 * (`torch.ops.hpu_kvq_ext::kv_rotate_quant`).
 *
 * The backend lowering (KvRotateQuant : OpBackend + KvRotateQuantMeta, full node-chain)
 * lives in kv_rotate_quant_backend_shared.h. This TU adds ONLY the EAGER frontend
 * (habana::eager::EagerOp) + schema/registration, mirroring the proven
 * msa_block_indexer_backend.cpp / moe_fused_expert_backend.cpp.
 *
 * OWN NAMESPACE `hpu_kvq_ext`, OWN .so. Does NOT touch the installed hpu_dsa_ext /
 * hpu_moe_ext / hpu_msa_ext ops or anything in /opt / vllm-active.
 *
 * WHY EAGER here: vllm-gaudi:next is the pytorch-UPSTREAM/eager build; importing
 * habana_frameworks.torch loads the eager plugin so habana::eager::EagerOp's non-inline
 * symbols resolve at runtime. Under LAZY serving use kv_rotate_quant_backend_lazy.cpp.
 *
 * Build CPU-only:  bash build_kv_rotate_quant.sh   (inside vllm-gaudi:next; host g++).
 * RUN ON THE HPU BOX ONLY via tests/test_turboquant_kv_op.py.
 */
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "kv_rotate_quant_backend_shared.h"
#include "habana_eager/ops/eager_op.h"  // habana::eager::EagerOp<...>
#include <torch/library.h>

#include <vector>

namespace habana {

// EAGER frontend. Returns the 4 outputs (k_fp8, k_scale, v_packed, v_scale) as a tuple.
// EagerOp<std::vector<at::Tensor>> builds a one-node JIT graph routed by the schema name
// to KvRotateQuant via KernelRegistry, then launches it.
static std::vector<at::Tensor> kv_rotate_quant_hpu(
    const at::Tensor& k,    // [N,D] bf16
    const at::Tensor& v,    // [N,D] bf16
    const at::Tensor& rot,  // [D,D] bf16
    int64_t v_bits) {       // 4 (k8v4) | 2 (k8v2)
  TORCH_CHECK(k.dim() == 2, "[kv_rotate_quant] k must be 2D [N,D], got ", k.dim());
  TORCH_CHECK(rot.dim() == 2, "[kv_rotate_quant] rot must be 2D [D,D], got ", rot.dim());
  const int64_t N = k.size(0);
  const int64_t D = k.size(1);
  TORCH_CHECK(v_bits == 4 || v_bits == 2,
              "[kv_rotate_quant] v_bits must be 4 or 2, got ", v_bits);
  const int64_t per_lane = (v_bits == 4) ? 8 : 16;
  TORCH_CHECK(D % per_lane == 0,
              "[kv_rotate_quant] head_dim D=", D, " must be a multiple of ", per_lane);
  const int64_t Lv = D / per_lane;

  std::vector<std::vector<int64_t>> out_shapes = {
      {N, D}, {N, 1}, {N, Lv}, {N, 1}};
  habana::eager::EagerOp<std::vector<at::Tensor>> hpu_op{
      "hpu_kvq_ext::kv_rotate_quant",
      {k, v, rot, v_bits},
      std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::Float8_e4m3fn, c10::ScalarType::Float,
                           c10::ScalarType::Int, c10::ScalarType::Float});
  return hpu_op.call();
}

} // namespace habana

// =============================================================================
// Registration (3 parts, one .so). OWN namespace hpu_kvq_ext.
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_kvq_ext, m) {
  m.def(
      "kv_rotate_quant(Tensor k, Tensor v, Tensor rot, int v_bits) "
      "-> Tensor[]");
}

TORCH_LIBRARY_IMPL(hpu_kvq_ext, HPU, m) {
  m.impl("kv_rotate_quant", habana::kv_rotate_quant_hpu);
}

static const auto& KvRotateQuantKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_kvq_ext::kv_rotate_quant", habana::KvRotateQuant);
