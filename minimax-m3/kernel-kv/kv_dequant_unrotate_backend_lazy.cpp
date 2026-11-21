/*
 * kv_dequant_unrotate_backend_lazy.cpp -- MiniMax-M3 TurboQuant KV-READ: the LAZY-
 * frontend variant of the dequant->inverse-rotate OpBackend op
 * (`torch.ops.hpu_kvq_ext::kv_dequant_unrotate`). Captures into the decode HPU-graph
 * under LAZY serving (PT_HPU_LAZY_MODE=1).
 *
 * BYTE-FOR-BYTE REUSE of KvDequantUnrotate : OpBackend + KvDequantUnrotateMeta from
 * kv_dequant_unrotate_backend_shared.h; only the frontend differs (LazyOp vs EagerOp).
 * OWN namespace hpu_kvq_ext, OWN .so. Does NOT touch installed plugins or /opt.
 *
 * Build CPU-only:  bash build_kv_dequant_unrotate_lazy.sh   (vllm-gaudi:next under
 * PT_HPU_LAZY_MODE=1). RUN ON THE HPU BOX ONLY.
 */
#include "kv_dequant_unrotate_backend_shared.h"
#include "habana_kernels/lazy_kernels.h"
#include <torch/library.h>

#include <vector>

namespace habana_lazy {

using namespace habana;

static std::vector<at::Tensor> kv_dequant_unrotate_hpu(
    const at::Tensor& k_fp8,
    const at::Tensor& k_scale,
    const at::Tensor& v_packed,
    const at::Tensor& v_scale,
    const at::Tensor& rot_t,
    int64_t v_bits,
    int64_t out_fp8) {
  LazyOp<std::vector<at::Tensor>> hpu_op{
      "hpu_kvq_ext::kv_dequant_unrotate",
      {k_fp8, k_scale, v_packed, v_scale, rot_t, v_bits, out_fp8}};
  hpu_op.SetOutputMetaFn(KvDequantUnrotateMeta);
  RUN_MAYBE_WITH_ACC_THREAD(kv_dequant_unrotate, hpu_op);
}

} // namespace habana_lazy

TORCH_LIBRARY_FRAGMENT(hpu_kvq_ext, m) {
  m.def(
      "kv_dequant_unrotate(Tensor k_fp8, Tensor k_scale, Tensor v_packed, "
      "Tensor v_scale, Tensor rot_t, int v_bits, int out_fp8) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(hpu_kvq_ext, HPU, m) {
  m.impl("kv_dequant_unrotate", habana_lazy::kv_dequant_unrotate_hpu);
}

static const auto& KvDequantUnrotateKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_kvq_ext::kv_dequant_unrotate", habana::KvDequantUnrotate);
