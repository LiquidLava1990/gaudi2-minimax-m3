/*
 * kv_rotate_quant_backend_lazy.cpp -- MiniMax-M3 TurboQuant KV-WRITE: the LAZY-frontend
 * variant of the rotate->quant->pack OpBackend op
 * (`torch.ops.hpu_kvq_ext::kv_rotate_quant`). It captures into the decode HPU-graph
 * under LAZY serving (PT_HPU_LAZY_MODE=1) -- the fast serving path (zero per-op host
 * dispatch), exactly as the shipped MoE op + the DSA/MSA lazy TUs do.
 *
 * BYTE-FOR-BYTE REUSE: the backend lowering KvRotateQuant : OpBackend (full AddNode
 * chain) + KvRotateQuantMeta, included UNCHANGED from kv_rotate_quant_backend_shared.h.
 * Only the frontend differs (LazyOp vs EagerOp).
 *
 * OWN namespace hpu_kvq_ext, OWN .so. Does NOT touch installed plugins or /opt.
 *
 * Build CPU-only:  bash build_kv_rotate_quant_lazy.sh   (inside vllm-gaudi:next under
 * PT_HPU_LAZY_MODE=1; links ONLY libhabana_pytorch_backend.so; habana_lazy::* syms
 * resolve at RUNTIME from the loaded lazy plugin). RUN ON THE HPU BOX ONLY.
 */
#include "kv_rotate_quant_backend_shared.h"
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<...>
#include <torch/library.h>

#include <vector>

namespace habana_lazy {

using namespace habana;  // scheduleAccTask + KvRotateQuantMeta resolve here

static std::vector<at::Tensor> kv_rotate_quant_hpu(
    const at::Tensor& k,    // [N,D] bf16
    const at::Tensor& v,    // [N,D] bf16
    const at::Tensor& rot,  // [D,D] bf16
    int64_t v_bits) {
  LazyOp<std::vector<at::Tensor>> hpu_op{
      "hpu_kvq_ext::kv_rotate_quant",
      {k, v, rot, v_bits}};
  hpu_op.SetOutputMetaFn(KvRotateQuantMeta);
  RUN_MAYBE_WITH_ACC_THREAD(kv_rotate_quant, hpu_op);
}

} // namespace habana_lazy

TORCH_LIBRARY_FRAGMENT(hpu_kvq_ext, m) {
  m.def(
      "kv_rotate_quant(Tensor k, Tensor v, Tensor rot, int v_bits) "
      "-> Tensor[]");
}

TORCH_LIBRARY_IMPL(hpu_kvq_ext, HPU, m) {
  m.impl("kv_rotate_quant", habana_lazy::kv_rotate_quant_hpu);
}

static const auto& KvRotateQuantKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_kvq_ext::kv_rotate_quant", habana::KvRotateQuant);
