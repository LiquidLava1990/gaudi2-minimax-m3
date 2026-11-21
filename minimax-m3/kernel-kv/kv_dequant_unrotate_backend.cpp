/*
 * kv_dequant_unrotate_backend.cpp -- MiniMax-M3 TurboQuant KV-READ: the EAGER-frontend
 * TU for the dequant->inverse-rotate OpBackend op
 * (`torch.ops.hpu_kvq_ext::kv_dequant_unrotate`). Folded into the SDPA read path.
 *
 * Backend lowering (KvDequantUnrotate : OpBackend + KvDequantUnrotateMeta) lives in
 * kv_dequant_unrotate_backend_shared.h. This TU adds ONLY the EAGER frontend
 * (habana::eager::EagerOp) + schema/registration. OWN namespace hpu_kvq_ext, OWN .so.
 * Does NOT touch installed plugins or /opt.
 *
 * Build CPU-only:  bash build_kv_dequant_unrotate.sh   (vllm-gaudi:next; host g++).
 * RUN ON THE HPU BOX ONLY via tests/test_turboquant_kv_op.py.
 */
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "kv_dequant_unrotate_backend_shared.h"
#include "habana_eager/ops/eager_op.h"
#include <torch/library.h>

#include <vector>

namespace habana {

// EAGER frontend. Returns (k_out, v_out). k_out dtype is fp8 (out_fp8==1) or bf16.
static std::vector<at::Tensor> kv_dequant_unrotate_hpu(
    const at::Tensor& k_fp8,    // [N,D] e4m3
    const at::Tensor& k_scale,  // [N,1] f32
    const at::Tensor& v_packed, // [N,D/L] i32
    const at::Tensor& v_scale,  // [N,1] f32
    const at::Tensor& rot_t,    // [D,D] bf16
    int64_t v_bits,             // 4 | 2
    int64_t out_fp8) {          // 0 -> bf16 K, 1 -> fp8 K
  TORCH_CHECK(k_fp8.dim() == 2, "[kv_dequant_unrotate] k_fp8 must be 2D [N,D]");
  TORCH_CHECK(rot_t.dim() == 2, "[kv_dequant_unrotate] rot_t must be 2D [D,D]");
  const int64_t N = k_fp8.size(0);
  const int64_t D = k_fp8.size(1);
  TORCH_CHECK(v_bits == 4 || v_bits == 2,
              "[kv_dequant_unrotate] v_bits must be 4 or 2, got ", v_bits);

  std::vector<std::vector<int64_t>> out_shapes = {{N, D}, {N, D}};
  habana::eager::EagerOp<std::vector<at::Tensor>> hpu_op{
      "hpu_kvq_ext::kv_dequant_unrotate",
      {k_fp8, k_scale, v_packed, v_scale, rot_t, v_bits, out_fp8},
      std::move(out_shapes)};
  const c10::ScalarType k_dt = (out_fp8 == 1) ? c10::ScalarType::Float8_e4m3fn
                                              : c10::ScalarType::BFloat16;
  hpu_op.set_scalar_types({k_dt, c10::ScalarType::BFloat16});
  return hpu_op.call();
}

} // namespace habana

TORCH_LIBRARY_FRAGMENT(hpu_kvq_ext, m) {
  m.def(
      "kv_dequant_unrotate(Tensor k_fp8, Tensor k_scale, Tensor v_packed, "
      "Tensor v_scale, Tensor rot_t, int v_bits, int out_fp8) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(hpu_kvq_ext, HPU, m) {
  m.impl("kv_dequant_unrotate", habana::kv_dequant_unrotate_hpu);
}

static const auto& KvDequantUnrotateKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_kvq_ext::kv_dequant_unrotate", habana::KvDequantUnrotate);
