/*
 * msa_sparse_attn_backend_lazy.cpp -- MiniMax-M3 MSA: the LAZY-frontend variant of
 * the gather-consume sparse-attention OpBackend op
 * (`torch.ops.hpu_msa_ext::sparse_attn`). It captures into the decode HPU-graph under
 * LAZY serving (PT_HPU_LAZY_MODE=1) -- the production serving path. The graph is
 * byte-identical to the validated DSA consume op (MSA-named, OWN `hpu_msa_ext`
 * namespace; never collides with the installed/serving hpu_dsa_ext::sparse_attn).
 *
 * Reuses the shared backend (`MsaSparseAttn : OpBackend` + `MsaSparseAttnMeta`) from
 * msa_sparse_attn_backend_shared.h byte-for-byte; only the frontend differs (LazyOp).
 * Verbatim mechanism from dsa_sparse_attn_backend_lazy.cpp.
 *
 * Build CPU-only:  bash build_msa_sparse_attn_lazy.sh   (vllm-gaudi:next under
 * PT_HPU_LAZY_MODE=1; links ONLY libhabana_pytorch_backend.so; habana_lazy::* syms
 * runtime-resolved from the loaded lazy plugin). RUN ON THE HPU BOX ONLY.
 */

// --- shared backend (AddNode chain + MsaSparseAttnMeta), VERBATIM. -----------
#include "msa_sparse_attn_backend_shared.h"

// --- frontend (LAZY op). Verbatim mechanism from dsa_sparse_attn_backend_lazy.cpp. ---
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

namespace habana_lazy {

// Mirror the DSA consume lazy TU's file-scope `using namespace habana;` (the
// RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask unqualified; MsaSparseAttnMeta
// is in namespace habana). Both resolve via this using-directive.
using namespace habana;

static at::Tensor sparse_attn_hpu(
    const at::Tensor& q,         // [T, H, 1, Dh]  bf16
    const at::Tensor& k,         // [S, H, Dh]     bf16  (GQA-expanded to H upstream)
    const at::Tensor& v,         // [S, H, Dv]     bf16
    const at::Tensor& token_idx, // [T, K]         int32 (-1 padded)
    double scale,
    int64_t gather_width) {
  LazyOp<at::Tensor> hpu_op{
      "hpu_msa_ext::sparse_attn",
      {q, k, v, token_idx, scale, gather_width}};
  hpu_op.SetOutputMetaFn(MsaSparseAttnMeta);
  RUN_MAYBE_WITH_ACC_THREAD(sparse_attn, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration. TORCH_LIBRARY_FRAGMENT (additive; coexists with the indexer lazy
// .so's TORCH_LIBRARY(hpu_msa_ext)). OWN namespace `hpu_msa_ext`.
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_msa_ext, m) {
  m.def(
      "sparse_attn(Tensor q, Tensor k, Tensor v, Tensor token_idx, "
      "float scale, int gather_width) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_msa_ext, HPU, m) {
  m.impl("sparse_attn", habana_lazy::sparse_attn_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& MsaSparseAttnKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_msa_ext::sparse_attn", habana::MsaSparseAttn);
