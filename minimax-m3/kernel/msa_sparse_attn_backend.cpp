/*
 * msa_sparse_attn_backend.cpp -- MiniMax-M3 MSA: the EAGER-frontend TU for the
 * gather-consume sparse-attention OpBackend op
 * (`torch.ops.hpu_msa_ext::sparse_attn`). It CONSUMES the block->token-expanded
 * indices (msa_expand_blocks_to_tokens, run upstream from the block-indexer op's
 * BLOCK ids) + GQA-expanded real K/V, and produces the MSA main attention output.
 *
 * The graph is BYTE-IDENTICAL to the validated DSA consume op; this is the DSA
 * consume copied UNCHANGED into this package (MSA-named) so the M3 kernel is
 * self-contained, with its OWN `hpu_msa_ext` op namespace (never collides with the
 * installed/serving `hpu_dsa_ext::sparse_attn`). The backend lowering
 * (`MsaSparseAttn : OpBackend` + `MsaSparseAttnMeta`) lives in
 * msa_sparse_attn_backend_shared.h. This TU adds ONLY the EAGER frontend + schema/
 * registration, mirroring dsa_sparse_attn_backend.cpp.
 *
 * Build CPU-only:  bash build_msa_sparse_attn.sh   (inside vllm-gaudi:next; host g++,
 * links torch + the shipped libhabana_pytorch_backend(.upstream).so + the cloned
 * bridge headers, -D_GLIBCXX_USE_CXX11_ABI=1, NEVER the plugin libs).
 * RUN ON THE HPU BOX ONLY via tests/test_msa_sparse_attn_op.py.
 */

// fmt header-only (see the shared header for the full rationale).
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- shared backend (AddNode chain + MsaSparseAttnMeta). ---------------------
#include "msa_sparse_attn_backend_shared.h"

// --- frontend (EAGER op). Identical include to dsa_sparse_attn_backend.cpp:48. ---
#include "habana_eager/ops/eager_op.h"  // habana::eager::EagerOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

#include <algorithm>
#include <vector>

namespace habana {

// =============================================================================
// EAGER frontend: build the eager op and call it. Out shape == [T,H,Dv] bf16
// (== MsaSparseAttnMeta). VERBATIM mechanism from dsa_sparse_attn_backend.cpp.
// =============================================================================
static at::Tensor sparse_attn_hpu(
    const at::Tensor& q,         // [T, H, 1, Dh]  bf16
    const at::Tensor& k,         // [S, H, Dh]     bf16  (GQA-expanded to H upstream)
    const at::Tensor& v,         // [S, H, Dv]     bf16
    const at::Tensor& token_idx, // [T, K]         int32 (-1 padded; K=topk_blocks*block_size)
    double scale,
    int64_t gather_width) {
  TORCH_CHECK(q.dim() == 4, "[msa_sparse_attn] q must be 4D [T,H,1,Dh], got ", q.dim());
  TORCH_CHECK(v.dim() == 3, "[msa_sparse_attn] v must be 3D [S,H,Dv], got ", v.dim());
  const int64_t T = q.size(0);
  const int64_t H = q.size(1);
  const int64_t Dv = v.size(2);

  std::vector<std::vector<int64_t>> out_shapes{{T, H, Dv}};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_msa_ext::sparse_attn",
      {q, k, v, token_idx, scale, gather_width},
      std::move(out_shapes)};
  return hpu_op.call();
}

} // namespace habana

// =============================================================================
// Registration. TORCH_LIBRARY_FRAGMENT (additive) so this .so can load ALONGSIDE
// the indexer .so (which owns TORCH_LIBRARY(hpu_msa_ext)) in one process without the
// "single TORCH_LIBRARY per namespace" abort -- exactly as the DSA consume used a
// fragment of hpu_dsa_ext. OWN namespace `hpu_msa_ext`.
//   1. TORCH_LIBRARY_FRAGMENT schema -> adds hpu_msa_ext::sparse_attn
//   2. TORCH_LIBRARY_IMPL(HPU)       -> routes through the eager frontend
//   3. KernelRegistry().add          -> binds the schema name to MsaSparseAttn
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_msa_ext, m) {
  m.def(
      "sparse_attn(Tensor q, Tensor k, Tensor v, Tensor token_idx, "
      "float scale, int gather_width) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_msa_ext, HPU, m) {
  m.impl("sparse_attn", habana::sparse_attn_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& MsaSparseAttnKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_msa_ext::sparse_attn", habana::MsaSparseAttn);
