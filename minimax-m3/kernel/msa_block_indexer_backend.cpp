/*
 * msa_block_indexer_backend.cpp -- MiniMax-M3 MSA: the EAGER-frontend TU for the
 * block-indexer top-k OpBackend op
 * (`torch.ops.hpu_msa_ext::msa_block_indexer_topk`).
 *
 * This is the FIRST M3 MSA kernel -- a GC-captured Synapse recipe that selects, per
 * (query, GQA-group), the top-k KV BLOCKS to attend (block-max-pool over the index
 * scores, then top-k). It is an ADDITIVE extension of the validated DSA indexer op:
 * the SAME op, config-driven (`mode`/`block_size`), also still does the DSA token-
 * sparse path (mode=0) byte-identically.
 *
 * The backend lowering (`MsaBlockIndexerTopk : OpBackend` + `MsaBlockIndexerTopkMeta`,
 * incl. the full node-chain GUIDs/params and the matched MSA oracle math) lives in
 * msa_block_indexer_backend_shared.h. This TU adds ONLY the EAGER frontend
 * (habana::eager::EagerOp) + the schema/registration, exactly mirroring the proven
 * dsa_indexer_backend.cpp / moe_fused_expert_backend.cpp. The lazy variant
 * (msa_block_indexer_backend_lazy.cpp) reuses the SAME shared backend byte-for-byte
 * and differs only in its frontend (habana_lazy::LazyOp).
 *
 * OWN NAMESPACE: registers `hpu_msa_ext::msa_block_indexer_topk` -- it does NOT
 * collide with the installed/serving `hpu_dsa_ext` ops (the running vllm-active serve
 * loads those; this package is separate, additive, with its own .so).
 *
 * WHY OpBackend (not UserCustomOpDescriptor): the indexer is a MULTI-node recipe
 * (batch_gemm -> [relu] -> mult -> [reduce_sum] -> masked_fill -> [reshape ->
 * reduce_max] -> topk). UserCustomOpDescriptor binds ONE guid; OpBackend in the
 * KernelRegistry is how the bridge ships its own multi-node ops -- the same
 * justification proven by the shipped hpu_moe_ext::fused_expert + the DSA ops.
 *
 * WHY EAGER here: the vllm-gaudi:next image is the pytorch-UPSTREAM/eager build;
 * `import habana_frameworks.torch` loads the EAGER plugin
 * (libhabana_pytorch2_plugin.so), so habana::eager::EagerOp's non-inline symbols
 * resolve at runtime, while habana_lazy::LazyOp is unavailable. (Under LAZY serving
 * use msa_block_indexer_backend_lazy.cpp -- it captures into the decode HPU-graph.)
 *
 * Build CPU-only:  bash build_msa_block_indexer.sh   (inside vllm-gaudi:next; host
 * g++, links torch + the shipped libhabana_pytorch_backend(.upstream).so + the cloned
 * bridge headers, -D_GLIBCXX_USE_CXX11_ABI=1, NEVER the plugin libs).
 * RUN ON THE HPU BOX ONLY via tests/test_msa_block_indexer_op.py.
 */

// fmt header-only (see the shared header for the full rationale).
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- shared backend (AddNode chain + MsaBlockIndexerTopkMeta). ---------------
#include "msa_block_indexer_backend_shared.h"

// --- frontend (EAGER op): the public, non-codegen path the bridge's own
//     user-custom-op frontend uses in the EAGER/pt2 build. Identical include to
//     dsa_indexer_backend.cpp:50.
#include "habana_eager/ops/eager_op.h"  // habana::eager::EagerOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

#include <algorithm>
#include <vector>

namespace habana {

// =============================================================================
// EAGER frontend: build the eager op from (schema, inputs, out-shape) and call
// it. EagerOp<...>.call() builds a one-node JIT graph whose schema name routes the
// lowering to MsaBlockIndexerTopk via KernelRegistry().get(opname), then launches
// it. VERBATIM mechanism from dsa_indexer_backend.cpp:69-88.
//
// Out shape (== MsaBlockIndexerTopkMeta):
//   mode==0 (DSA): [Sq, min(index_topk, Sk)]              int32
//   mode==1 (MSA): [H, Sq, min(index_topk, n_blocks)]     int32   (n_blocks=Sk/block_size)
// =============================================================================
static at::Tensor msa_block_indexer_topk_hpu(
    const at::Tensor& q,        // [H, Sq, D]  bf16
    const at::Tensor& k,        // [H, Sk, D]  bf16
    const at::Tensor& weights,  // [H, Sq, 1]  bf16
    const at::Tensor& cmask,    // [Sq, Sk]    bool
    int64_t index_topk,
    int64_t mode,               // 0 = DSA token-sparse, 1 = MSA block-sparse
    int64_t block_size) {       // tokens/block (MSA); ignored if mode==0
  TORCH_CHECK(q.dim() == 3, "[msa_block_indexer] q must be 3D [H,Sq,D], got ", q.dim());
  TORCH_CHECK(k.dim() == 3, "[msa_block_indexer] k must be 3D [H,Sk,D], got ", k.dim());
  const int64_t H  = q.size(0);
  const int64_t Sq = q.size(1);
  const int64_t Sk = k.size(1);

  std::vector<std::vector<int64_t>> out_shapes;
  if (mode == 0) {
    const int64_t kk = std::min<int64_t>(index_topk, Sk);
    out_shapes = {{Sq, kk}};
  } else {
    TORCH_CHECK(block_size > 0, "[msa_block_indexer] MSA mode needs block_size>0");
    TORCH_CHECK(Sk % block_size == 0,
                "[msa_block_indexer] MSA needs Sk(", Sk, ") a multiple of block_size(",
                block_size, "); pad the key axis (and cmask) upstream");
    const int64_t n_blocks = Sk / block_size;
    const int64_t kk = std::min<int64_t>(index_topk, n_blocks);
    out_shapes = {{H, Sq, kk}};
  }

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_msa_ext::msa_block_indexer_topk",
      {q, k, weights, cmask, index_topk, mode, block_size},
      std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::Int});  // indices are int32
  return hpu_op.call();
}

} // namespace habana

// =============================================================================
// Registration (3 parts, one .so; modeled on dsa_indexer_backend.cpp:101-117):
//   1. TORCH_LIBRARY schema    -> makes torch.ops.hpu_msa_ext.msa_block_indexer_topk
//   2. TORCH_LIBRARY_IMPL(HPU) -> routes through the eager frontend (EagerOp)
//   3. KernelRegistry().add    -> binds the schema name to MsaBlockIndexerTopk
// OWN namespace `hpu_msa_ext` -- never touches the installed hpu_dsa_ext library.
// =============================================================================
TORCH_LIBRARY(hpu_msa_ext, m) {
  m.def(
      "msa_block_indexer_topk(Tensor q, Tensor k, Tensor weights, Tensor cmask, "
      "int index_topk, int mode, int block_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_msa_ext, HPU, m) {
  m.impl("msa_block_indexer_topk", habana::msa_block_indexer_topk_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& MsaBlockIndexerTopkKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_msa_ext::msa_block_indexer_topk", habana::MsaBlockIndexerTopk);

/*
 * CONSUME: the MSA gather-consume op is the DSA `sparse_attn` op fed (a) GQA-expanded
 * real K/V and (b) the block->token-expanded indices (msa_expand_blocks_to_tokens),
 * with gather_width = topk_blocks*block_size. It is copied UNCHANGED into this package
 * as hpu_msa_ext::sparse_attn (msa_sparse_attn_backend{,_lazy}.cpp) so the M3 kernel
 * is self-contained, but its graph is byte-identical to DSA's (block->token expansion
 * is done upstream in Python). The installed hpu_dsa_ext::sparse_attn stays untouched.
 */
