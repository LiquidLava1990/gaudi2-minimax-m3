/*
 * msa_block_indexer_backend_lazy.cpp -- MiniMax-M3 MSA: the LAZY-frontend variant of
 * the block-indexer top-k OpBackend op
 * (`torch.ops.hpu_msa_ext::msa_block_indexer_topk`). It exists so the op CAPTURES
 * into the decode HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1) -- the fast
 * serving path (zero per-op host dispatch), exactly as the shipped MoE op + the DSA
 * lazy TUs do. Serving uses lazy, so this is the production frontend.
 *
 * WHAT IS REUSED BYTE-FOR-BYTE: the backend lowering `MsaBlockIndexerTopk : OpBackend`
 * (its full AddNode chain, both modes) + `MsaBlockIndexerTopkMeta`, included UNCHANGED
 * from msa_block_indexer_backend_shared.h. The SRAM-resident batch_gemm->...->topk
 * recipe is identical; ONLY the frontend differs (LazyOp vs EagerOp).
 *
 * WHY A SEPARATE LAZY TU (vs the eager .cpp): under PT_HPU_LAZY_MODE=1 the process
 * loads the LAZY plugin (libhabana_pytorch_plugin.so), not the eager one, so (1) the
 * eager .so would fail to load (undefined EagerOpBase::validate_inputs) and (2) even
 * if it loaded, EagerOp's path bypasses the lazy IR the HPU-graph capture engine
 * records. habana_lazy::LazyOp fixes both. (Verbatim from the DSA indexer lazy TU.)
 *
 * SELF-CONTAINED: this single .so carries the schema (TORCH_LIBRARY), the lazy
 * frontend binding (TORCH_LIBRARY_IMPL(HPU)) and the KernelRegistry registration.
 * Loading ONLY this .so under lazy is sufficient. Load the lazy .so XOR the eager .so
 * by mode (never both), so one TORCH_LIBRARY def is fine. OWN namespace `hpu_msa_ext`.
 *
 * Build CPU-only:  bash build_msa_block_indexer_lazy.sh   (inside vllm-gaudi:next
 * under PT_HPU_LAZY_MODE=1 so the lazy plugin is referenced; links ONLY the shipped
 * libhabana_pytorch_backend.so; the habana_lazy::* frontend symbols resolve at
 * RUNTIME from the loaded lazy plugin). RUN ON THE HPU BOX ONLY.
 */

// --- shared backend (AddNode chain + MsaBlockIndexerTopkMeta), VERBATIM. ------
#include "msa_block_indexer_backend_shared.h"

// --- frontend (LAZY op) -- modeled VERBATIM on the DSA indexer lazy TU /
//     moe_fused_expert_backend_lazy.cpp:43-58:
//        LazyOp<at::Tensor> hpu_op{"hpu_msa_ext::msa_block_indexer_topk", inputs};
//        hpu_op.SetOutputMetaFn(MsaBlockIndexerTopkMeta);
//        RUN_MAYBE_WITH_ACC_THREAD(msa_block_indexer_topk, hpu_op);
//     LazyOp lives in habana_kernels/lazy_kernels.h; RUN_MAYBE_WITH_ACC_THREAD +
//     scheduleAccTask are in hpu_ops/hpu_op_helper.h (pulled by the shared header).
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

namespace habana_lazy {

// Mirror the DSA indexer lazy TU's file-scope `using namespace habana;`: the
// RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask unqualified (namespace
// habana, hpu_op_helper.h); MsaBlockIndexerTopkMeta is also in namespace habana (the
// shared header). Both resolve via this using-directive.
using namespace habana;

// =============================================================================
// LAZY frontend: build the LazyOp from (qualified-name, inputs), set the
// output-meta fn so shape inference works, dispatch through the acc-thread/call
// machinery. LazyOp::call()/get_result() emit into the lazy IR, so the op is
// recorded by the HPU-graph capture engine (the whole point). The schema name
// "hpu_msa_ext::msa_block_indexer_topk" routes the lowering to MsaBlockIndexerTopk
// via KernelRegistry().get(opname). VERBATIM mechanism from the DSA indexer lazy TU.
// =============================================================================
static at::Tensor msa_block_indexer_topk_hpu(
    const at::Tensor& q,        // [H, Sq, D]  bf16
    const at::Tensor& k,        // [H, Sk, D]  bf16
    const at::Tensor& weights,  // [H, Sq, 1]  bf16
    const at::Tensor& cmask,    // [Sq, Sk]    bool
    int64_t index_topk,
    int64_t mode,               // 0 = DSA token-sparse, 1 = MSA block-sparse
    int64_t block_size) {       // tokens/block (MSA); ignored if mode==0
  LazyOp<at::Tensor> hpu_op{
      "hpu_msa_ext::msa_block_indexer_topk",
      {q, k, weights, cmask, index_topk, mode, block_size}};
  // Set the output-meta fn (== MsaBlockIndexerTopkMeta) so LazyOp::get_result()
  // sizes/types the lazy result tensor.
  hpu_op.SetOutputMetaFn(MsaBlockIndexerTopkMeta);
  RUN_MAYBE_WITH_ACC_THREAD(msa_block_indexer_topk, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts, one .so; modeled on the DSA indexer lazy TU :97-110).
//   1. TORCH_LIBRARY schema    -> makes torch.ops.hpu_msa_ext.msa_block_indexer_topk
//   2. TORCH_LIBRARY_IMPL(HPU) -> routes through the LAZY frontend (LazyOp)
//   3. KernelRegistry().add    -> binds the schema name to MsaBlockIndexerTopk
// OWN namespace `hpu_msa_ext` -- never touches the installed hpu_dsa_ext library.
// =============================================================================
TORCH_LIBRARY(hpu_msa_ext, m) {
  m.def(
      "msa_block_indexer_topk(Tensor q, Tensor k, Tensor weights, Tensor cmask, "
      "int index_topk, int mode, int block_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_msa_ext, HPU, m) {
  m.impl("msa_block_indexer_topk", habana_lazy::msa_block_indexer_topk_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& MsaBlockIndexerTopkKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_msa_ext::msa_block_indexer_topk", habana::MsaBlockIndexerTopk);
