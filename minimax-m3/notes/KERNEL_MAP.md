# MiniMax-M3 MSA on Gaudi 2 — extending the DSA OpBackend kernel (config-driven)

Status: PLAN (read-only investigation, 2026-06-14). No code modified.
Goal: extend the **already-built, on-card-validated** DeepSeek/GLM **DSA** OpBackend
sparse-attention kernel so it ALSO serves **MiniMax-M3 MSA**, keeping every dim
config-driven (an `MsaDims` reader analogous to `DsaDims`) and a `mode` flag selecting
**MLA-token-sparse (DSA)** vs **GQA-block-sparse (MSA)**.

This plans against the REAL kernel sources, not the design doc's hypothetical:
- DSA kernels are BUILT: `$HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/dsa/`
  (`dsa_indexer_backend{,_lazy}.cpp` + `_shared.h`; `dsa_sparse_attn_backend{,_lazy}.cpp`
  + `_shared.h`; `build_dsa_*.sh`; `tests/test_dsa_*_op.py`; `build/libdsa_*.so` all present).
- Torch oracles/spec: `$HOME/AI/hpu-dsa-ext/hpu_dsa_ext/` (`sparse_indexer_hpu.py`,
  `sparse_mla_consume.py`, `indexer_cache_hpu.py`, `indexer_metadata_hpu.py`,
  `glm_moe_dsa_wire.py`).
- NOTE: `$HOME/AI/minimax-m3/{ref,kernel,vllm,quant}` are EMPTY — there is no M3
  reference code on the box yet. The MSA math below is taken from the task spec and must be
  pinned to the real HF/vLLM M3 modeling files before MVP-1 closes (analogous to how DSA was
  pinned to `modeling_glm_moe_dsa.py` — see "Open confirmations").

---

## 0. TL;DR

MSA is a **strict superset** of the DSA node-chain on the **indexer** side (one extra
block-max-pool reduce before the top-k) and a **near-identical** chain on the **consume**
side (same `gather_fwd` + `sdpa_fwd`, just real GQA K/V and a block→token index expansion
done UPSTREAM). **Every bridge GUID MSA needs already exists** — confirmed below with
file:line. **No new TPC ISA kernel, no missing bridge op.** New surface is: 2 indexer nodes
(reshape + `reduce_max_multi_dim_fwd`) gated by a `score_type` branch, an `MsaDims` config
reader, a block→token gather-index expansion (Python or a tiny op), and a `test_msa_synthetic`
CPU parity harness. Build/validate order mirrors DSA: CPU parity first, then eager `.so`
on-card, then lazy `.so`.

---

## 1. DSA torch functions ↔ OpBackend node-chains (the map)

### 1a. Indexer — `lightning_indexer_topk` → `hpu_dsa_ext::indexer_topk`

Torch spec (`sparse_indexer_hpu.py:40-76`; oracle `reference_indexer_topk` :79-92):
```
logits[t,h,s] = q[t,h,:]·k[s,:]                # per-head dot          (:66)
logits        = ReLU(logits)                   # .clamp_(min=0)        (:66)
score[t,s]    = Σ_h weights[t,h]·logits[t,h,s] # weighted head-sum     (:67)
score         = masked_fill(key_pos>abs_pos,-inf)  # causal s<=abs_pos (:71)
idx[t]        = score.topk(k).indices          # top-k TOKEN indices   (:72)
```
OpBackend chain (`dsa_indexer_backend_shared.h:171-318`), PyTorch row-major, head axis H is the
leading bmm-batch (MQA key expanded to H upstream, so per-head Q·Kᵀ is ONE `batch_gemm`):

| # | node | GUID (file:line of GUID) | params | shape |
|---|---|---|---|---|
| 1 | per-head Q·Kᵀ | `batch_gemm` (`matmul.cpp` bmm idiom; `bmm_gen.cpp:42-60`) | `synGEMMParams{false,true}` | `[H,Sq,D]·[H,Sk,D]→[H,Sq,Sk]` |
| 2 | ReLU | `relu_fwd` (`addmm_gen.cpp:219`) | — | `[H,Sq,Sk]` |
| 3 | per-head weight | `mult_fwd` (`moe_…_shared.h:289`) | broadcast `[H,Sq,1]` over FCD | `[H,Sq,Sk]` |
| 4 | sum over heads | `reduce_sum_multi_dim_fwd` (`index_put_gen.cpp:236`, `convolution_backward_overrideable_gen.cpp:62`) | `ns_Reduction::ParamsV2{mask=1<<(3-0-1)=4, keepDim=false}` | `[Sq,Sk]` |
| 5 | causal mask | `masked_fill_fwd` (`masked_fill_gen.cpp:112-114`) | `ns_MaskedFill::ParamsV2{value.f=-inf}` | `[Sq,Sk]` |
| 6 | top-k (FINAL) | `topk` (`topk_gen.cpp:96-104`) | `ns_TopkNodeV2::ParamsV4{bsw=kk, axis=get_dim_in_tpc_order(1,2)=0, bottomK=false, kType=K_TENSOR_NONE}` | values `[Sq,kk]` (discard) + indices `[Sq,kk]` **int32** |

`DsaIndexerTopkMeta` (`:137-151`): out `[Sq, min(index_topk,Sk)]` int32.
**Where dims come from:** `H,Sq,D` from `q.sizes()`, `Sk` from `k.sizes()`, `kk=min(topk,Sk)`
(`:142-145`). `index_topk` is the only scalar arg (schema `:101-105`). NOTHING hardcoded to GLM —
H=32 (GLM) vs 64 (DeepSeek) "just work" (`_shared.h:36-37`).

### 1b. Consume — `sparse_mla_gather_attention` → `hpu_dsa_ext::sparse_attn`

Torch spec (`sparse_mla_consume.py:165-229`; oracle `sparse_mla_masked_attention` :131-162;
gather==masked diff 0.011). OpBackend chain (`dsa_sparse_attn_backend_shared.h:197-446`):

| # | node | GUID (file:line) | note |
|---|---|---|---|
| 1 | pad detector `valid=(idx>=0)` | `greater_equal_fwd` (`compare_gen.cpp:45`) | vs `ConstantHelper(0)` |
| 2 | gather id `gid=where(valid,idx,S)` | `where_fwd` (`nantonum_gen.cpp:145`) | sentinel row S, never clamp(0) |
| 3 | pad bias `where(valid,0,-inf)`→`[T,1,1,K]` | `where_fwd` + `ReshapeHelper` | `NEG_BIG=bf16::lowest()` (finite, `:252`) |
| 4 | append zero KV row | `concat` (`synConcatenateParams.axis=tpc(0,3)=2`) | `k_ext,v_ext = [S+1,H,D*]` |
| 5 | row gather (flattened key axis) | `gather_fwd` (`index_select_gen.cpp:22-29`) | `ns_GatherKernel::Params{axis=tpc(0,2)=1}`; reshape `[S+1,H*D]`, gid `[T*K]` → `[T,K,H,D]` |
| 6 | permute K/V to `[T,H,K,D]` | `transpose` via `PermuteHelper` (`op_backend.h:316`) | perm `{0,2,1,3}`; pass INPUT shape |
| 7 | fused SDPA | `sdpa_fwd` (`sdpa_gen.cpp:319-384`) | `ns_Sdpa::ParamsV7{scale,is_causal=false,softmax_mode=SDPA_FP32_SOFTMAX}`; inputs `{q,k,v,mask,seed=null,+7 null}` |
| 8 | drop L=1 (FINAL) | `reshape` via `ReshapeHelper` | `[T,H,1,Dv]→[T,H,Dv]` |

`DsaSparseAttnMeta` (`:164-177`): out `[T,H,Dv]` bf16. **Where dims come from:** `T,H,Dh` from
`q.sizes()`; `S,Dv` from `k/v.sizes()`; `K` from `topk_idx.size(1)` (`:220-231`). `scale` is the
only float arg. Causality is ALREADY in the gather geometry (indexer pre-masked) → consume masks
ONLY pad slots (`_shared.h:46-50`).

### 1c. The Python wire / cache / metadata (NOT kernel; reused or M3-analog)

- `DsaDims`/`read_dsa_dims` (`glm_moe_dsa_wire.py:71-147`): reads every dim from `hf_config`;
  `layer_is_dense`/`layer_skips_topk` cadence.
- `IndexerKCache` (`indexer_cache_hpu.py`): flat bf16 `(num_blocks*block_size, head_dim)` cache;
  `.insert/.gather_blocks/.gather_rows` + runner `get_kv_cache_spec`/`bind_kv_cache` patches.
- `build_indexer_metadata` (`indexer_metadata_hpu.py`): derives abs_pos / cu_seqlens /
  slot_mapping / block layout from the trimmed `HPUMLAMetadata` (`:1-35`).
- `forward_oot` (`sparse_indexer_hpu.py:223-322`): per-request key gather in ABSOLUTE order →
  scorer → writes `topk_indices_buffer`.

---

## 2. The MSA plan (concrete)

**Backbone difference.** DSA: token-sparse over MLA (latent absorption). MSA: **top-16 BLOCK**
selection (`block_size=128`) over **GQA** with **REAL uncompressed K/V** (`num_key_value_heads=4`),
no MLA absorption. Block scoring uses `score_type: max` — per-token index scores are pooled per
128-token block by reduce-max, top-16 over blocks, then each block id expands to its 128 token
positions for the gather. Consume gather width `K = 16*128 = 2048`.

### 2a. Block-max-pool scoring node — `reduce_max_multi_dim_fwd` CONFIRMED

The block-max-pool reduce is a REAL bridge GUID: **`reduce_max_multi_dim_fwd`**, emitted by the
`amax`/`AminAmax` lowering — `amax_amin_gen.cpp:71` (`AmaxSharedMeta`), `:86`/`:118`
(`get_guid_with_precision("reduce_max_multi_dim_fwd", dtype)` in `Aminmax::AddNode`), and the
`max`/`min` dim form `max_min_gen.cpp:122-137` (`MaxSharedMeta`→`"reduce_max_multi_dim_fwd"`).
Params struct: `ns_Reduction::ParamsV2`, filled by `FillReductionParams(rank, dim_vec, keepdim)`
where the reduce-dim bit is `1 << (ndims - d - 1)` (`reduction_template.cpp:22-38`, esp `:33`) —
the SAME formula DSA NODE 4 already uses (`dsa_indexer_backend_shared.h:254-256`).

**Insert between indexer NODE 5 (masked `score[Sq,Sk]`) and NODE 6 (topk):**
```
5.   masked  = masked_fill_fwd(score, cmask, -inf)              [Sq, Sk]   (unchanged)
5a.  scoreB  = ReshapeHelper(masked, [Sq, n_blocks, block_size])           (op_backend.h:302)
5b.  blkmax  = reduce_max_multi_dim_fwd(scoreB, dim=2, keepDim=false)  [Sq, n_blocks]
                ns_Reduction::ParamsV2{mask = 1<<(3-2-1) = 1<<0 = 1, keepDim=false}
6.   (vals,bid) = topk(blkmax, k=16, axis=tpc(1,2)=0, largest)   [Sq, 16] int32  ← BLOCK ids
```
- `n_blocks = ceil(Sk / block_size)`; `block_size` and `top_blocks(=16)` come from `MsaDims`
  (config), NOT literals. **Sk MUST be a multiple of block_size** for the reshape — pad the key
  axis (and the `cmask`) up to `n_blocks*block_size` upstream; padded tail tokens are non-causal
  for every query so `masked_fill`→ -inf, and a fully-padded block reduce-maxes to -inf → never
  selected (matches MSA semantics: a partial last block scores by its real tokens only because
  the pad columns are -inf). This pad is the MSA analog of DSA's clamp `kk=min(topk,Sk)`.
- Reusing `masked` (post-causal-mask, -inf on non-causal) as the reduce input is correct: a block
  whose only causal members are masked contributes -inf and is correctly dropped.

### 2b. Top-16 BLOCK topk + block→token expansion

- **Block top-k** is the SAME `topk` node as DSA NODE 6, only `bsw=top_blocks(16)` and the axis is
  `n_blocks` instead of `Sk`. Output `bid[Sq,16]` int32 = selected BLOCK ids.
- **Block→token expansion** (each block id → its `block_size` consecutive token positions). Two
  equally-valid placements; pick ONE per the generality discipline:
  - **(A, recommended for MVP) UPSTREAM in Python** — emit `token_idx[Sq, 16*block_size]` =
    `bid.unsqueeze(-1)*block_size + arange(block_size)` (a `range`+broadcast+reshape), clamp
    out-of-range to a `-1` sentinel, then feed the EXISTING `sparse_attn` op unchanged with
    `K=16*block_size=2048`. This is the cheapest path and keeps the consume op byte-identical to
    DSA. `range` GUID exists (`arange_gen.cpp:166` `get_guid_with_precision("range")`); broadcast
    is `BroadcastHelper`/`"broadcast"` (`op_backend.h`; `expand_gen.cpp:78`).
  - **(B, fused) IN-GRAPH** — add an expansion sub-chain to a new `msa_sparse_attn` op:
    `range(block_size)` → broadcast to `[Sq,16,block_size]` → `bid[Sq,16,1]*block_size` (`mult_fwd`
    + add) → reshape `[Sq, 2048]`. All GUIDs exist; this is a follow-up once (A) validates.

### 2c. GQA gather-consume — reuse `gather_fwd` + `sdpa_fwd` CONFIRMED

**The DSA `sparse_attn` consume op already works for MSA with ZERO graph change** when fed
`token_idx[T,2048]` (path A), because:
- It already requires K/V **expanded to H** upstream (`_shared.h:233-234` asserts `Hk==H && Hv==H`).
  GQA → MHA expansion (repeat each of the 4 KV heads to the query-head count) is the SAME upstream
  step DSA does for its MQA indexer key (`test_dsa_indexer_op.py:153`
  `k_sd.unsqueeze(0).expand(H,-1,-1)`). So MSA's `num_key_value_heads=4` is handled by an upstream
  `expand`/`broadcast`, exactly like DSA's single MQA head.
- `sdpa_fwd` takes q/k/v as plain shaped inputs + an OPTIONAL additive `attention_mask`
  (`sdpa_gen.cpp:330-358`; mask pushed at `:354-356`); output via `SDPAFwdOutputShapeCommon`
  (`:96-160`). It does not special-case GQA — heads are just a batch dim — so the `[T,H,K,D]`
  layout (`_shared.h:393-394`) is identical for MSA. The pad-only additive bias `[T,1,1,K]`
  (`_shared.h:294-296`) masks any `-1`-padded token slots (partial last block / short rows).
- Real (uncompressed) K/V means **no MLA latent absorption** on the consume side — strictly
  SIMPLER than DSA (DSA's `oot_mla.py` `W_UK_T`/`W_UV` absorption is skipped entirely). The op
  already consumes per-head real K/V; nothing to remove from the graph.

So: **MSA consume = DSA `sparse_attn` op AS-IS, fed expanded GQA K/V + the block-expanded token
indices, with `K=2048` and `scale = head_dim**-0.5`.** A dedicated `msa_sparse_attn` op is needed
ONLY if path B (in-graph block expansion) is chosen.

---

## 3. Reuse vs new — precise inventory

### Reused AS-IS (no change)
- `dsa_sparse_attn_backend_shared.h` + `dsa_sparse_attn_backend{,_lazy}.cpp` + `build_dsa_sparse_attn*.sh`
  — the consume op (path A feeds it expanded GQA K/V + token indices, `K=2048`).
- All bridge GUIDs/helpers: `batch_gemm, relu_fwd, mult_fwd, reduce_sum_multi_dim_fwd,
  masked_fill_fwd, topk, greater_equal_fwd, where_fwd, concat, gather_fwd, sdpa_fwd,
  reduce_max_multi_dim_fwd, range, broadcast`; `ReshapeHelper/ConstantHelper/PermuteHelper/
  BroadcastHelper`.
- The OpBackend mechanism: `struct : OpBackend` + `AddNode` BuildNode chain + `SetOutputMetaFn`,
  eager `EagerOp` / lazy `LazyOp` frontends, `*_shared.h` split, 3-line registration
  (`moe_fused_expert_backend{,_lazy}.cpp`, `dsa_indexer_backend.cpp:69-117`).
- The generality discipline §5 (`DSA_OPBACKEND_KERNEL_DESIGN.md:387-411`): all dims from
  shapes/config; mode/score_type/block_size = data.
- `indexer_metadata_hpu.py` abs_pos/causal-bound/slot derivation (M3 uses standard paged GQA KV —
  the existing `HPUMLAMetadata` derivation largely transfers; see Open confirmations for the
  GQA-vs-MLA cache-shape delta).

### Need an MSA branch / mode (modify)
- **`dsa_indexer_backend_shared.h`** → add a `score_type`/`mode` arg to the schema and a branch:
  after NODE 5, if `block_pool` (MSA): emit NODE 5a (reshape) + 5b (`reduce_max_multi_dim_fwd`)
  and set NODE 6's `bsw=top_blocks`, axis over `n_blocks`; else (DSA token-sparse) the chain is
  unchanged. Output meta becomes `[Sq, top_blocks]` int32 in MSA mode (block ids) vs `[Sq,kk]`
  in DSA mode. Keep ONE shared header, one op, two modes (the §5 "one arch path" rule). Schema gains
  `int block_size` (0 ⇒ token-sparse/DSA) + `int top_k` (already there as `index_topk`/blocks).
- **The wire layer** (`glm_moe_dsa_wire.py` analog) → add `MsaDims` (below) and route `mode`.

### NEW torch reference functions (the executable spec + oracle — author FIRST, CPU-gated)
In a new `hpu-msa-ext/` (mirroring `hpu-dsa-ext/`), or an `msa_*` module beside the DSA ones:
1. `msa_block_indexer_topk(q,k,weights,block_size,top_blocks,abs_pos)` → `[Sq,top_blocks]` int32
   block ids. Body = DSA scorer through the masked `score[Sq,Sk]`, then
   `score.view(Sq,n_blocks,block_size).amax(-1)` → `.topk(top_blocks).indices`. (the 5a/5b spec).
2. `msa_reference_block_topk(...)` — fp32 oracle (no reshape tricks, explicit per-block max), the
   parity reference (analog of `reference_indexer_topk`).
3. `msa_expand_blocks_to_tokens(block_ids, block_size, seq_len)` → `[Sq, top_blocks*block_size]`
   int32 token ids, `-1` for out-of-range (the path-A expansion; pure torch, graph-friendly).
4. `msa_gqa_gather_attention(q,k,v,token_idx,abs_pos,scale)` — thin wrapper that GQA-expands K/V to
   H then calls the EXISTING `sparse_mla_gather_attention` (reuse, not reimplement) — its output is
   the consume oracle.

### NEW OpBackend nodes
- **Indexer (MSA mode):** exactly TWO new nodes inside the branch — `ReshapeHelper` ([Sq,Sk]→
  [Sq,n_blocks,block_size]) and `reduce_max_multi_dim_fwd` ([Sq,n_blocks]). The topk node is the
  existing one with different `bsw`/axis. **No other new indexer node.**
- **Consume:** ZERO new nodes for path A. (Path B adds a `range`+`broadcast`+`mult_fwd`+`add_fwd`+
  `reshape` expansion sub-chain to an `msa_sparse_attn` op — all existing GUIDs.)

### NEW config + harness
- **`MsaDims`** (frozen dataclass, analog of `DsaDims` `glm_moe_dsa_wire.py:71-105`): reads from
  M3 `hf_config` — `num_key_value_heads`, `block_size`(128), `top_blocks`(16), `index_n_heads`,
  `index_head_dim`, `num_attention_heads`, `head_dim`, `first_k_dense_replace`(if any), the
  `score_type`("max"). A `mode` enum {`MLA_TOKEN_SPARSE`, `GQA_BLOCK_SPARSE`}. Validate-loud like
  `assert_sparse_config`.
- **`tests/test_msa_synthetic_hpu.py`** (analog of `test_dsa_synthetic_hpu.py`): tiny GLM-shrunk
  config, asserts (a) `msa_block_indexer_topk` block-set == `msa_reference_block_topk` (Jaccard/
  recall ≥0.98), (b) `msa_gqa_gather_attention` == masked-MHA oracle on selected blocks (~1e-2),
  (c) `-1` pad / partial-last-block semantics, (d) optional on-card variant.
- **On-card op tests** `test_msa_indexer_op.py` / (if path B) `test_msa_sparse_attn_op.py`,
  analogs of the DSA op tests (set-match vs fp32 oracle + HPU-graph capture check).
- **Build scripts** `build_msa_indexer{,_lazy}.sh` (copy of `build_dsa_indexer*.sh`, swap SRC/OUT
  + op name), and only if path B, `build_msa_sparse_attn*.sh`.

---

## 4. Build / validate order (mirrors DSA §7)

Gate EVERY step on the CPU parity harness before any on-card run; keep MSA BLOCKED behind a
`MSA_READY=False` env gate until the full path validates (the `platform_patch.py` safety contract —
never silently dense).

- **MVP-0 — pin the spec.** Read M3's HF `modeling_*`/vLLM modeling for MSA, confirm: block scoring
  is per-128 reduce-max, top-16 blocks, GQA `num_key_value_heads=4`, real K/V, and the exact
  index-head projection (so `q/k/weights` fed to the scorer match M3, like DSA pinned to
  `modeling_glm_moe_dsa.py`). Write `MsaDims` + the 4 torch reference fns.
- **MVP-1 — CPU parity.** `test_msa_synthetic_hpu.py` green: block-indexer set == oracle; GQA
  gather == masked oracle; pad/partial-block semantics. (No device.)
- **MVP-2 — `msa_indexer` eager `.so`, on-card.** Add the `score_type` branch (5a/5b +
  block-topk) to the indexer shared header; build `libmsa_indexer.so`; `test_msa_indexer_op.py`:
  block-id SET == fp32 oracle (Jaccard/recall tol like DSA 0.97/0.98) + HPU-graph capture.
- **MVP-3 — consume on-card (path A).** Feed `msa_expand_blocks_to_tokens` output + GQA-expanded
  K/V into the EXISTING `libdsa_sparse_attn.so` with `K=2048`; validate vs `msa_gqa_gather_attention`
  oracle (the DSA consume op needs no rebuild). If fusing, build `libmsa_sparse_attn.so` (path B)
  and validate the same way.
- **MVP-4 — integrate + flip.** Wire `MsaDims`/mode into the model runner (M3 arch binding,
  GQA paged KV cache spec — reuse `indexer_cache_hpu` shape logic adjusted for `num_kv_heads`),
  flip `MSA_READY` after: (i) indexer-fired counter>0 on ctx>block_size, (ii) ctx≤K sparse==dense
  GQA baseline, (iii) ctx≫K sparse≠dense (not-silently-dense).
- **Phase 5 — lazy `.so`** (`*_lazy.cpp` sharing the header) so both ops capture into the decode
  HPU-graph under `PT_HPU_LAZY_MODE=1`. Re-validate coherence.
- **Phase 6 (opt) — fp8 index logits / in-graph block expansion (path B) / decode paged consume.**

---

## 5. Bridge-GUID confirmation table (every op MSA relies on — cited, all EXIST)

| MSA need | GUID | gen file:line | exists? |
|---|---|---|---|
| per-head Q·Kᵀ (indexer) | `batch_gemm` | `bmm_gen.cpp:42-60`; used `dsa_indexer_backend_shared.h:219-224` | ✅ |
| ReLU | `relu_fwd` | `addmm_gen.cpp:219` | ✅ |
| per-head weight / block scale | `mult_fwd` | `moe_fused_expert_backend_shared.h:289-295` | ✅ |
| head-sum | `reduce_sum_multi_dim_fwd` | `index_put_gen.cpp:236`, `convolution_backward_overrideable_gen.cpp:62` | ✅ |
| causal mask | `masked_fill_fwd` | `masked_fill_gen.cpp:112-114` | ✅ |
| **block-max pool** | **`reduce_max_multi_dim_fwd`** | **`amax_amin_gen.cpp:118` (also `:71,:86`); `max_min_gen.cpp:125`** | ✅ |
| reduction params | `ns_Reduction::ParamsV2` via `FillReductionParams` (bit `1<<(ndims-d-1)`) | `reduction_template.cpp:22-38` | ✅ |
| block top-k / token top-k | `topk` (`ns_TopkNodeV2::ParamsV4`, dyn-K capable) | `topk_gen.cpp:96-104` (`kType=K_TENSOR_SHAPE` `:91-93`) | ✅ |
| block→token expand | `range` + `broadcast` | `arange_gen.cpp:166,180`; `expand_gen.cpp:78` (`BroadcastHelper` `op_backend.h`) | ✅ |
| pad detector | `greater_equal_fwd` | `compare_gen.cpp:45` | ✅ |
| select / pad-bias | `where_fwd` | `nantonum_gen.cpp:145-157` | ✅ |
| sentinel KV row | `concat` (`synConcatenateParams`) | used `dsa_sparse_attn_backend_shared.h:313-324` | ✅ |
| KV gather | `gather_fwd` (index_select; `ns_GatherKernel::Params`) | `index_select_gen.cpp:22-29`; full-gather `gather_elements_fwd` `gather_gen.cpp:121-127` | ✅ |
| GQA SDPA + additive mask | `sdpa_fwd` (`ns_Sdpa::ParamsV7`, optional mask, `valid_seq_len`, window) | `sdpa_gen.cpp:319-384` (mask `:354-356`) | ✅ |
| reshape / permute helpers | `ReshapeHelper`, `PermuteHelper`, `ConstantHelper`, `BroadcastHelper` | `op_backend.h:287-330` | ✅ |

**Missing bridge ops: NONE.** Every MSA primitive — including the block-max-pool — is a real
GC-lowered node in the version-exact bridge (`tag v1.24.0-1007`). The one composition (not a
missing GUID) is the **block→token index expansion**, done as `range`+`broadcast`+arithmetic
(path A in Python, or path B in-graph) — all from existing ops, exactly as the design's §3a
"compose, don't ISA-write" rule. (The CUDA-only `top_k_per_row`/`*_mqa_logits`/`*_quant_and_cache`
remain irrelevant — replaced by `topk`/`batch_gemm`/`gather` as in DSA.)

---

## 6. Open confirmations (pin before MVP-1 closes)

1. **M3 modeling source.** No M3 ref on the box (`minimax-m3/ref` empty). Confirm against the real
   HF/vLLM M3 files: block_size=128, top_blocks=16, `score_type:max`, `num_key_value_heads=4`, the
   index-head projection that yields the scorer's `q/k/weights` (so the fold of softmax_scale into
   weights, the ReLU positive-homogeneity argument from `sparse_indexer_hpu.py:13-17`, still holds
   — verify M3 uses ReLU/positive weights; if it uses a different index activation, NODE 2 changes).
2. **GQA KV cache shape.** `indexer_cache_hpu`/`HPUMLAMetadata` are MLA-shaped. M3 uses ordinary
   paged GQA KV `(num_blocks*block_size, num_kv_heads=4, head_dim)`; confirm the runner
   `get_kv_cache_spec` allocates standard GQA KV (M3 is NOT MLA), so the consume reads real K/V
   blocks — likely SIMPLER than the DSA indexer-cache patch, but verify slot_mapping/block_table
   derivation for GQA.
3. **block_size | Sk.** The 5a reshape needs `Sk` padded to `n_blocks*block_size`; confirm where to
   pad (metadata builds a `cmask` padded to `n_blocks*block_size`, padded cols=non-causal→-inf).
4. **scale.** MSA `scale = head_dim**-0.5` over the GQA head_dim (the index `softmax_scale` for the
   indexer score is `index_head_dim**-0.5`, like DSA); confirm both from config.
