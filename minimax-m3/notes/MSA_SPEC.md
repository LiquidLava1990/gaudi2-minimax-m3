# MiniMax-M3 MSA (MiniMax Sparse Attention) — Exact Forward Spec

**Purpose:** parity oracle for a Gaudi kernel. Every claim is cited to a file:line or URL.
We serve the **text tower only** (`MiniMaxM3SparseForCausalLM`); vision is skipped.

**Model:** `model_type=minimax_m3_vl`, arch `MiniMaxM3SparseForConditionalGeneration`,
428B MoE, released ~June 2026.

## Primary sources

| Tag | Source |
|---|---|
| **CFG** | `/mnt/data2/minimax-m3-bf16/config.json` (local checkpoint) |
| **CFGPY** | `/mnt/data2/minimax-m3-bf16/configuration_minimax_m3_vl.py` (bundled, config classes only — no math) |
| **PAPER** | "MiniMax Sparse Attention", arXiv:2606.13392 — https://arxiv.org/html/2606.13392v1 (Eq. 5–8, §3.1, §5.1) |
| **MSA** | `MiniMax-AI/MSA` GitHub (cloned to `/tmp/MSA_repo`) — the official SM100 kernel + **PyTorch reference**. Algorithm PDF at `docs/MiniMaxSparseAttention.pdf`. https://github.com/MiniMax-AI/MSA |
| **BLOG** | "MiniMax Goes Sparse: Decoding M3's Attention from a Single Diagram" — https://huggingface.co/blog/AtlasCloud-AI/minimax-goes-sparse |
| **M2** | `transformers/main` `models/minimax_m2/modeling_minimax_m2.py` (for the diff) |

> NOTE: the MSA repo is the **consume + select kernel** (it takes already-pooled
> block `max_score` and `q2k_indices` as *inputs*). The **indexer projections + scoring**
> live in the model's attention module (PAPER Eq. 5–7); they are not in the MSA repo.

---

## 0. Raw config values (CFG, all under `text_config`)

```
hidden_size              6144         CFG:10
num_hidden_layers        60           CFG:12
num_attention_heads      64           CFG:13   (H_q)
num_key_value_heads      4            CFG:14   (H_kv  -> GQA group size G = 16)
head_dim                 128          CFG:15   (d_h)
max_position_embeddings  1048576      CFG:17   (1M context)
rope_theta               5000000      CFG:21
rotary_dim               64           CFG:22   (PARTIAL rope: only first 64 of 128 dims rotated)
partial_rotary_factor    0.5          CFG:23
use_qk_norm              true         CFG:25
qk_norm_type             per_head     CFG:96
use_gemma_norm           true         CFG:19
attention_output_gate    false        CFG:20   (NO sigmoid output gate)

sparse_attention_config (CFG:102-235):
  use_sparse_attention   true         CFG:103
  sparse_index_dim       128          CFG:104  (d_idx)
  sparse_num_index_heads 4            CFG:105  (index Q heads = H_kv; index K head = 1, shared)
  sparse_topk_blocks     16           CFG:106  (k)
  sparse_block_size      128          CFG:107  (B_k, tokens per KV block == page size)
  sparse_score_type      "max"        CFG:170  (BlockMaxPool: block score = max token score)
  sparse_init_block      0            CFG:171  (# forced SINK blocks at start -> ZERO for M3)
  sparse_local_block     1            CFG:172  (# forced LOCAL/recent blocks -> ONE, the block holding query)
  sparse_disable_index_value  [0,0,0,1,...,1]  CFG:108-169  (per-layer; 1 = sparse layer)
  sparse_attention_freq       [0,0,0,1,...,1]  CFG:173-234  (per-layer; 1 = sparse layer)
```

### Dense vs sparse layers
- `sparse_attention_freq` and `sparse_disable_index_value` are length-60 arrays: **first 3 entries
  are 0, remaining 57 are 1** (CFG:173-234, CFG:108-169). → **Layers 0,1,2 use DENSE full GQA
  attention (no indexer/selection); layers 3..59 use MSA sparse attention.**
- `moe_layer_freq` is identically `[0,0,0,1,...,1]` (CFG:34-95) → same 3 dense-MLP + 57 MoE split.
- PAPER §5.1 confirms the pattern for the 109B research model: *"the first three layers are
  dense layers, and the remaining 38 layers are MoE layers."* M3-428B uses the SAME `first-3-dense`
  rule (3 dense, 57 sparse here).
- M3 has **no `first_k_dense_replace` scalar**; the dense prefix is encoded by the leading zeros
  in these per-layer arrays. Effective `first_k_dense = 3`.

---

## 1. The attention is GQA over REAL K/V — NOT MLA

Confirmed by:
- CFG: `num_key_value_heads=4`, `head_dim=128`, `num_attention_heads=64` (CFG:13-15) — ordinary GQA,
  no `kv_lora_rank` / `q_lora_rank` / `qk_rope_head_dim` fields (those would mark MLA). Absent.
- PAPER Eq. 8 attends over `K^(r)[I]`, `V^(r)[I]` — the **real, uncompressed** GQA K/V, gathered at
  selected block indices. (PAPER: *"each key-value head serves G = H_q/H_kv query heads"*.)
- MSA reference `sparse_ref(...)` gathers full `k_pages[blk, kv_h]` / `v_pages[blk, kv_h]`
  (`/tmp/MSA_repo/tests/regression/test_sparse_attn.py:51-69`) — real per-block K/V, head_dim 128.
- MSA cute defaults: `DEFAULT_HEAD_KV=4`, `DEFAULT_QHEAD_PER_KV=16`, `DEFAULT_DIM=128`,
  `DEFAULT_BLK_KV=128`, `DEFAULT_TOPK=16` (`/tmp/MSA_repo/python/fmha_sm100/cute/test_sparse_atten.py:40-49`)
  — exactly the M3 text config.

So: the indexer compresses **scores**, never the K/V values. Consume reads full-precision K/V.

---

## 2. Index / selection branch (PAPER Eq. 5–7)

The indexer is a tiny extra attention head that decides **which KV blocks** the real attention reads.

### 2a. Index projections (Eq. 5)
```
Q_idx = X · W_q^idx  ∈ ℝ^(N × H_kv × d_idx)     # one index-query head PER GQA GROUP (H_kv = 4)
K_idx = X · W_k^idx  ∈ ℝ^(N × 1   × d_idx)     # ONE index-key head, shared across all groups
```
- `d_idx = sparse_index_dim = 128` (CFG:104); `H_kv = sparse_num_index_heads = 4` (CFG:105).
- BLOG confirms: *"all heads share the same index key"*, `K_idx ∈ ℝ^{n×1×d}`.
- During training only, inputs are detached: `Q_idx = stopgrad(X)·W_q^idx`, `K_idx = stopgrad(X)·W_k^idx`
  (PAPER Eq. 11). **Irrelevant for inference/kernel** (weights are frozen) — listed for completeness.

### 2b. Token-level index score — CORRECTED 2026-06-14 against the OFFICIAL modeling

> **AUTHORITATIVE SOURCE (supersedes the paper for every index-branch detail):**
> `transformers 5.12.0` :: `models/minimax_m3_vl/modeling_minimax_m3_vl.py` ::
> `MiniMaxM3VLIndexer` + `MiniMaxM3VLRMSNorm` + `apply_rotary_pos_emb`
> (installed at `$HOME/.venvs/m3calib`; read verbatim). The paper-derived math below was WRONG
> in three load-bearing ways. **Validated**: our corrected oracle `ref/msa_reference.py` reproduces
> the REAL `MiniMaxM3VLIndexer` block_scores to |Δ|≤4.8e-07 and selection-set Jaccard 1.0
> (`tests/test_msa_hf_parity.py`); the wire reproduces the corrected oracle bit-for-bit
> (`tests/test_msa_wire_parity.py`).

The OFFICIAL per-index-head order of ops (modeling lines 559–580):
`q_proj/k_proj → view(B,Sq,H_idx|1,d_idx) → Gemma-QK-norm → partial-RoPE → raw dot → causal mask →
block-max-pool → amax over index heads (HEAD-COLLAPSE) → force-local → top-k → -1 pad`.

```
# per index head h (H_idx = sparse_num_index_heads = 4, single shared index key):
q̃_idx[h] = gemma_rms_norm(Q_idx[h])            # x*(1+w), weight zero-init, eps=1e-6, over d_idx
k̃_idx    = gemma_rms_norm(K_idx)               #   (same Gemma norm on the single shared K head)
q̂_idx[h], k̂_idx = partial_rope(q̃_idx[h], k̃_idx, cos[:,:64], sin[:,:64])   # theta=5e6, SAME as main
S_idx[h]_{i,j} = q̂_idx[h]_i · k̂_idx_j^T        # RAW matmul — NO 1/sqrt(d_idx)  (scale = 1.0)
```

1. **Gemma QK-norm `x*(1+w)` on the index q/k (NOT absent, NOT plain `x*w`).** `index_q_norm` /
   `index_k_norm` are `MiniMaxM3VLRMSNorm` (Gemma-style: fp32 normalize, scale by `1 + weight`,
   **weight inits to ZEROS**), applied per index head over `d_idx=128` **after** the reshape-to-heads,
   **before** rope, `eps = rms_norm_eps = 1e-6` (modeling 137–153, 548, 559–562). The checkpoint
   `index_{q,k}_norm.weight` cluster near 0 (mean ~0.2, some **negative**) — impossible for plain
   `x*w`, exactly right for `x*(1+w)`. The OLD spec modeled NO index norm — **wrong**.
2. **Partial RoPE IS applied to the index q/k** — `apply_rotary_pos_emb(idx_q, idx_k,
   cos[...,:head_dim], sin[...,:head_dim])` with the **same `cos`/`sin` the main attention uses**
   (`position_embeddings` is computed once per layer and passed in), `rotary_dim=64`, `theta=5e6`,
   applied **after** the norm, **before** the dot (modeling 563–564). `cos`/`sin` last-dim is only
   `rotary_dim=64` (`partial_rotary_factor=0.5`, `head_dim=128`), so `cos[...,:128]` is a no-op slice
   and exactly the **first 64** of the 128-dim index q/k are rotated — identical positional treatment
   to the main branch. The OLD "no index RoPE" was **wrong** (this changes which blocks win).
3. **NO activation** (no ReLU/softmax) between dot and pool — matches paper §4 "exp-free". (Differs
   from DeepSeek DSA's lightning-indexer, which uses ReLU.)
4. **Index score scale = 1.0 (RAW matmul), NOT `1/sqrt(d_idx)`.** Official: `scores =
   matmul(idx_q.float(), idx_k.float().transpose(-1,-2))` — raw, fed straight to amax/topk (modeling
   573). A single positive global constant is selection-neutral (cannot change topk ordering), so the
   old `1/sqrt(128)` was not a *selection* bug, but it does **not match the code** → `MsaDims.index_scale
   = 1.0`. The MAIN-branch scale `1/sqrt(head_dim)` is unchanged.

### 2c. Block Max-Pool → per-block relevance (Eq. 6, `score_type="max"`)
```
M_idx[h]_{i,b} = max_{ j ∈ block_b , j ≤ i }  S_idx[h]_{i,j}     # per INDEX head h (pre head-collapse)
```
- Per (query i, **index head** h, KV block b): the block's score is the **MAX** over the (causally
  valid) token scores inside that 128-token block (modeling `scores.amax(dim=-1)`, 580). This is
  **pre** head-collapse — the `amax` over the 4 index heads happens next in §2d. CFG:170
  `sparse_score_type="max"`.
- Causality is enforced **inside** the pool by the `j ≤ i` constraint (PAPER Eq. 6, *"enforces
  causality via j≤i"*). A block partially above the diagonal still contributes via its visible tokens.
- In the kernel this is computed as a cheap dense pass over the **proxy** index cache with
  `output_maxscore=True`, which returns exactly this per-block max (MSA README "Dense proxy pass…
  output_maxscore=True"; `/tmp/MSA_repo/tests/integration/test_proxy_kv_e2e.py:1-40` "Dense pass
  fmha_sm100(num_kv_heads=1) → max_score"). The proxy K cache stores `K_idx` (single head).

### 2d. Top-k block selection (Eq. 7) — HEAD-COLLAPSE corrected 2026-06-14

```
# OFFICIAL: collapse the index heads BEFORE topk -> ONE block set per query, shared by ALL groups
M_idx[i,b]   = amax_{h ∈ index heads} ( max_{j ∈ block_b, j ≤ i} S_idx[h]_{i,j} )   # [Sq, n_blk]
I_i          = TopK_{b}( M_idx[i,·] , k )            # k = 16 ; ONE shared set, NO head axis
```
- **HEAD-COLLAPSE (corrected).** The official does `block_scores = scores.amax(dim=-1).amax(dim=1)`
  (modeling 580): after the block-max-pool it takes a **second `amax` over the index-head axis
  (dim=1)**, collapsing the `index_n_heads = 4` index heads into **ONE** `[B, Sq, n_blk]` score
  vector per query. Force-local (`block_scores.scatter_(-1, local_idx, inf)`, 587) and `topk`
  (593) then run on that single vector → **`[B, Sq, topk]`: ONE shared block set per query, used by
  EVERY GQA group / every query head** (`build_block_mask` broadcasts it to all heads via
  `.unsqueeze(1)`, 618). The OLD spec's **per-GQA-group (4 independent top-k sets)** was **WRONG**.
  Kernel/consume implication: there is exactly **one gather per query** (not 4), so the consume is a
  SINGLE op call over all heads — see §3 / §7.5 and the wire (`msa_m3_wire.py` step 6). `k =
  sparse_topk_blocks = 16` (CFG:106); the official uses `topk = min(topk_blocks, num_key_blocks)`.
- **Forced blocks**:
  - `sparse_init_block = 0` (CFG:171) → **NO sink block is forced**. (The official indexer has no
    init-force path at all; only the local force exists.)
  - `sparse_local_block = 1` (CFG:172) → the local block(s) `q_block = position_ids // block_size`
    (walking back `local_blocks-1` more) are forced to `+inf` and **ALWAYS selected** regardless of
    score (modeling 582–587; PAPER §3.1). `clamp(min=0)` collapses early-query local offsets to block 0.
  - Output indices are `-1`-padded when a slot lands on `-inf` (fewer than k visible blocks)
    (`topk_indices.masked_fill(topk_scores == -inf, -1)`, modeling 593–594). **Order note:** the
    official returns top-k in **score order with forced (+inf) blocks first**, NOT a re-sorted
    ascending order — only the **SET** is load-bearing (the block-sparse kernel counts valid entries
    and reads them; our oracle/wire canonicalize to ascending, which is set-equivalent).
- **No sliding window / no separate sink stream.** MSA = pure top-k block-sparse + 1 forced local
  block, ONE set per query.

---

## 3. Consume: GQA SDPA over selected blocks (PAPER Eq. 8)

```
O_i^(h) = softmax( Q_i^(h) · ( K^(r)[I_i^(r)] )^T / sqrt(d_h) ) · V^(r)[I_i^(r)]
          for every query head h in group r ,  d_h = head_dim = 128 , scale = 1/sqrt(128)
```
- **scale = `1.0 / sqrt(head_dim)` = 1/sqrt(128)** — both references hard-code this
  (`test_sparse_attn.py:62` `/ math.sqrt(head_dim)`; cute `sm_scale=1.0/math.sqrt(head_dim)`,
  `test_sparse_atten.py:299` default `softmax_scale = 1.0/math.sqrt(dim)`).
- **Block → token expansion:** selected block id `b` expands to token range
  `[b*128, (b+1)*128)`; the reference does `kv_pos.extend(range(blk*page_size,(blk+1)*page_size))`
  (`test_sparse_attn.py:64-66`).
- **Causal mask uses ORIGINAL absolute positions** (bottom-right alignment): a key token at
  absolute position `ki` is masked iff `ki > qi` where `qi = q_local + qo_offset`
  (`test_sparse_attn.py:60-67`: `scores.masked_fill_(qi < ki, -inf)`). So gathered blocks are
  NOT all-visible; the per-token causal mask is re-applied after the gather. The cute golden
  reference computes the causal visible-block bound as `(q_pos + (seqlen_k - seqlen_q))//blk +1`
  (`test_sparse_atten.py:127-133`).
- GQA mapping: `kv_h = h // (H_q // H_kv) = h // 16` (`test_sparse_attn.py:58`).
- Softmax is over the **gathered** key set only (concatenated selected blocks), per head, in fp32
  (`test_sparse_attn.py:68`). Heads in a group reuse the same gathered K/V but each head has its
  own Q and its own softmax.
- **HEAD-COLLAPSE consume (corrected 2026-06-14):** because selection is now ONE shared block set
  per query (§2d), **every** GQA group/head gathers the **same** token set — one gather per query,
  a SINGLE consume call over all `H_q` heads. (The earlier per-group plan implied 4 gathers / a
  4-call loop; that is gone. The wire fans the 4 real kv-heads to `H_q` via an `expand` view, no
  16× physical copy, and issues one `sparse_attn` call — `msa_m3_wire.py` step 6.)

### Two-stage decode/prefill pipeline (kernel realization — MSA README "Usage", proxy_kv_e2e)
1. **Proxy pass** (cheap): dense MQA attention with `num_kv_heads=1` over the index K cache
   (`K_idx`), `output_maxscore=True` → `max_score` shape `(H_kv, K_blocks, T)`. == BlockMaxPool.
2. **Select**: `sparse_topk_select(max_score, topk=16, force_begin=0, force_end=1, num_valid_pages)`
   → `kv_block_indexes` shape `(T, H_kv, 16)`, int32, ascending, `-1`-padded.
3. **Sparse pass**: `fmha_sm100(q, k_pages, v_pages, …, kv_block_indexes, sm_scale=1/sqrt(128))`
   over the **real** GQA KV cache → output. (Each query token is one "batch" row in the kernel;
   `num_kv_heads = 4`.)

The index (proxy) K cache is stored **alongside** the real KV cache and must be kept in sync for
decode (BLOG/search: *"keeps a K-only index buffer alongside the main KV cache … to avoid reading
stale state"*). Only one extra **K** stream (1 head × 128 dims) per token — no extra V.

---

## 4. Exact dimensions table

| Quantity | Symbol | Value | Source |
|---|---|---|---|
| Hidden size | — | 6144 | CFG:10 |
| Layers (total) | L | 60 | CFG:12 |
| Dense attention layers | — | 3 (idx 0–2) | CFG:173-234 leading zeros; PAPER §5.1 |
| Sparse (MSA) layers | — | 57 (idx 3–59) | CFG:173-234 |
| Query heads | H_q | 64 | CFG:13 |
| KV heads | H_kv | 4 | CFG:14 |
| GQA group size | G | 16 (=64/4) | derived |
| Head dim | d_h | 128 | CFG:15 |
| Attn scale | — | 1/√128 ≈ 0.08839 | references §3 |
| RoPE θ | — | 5,000,000 | CFG:21 |
| Rotary dim (partial) | — | 64 (factor 0.5) | CFG:22-23 |
| Index Q heads | — | 4 (= H_kv) | CFG:105 |
| Index K heads (shared) | — | 1 | PAPER Eq.5; BLOG |
| Index dim | d_idx | 128 | CFG:104 |
| Index scale | — | **1.0** (raw matmul, NO 1/√d_idx) | OFFICIAL modeling 573 (was 1/√128) |
| Index QK-norm | — | Gemma `x*(1+w)`, zeros-init, eps 1e-6 | OFFICIAL modeling 137-153, 548 |
| Index RoPE | — | partial, rotary_dim 64, θ 5e6 (same as main) | OFFICIAL modeling 563-564 |
| Index-head collapse | — | `amax` over 4 idx heads → ONE set/query | OFFICIAL modeling 580 |
| Block size (tokens) | B_k | 128 | CFG:107 |
| Top-k blocks | k | 16 | CFG:106 |
| Block pooling | — | MAX | CFG:170 |
| Forced sink blocks | — | 0 | CFG:171 |
| Forced local blocks | — | 1 | CFG:172 |
| Max context | — | 1,048,576 | CFG:17 |

At 1M tokens: 8192 KV blocks, of which only 16 are read → ~512× block reduction (matches the
paper's ~28× per-token attention-FLOP reduction once head-dim and the dense proxy pass are folded in).

---

## 5. Differences from MiniMax-M2 attention (M2 = the box's transformers reference)

M2 (`modeling_minimax_m2.py`, confirmed via transformers/main):

| Aspect | M2 | M3 |
|---|---|---|
| **Sparsity** | **None** — fully dense GQA, `eager_attention_forward` does full `Q·Kᵀ` with causal mask, every layer | **MSA block-sparse** on layers 3–59: indexer + top-16-block selection; only 3 dense layers |
| **Indexer branch** | absent | present (`W_q^idx`, `W_k^idx`, d_idx=128, 4 idx-Q + 1 shared idx-K head, BlockMaxPool, TopK) |
| **KV read per query** | all past tokens | ≤ 16 blocks × 128 + 1 forced local block |
| **Index/proxy K cache** | none | extra K-only buffer (1 head × 128) per token, kept beside KV cache |
| GQA heads | GQA (H_q/H_kv) | GQA, **H_q=64, H_kv=4** |
| QK-norm | yes (`q_norm`/`k_norm` RMSNorm) | yes (`use_qk_norm`, `qk_norm_type=per_head`, CFG:25/96) |
| RoPE | applied to Q/K (full rotary in the M2 forward) | applied to MAIN Q/K, **PARTIAL** (rotary_dim=64 / factor 0.5, θ=5e6); **index branch ALSO gets the same partial RoPE** (OFFICIAL modeling 563-564 — the paper-derived "no index RoPE" was WRONG) |
| Output gate | none (`o_proj` only) | none (`attention_output_gate=false`, CFG:20) |
| Scale | `head_dim**-0.5` | `head_dim**-0.5` (same) |
| MoE split | per its own config | 3 dense + 57 MoE (CFG:34-95) |

Same as M2: GQA substrate, `1/√d_h` scale, per-head QK-norm, no output gate, causal.
New in M3: **everything to do with block selection** + partial rope + the dense-prefix-of-3.

For the **3 dense M3 layers (0,1,2)** the attention is essentially M2-style dense GQA (no indexer,
no selection) — implement them with the ordinary dense GQA path.

---

## 6. PURE-TORCH REFERENCE (executable spec / parity oracle) — CORRECTED 2026-06-14

> **The AUTHORITATIVE, parity-tested oracle is `ref/msa_reference.py`** (validated vs the REAL
> `MiniMaxM3VLIndexer`: block_scores `|Δ|≤4.8e-07`, selection Jaccard 1.0 — `tests/test_msa_hf_parity.py`).
> Use it, not this snippet, as the contract. The block below is a faithful condensation of that oracle's
> index/selection path, **corrected for the four index-branch fixes** (Gemma norm `x*(1+w)`; partial
> index RoPE `rotary_dim=64, theta=5e6`; **raw** index scale 1.0; **head-collapse → ONE shared block
> set per query**). The pre-2026-06-14 version of this snippet (no rope, `/sqrt(d_idx)`, per-GQA-group
> selection) was paper-derived and **WRONG** — do not resurrect it.

This is the contract the Gaudi kernel must match bitwise-modulo-fp. **Index branch:** the indexer
takes the RAW projected `q_idx`/`k_idx` and applies the Gemma index norm + partial RoPE itself (it
reuses the **main** `cos`/`sin`). **Main branch:** RoPE on main Q/K and the main Gemma QK-norm are
assumed **already applied** to `q`/`k` before these functions (kernel boundary == post-rope,
post-norm). Single-sequence, absolute positions; batch by looping.

```python
import math, torch

# ---- dims for M3 text tower ----
HEAD_DIM   = 128   # d_h
H_Q        = 64    # query heads
H_KV       = 4     # kv / index-query heads ; GQA group G = 16
D_IDX      = 128   # index dim
BLOCK      = 128   # sparse_block_size (tokens per KV block)
TOPK       = 16    # sparse_topk_blocks
FORCE_INIT = 0     # sparse_init_block  (forced sink blocks)   -> M3 = 0
FORCE_LOCAL= 1     # sparse_local_block (forced local blocks)  -> M3 = 1


ROPE_THETA = 5_000_000.0   # rope_theta ; the index branch reuses the MAIN cos/sin
ROTARY_DIM = 64            # rotary_dim ; partial rope (first 64 of 128 rotated)
IDX_SCALE  = 1.0           # OFFICIAL index score scale: RAW matmul, NO 1/sqrt(d_idx)


def gemma_rms_norm(x, weight, eps=1e-6):
    """MiniMaxM3VLRMSNorm: fp32 normalize, scale by (1 + weight). weight zero-init (Gemma)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).to(x.dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def index_cos_sin(causal_pos, rotary_dim=ROTARY_DIM, theta=ROPE_THETA):
    """The MAIN cos/sin the indexer reuses (default rope). returns cos,sin : [S, rotary_dim]."""
    pos = causal_pos.float()
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)            # [S, rotary_dim]
    return emb.cos(), emb.sin()


def apply_index_rope(q_idx, k_idx, cos, sin):
    """Partial RoPE on index q/k (modeling apply_rotary_pos_emb): rotate first rotary_dim, pass rest.
    q_idx [S,H_idx,D], k_idx [S,1,D], cos/sin [S,rotary_dim] (broadcast over heads)."""
    rd = min(cos.shape[-1], q_idx.shape[-1])           # == official cos[...,:head_dim] (no-op for M3)
    cf, sf = cos.float()[..., :rd].unsqueeze(1), sin.float()[..., :rd].unsqueeze(1)
    def _r(x):
        xf = x.float(); rot, pas = xf[..., :rd], xf[..., rd:]
        return torch.cat([(rot * cf) + (_rotate_half(rot) * sf), pas], dim=-1)
    return _r(q_idx), _r(k_idx)


def msa_index_scores(q_idx, k_idx, causal_pos, block_size=BLOCK):
    """BlockMaxPool of the index branch (OFFICIAL MiniMaxM3VLIndexer.forward), PRE head-collapse.

    q_idx : [S, H_idx, D_idx]   Gemma-NORMED index query (H_idx = index_n_heads = 4)
    k_idx : [S, 1,    D_idx]   Gemma-NORMED single shared index key head
    Partial RoPE is applied HERE (after the norm, before the dot). RAW dot (scale 1.0), NO relu.
    returns block_scores : [S, H_idx, n_blk]  per-(query, index-head, block) MAX token score.
    """
    S = q_idx.shape[0]
    n_blk = (S + block_size - 1) // block_size
    cos, sin = index_cos_sin(causal_pos)
    q_idx, k_idx = apply_index_rope(q_idx, k_idx, cos, sin)               # partial rope (theta=5e6)
    qf = q_idx.float(); kf = k_idx.float()[:, 0, :]                       # k: [S, D_idx]
    s = torch.einsum("qhd,kd->qhk", qf, kf) * IDX_SCALE                   # RAW (1.0), NO /sqrt(d_idx)
    k_pos = torch.arange(S, device=q_idx.device)                         # key abs pos = row idx
    s = s.masked_fill((k_pos[None, :] > causal_pos[:, None])[:, None, :], float("-inf"))  # j<=i
    pad = n_blk * block_size - S
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    return s.view(S, q_idx.shape[1], n_blk, block_size).amax(dim=-1)     # [S, H_idx, n_blk]


def msa_block_select(block_scores, block_size, topk_blocks, causal_pos,
                     force_init=FORCE_INIT, force_local=FORCE_LOCAL):
    """HEAD-COLLAPSE + top-k KV-block selection.  ONE shared set per query (OFFICIAL, modeling 580).

    block_scores : [S, H_idx, n_blk]  ->  collapse index heads (amax over dim=1)  ->  topk
    returns block_idx : [S, topk_blocks] int64  (NO head axis) — ascending, -1 padded.
        Forced: first `force_init` blocks (sink; M3=0) and the local block holding the query
        (plus `force_local-1` preceding) are ALWAYS included.
    """
    scores = block_scores.float().amax(dim=1)                                  # [S, n_blk] HEAD-COLLAPSE
    S, n_blk = scores.shape
    dev = scores.device
    qpos = causal_pos
    # a block b is selectable iff b*block_size <= qpos  (>=1 causally-visible token)
    blk_first = torch.arange(n_blk, device=dev) * block_size                   # [n_blk]
    visible = blk_first[None, :] <= qpos[:, None]                              # [S, n_blk]
    scores = scores.masked_fill(~visible, float("-inf"))                       # never pick future blocks

    forced = torch.zeros(S, n_blk, dtype=torch.bool, device=dev)
    if force_init > 0:
        forced[:, :force_init] = True                                         # sink (M3: none)
    if force_local > 0:
        local_blk = torch.div(qpos, block_size, rounding_mode="floor")        # block holding query i
        for d in range(force_local):
            forced[torch.arange(S, device=dev), (local_blk - d).clamp(min=0)] = True
    forced = forced & visible                                                 # never force a future block
    scores = scores.masked_fill(forced, float("inf"))

    k = min(topk_blocks, n_blk)
    top = torch.topk(scores, k=k, dim=-1).indices                             # [S, k]
    picked = torch.gather(scores, -1, top)
    block_idx = torch.where(torch.isneginf(picked), torch.full_like(top, -1), top)
    sort_key = torch.where(block_idx < 0, torch.full_like(block_idx, n_blk), block_idx)
    block_idx = torch.gather(block_idx, -1, torch.argsort(sort_key, dim=-1))   # ascending, -1 tail
    if k < topk_blocks:
        pad = torch.full((S, topk_blocks - k), -1, dtype=block_idx.dtype, device=dev)
        block_idx = torch.cat([block_idx, pad], dim=-1)
    return block_idx.to(torch.int64)                                          # [S, topk]  ONE set


def msa_attend(q, k, v, block_idx, block_size, scale=None, causal_pos=None):
    """GQA SDPA over the selected blocks' REAL K/V.  PAPER Eq. 8.  ONE shared set per query.

    q : [S, H_q,  D]      post-rope/post-norm query
    k : [S, H_kv, D]      post-rope/post-norm key   (real, uncompressed)
    v : [S, H_kv, D]      real value
    block_idx : [S, topk]  int64 ONE shared selected-block set per query (-1 = pad); every group
                gathers it (head h uses kv-head h//G). main scale = 1/sqrt(D).
    """
    S, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    if scale is None:
        scale = 1.0 / math.sqrt(D)                      # 1/sqrt(128)
    if causal_pos is None:
        causal_pos = torch.arange(S, device=q.device)
    out = torch.zeros(S, Hq, D, device=q.device, dtype=torch.float32)
    kf, vf = k.float(), v.float()
    for i in range(S):                                  # one query token at a time (clarity oracle)
        qi = int(causal_pos[i])
        blks = block_idx[i]                             # ONE set, shared by all heads
        blks = blks[blks >= 0].tolist()
        if not blks:
            continue
        tok = []
        for b in blks:
            tok.extend(range(b * block_size, (b + 1) * block_size))
        tok = torch.tensor(tok, device=q.device, dtype=torch.long)
        tok = tok[tok < S]                              # clip padded tail beyond seq
        for h in range(Hq):
            kv_h = h // G
            kg, vg = kf[tok, kv_h], vf[tok, kv_h]       # [n_sel, D]
            s = (q.float()[i, h] @ kg.T) * scale        # [n_sel]
            s = s.masked_fill(tok > qi, float("-inf"))  # causal on ABSOLUTE positions (bottom-right)
            out[i, h] = torch.softmax(s, dim=-1) @ vg
    return out.to(q.dtype)


# ---- end-to-end oracle (one sparse layer) ----
# q_idx_raw/k_idx_raw are the RAW projected index q/k; the indexer Gemma-norms (with the stored
# index_{q,k}_norm weights) and partial-ropes them itself. See ref/msa_reference.py::msa_forward.
def msa_forward(q, k, v, q_idx_raw, k_idx_raw, q_norm_w, k_norm_w, causal_pos=None):
    S = q.shape[0]
    if causal_pos is None:
        causal_pos = torch.arange(S, device=q.device)
    q_idx = gemma_rms_norm(q_idx_raw, q_norm_w)         # x*(1+w), per head over d_idx
    k_idx = gemma_rms_norm(k_idx_raw, k_norm_w)
    block_scores = msa_index_scores(q_idx, k_idx, causal_pos, BLOCK)   # [S, H_idx, n_blk] (rope inside)
    block_idx = msa_block_select(block_scores, BLOCK, TOPK, causal_pos)  # [S, topk]  ONE set (collapse)
    return msa_attend(q, k, v, block_idx, BLOCK, 1.0 / math.sqrt(q.shape[-1]), causal_pos)
```

**Parity checks to run against the kernel** (mirror MSA's own tests):
- selection: gathered top-k score multiset == `torch.topk` multiset, with `-1`↔`-inf`
  (`test_proxy_kv_e2e.py:verify_topk_selection`).
- consume: cosine-sim > 0.9999 (bf16) vs `msa_attend`, NaN-pattern match
  (`test_sparse_attn.py:check`).
- forced-block invariant: local block (and any `force_init`) always present, ascending, `-1` pad
  (`test_sparse_topk_forced.py`).

---

## 7. Ambiguities — RESOLVED against the OFFICIAL modeling (2026-06-14)

> The four load-bearing index-branch questions below are now **CONFIRMED by code** against
> `transformers 5.12.0` :: `modeling_minimax_m3_vl.py` (`MiniMaxM3VLIndexer` /
> `MiniMaxM3VLRMSNorm` / `MiniMaxM3VLAttention` / `apply_rotary_pos_emb`, installed at
> `$HOME/.venvs/m3calib`, read verbatim) **and VALIDATED**: our corrected oracle
> `ref/msa_reference.py` reproduces the REAL `MiniMaxM3VLIndexer` to block_scores `|Δ|≤4.8e-07`
> and **selection-set Jaccard 1.0** on every query (`tests/test_msa_hf_parity.py`, all 4 sub-tests +
> 3 control divergences PASS); the wire `vllm/msa_m3_wire.py` reproduces the corrected oracle
> **bit-for-bit** (`tests/test_msa_wire_parity.py`, all PASS). The earlier "ambiguous" wording
> below was paper-derived and is superseded.

1. **RoPE on the index branch — RESOLVED: YES (partial RoPE).** The old assumption ("no index RoPE",
   DSA-style rope-free indexer) was **WRONG**. `MiniMaxM3VLIndexer.forward` (modeling 563–564) calls
   `apply_rotary_pos_emb(idx_q, idx_k, cos[...,:head_dim], sin[...,:head_dim])` with the **same
   `cos`/`sin` the main attention uses** (`rotary_dim=64`, `theta=5e6`, `partial_rotary_factor=0.5`),
   **after** the Gemma norm and **before** the dot. The first 64 of the 128-dim index q/k are rotated
   (the `cos[...,:128]` slice is a no-op since `cos` is only 64 wide). See §2b.2. Oracle/wire now apply
   it (`build_index_cos_sin`/`apply_index_rope`).
2. **Index-score activation — RESOLVED: NONE (raw dot).** Confirmed in code: `scores =
   matmul(idx_q.float(), idx_k.float().transpose(-1,-2))` fed straight to amax/topk — no ReLU/sigmoid/
   softmax (modeling 573). Matches paper §4 "exp-free"; differs from DeepSeek DSA (ReLU). ALSO RESOLVED
   here: the score scale is **1.0 (raw), NOT `1/sqrt(d_idx)`** — `MsaDims.index_scale = 1.0` (§2b.4;
   selection-neutral but matches the code).
3. **Index norm form — RESOLVED: Gemma RMSNorm `x*(1+w)`, weight zero-init.** (Was implicitly absent /
   plain `x*w`.) `index_q_norm`/`index_k_norm` are `MiniMaxM3VLRMSNorm` (Gemma `x*(1+w)`, fp32, eps=1e-6,
   per head over d_idx=128, after reshape, before rope). Checkpoint weights cluster near 0 (mean ~0.2,
   some negative) — impossible for plain `x*w`. See §2b.1.
4. **Selection granularity / head-collapse — RESOLVED: ONE shared block set per query.** (Was per-GQA-
   group, 4 independent top-k sets — **WRONG**.) `block_scores = scores.amax(dim=-1).amax(dim=1)`
   (modeling 580) collapses the 4 index heads with a second `amax` **before** topk → `[B,Sq,topk]`, ONE
   set used by every GQA group/head (`build_block_mask` broadcasts via `.unsqueeze(1)`). **Kernel/consume
   implication:** exactly one gather per query → the consume is a SINGLE op call over all `H_q` heads
   (the prior per-group 4-call loop is gone), and the on-card `block_indexer_topk` (mode=1) needs one
   `reduce_max` over the head axis inserted before its final topk to emit `[Sq,kk]` (see §2d / §3 and
   `vllm/msa_m3_wire.py` `_cpu_block_indexer_topk` KERNEL IMPLICATION). No consume-schema change needed.

### Still genuinely open (orthogonal to the index-math fixes above)
- **Exact local-block definition vs `force_end`.** `sparse_local_block=1` → the block holding the
  query (`position_ids // block_size`, walking back `local_blocks-1`); matches the kernel's
  `force_end`. Low risk (coincide for the diagonal/decode case). Confirmed against the official
  `scatter_(local_idx, inf)` (modeling 582–587).
- **Dense-layer attention details (layers 0–2).** Assumed identical to M2 dense GQA (full causal,
  partial rope, Gemma per-head QK-norm — same `MiniMaxM3VLRMSNorm`, no gate). Not separately specced.
- **Decode / paged (T≠S).** The validated wire is PREFILL-first (T==S, abs pos from `positions`).
  Paged-history decode with the proxy K-index cache is a follow-up (§3 two-stage pipeline; not yet
  parity-gated against the official `past_key_values.update_index`).
- **Per-token vs per-block-of-queries selection.** Official selects per query token; output parity is
  unaffected if the kernel selects per-token. Flag if a coarser query-block granularity is used.

— end MSA_SPEC.md —
