# MSA index/selection branch — reconfirm vs OFFICIAL modeling code

**Verdict: FIX-NEEDED (two independent corruptions on the index branch).**
**Authoritative source: `huggingface/transformers` @ `main` — `src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py` (class `MiniMaxM3VLIndexer`).** This file EXISTS now (it did not when `ref/msa_reference.py` / `MSA_SPEC.md` were written from the PAPER). It supersedes the paper for every index-branch detail.
**Confidence: VERY HIGH.** Official PyTorch modeling code read verbatim, cross-checked against the landed checkpoint shapes AND the actual stored norm-weight statistics. Date: 2026-06-14.

---

## TL;DR — what the official indexer actually does (and where we are wrong)

| # | Index-branch op | OUR impl (`msa_m3_wire.py` / oracle) | OFFICIAL `MiniMaxM3VLIndexer` | Status |
|---|---|---|---|---|
| 1 | RoPE on index q/k | **NONE** (oracle + wire: "NO rope") | **YES** — partial RoPE, **same cos/sin as main** (rotary_dim=64, theta=5e6), applied AFTER the norm | ❌ **BUG (missing rope)** |
| 2 | index_q_norm/index_k_norm form | plain RMSNorm `x * w` (`_IndexRMSNorm`) | **Gemma RMSNorm `x * (1 + w)`** (`MiniMaxM3VLRMSNorm`) | ❌ **BUG (wrong norm form)** |
| 2b | norm placement | per-head, over d_idx=128, AFTER reshape to heads, BEFORE dot | per-head, over d_idx=128, AFTER reshape to heads, **BEFORE rope**, BEFORE dot | ✅ placement correct |
| 2c | norm eps | 1e-6 | `config.rms_norm_eps` = **1e-6** | ✅ |
| 3 | index score scale | `* 1/sqrt(d_idx)` (=1/sqrt(128)) | **NO scaling** — raw `matmul`, fed straight to amax/topk | ⚠️ benign for selection (see §3) but **does not match code**; remove for exactness |
| 4 | activation on scores | none | none (raw dot → amax → topk) | ✅ |
| 5 | block max-pool / topk / force-local / -1 pad | matches | `scores.amax(-1).amax(dim=1)`; topk; local via `scatter_(inf)`; `topk_scores==-inf → -1` | ✅ semantics match |

Two of these (RoPE, Gemma-norm) **change which blocks are selected** and silently corrupt MSA at long context. **Do not flip `MSA_READY` until both are fixed.**

---

## The official indexer, verbatim (transformers `main`)

`MiniMaxM3VLRMSNorm` — **Gemma-style** (note `weight` inits to **zeros**, scale is `1 + weight`):

```python
class MiniMaxM3VLRMSNorm(nn.Module):
    """Gemma-style RMSNorm: normalizes in fp32 and scales by `weight + 1`."""
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))   # <-- zeros, not ones
        self.eps = eps
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())  # <-- (1 + w), Gemma form
        return output.type_as(x)
```

`MiniMaxM3VLIndexer.__init__`:

```python
self.head_dim    = config.index_head_dim        # 128
self.num_heads   = config.index_n_heads         # 4   (== num_key_value_heads)
self.block_size  = config.index_block_size      # 128
self.topk_blocks = config.index_topk_blocks     # 16
self.local_blocks= config.index_local_blocks    # 1
self.q_proj = nn.Linear(hidden, index_n_heads * index_head_dim, bias=False)  # [512, 6144]
self.k_proj = nn.Linear(hidden, index_head_dim, bias=False)                  # [128, 6144]
self.q_norm = MiniMaxM3VLRMSNorm(index_head_dim, eps=config.rms_norm_eps)
self.k_norm = MiniMaxM3VLRMSNorm(index_head_dim, eps=config.rms_norm_eps)
```

`MiniMaxM3VLIndexer.forward` (the load-bearing 6 lines):

```python
idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
idx_q = self.q_norm(idx_q).transpose(1, 2)            # [B, H_idx, Sq, D]   norm BEFORE rope
idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
idx_k = self.k_norm(idx_k).transpose(1, 2)            # [B, 1,  Sq, D]
cos, sin = position_embeddings
idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., :self.head_dim], sin[..., :self.head_dim])
# ... causal mask (k_pos > position_ids -> -inf), pad to block, then:
scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))   # NO 1/sqrt(d) scale
scores = scores.view(B, num_heads, q_len, num_key_blocks, block_size)
block_scores = scores.amax(dim=-1).amax(dim=1)        # block max-pool, then max over index heads
# local force: block_scores.scatter_(-1, local_idx, inf)
topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
return topk_indices.masked_fill(topk_scores == -inf, -1)
```

Order of ops on each index head: **proj → reshape-to-heads → Gemma-RMSNorm → partial-RoPE → dot → causal mask → block-max → (max over index heads) → +inf force-local → topk → -1 pad.**

---

## 1. RoPE on the index branch — IT IS APPLIED (paper was wrong/incomplete)

`apply_rotary_pos_emb(idx_q, idx_k, cos[..., :head_dim], sin[..., :head_dim])` is the **same** function and the **same `cos`/`sin`** the main attention uses (`position_embeddings` is computed once per layer and passed in). The PAPER's "plain linear projection, bare dot" (Eq. 5/6) is **incomplete** — exactly the failure mode the FLAG anticipated.

**Exactly how much gets rotated (the subtle part):** `head_dim=128`, `rotary_dim=64`, `partial_rotary_factor=0.5`.
- The rotary embedding builds `dim = head_dim * partial = 64`, `inv_freq` length 32, `freqs` width 32, `emb = cat((freqs,freqs))` ⇒ **`cos`/`sin` last-dim = 64**.
- `apply_rotary_pos_emb` computes `rotary_dim = cos.shape[-1]` (= 64) and rotates `q[..., :64]`, passing `q[..., 64:]` through (`rotate_half` splits the rotated 64 at 32). This is the partial RoPE.
- The indexer slices `cos[..., :self.head_dim]` = `cos[..., :128]`, but `cos` is only 64 wide, so it gets **all 64 columns** — i.e. the slice is a **no-op**. The indexer therefore rotates the **first 64 of its 128-dim** index q/k, identical to the main branch.

**Net:** index branch gets **partial RoPE, rotary_dim=64, theta=5,000,000, applied after the norm, before the dot** — the same positional treatment as main q/k. Config: `rope_theta=5000000`, `rotary_dim=64`, `partial_rotary_factor=0.5` (all verified in `/mnt/data2/minimax-m3-bf16/config.json text_config`).

> Why the paper-derived "no rope" survived this long: the SET-match parity gate compares against the *dense oracle*, which also has no index rope, so the two agree with each other while both diverge from the real checkpoint. RoPE changes the per-token dot, hence which blocks win — invisible to a self-consistent oracle, visible only at real-context eval.

## 2. index_q_norm / index_k_norm — Gemma RMSNorm `(1 + w)`, NOT plain `w`

The norm class is **Gemma-style** (`x * (1 + weight)`), and its `weight` initializes to **zeros**. Our `_IndexRMSNorm` is plain `x * weight`.

**Checkpoint proof (decoded the actual bf16 weights from `model-00003-of-00059.safetensors`, layer 3):**

```
index_q_norm.weight  mean=+0.1914 std=0.1243 min=-0.2793 max=+0.5117   first6=[0.281,0.190,-0.033,0.313,0.165,0.039]
index_k_norm.weight  mean=+0.2274 std=0.0806 min=-0.0383 max=+0.6055
q_norm.weight  (main) mean=+0.1951 std=0.0836 min=-0.3164 max=+0.6914
```

Weights cluster near **0.0** (and include **negative** values) — impossible for a plain `x*w` norm (whose learned weights sit near 1.0), exactly right for Gemma `x*(1+w)`. Config corroborates: `text_config.use_gemma_norm = true`, `qk_norm_type` present, `use_qk_norm` set. The same Gemma norm is used by the **main** q_norm/k_norm too (same stats), so this also confirms the main qk-norm handling must be Gemma `(1+w)` on the box.

**Magnitude of the bug if unfixed:** with `w ≈ 0.2`, the true scale is `1+w ≈ 1.2`; our plain `x*w` applies `≈ 0.2`. That is a **~6× shrink** of the normalized index vector, per-element re-weighted (some dims go *negative* where `w<0`, flipping sign). The dot product `q_idx·k_idx` is mangled non-monotonically ⇒ wrong topk blocks. This is the single most dangerous item: not a global rescale (which topk would absorb) but a per-coordinate distortion of both operands.

**Placement / eps are correct in our wire:** per-head, over the last (d_idx=128) axis, after the `view(...,n_heads,d_idx)` reshape, before the dot; `eps = config.rms_norm_eps = 1e-6`. Only the **functional form** (and the implied weight-init/centering) is wrong.

## 3. index score scaling — code uses NO `1/sqrt(d_idx)`

Official: `scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1,-2))` — **raw**, no scale, straight into `amax`/`topk`. Our oracle and wire multiply by `index_scale = 1/sqrt(128)`.

This is a single **positive global constant** applied to every score, so it **cannot change the argmax/topk ordering** — selection is identical with or without it. So it is **not a correctness bug for block selection**. BUT it does not match the reference and is pointless work; for exact-match with the modeling code (and to avoid confusion), **drop the `1/sqrt(d_idx)` on the index branch** (set `index_scale = 1.0`). The main-branch scale `1/sqrt(head_dim)` is unaffected and stays. (No bias, no activation anywhere on the index path — confirmed.)

## Other index-branch ops — checked, all clean
- **No bias** on `q_proj`/`k_proj` (`bias=False`) — matches `ReplicatedLinear(bias=False)`.
- **No activation** (no ReLU/sigmoid/softmax) before pooling — matches oracle/PAPER §4 exp-free.
- **Block max-pool**: `scores.amax(-1)` over the 128 tokens in a block — matches.
- **Max over index heads**: official does `.amax(dim=1)` ⇒ collapses the 4 index heads to ONE block score per query *before* topk (`block_scores: [B, Sq, num_key_blocks]`, topk shared across heads). **Re-examine our per-GQA-group selection** (see ⚠️ below).
- **force-local** = `index_local_blocks = 1` recent block via `scatter_(inf)`; **no init/sink** (`index_init_blocks` absent ⇒ 0). Matches `sparse_init_block=0, sparse_local_block=1`.
- **-1 pad** when a topk slot is `-inf` — matches.

---

## ⚠️ SECONDARY FINDING (verify, possibly a third fix) — selection granularity

Official `block_scores = scores.amax(dim=-1).amax(dim=1)` reduces over **dim=1 = the index-head axis**, producing `[B, Sq, num_key_blocks]` — a **single block set per query, shared by ALL heads** (one topk for the whole layer at that position). Our impl selects **per GQA group** (`[T, H_kv, n_blk]`, 4 independent topk sets, consume loops per group). With `index_n_heads == num_key_value_heads == 4`, the official code computes the 4 index-head scores then **max-collapses them to one** before topk; we keep 4 separate selections.

If the reference truly shares one block set across all 4 groups, our per-group selection is a **different (finer) selection** and would diverge. This needs confirmation against the consume side (does main attention gather one shared block set, or per-kv-head?). **Flagged for a dedicated follow-up** — it is orthogonal to the norm/rope fixes but is in the same "selection silently wrong" risk class. Pull the `MiniMaxM3VLAttention.forward` gather/`block_indices` plumbing to settle it.

---

## EXACT FIX

### Fix A — index norm to Gemma `(1 + w)` (`vllm/msa_m3_wire.py`, `_IndexRMSNorm`)

```python
class _IndexRMSNorm(torch.nn.Module):
    """Gemma-style RMSNorm (MiniMaxM3VLRMSNorm): fp32 normalize, scale by (1 + weight).
    weight inits to ZEROS (Gemma), NOT ones. Matches transformers main
    modeling_minimax_m3_vl.MiniMaxM3VLRMSNorm; checkpoint index_{q,k}_norm weights
    cluster near 0 (mean ~0.2, some negative) — confirms (1+w)."""
    def __init__(self, d_idx: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(d_idx))   # was torch.ones
        self.variance_epsilon = float(eps)
    def forward(self, x):
        orig = x.dtype
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.variance_epsilon)
        return (xf * (1.0 + self.weight.to(torch.float32))).to(orig)   # (1 + w)
```

### Fix B — apply partial RoPE to index q/k, AFTER the norm, BEFORE the dot

The index branch must rotate the first `rotary_dim=64` dims of both `q_idx` and `k_idx` with the SAME cos/sin (theta=5e6) the main branch uses. In `compute_index_qk` (after the norm), or in `msa_wire` before building `q_op`/`k_op`, insert the rotation. The wire currently receives main q/k **already roped**; it must rope the index q/k with the same `positions`→cos/sin. Easiest: compute cos/sin from `positions` (rotary_dim=64, base=5e6) once and apply `rotate_half`-style partial rope to `q_idx[...,:64]` / `k_idx[...,:64]`, pass-through `[...,64:]`. The oracle `msa_index_scores` must take **roped** q_idx/k_idx (drop the "NO rope" contract; add a rope step or accept pre-roped inputs). Update `MSA_SPEC.md §2b/§7.1`, the oracle docstring, and `msa_reference.py` lines 55-64 (the "(1) RoPE … -> NO" block) — that conclusion is now **WRONG**.

### Fix C — drop the index-branch `1/sqrt(d_idx)` scale (exactness, not correctness)

`MsaDims.index_scale` → return `1.0` (or remove the multiply / set the op `weights` to 1.0). Selection is unchanged either way; this just matches the reference and removes a misleading factor. Update both `MsaDims.index_scale` docstrings (oracle + wire) which cite "PAPER Eq.6".

### Keep `MSA_APPLY_INDEX_NORM` default True — but it is now NON-optional
The norm is real and load-bearing; the toggle should remain only for the bare-dot parity path A, and the default-True path must use the **Gemma** form above.

---

## Net answer

- **RETURN: FIX-NEEDED.**
- **Fixes:** (A) index_{q,k}_norm → **Gemma RMSNorm `x*(1+w)`**, weight-init zeros [critical]; (B) **apply partial RoPE** (rotary_dim=64, theta=5e6, same cos/sin as main) to index q/k after the norm, before the dot — our "NO rope" oracle/wire/spec are **wrong** [critical]; (C) **remove the `1/sqrt(d_idx)`** index-score scale [exactness only, selection-neutral]. Plus **verify** whether selection is one shared block set per query (official `amax(dim=1)` over index heads) vs our per-GQA-group selection [secondary, possibly a 4th fix].
- **Norm placement/eps were already correct** (per-head over d_idx=128, after reshape, eps=1e-6).
- **Authoritative source:** `huggingface/transformers` @ `main`, `src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py`, class `MiniMaxM3VLIndexer` + `MiniMaxM3VLRMSNorm` + `apply_rotary_pos_emb`/`compute_default_rope_parameters`. Config: `configuration_minimax_m3_vl.py` (`index_head_dim=128, index_n_heads=4, index_block_size=128, index_topk_blocks=16, index_local_blocks=1, rotary_dim=64`) and local `/mnt/data2/minimax-m3-bf16/config.json` (`use_gemma_norm=true`, `rope_theta=5e6`, `partial_rotary_factor=0.5`, `rms_norm_eps=1e-6`).
- **Confidence: VERY HIGH** — official PyTorch code read verbatim; Gemma-norm independently proven by decoding the checkpoint's stored norm weights (cluster near 0, include negatives); RoPE math traced end-to-end through the actual `cos`-width=64 derivation.

### Citations
- transformers `main` modeling: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py` (raw: `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py`) — `MiniMaxM3VLIndexer`, `MiniMaxM3VLRMSNorm`, `apply_rotary_pos_emb`, `rotate_half`, `MiniMaxM3VLRotaryEmbedding.compute_default_rope_parameters`.
- transformers `main` config: `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/minimax_m3_vl/configuration_minimax_m3_vl.py`.
- Local checkpoint: `/mnt/data2/minimax-m3-bf16/config.json` (`text_config`), and decoded bf16 norm weights from `model-00003-of-00059.safetensors` (`language_model.model.layers.3.self_attn.index_q_norm/index_k_norm/q_norm`).
- vLLM: NO `minimax_m3` model in `model_executor/models` on `main` yet (M3 support still in development; recipes only) — transformers is the sole authoritative modeling source. SGLang/official MiniMax-AI/MSA repo are the consume/select kernels, not the index projections.
