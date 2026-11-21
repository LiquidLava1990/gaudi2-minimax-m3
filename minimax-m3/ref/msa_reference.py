"""MiniMax-M3 MSA (MiniMax Sparse Attention) — pure-torch executable spec + parity oracle.

This is the numerically-authoritative reference the Gaudi MSA OpBackend kernel is validated
against (mirrors how DeepSeek/GLM DSA was done on this box: see
``hpu-dsa-ext/hpu_dsa_ext/sparse_indexer_hpu.py`` :: ``reference_indexer_topk`` (oracle) and
``sparse_mla_consume.py`` :: ``sparse_mla_masked_attention`` (oracle) /
``sparse_mla_gather_attention`` (product gather form)). Wrong math here => word-salad on the box,
so every nontrivial choice is cited to the official source, file:line / URL.

================================================================================================
WHAT MSA IS (one sparse layer, layers 3..59; layers 0..2 are plain dense GQA)
================================================================================================
A GQA layer with REAL (uncompressed) K/V — NOT MLA. A lightweight "Index Branch" scores KV blocks
(block_size=128 tokens) with one extra index-query head per GQA group + a single shared index-key
head, max-pools the per-token index scores to per-block scores, and top-k selects (k=16) the blocks
each (query, GQA-group) attends to. The local block containing the query is always force-included.
The "Main Branch" then runs standard scaled-dot-product GQA softmax attention over only the
gathered tokens of the selected blocks, causal on ABSOLUTE positions.

The kernel boundary is POST-rope / POST-qk-norm for the MAIN q/k (those are applied by the model
before these functions). The INDEX branch q/k are POST index-projection; this oracle applies the
index Gemma-QK-norm and the index partial-RoPE itself (it takes the RAW projected q/k and the main
cos/sin), matching the OFFICIAL ``MiniMaxM3VLIndexer.forward``. NO activation on the index scores.

>>> 2026-06-14 CORRECTION (authoritative source now exists; paper-derived math was WRONG) <<<
The official PyTorch modeling ``modeling_minimax_m3_vl.MiniMaxM3VLIndexer`` /
``MiniMaxM3VLRMSNorm`` / ``MiniMaxM3VLAttention`` (transformers 5.12.0, installed at
$HOME/.venvs/m3calib) supersedes the paper for EVERY index-branch detail. Four corrections,
all reconfirmed against that code (see notes/reviews/index-norm-reconfirm.md):
  (1) index_q_norm / index_k_norm are Gemma RMSNorm ``x*(1+w)`` (weight zero-init), per index head
      over d_idx=128, eps=1e-6, applied AFTER reshape-to-heads, BEFORE rope, BEFORE the dot. The
      old oracle modeled NO index norm at all — wrong.
  (2) The index branch GETS partial RoPE (rotary_dim=64, theta=5e6), the SAME cos/sin as the main
      attention, applied AFTER the norm, BEFORE the dot. The old "NO rope" was wrong.
  (3) Index scores are a RAW matmul — NO 1/sqrt(d_idx). ``MsaDims.index_scale`` is now 1.0
      (selection-neutral, but matches the code).
  (4) HEAD-COLLAPSE: official is ``block_scores = scores.amax(-1).amax(dim=1)`` -> ONE shared block
      set per query, collapsing the 4 index heads BEFORE topk/force-local. The old per-GQA-group
      (4 independent sets) was wrong. Selection is now ``[T, n_blk] -> [T, topk]`` (one set), and the
      consume gathers that ONE set for every GQA group (one gather/query, not 4).

================================================================================================
SOURCES (priority: official MSA repo PyTorch refs > MSA paper > vLLM/SGLang model code)
================================================================================================
[CFG]   local checkpoint config  /mnt/data2/minimax-m3-bf16/config.json  (all dims under text_config)
[PAPER] "MiniMax Sparse Attention" — local PDF  /tmp/MSA_repo/docs/MiniMaxSparseAttention.pdf
        (pdftotext -> /tmp/msa_paper.txt; line numbers below are from that extraction);
        public mirror https://arxiv.org/abs/2606.13392 / https://huggingface.co/papers/2606.13392
[MSA]   official MiniMax-AI/MSA GitHub, cloned to /tmp/MSA_repo (https://github.com/MiniMax-AI/MSA).
        This repo is the CONSUME + SELECT kernel; it takes ALREADY-pooled block ``max_score`` as
        input. The index PROJECTIONS + per-token scoring live in the model attention module
        (PAPER Eq. 5-7), NOT in this repo — so the consume/select/force semantics are pinned to
        the MSA repo, and the rope/activation questions are pinned to PAPER.
        - sparse_ref (gather selected blocks, causal on absolute pos):
          /tmp/MSA_repo/tests/regression/test_sparse_attn.py:46-75
        - sparse_topk_select (force_begin/force_end, -1 pad, ascending):
          /tmp/MSA_repo/python/fmha_sm100/api.py:1179-1291
        - forced-block invariant test: /tmp/MSA_repo/tests/smoke/test_sparse_topk_forced.py
        - q2k shape [batch, head_kv, Sq, topK], -1 padded:
          /tmp/MSA_repo/python/fmha_sm100/cute/sparse_index_utils.py:16-25

NOTE on availability: at the time of writing, vLLM `main` has NO minimax_m3 model file
(model_executor/models/registry.py has MiniMaxM1/M2/VL01 only — verified by fetching
raw.githubusercontent.com/vllm-project/vllm/main/.../registry.py, no `m3` arch), and the HF repo
exposes no remote-code modeling_minimax_m3*.py (config auto_map points only at
configuration_minimax_m3_vl.py — config classes, no math; modeling_* resolve -> HTTP 404). The
PAPER is therefore the authoritative source for the index-branch math; it is unambiguous on all
three points below.

================================================================================================
RESOLVED AMBIGUITIES (MSA_SPEC.md §7) — official source wins; corrections to the spec noted
================================================================================================
(1) RoPE on the INDEX / selection branch?  -> YES (paper-derived "NO" was WRONG; code confirms).
    OFFICIAL MiniMaxM3VLIndexer.forward (m3calib, modeling line ~564):
        idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., :head_dim], sin[..., :head_dim])
    applied AFTER the Gemma norm, BEFORE the dot, with the SAME cos/sin the main attention uses
    (position_embeddings is computed once per layer and passed in). partial_rotary_factor=0.5,
    head_dim=128 => rope `dim=64`, so cos/sin last-dim == 64; the indexer's `cos[..., :128]` slice
    is a no-op (cos is only 64 wide) so it rotates the FIRST 64 of the 128-dim index q/k, identical
    to the main branch (rotary_dim=64, theta=5e6, CFG:21-23). The PAPER (Eq.5/6) is incomplete.
    => THIS ORACLE applies partial RoPE to q_idx/k_idx (msa_index_scores takes raw q/k + cos/sin).
       MSA_SPEC.md §2b/§7.1 "no index rope" is now WRONG.

(2) Activation (ReLU / nonlinearity) on index scores before block-max-pool?  -> NONE.
    PAPER Eq. 6 (msa_paper.txt:191-231): score is the raw scaled dot product, then a plain max over
    the in-block (causally-visible) token scores. No ReLU/exp/sigmoid. PAPER §4 "Exp-free selection"
    (msa_paper.txt:449-452) verbatim: "the index module ranks the index scores s directly. Since
    softmax is order-preserving ... The forward pass, therefore, bypasses the max/exp/sum steps of
    softmax and passes raw scores directly to selection." The ONLY ReLU in the paper (:1261) is in
    related-work describing DeepSeek's lightning indexer, explicitly contrasted with MSA.
    => Confirms MSA_SPEC.md §2b/§7.2. THIS DIFFERS FROM DSA: the DSA reference
       (sparse_indexer_hpu.py:66,86) applies ReLU to the per-head logits before the weighted
       head-sum; MSA does NOT. There is also no per-head weighting and no head-sum in MSA — the
       index head per GQA group is used directly (each group selects independently).

(3) Force-local / force-init block semantics (which blocks are always included)?
    Config: sparse_init_block = 0 (CFG:171) -> NO forced sink/begin blocks.
            sparse_local_block = 1 (CFG:172) -> exactly ONE forced LOCAL block.
    PAPER §3.2 "Local Block" (msa_paper.txt:390-392) verbatim: "For each query position i and GQA
    group r, the local block containing i is always selected as part of I_i^(r) during both training
    and inference. This fixed allocation reserves one block slot ...". Figure 1 caption (:28-29):
    "the local block is always included regardless of its score."
    MSA kernel semantics (api.py:1210-1217): force_begin_blocks = blocks at indices [0, N) always
    kept (sink); force_end_blocks = "blocks at the end of the VALID sequence (indices
    nvp-N .. nvp-1, closest to the current query)" — i.e. the last valid (causally-trimmed) block(s),
    which for a query at position i is exactly the block containing i. So for M3: force_begin = 0,
    force_end = sparse_local_block = 1, and the forced block is block(i) = floor(i / block_size).
    => Confirms MSA_SPEC.md §2d/§7.3. We force-include the block containing the query position
       (`force_local` of them, walking back from block(i)); with force_init>0 we'd also force the
       first `force_init` blocks. M3 uses force_init=0, force_local=1.

(4) Index-score scaling + selection granularity (paper/MSA-kernel-derived facts were WRONG).
    - SCALE: official scores = matmul(idx_q, idx_k^T) with NO 1/sqrt(d_idx) (modeling line ~573).
      The old oracle used 1/sqrt(128). A single positive global constant cannot change topk
      ordering (selection-neutral), but to match the code MsaDims.index_scale is now 1.0.
    - HEAD-COLLAPSE: official ``block_scores = scores.amax(dim=-1).amax(dim=1)`` (modeling line ~580)
      collapses the 4 index heads (dim=1) into ONE block-score vector per query [B, Sq, n_blk]
      BEFORE topk; force-local (`block_scores.scatter_(-1, local_idx, inf)`, line ~587) and topk
      then run on that ONE collapsed vector, returning [B, Sq, topk] — ONE shared block set per
      query, ALL heads/groups (build_block_mask, line ~618, broadcasts it to every head via
      `.unsqueeze(1)`). The old oracle's per-GQA-group (4 independent sets) was WRONG. This oracle
      now returns ONE set per query and the consume gathers that SAME set for every group.

Other M3 facts pinned: main scale = 1/sqrt(head_dim)=1/sqrt(128) (PAPER Eq. 8 / test_sparse_attn.py:64
`/ math.sqrt(head_dim)`); INDEX scale = 1.0 (raw matmul, official). GQA: H_q=64, H_kv=4, group G=16
(CFG:13-14). Block->token: block b -> tokens [b*B, (b+1)*B) (test_sparse_attn.py:67-68). Causal mask
in consume uses ORIGINAL absolute positions, key ki masked iff ki>qi (test_sparse_attn.py:65-70
`scores.masked_fill_(qi<ki,-inf)`). Selection is ONE shared block set per query (official amax over
the index-head axis), gathered by every GQA group. Output indices ascending, -1 padded
(api.py:1222-1223, test_sparse_topk_forced.py:106-113; official `masked_fill(topk_scores==-inf,-1)`).

================================================================================================
DESIGN (mirrors DSA reference style)
================================================================================================
- fp32 math throughout (the oracle).
- Config-driven dims via ``MsaDims`` (read from the HF config dict); NO magic literals in the
  forward functions — block_size/topk/heads/dims all come from the config or tensor shapes.
- Two consume forms, numerically equivalent, both CPU-testable:
    * ``msa_dense_oracle`` — full causal GQA over ALL keys (the limit topk*block>=S);
    * ``msa_gqa_gather_attention`` — the PRODUCT gather form (gathers only selected tokens),
      using a SENTINEL key/value row for -1 / out-of-range slots (NEVER clamp(min=0)), the same
      discipline as DSA's sparse_mla_gather_attention. This is the consume oracle the kernel diffs.
- Selection mirrors DSA's ``reference_indexer_topk``: an explicit fp32 per-block-max + topk, with
  forced blocks injected as +inf and future blocks masked to -inf, ascending + -1 padded output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

NEG_INF = float("-inf")


# ================================================================================================
# Config reader (config-driven dims — no literals in the forward path). Analog of DsaDims.
# ================================================================================================
@dataclass(frozen=True)
class MsaDims:
    """All MSA dims, read from the HF config dict (``config["text_config"]``). Mirrors DsaDims.

    Defaults match /mnt/data2/minimax-m3-bf16/config.json text_config (used only when a key is
    absent; the from_config reader pulls every value from the dict when present)."""
    hidden_size: int = 6144            # CFG:10
    num_attention_heads: int = 64      # CFG:13   H_q
    num_key_value_heads: int = 4       # CFG:14   H_kv  (GQA group G = H_q/H_kv = 16)
    head_dim: int = 128                # CFG:15   d_h
    rotary_dim: int = 64               # CFG:22   partial RoPE dim (index branch reuses this)
    rope_theta: float = 5000000.0      # CFG:21   RoPE base (index branch reuses main cos/sin)
    # sparse_attention_config (CFG:102-235)
    sparse_index_dim: int = 128        # CFG:104  d_idx
    sparse_num_index_heads: int = 4    # CFG:105  index-Q heads == H_kv (index-K head=1, shared)
    sparse_topk_blocks: int = 16       # CFG:106  k
    sparse_block_size: int = 128       # CFG:107  B_k (tokens per KV block)
    sparse_score_type: str = "max"     # CFG:170  BlockMaxPool
    sparse_init_block: int = 0         # CFG:171  forced sink/begin blocks  -> 0 for M3
    sparse_local_block: int = 1        # CFG:172  forced local blocks       -> 1 for M3

    @property
    def gqa_group(self) -> int:
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"H_q={self.num_attention_heads} not divisible by H_kv={self.num_key_value_heads}")
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def main_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)       # PAPER Eq. 8 ; test_sparse_attn.py:64

    @property
    def index_scale(self) -> float:
        # OFFICIAL MiniMaxM3VLIndexer: scores = matmul(idx_q, idx_k^T) — RAW, NO 1/sqrt(d_idx).
        # (The paper's Eq.6 1/sqrt(128) is selection-neutral but does NOT match the code.) 1.0.
        return 1.0

    @classmethod
    def from_config(cls, config: dict) -> "MsaDims":
        """Build from a parsed HF config.json dict (top-level or already the text_config sub-dict)."""
        tc = config.get("text_config", config)
        sa = tc.get("sparse_attention_config", {})
        d = cls(
            hidden_size=int(tc.get("hidden_size", cls.hidden_size)),
            num_attention_heads=int(tc.get("num_attention_heads", cls.num_attention_heads)),
            num_key_value_heads=int(tc.get("num_key_value_heads", cls.num_key_value_heads)),
            head_dim=int(tc.get("head_dim", cls.head_dim)),
            rotary_dim=int(tc.get("rotary_dim", cls.rotary_dim)),
            rope_theta=float(tc.get("rope_theta", cls.rope_theta)),
            sparse_index_dim=int(sa.get("sparse_index_dim", cls.sparse_index_dim)),
            sparse_num_index_heads=int(sa.get("sparse_num_index_heads", cls.sparse_num_index_heads)),
            sparse_topk_blocks=int(sa.get("sparse_topk_blocks", cls.sparse_topk_blocks)),
            sparse_block_size=int(sa.get("sparse_block_size", cls.sparse_block_size)),
            sparse_score_type=str(sa.get("sparse_score_type", cls.sparse_score_type)),
            sparse_init_block=int(sa.get("sparse_init_block", cls.sparse_init_block)),
            sparse_local_block=int(sa.get("sparse_local_block", cls.sparse_local_block)),
        )
        d.validate()
        return d

    def validate(self) -> None:
        # LOUD failures (DSA build quality bar). The kernel must not silently mis-shape.
        assert self.sparse_score_type == "max", (
            f"only score_type='max' (BlockMaxPool) is implemented; got {self.sparse_score_type!r}")
        assert self.sparse_num_index_heads == self.num_key_value_heads, (
            "MSA: one index-query head per GQA group => sparse_num_index_heads must == "
            f"num_key_value_heads ({self.sparse_num_index_heads} != {self.num_key_value_heads})")
        assert self.sparse_block_size > 0 and self.sparse_topk_blocks > 0
        assert self.sparse_init_block >= 0 and self.sparse_local_block >= 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        # forced blocks cannot exceed the budget (api.py:1255-1258 force_begin+force_end<=topk).
        assert self.sparse_init_block + self.sparse_local_block <= self.sparse_topk_blocks, (
            f"forced blocks ({self.sparse_init_block}+{self.sparse_local_block}) exceed "
            f"topk_blocks={self.sparse_topk_blocks}")


# ================================================================================================
# INDEX-BRANCH norm + rope (OFFICIAL MiniMaxM3VLIndexer / MiniMaxM3VLRMSNorm / apply_rotary_pos_emb).
# These reproduce the modeling code VERBATIM (transformers 5.12.0 @ m3calib). fp32.
# ================================================================================================
def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gemma-style RMSNorm == MiniMaxM3VLRMSNorm: fp32 normalize, scale by (1 + weight).

    weight is the stored parameter (Gemma-init ZEROS, so an unset norm is the identity scale 1.0,
    NOT 0.0). The checkpoint index_{q,k}_norm weights cluster near 0 (mean ~0.2, some negative) ->
    confirms (1+w). Normalizes over the LAST axis (d_idx). Matches modeling lines 145-153.
    """
    orig = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    out = xf * (1.0 + weight.to(torch.float32))
    return out.to(orig)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """modeling rotate_half: split last dim in half, return cat((-x2, x1))."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def build_index_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float,
                        device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The MAIN-attention cos/sin the indexer reuses (MiniMaxM3VLRotaryEmbedding.forward, default
    rope). positions [T] long -> (cos, sin) each [T, rotary_dim], fp32.

    inv_freq = 1/theta^(arange(0,rotary_dim,2)/rotary_dim) (len rotary_dim/2); freqs = pos⊗inv_freq;
    emb = cat((freqs, freqs)) so the last dim == rotary_dim (== 64 for M3). The indexer slices
    cos[..., :head_dim(128)], a no-op since cos is only rotary_dim(64) wide -> rotates the first 64
    of the 128-dim index q/k. Matches modeling lines 300-326 + 564.
    """
    dev = device if device is not None else positions.device
    pos = positions.to(dev).float()                                       # [T]
    inv_freq = 1.0 / (float(theta) ** (torch.arange(0, rotary_dim, 2, device=dev).float() / float(rotary_dim)))
    freqs = torch.outer(pos, inv_freq)                                    # [T, rotary_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                              # [T, rotary_dim]
    return emb.cos(), emb.sin()


def apply_index_rope(q_idx: torch.Tensor, k_idx: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial RoPE on the index q/k (modeling apply_rotary_pos_emb, lines 366-399). Rotates the
    first ``rotary_dim`` dims, passes the rest through.

    q_idx [T, H_kv, D_idx], k_idx [T, 1, D_idx], cos/sin [T, rotary_dim]. Broadcasts cos/sin over the
    head axis (unsqueeze at dim 1). Returns roped q/k same shape, fp32.

    OFFICIAL clamp (MiniMaxM3VLIndexer.forward line 564): the indexer calls
    ``apply_rotary_pos_emb(idx_q, idx_k, cos[..., :head_dim], sin[..., :head_dim])`` — it SLICES
    cos/sin to the index head dim ``D_idx`` first, so the effective rotary width is
    ``min(rotary_dim, D_idx)``. For M3 (rotary_dim=64 < D_idx=128) the slice is a no-op (cos is only
    64 wide). We replicate that slice so a config with ``rotary_dim > D_idx`` (e.g. shrunk test dims)
    cannot index past the vector — exactly the official's behaviour, never a silent over-rotate.
    """
    D = q_idx.shape[-1]                                 # index head dim D_idx
    rd = min(cos.shape[-1], D)                          # == official cos[..., :head_dim] width
    cf = cos.to(torch.float32)[..., :rd].unsqueeze(1)   # [T, 1, rd]
    sf = sin.to(torch.float32)[..., :rd].unsqueeze(1)

    def _rope(x):
        xf = x.to(torch.float32)
        x_rot, x_pass = xf[..., :rd], xf[..., rd:]
        x_embed = (x_rot * cf) + (_rotate_half(x_rot) * sf)
        return torch.cat([x_embed, x_pass], dim=-1)

    return _rope(q_idx), _rope(k_idx)


# ================================================================================================
# 1. INDEX BRANCH — per-block scores (BlockMaxPool).  OFFICIAL MiniMaxM3VLIndexer.forward.
# ================================================================================================
def msa_index_scores(
    q_idx: torch.Tensor,        # [T, H_kv, D_idx]  index query (RAW post-projection; norm+rope here)
    k_idx: torch.Tensor,        # [T, 1, D_idx] or [T, D_idx]  single SHARED index key head (RAW)
    scale: float,               # 1.0 (OFFICIAL: raw matmul, no 1/sqrt(d_idx))
    block_size: int,            # B_k  (config-driven; not a literal)
    causal_pos: torch.Tensor | None = None,  # [T] long, abs pos per query row (default arange)
    *,
    cos: torch.Tensor | None = None,   # [T, rotary_dim] index rope cos (None => no rope, legacy)
    sin: torch.Tensor | None = None,   # [T, rotary_dim] index rope sin
) -> torch.Tensor:              # block_scores [T, H_kv, n_blk]  == M_idx (PER index head, pre-collapse)
    """BlockMaxPool of the index branch (OFFICIAL MiniMaxM3VLIndexer.forward).

    The caller passes the GEMMA-NORMED index q/k (gemma_rms_norm already applied per head over D_idx).
    If cos/sin are given, partial RoPE (apply_index_rope) is applied here AFTER the norm, BEFORE the
    dot — the SAME cos/sin the main attention uses. Per (query i, index head h, KV block b):
        S_idx[h]_{i,j} = (q_idx[i,h] . k_idx[j])         # RAW dot (scale=1.0), NO activation
        M_idx[h]_{i,b} = max over j in block_b with j <= i  of  S_idx[h]_{i,j}    # block max, causal
    Returns the per-index-head block scores [T, H_idx, n_blk]; the HEAD-COLLAPSE (.amax over heads)
    that the official code does before topk is in ``msa_collapse_index_heads`` / ``msa_block_select``.
    Blocks with no causally-visible token score -inf; the padded tail of a partial last block also
    scores -inf (a partial block scores by its REAL tokens only).
    """
    assert q_idx.dim() == 3, f"q_idx must be [T,H_kv,D_idx], got {tuple(q_idx.shape)}"
    T, Hkv, D = q_idx.shape
    if k_idx.dim() == 3:
        assert k_idx.shape[1] == 1, f"k_idx must have a single shared head, got {k_idx.shape[1]}"
        k3 = k_idx
    else:
        k3 = k_idx.unsqueeze(1)
    assert k3.shape == (T, 1, D), f"k_idx must be [T,(1,)D_idx]={ (T,1,D) }, got {tuple(k_idx.shape)}"
    dev = q_idx.device
    if causal_pos is None:
        causal_pos = torch.arange(T, device=dev)
    causal_pos = causal_pos.to(dev).long()

    # partial RoPE (after the norm, before the dot) — OFFICIAL applies the main cos/sin to index q/k.
    if cos is not None and sin is not None:
        q_idx, k3 = apply_index_rope(q_idx, k3, cos, sin)

    n_blk = (T + block_size - 1) // block_size
    qf = q_idx.float()
    kf = k3[:, 0, :].float()
    # token-level dot product: [T_q, H_idx, T_k].  RAW (scale=1.0), NO ReLU / NO softmax (exp-free).
    s = torch.einsum("qhd,kd->qhk", qf, kf) * float(scale)

    # causality (j <= i): mask keys whose absolute position > query abs position.
    key_pos = torch.arange(T, device=dev)                      # key abs positions == row index
    causal_bad = key_pos[None, :] > causal_pos[:, None]        # [T_q, T_k]
    s = s.masked_fill(causal_bad[:, None, :], NEG_INF)

    # pad the key axis up to n_blk*block_size with -inf, then max-pool within each block.
    pad = n_blk * block_size - T
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=NEG_INF)
    s = s.view(T, Hkv, n_blk, block_size)
    block_scores = s.amax(dim=-1)                              # [T, H_idx, n_blk]  == M_idx (per head)
    return block_scores


def msa_collapse_index_heads(block_scores: torch.Tensor) -> torch.Tensor:
    """OFFICIAL head-collapse: ``block_scores.amax(dim=1)`` over the index-head axis (modeling line
    580). [T, H_idx, n_blk] -> [T, n_blk]. ONE shared block-score vector per query, feeding a single
    topk/force-local selection shared by ALL GQA groups."""
    assert block_scores.dim() == 3, f"expected [T,H_idx,n_blk], got {tuple(block_scores.shape)}"
    return block_scores.amax(dim=1)                            # [T, n_blk]


# ================================================================================================
# 2. TOP-K BLOCK SELECTION.  PAPER Eq. 7 + forced local/init blocks.
# ================================================================================================
def msa_block_select(
    index_scores: torch.Tensor,   # [T, n_blk] collapsed M_idx (msa_collapse_index_heads) OR
                                  #   [T, H_idx, n_blk] (auto-collapsed here for back-compat)
    block_size: int,
    topk_blocks: int,
    causal_pos: torch.Tensor | None = None,   # [T] long abs pos per query (default arange)
    force_local: bool = True,     # force-include the block containing the query (PAPER §3.2)
    force_local_blocks: int = 1,  # sparse_local_block (how many recent blocks to force)
    force_init_blocks: int = 0,   # sparse_init_block  (how many leading sink blocks to force)
) -> torch.Tensor:                # block_idx [T, topk_blocks] int64, ascending, -1 padded (ONE set)
    """Top-k block selection on the COLLAPSED index scores (OFFICIAL MiniMaxM3VLIndexer.forward).

    The official code collapses the index heads (``scores.amax(dim=1)``) to ONE [T, n_blk] vector,
    then force-local (``scatter_(-1, local_idx, inf)``) + topk on that single vector -> ONE block
    set per query, shared by ALL GQA groups. This function therefore returns [T, topk] (no head axis).
    For back-compat it accepts the per-head [T, H_idx, n_blk] and collapses internally.

    Returns ascending block ids per query, -1 padded when fewer than topk valid (causally visible)
    blocks exist. Forced blocks (PAPER §3.2 "Local Block": the block containing the query; optionally
    `force_init_blocks` leading sink blocks) are ALWAYS present regardless of score. A block b is
    selectable iff it has >= 1 causally-visible token, i.e. b*block_size <= query_abs_pos.
    """
    if index_scores.dim() == 3:                                              # per-head -> collapse
        index_scores = msa_collapse_index_heads(index_scores)
    assert index_scores.dim() == 2, f"index_scores must be [T,n_blk] (or [T,H,n_blk]), got {index_scores.shape}"
    T, n_blk = index_scores.shape
    dev = index_scores.device
    if causal_pos is None:
        causal_pos = torch.arange(T, device=dev)
    qpos = causal_pos.to(dev).long()

    scores = index_scores.float().clone()                                    # [T, n_blk]

    # never select a future block: block b visible iff b*block_size <= qpos (>=1 causal token).
    blk_first = torch.arange(n_blk, device=dev) * block_size                  # [n_blk]
    visible = blk_first[None, :] <= qpos[:, None]                             # [T, n_blk]
    scores = scores.masked_fill(~visible, NEG_INF)

    # forced blocks -> +inf so topk always keeps them; AND with `visible` (never force a future blk).
    forced = torch.zeros(T, n_blk, dtype=torch.bool, device=dev)
    n_init = int(force_init_blocks)
    if n_init > 0:
        forced[:, :min(n_init, n_blk)] = True                                # sink (M3: 0)
    n_local = int(force_local_blocks) if force_local else 0
    if n_local > 0:
        local_blk = torch.div(qpos, block_size, rounding_mode="floor")       # block holding query i
        rows = torch.arange(T, device=dev)
        for d in range(n_local):                                             # walk back n_local blks
            b = (local_blk - d).clamp(min=0)
            forced[rows, b] = True
    forced = forced & visible
    scores = scores.masked_fill(forced, float("inf"))

    k = min(int(topk_blocks), n_blk)
    top = torch.topk(scores, k=k, dim=-1).indices                            # [T, k]
    # a slot that landed on a still-(-inf) entry means there were < k visible blocks -> mark pad -1.
    picked = torch.gather(scores, -1, top)
    block_idx = torch.where(torch.isneginf(picked), torch.full_like(top, -1), top)

    # canonical: ascending block id, -1 padded to the tail (api.py:1222-1223; sparse_index_utils:154).
    sort_key = torch.where(block_idx < 0, torch.full_like(block_idx, n_blk), block_idx)
    order = torch.argsort(sort_key, dim=-1)
    block_idx = torch.gather(block_idx, -1, order)
    if k < int(topk_blocks):                                                 # pad to fixed width
        pad = torch.full((T, int(topk_blocks) - k), -1, dtype=block_idx.dtype, device=dev)
        block_idx = torch.cat([block_idx, pad], dim=-1)
    return block_idx.to(torch.int64)                                         # [T, topk]


# ================================================================================================
# 3. BLOCK -> TOKEN EXPANSION (the KERNEL_MAP path-A expansion). Pure torch, graph-friendly.
# ================================================================================================
def msa_expand_blocks_to_tokens(
    block_idx: torch.Tensor,    # [T, topk] int  selected block ids (-1 padded), ONE set/query
    block_size: int,
    seqlen: int,                # S; token positions >= S are out-of-range -> -1
) -> torch.Tensor:             # token_idx [T, topk*block_size] int64 (-1 for pad / OOB)
    """Expand each selected block id b -> its block_size consecutive token positions
    [b*block_size, (b+1)*block_size) (test_sparse_attn.py:67-68). Padded blocks (-1) and any token
    position >= seqlen (a partial last block) become -1 so the consume side masks them.

    ONE shared block set per query (official head-collapse), so this is [T, topk] -> [T, topk*B]
    (no head axis). For back-compat it also accepts [T, H_kv, topk] and keeps the head axis.
    """
    dev = block_idx.device
    bi = block_idx.long()
    offs = torch.arange(block_size, device=dev)                              # [block_size]
    base = bi.unsqueeze(-1) * block_size + offs                             # [..., K, block_size]
    valid_block = (bi >= 0).unsqueeze(-1)                                    # pad block -> all -1
    tok = torch.where(valid_block, base, torch.full_like(base, -1))
    tok = torch.where(tok < seqlen, tok, torch.full_like(tok, -1))          # OOB tail -> -1
    return tok.reshape(*block_idx.shape[:-1], block_idx.shape[-1] * block_size)


# ================================================================================================
# 4. CONSUME — GQA gather attention over selected tokens.  PAPER Eq. 8 (PRODUCT form).
# ================================================================================================
def msa_gqa_gather_attention(
    q: torch.Tensor,            # [T, H_q,  D]  post-rope/post-norm main query
    k: torch.Tensor,            # [S, H_kv, D]  REAL (uncompressed) main key
    v: torch.Tensor,            # [S, H_kv, D]  REAL main value
    token_idx: torch.Tensor,    # [T, W] int  ONE shared selected-token set per query, -1 pad
                                #   (back-compat: [T, H_kv, W] per-group also accepted)
    scale: float,               # 1/sqrt(D)  (PAPER Eq. 8)
    causal_pos: torch.Tensor | None = None,  # [T] long abs pos per query (default arange)
) -> torch.Tensor:              # out [T, H_q, D]  fp32
    """Gather the REAL GQA K/V at the selected token positions and run SDPA, causal on ABSOLUTE
    positions, with GQA group expansion (head h uses kv head h // G).  PAPER Eq. 8.

    OFFICIAL head-collapse => ONE shared token set per query (token_idx [T, W]); EVERY group/head
    gathers that SAME set (one gather/query). Mirrors DSA's ``sparse_mla_gather_attention``: a
    SENTINEL key/value row at index S absorbs every -1 / out-of-range slot (NEVER clamp(min=0)).
    Gathered sentinel slots, and any gathered token whose absolute position > the query's, are masked
    to -inf. NaN guard (all-(-inf) row -> 0). For back-compat a per-group token_idx [T, H_kv, W] is
    still accepted (gather per group), but M3 uses the single shared set.

    Reference (MSA): test_sparse_attn.py:46-75 (gather selected blocks' tokens, scores/ sqrt(d),
    masked_fill_(qi<ki,-inf), softmax, @ v).
    """
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    T, Hq, D = q.shape
    S, Hkv, Dk = k.shape
    assert Dk == D and v.shape == (S, Hkv, v.shape[-1]), "k/v head_dim must match q"
    Dv = v.shape[-1]
    assert Hq % Hkv == 0, f"H_q={Hq} not divisible by H_kv={Hkv}"
    G = Hq // Hkv
    dev = q.device
    if causal_pos is None:
        causal_pos = torch.arange(T, device=dev)
    qpos = causal_pos.to(dev).long()                         # [T]

    # Normalize token_idx to a per-group view [T, H_kv, W]: the shared set [T, W] broadcasts to all
    # groups; a per-group [T, H_kv, W] passes through (back-compat).
    if token_idx.dim() == 2:
        per_group = token_idx.unsqueeze(1).expand(T, Hkv, token_idx.shape[-1])
    else:
        assert token_idx.dim() == 3 and token_idx.shape[:2] == (T, Hkv), (
            f"token_idx must be [T,W] or [T,H_kv,W], got {tuple(token_idx.shape)}")
        per_group = token_idx
    W = per_group.shape[-1]

    qf = q.float()
    # SENTINEL row at index S: route -1 / out-of-range token ids there; mask them out afterwards.
    k_ext = torch.cat([k.float(), torch.zeros(1, Hkv, D, device=dev)], dim=0)    # [S+1, H_kv, D]
    v_ext = torch.cat([v.float(), torch.zeros(1, Hkv, Dv, device=dev)], dim=0)   # [S+1, H_kv, Dv]

    tok = per_group.long()
    valid = (tok >= 0) & (tok < S)                           # [T, H_kv, W]
    sink = torch.where(valid, tok, torch.full_like(tok, S))  # -1/OOB -> sentinel row S

    out = torch.zeros(T, Hq, Dv, device=dev, dtype=torch.float32)
    # gather per GQA group, then expand to the G heads in the group (each head keeps its own Q).
    for r in range(Hkv):
        idx_r = sink[:, r, :]                                # [T, W] token rows for this group
        kg = k_ext[:, r, :][idx_r]                           # [T, W, D]
        vg = v_ext[:, r, :][idx_r]                           # [T, W, Dv]
        kpos_r = torch.where(valid[:, r, :], tok[:, r, :], torch.full_like(tok[:, r, :], -1))  # [T,W]
        # mask: slot is a sentinel/pad OR its key abs pos > query abs pos (causal on absolute pos).
        bad = (~valid[:, r, :]) | (kpos_r > qpos[:, None])   # [T, W]
        bias = torch.where(bad, torch.full_like(kg[..., 0], NEG_INF),
                           torch.zeros_like(kg[..., 0]))     # [T, W]
        for g in range(G):
            h = r * G + g
            s = torch.einsum("td,twd->tw", qf[:, h, :], kg) * float(scale)   # [T, W]
            s = s + bias
            p = torch.softmax(s, dim=-1)
            p = torch.nan_to_num(p, nan=0.0)                 # NaN guard (fully-masked row -> 0)
            out[:, h, :] = torch.einsum("tw,twd->td", p, vg)
    return out


# ================================================================================================
# 5. DENSE ORACLE — full causal GQA over ALL keys (the limit topk*block_size >= S).  PAPER Eq. 1/8.
# ================================================================================================
def msa_dense_oracle(
    q: torch.Tensor,            # [T, H_q,  D]
    k: torch.Tensor,            # [S, H_kv, D]
    v: torch.Tensor,            # [S, H_kv, D]
    scale: float,               # 1/sqrt(D)
    causal_pos: torch.Tensor | None = None,
) -> torch.Tensor:              # [T, H_q, D] fp32
    """Full causal GQA softmax attention over every key (no selection). This is what MSA must equal
    when the selection budget covers the whole context (topk_blocks*block_size >= S), i.e. every
    causally-visible block is selected. Used as the dense limit for the parity tests and as the
    layers-0..2 dense path (MSA_SPEC.md §5). fp32. Causal on absolute positions: key s visible to
    query t iff s <= abs_pos(t)."""
    T, Hq, D = q.shape
    S, Hkv, _ = k.shape
    Dv = v.shape[-1]
    G = Hq // Hkv
    dev = q.device
    if causal_pos is None:
        causal_pos = torch.arange(T, device=dev)
    qpos = causal_pos.to(dev).long()
    key_pos = torch.arange(S, device=dev)
    causal_bad = key_pos[None, :] > qpos[:, None]            # [T, S]  (key abs pos > query abs pos)

    qf = q.float()
    out = torch.zeros(T, Hq, Dv, device=dev, dtype=torch.float32)
    for r in range(Hkv):
        kf = k[:, r, :].float()                             # [S, D]
        vf = v[:, r, :].float()                             # [S, Dv]
        for g in range(G):
            h = r * G + g
            s = torch.einsum("td,sd->ts", qf[:, h, :], kf) * float(scale)    # [T, S]
            s = s.masked_fill(causal_bad, NEG_INF)
            p = torch.softmax(s, dim=-1)
            p = torch.nan_to_num(p, nan=0.0)
            out[:, h, :] = torch.einsum("ts,sd->td", p, vf)
    return out


# ================================================================================================
# 6. END-TO-END — one full MSA sparse layer.  PAPER Algorithm 1 (inference forward).
# ================================================================================================
def msa_forward(
    q: torch.Tensor,            # [T, H_q,  D]  post-rope/post-norm main query
    k: torch.Tensor,            # [S, H_kv, D]  real main key
    v: torch.Tensor,            # [S, H_kv, D]  real main value
    q_idx: torch.Tensor,        # [T, H_kv, D_idx]  RAW index query (post-projection; norm+rope here)
    k_idx: torch.Tensor,        # [T, 1, D_idx] or [T, D_idx]  RAW shared index key
    dims: MsaDims,
    causal_pos: torch.Tensor | None = None,   # [T] long abs pos per query (default arange)
    return_selection: bool = False,
    *,
    index_q_norm_weight: torch.Tensor | None = None,  # [D_idx] Gemma weight (None => identity, w=0)
    index_k_norm_weight: torch.Tensor | None = None,  # [D_idx]
    index_rope: bool = True,    # apply partial index RoPE (OFFICIAL); needs dims.rotary_dim/rope_theta
    rms_norm_eps: float = 1e-6,
):
    """One sparse MSA layer (OFFICIAL MiniMaxM3VLIndexer + MiniMaxM3VLAttention, inference):
    gemma-qk-norm -> partial-rope -> index-score -> HEAD-COLLAPSE -> top-k block select (ONE set) ->
    block->token expand -> GQA gather SDPA (every group gathers that one set).

    Takes RAW projected q_idx/k_idx (as the model produces them); applies the Gemma index norm
    (x*(1+w); pass the stored index_{q,k}_norm weights, or None for the zero-init identity scale)
    and, if index_rope, the partial RoPE (rotary_dim, rope_theta from dims) using the SAME cos/sin
    the main attention uses. For M3 prefill T==S and causal_pos==arange(S).

    Returns out [T, H_q, D] (fp32). With return_selection=True also returns (block_idx [T,topk],
    token_idx [T,topk*B]) — the ONE shared selection per query.
    """
    T = q.shape[0]
    dev = q.device
    if causal_pos is None:
        causal_pos = torch.arange(T, device=dev)

    # Gemma QK-norm on the index branch (per head over D_idx), then partial RoPE (in msa_index_scores).
    D_idx = dims.sparse_index_dim
    wq = index_q_norm_weight if index_q_norm_weight is not None else torch.zeros(D_idx, device=dev)
    wk = index_k_norm_weight if index_k_norm_weight is not None else torch.zeros(D_idx, device=dev)
    q_idx_n = gemma_rms_norm(q_idx, wq, eps=rms_norm_eps)
    k_idx_n = gemma_rms_norm(k_idx, wk, eps=rms_norm_eps)
    cos = sin = None
    if index_rope:
        cos, sin = build_index_cos_sin(causal_pos, dims.rotary_dim, dims.rope_theta, device=dev)

    block_scores = msa_index_scores(
        q_idx_n, k_idx_n, dims.index_scale, dims.sparse_block_size, causal_pos,
        cos=cos, sin=sin)                                                          # [T,H_idx,n_blk]
    block_idx = msa_block_select(
        block_scores, dims.sparse_block_size, dims.sparse_topk_blocks, causal_pos,
        force_local=(dims.sparse_local_block > 0),
        force_local_blocks=dims.sparse_local_block,
        force_init_blocks=dims.sparse_init_block,
    )                                                                              # [T,topk]  ONE set
    token_idx = msa_expand_blocks_to_tokens(
        block_idx, dims.sparse_block_size, seqlen=k.shape[0])                      # [T,topk*B]
    out = msa_gqa_gather_attention(q, k, v, token_idx, dims.main_scale, causal_pos)  # [T,H_q,D]
    if return_selection:
        return out, block_idx, token_idx
    return out


# ---- brute-force selection oracle (for the test): argmax-topk over block-max-pooled scores -----
def msa_reference_block_topk(
    index_scores: torch.Tensor,   # [T, n_blk] collapsed M_idx, OR [T, H_idx, n_blk] (collapsed here)
    topk_blocks: int,
    causal_pos: torch.Tensor | None = None,
    force_local_blocks: int = 0,
    force_init_blocks: int = 0,
    block_size: int = 128,
) -> list:
    """Naive python oracle: for each query t return the ONE SET of selected block ids (official head-
    collapse), computed by an explicit causal-mask + forced-set + argsort top-k. Mirrors the DSA
    test's brute-force `reference_indexer_topk` SET comparison. Accepts the collapsed [T, n_blk] or
    the per-head [T, H_idx, n_blk] (collapsed via amax over heads). Returns a list [T] -> set."""
    if index_scores.dim() == 3:
        index_scores = index_scores.amax(dim=1)                              # head-collapse
    T, n_blk = index_scores.shape
    dev = index_scores.device
    if causal_pos is None:
        causal_pos = torch.arange(T, device=dev)
    qpos = causal_pos.to(dev).long().tolist()
    sc = index_scores.float().cpu()
    out = []
    for t in range(T):
        p = qpos[t]
        vis = [b for b in range(n_blk) if b * block_size <= p]                # causally-visible blks
        forced = set()
        for b in range(min(force_init_blocks, n_blk)):
            if b in vis:
                forced.add(b)
        if force_local_blocks > 0:
            lb = p // block_size
            for d in range(force_local_blocks):
                b = max(0, lb - d)
                if b in vis:
                    forced.add(b)
        scored = [(float(sc[t, b]), b) for b in vis]
        # forced blocks always in; fill the remaining slots by descending score (stable by id).
        chosen = set(forced)
        rest = sorted([bs for bs in scored if bs[1] not in forced], key=lambda x: (-x[0], x[1]))
        for _, b in rest:
            if len(chosen) >= min(topk_blocks, len(vis)):
                break
            chosen.add(b)
        out.append(chosen)
    return out
