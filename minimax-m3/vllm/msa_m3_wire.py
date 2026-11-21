"""MiniMax-M3 MSA wire — connect the validated model class (minimax_m3.py) to the
compiled hpu_msa_ext kernel ops. This is the *integration glue*, the MSA analog of
hpu-dsa-ext's ``sparse_mla_wire.py`` / ``glm_moe_dsa_wire.py``: it re-implements the
oracle's ``ref/msa_reference.py::msa_forward`` with the TWO compiled kernel ops replacing
the two torch hot loops, and installs itself via ``minimax_m3.set_msa_wire``.

CARD-FREE: nothing here touches the running serve, /opt, or the installed hpu-{moe,dsa}-ext.
This module's pure-torch orchestration is the thing the CPU parity harness gates; the two
kernel-op call-sites are swapped for small torch impls (``MSA_OP_IMPLS``) in the CPU test so
the WHOLE Python chain is proven bit-for-bit against the oracle before any on-card run.

==================================================================================================
THE KERNEL BOUNDARY (minimax_m3.py:412-439 / MSA_SPEC.md §6)
==================================================================================================
``set_msa_wire(fn)`` installs ``fn(attn_module, positions, q, k, v) -> attn_output`` where, on a
sparse layer with the gate open, the model hands the wire:
  * q : [T, H_q*head_dim]    flattened MAIN query, POST qk-norm + POST partial-rope
  * k : [T, H_kv*head_dim]   flattened MAIN key,   POST qk-norm + POST partial-rope
  * v : [T, H_kv*head_dim]   flattened MAIN value
  * positions : [T] long     absolute position of every query row (prefill: arange; the runner
                              supplies real abs pos for paged/decode — out of MVP scope, see §C).
The INDEX branch (q_idx/k_idx) is NOT in q/k/v — it is computed HERE from the hidden states via
the per-layer index projections (``attn_module.index_q_proj`` / ``index_k_proj`` and the index
RMSNorms ``index_q_norm`` / ``index_k_norm``): Gemma QK-norm (x*(1+w)) then partial RoPE
(rotary_dim=64, theta=5e6, SAME cos/sin as main), NO relu — matching OFFICIAL MiniMaxM3VLIndexer.

>>> 2026-06-14 CORRECTIONS vs OFFICIAL modeling_minimax_m3_vl (transformers 5.12.0 @ m3calib) <<<
The paper-derived index math was WRONG in load-bearing ways. Four fixes (all in this module + the
oracle, reconfirmed against MiniMaxM3VLIndexer/MiniMaxM3VLRMSNorm/MiniMaxM3VLAttention):
  (1) index_q_norm/index_k_norm are Gemma RMSNorm x*(1+w) (weight ZERO-init), per head over d_idx,
      eps=1e-6 — was plain x*w with ones-init.  (2) index branch GETS partial RoPE (rotary_dim=64,
      theta=5e6, same cos/sin as main), after the norm, before the dot — was "NO rope".  (3) index
      scores are a RAW matmul, NO 1/sqrt(d_idx) — index_scale is now 1.0.  (4) HEAD-COLLAPSE: the
      official does ``block_scores = scores.amax(-1).amax(dim=1)`` -> ONE shared block set per query
      (collapsing the 4 index heads BEFORE topk), NOT a per-GQA-group set. The wire now selects ONE
      set and the consume is a SINGLE op call over all heads (the 4-call loop is gone).

==================================================================================================
INDEX PROJECTIONS — sourced from the checkpoint (verified, 2026-06-14)
==================================================================================================
Probing the landed safetensors shards of /mnt/data2/minimax-m3-bf16 (model mid-download, but the
early sparse layers + their index params landed; ``model.safetensors.index.json`` not yet written)
gives the EXACT per-sparse-layer weight names + shapes (prefix ``language_model.model.`` is the
multimodal wrapper; vLLM strips ``language_model.`` so the model sees ``model.layers.N.self_attn.*``):

    self_attn.index_q_proj.weight   [512, 6144]   = [sparse_num_index_heads(4) * sparse_index_dim(128), hidden]
    self_attn.index_k_proj.weight   [128, 6144]   = [1 * sparse_index_dim(128), hidden]   (single shared K head)
    self_attn.index_q_norm.weight   [128]         = per-index-head RMSNorm over sparse_index_dim
    self_attn.index_k_norm.weight   [128]         = per-index-head RMSNorm over sparse_index_dim

Dense layers (0,1,2) carry NO index_* tensors — confirming the sparse-layer gate.

*** RECONFIRMED against OFFICIAL modeling (2026-06-14) ***  The checkpoint carries index_q_norm AND
index_k_norm on the index branch, AND the official applies partial RoPE to index q/k. Confirmed
verbatim against ``modeling_minimax_m3_vl.MiniMaxM3VLIndexer`` (transformers 5.12.0 @ m3calib): the
norm is Gemma-style ``x*(1+w)`` (weight zero-init) per head over d_idx, eps=1e-6, applied AFTER the
reshape-to-heads and BEFORE rope; partial RoPE (rotary_dim=64, theta=5e6, same cos/sin as main) then
applies before the dot; the dot is RAW (no 1/sqrt(d_idx)); and the 4 index heads collapse via
``amax`` to ONE shared block set per query before topk. ``MSA_APPLY_INDEX_NORM`` (default True) is
kept only for the bare-dot parity path; the default path uses the Gemma form. See ``MsaDims`` /
``add_index_projections`` / ``compute_index_qk``.

==================================================================================================
THE WIRE FLOW (== oracle ``msa_forward``, kernel ops replacing the two torch hot loops)
==================================================================================================
Given main q/k/v + the per-layer index projections, for one prefill (T == S, abs pos == arange):
  1. q_idx = index_q_norm(reshape(hidden @ index_q_proj^T))   [T, H_kv, D_idx]   Gemma x*(1+w)
     k_idx = index_k_norm(reshape(hidden @ index_k_proj^T))   [T, 1,   D_idx]
     then partial RoPE(q_idx,k_idx, cos[..:64],sin[..:64]) (theta=5e6) — AFTER norm, BEFORE dot.
  2. pad the key axis to n_blocks*block_size; build cmask[Sq,Sk] (TRUE where key non-causal:
     key_pos > abs_pos), padded tail columns are non-causal for every query -> TRUE -> -inf.
  3. block_ids[Sq,kk] = msa_block_indexer_topk(q_idx^[H,Sq,D], k_idx expanded to H, weights=1.0,
     cmask, index_topk=top_blocks, mode=1, block_size)   ← KERNEL OP (raw dot, block-max, AMAX over
     index heads (head-collapse), then top-k) -> ONE shared set per query.
  4. force-local (block holding the query, sparse_local_block) + ascending sort + (-1) right-pad
     -> block_ids[Sq,top_blocks]  (the oracle's msa_block_select tail; UPSTREAM of the op).
  5. token_idx[Sq,top_blocks*block_size] = expand each block id to its block_size token positions;
     (-1) for pad-block / token >= seqlen / token > abs_pos (oracle msa_expand_blocks_to_tokens +
     within-block causal cut). ONE shared set per query.
  6. consume — ONE call over ALL heads (head-collapse => one shared token set, so every head gathers
     the same tokens == the compiled op's native [T,K] contract). Fan the 4 real kv-heads to H_q via
     an expand-view (NO 16x physical copy): out[T,H_q,Dv] = sparse_attn(q[T,H_q,1,Dh], k_h[S,H_q,Dh],
     v_h, token_idx[T,K], scale=head_dim**-0.5, gather_width=K)  ← KERNEL OP (gather + SDPA). The
     prior per-group 4-call loop is GONE (one set, one call).
This equals the oracle's ``msa_forward`` (gemma-norm+rope -> msa_index_scores -> head-collapse ->
msa_block_select -> msa_expand_blocks_to_tokens -> msa_gqa_gather_attention), with steps 3 and 6
done by the kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

NEG_INF = float("-inf")

# Toggle: apply the checkpoint's index_q_norm / index_k_norm RMSNorms on the index branch.
# Default True (the real M3 weights carry them). See the module-header FLAG: the oracle does not
# model these; flip to False only to reproduce the bare-dot oracle exactly (parity-test path A).
MSA_APPLY_INDEX_NORM = True

# The two compiled-kernel call targets, indirected so the CPU parity harness can substitute small
# torch impls (proving the Python orchestration around the ops is correct without a card). The
# live plugin sets these to the loaded ``torch.ops.hpu_msa_ext.*`` ops (see ``bind_kernel_ops``).
#   "block_indexer_topk": fn(q[H,Sq,D], k[H,Sk,D], weights[H,Sq,1], cmask[Sq,Sk],
#                            index_topk:int, mode:int, block_size:int) -> block_ids[Sq,kk] int32
#                            (MSA mode=1 head-collapses the H index heads via amax BEFORE topk -> ONE
#                             shared set [Sq,kk]; the compiled .so needs that reduce inserted, see
#                             _cpu_block_indexer_topk KERNEL IMPLICATION.)
#   "sparse_attn":        fn(q[T,H,1,Dh], k[S,H,Dh], v[S,H,Dv], token_idx[T,K] int32,
#                            scale:float, gather_width:int) -> out[T,H,Dv]
#                            (ONE shared token set per query, so a SINGLE call over all H_q heads.)
MSA_OP_IMPLS: dict[str, Optional[Callable[..., torch.Tensor]]] = {
    "block_indexer_topk": None,
    "sparse_attn": None,
}


def bind_kernel_ops(block_indexer_topk=None, sparse_attn=None) -> None:
    """Point the wire at concrete op implementations.

    The live plugin calls this with the loaded GC ops::

        torch.ops.load_library(".../libmsa_block_indexer.so")
        torch.ops.load_library(".../libmsa_sparse_attn.so")
        bind_kernel_ops(torch.ops.hpu_msa_ext.msa_block_indexer_topk,
                        torch.ops.hpu_msa_ext.sparse_attn)

    The CPU parity harness calls it with the small torch reference impls in this module
    (``_cpu_block_indexer_topk`` / ``_cpu_sparse_attn``) so the EXACT same wire code runs."""
    if block_indexer_topk is not None:
        MSA_OP_IMPLS["block_indexer_topk"] = block_indexer_topk
    if sparse_attn is not None:
        MSA_OP_IMPLS["sparse_attn"] = sparse_attn


# ==================================================================================================
# MsaDims — config reader (all dims from config; NO literals in the forward path). Analog of DsaDims
# in glm_moe_dsa_wire.py; a thin adapter over ref/msa_reference.MsaDims so the wire and the oracle
# read the SAME numbers. Reads from a vLLM ``hf_config`` (top multimodal config OR the text config)
# OR a parsed config dict, so it works both at serve time and in the CPU test.
# ==================================================================================================
@dataclass(frozen=True)
class MsaDims:
    """Every MSA dimension the wire needs, read straight from the M3 (text) config.

    Mirrors ref/msa_reference.MsaDims field-for-field (so the wire and oracle never disagree) plus
    ``num_hidden_layers`` and the per-layer ``sparse_attention_freq`` cadence for the runner.
    """
    hidden_size: int
    num_attention_heads: int        # H_q
    num_key_value_heads: int        # H_kv  (GQA group G = H_q / H_kv ; also == index Q heads)
    head_dim: int                   # d_h (main attention head dim)
    rotary_dim: int                 # partial RoPE dim (64); index branch reuses the main cos/sin
    rope_theta: float               # RoPE base (5e6); index branch reuses the main cos/sin
    sparse_index_dim: int           # d_idx
    sparse_num_index_heads: int     # index-Q heads (== H_kv); index-K is a single shared head
    sparse_topk_blocks: int         # k (top-k BLOCKS)
    sparse_block_size: int          # B_k (tokens per KV block)
    sparse_score_type: str          # "max" (BlockMaxPool)
    sparse_init_block: int          # forced sink blocks (M3: 0)
    sparse_local_block: int         # forced local blocks (M3: 1)
    num_hidden_layers: int          # 60
    sparse_attention_freq: tuple[int, ...]  # per-layer 0/1 (1 == sparse MSA layer)

    @property
    def gqa_group(self) -> int:
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def main_scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)        # PAPER Eq.8 ; sparse_attn `scale`

    @property
    def index_scale(self) -> float:
        # OFFICIAL MiniMaxM3VLIndexer: scores = matmul(idx_q, idx_k^T) — RAW, NO 1/sqrt(d_idx).
        # (selection-neutral global constant, but matches the code). The op `weights` broadcast = 1.0.
        return 1.0

    def layer_is_sparse(self, layer_id: int) -> bool:
        if not (0 <= layer_id < len(self.sparse_attention_freq)):
            return False
        return bool(self.sparse_attention_freq[layer_id])

    @classmethod
    def from_config(cls, config: Any) -> "MsaDims":
        """Build from a vLLM PretrainedConfig (top OR text) or a parsed dict.

        Resolves the text sub-config for the nested ``minimax_m3_vl`` config, then reads every dim
        from ``sparse_attention_config`` + the LM fields. Hardcodes NOTHING (defaults only fill a
        genuinely-absent key, and each is the M3 config value so a stripped config still loads)."""
        tc = _resolve_text_config(config)
        g = lambda k, d=None: _cfg_get(tc, k, d)  # noqa: E731
        sa_raw = g("sparse_attention_config", {}) or {}
        sa = sa_raw if isinstance(sa_raw, dict) else getattr(sa_raw, "__dict__", {}) or {}
        sag = lambda k, d=None: sa.get(k, d)  # noqa: E731
        freq = sag("sparse_attention_freq", None)
        d = cls(
            hidden_size=int(g("hidden_size", 6144)),
            num_attention_heads=int(g("num_attention_heads", 64)),
            num_key_value_heads=int(g("num_key_value_heads", 4)),
            head_dim=int(g("head_dim", 128)),
            rotary_dim=int(g("rotary_dim", 64)),
            rope_theta=float(g("rope_theta", 5000000.0)),
            sparse_index_dim=int(sag("sparse_index_dim", 128)),
            sparse_num_index_heads=int(sag("sparse_num_index_heads", 4)),
            sparse_topk_blocks=int(sag("sparse_topk_blocks", 16)),
            sparse_block_size=int(sag("sparse_block_size", 128)),
            sparse_score_type=str(sag("sparse_score_type", "max")),
            sparse_init_block=int(sag("sparse_init_block", 0)),
            sparse_local_block=int(sag("sparse_local_block", 1)),
            num_hidden_layers=int(g("num_hidden_layers", 60)),
            sparse_attention_freq=tuple(int(x) for x in freq) if freq is not None else tuple(),
        )
        d.validate()
        return d

    def validate(self) -> None:
        # LOUD failures (DSA build quality bar). Never silently mis-shape the kernel.
        assert self.sparse_score_type == "max", (
            f"[msa] only score_type='max' (BlockMaxPool) implemented; got {self.sparse_score_type!r}")
        assert self.sparse_num_index_heads == self.num_key_value_heads, (
            f"[msa] sparse_num_index_heads({self.sparse_num_index_heads}) must == "
            f"num_key_value_heads({self.num_key_value_heads}) — one index-Q head per GQA group")
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"[msa] H_q({self.num_attention_heads}) not divisible by H_kv({self.num_key_value_heads})")
        assert self.sparse_block_size > 0 and self.sparse_topk_blocks > 0
        assert self.sparse_init_block >= 0 and self.sparse_local_block >= 0
        assert self.sparse_init_block + self.sparse_local_block <= self.sparse_topk_blocks, (
            f"[msa] forced blocks ({self.sparse_init_block}+{self.sparse_local_block}) exceed "
            f"topk_blocks={self.sparse_topk_blocks}")


def _resolve_text_config(config: Any) -> Any:
    """Return the LM (text) sub-config for a nested minimax_m3_vl config; pass through a flat one."""
    if isinstance(config, dict):
        return config.get("text_config", config)
    fn = getattr(config, "get_text_config", None)
    if callable(fn):
        try:
            tc = fn()
            if tc is not None:
                return tc
        except Exception:  # noqa: BLE001 — flat configs have no text_config
            pass
    return getattr(config, "text_config", config)


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ==================================================================================================
# INDEX-BRANCH projections (the patch the model attention lacks). minimax_m3.py's
# HpuMiniMaxM3Attention builds main qkv/o/qk-norm but NO index projections. ``add_index_projections``
# attaches them so the per-sparse-layer index params load + run. The index branch applies Gemma
# QK-norm (x*(1+w)) then partial RoPE (rotary_dim=64, theta=5e6, same cos/sin as main), NO relu —
# matching OFFICIAL MiniMaxM3VLIndexer (compute_index_qk does norm+rope).
#
# The index RMSNorms (index_q_norm / index_k_norm, [d_idx]) are Gemma-style, applied per index head
# over the d_idx axis (the SAME discipline the main Gemma qk-norm uses). Gated by
# MSA_APPLY_INDEX_NORM (default True; the bare-dot path is parity-only).
# ==================================================================================================
class _IndexRMSNorm(torch.nn.Module):
    """Gemma-style RMSNorm (== MiniMaxM3VLRMSNorm): fp32 normalize, scale by (1 + weight).

    weight inits to ZEROS (Gemma), NOT ones. Matches transformers 5.12.0
    modeling_minimax_m3_vl.MiniMaxM3VLRMSNorm (m3calib) lines 145-153:
        out = x*rsqrt(mean(x^2)+eps); out = out * (1.0 + weight); cast back.
    The checkpoint index_{q,k}_norm weights cluster near 0 (mean ~0.2, some NEGATIVE) — impossible
    for a plain x*w norm (weights near 1), exactly right for Gemma x*(1+w). Per-head over the last
    (d_idx) axis; the index norm is tiny (d_idx=128) and not sharded. Weight name
    ``self_attn.index_{q,k}_norm.weight`` ([d_idx])."""

    def __init__(self, d_idx: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(d_idx))   # Gemma: zeros (1+w == identity)
        self.variance_epsilon = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [..., d_idx]
        orig = x.dtype
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.variance_epsilon)
        return (xf * (1.0 + self.weight.to(torch.float32))).to(orig)   # (1 + w) — Gemma form


def add_index_projections(attn_module: torch.nn.Module, dims: MsaDims, *,
                          rms_norm_eps: float = 1e-6, quant_config=None) -> None:
    """Attach the MSA index branch (index_q_proj / index_k_proj + index_q_norm / index_k_norm) to a
    sparse-layer attention module so the per-layer index params load and the wire can run.

    Weight names (checkpoint, after vLLM strips ``language_model.``):
      ``...self_attn.index_q_proj.weight`` [n_idx_heads*d_idx, hidden]
      ``...self_attn.index_k_proj.weight`` [1*d_idx, hidden]
      ``...self_attn.index_q_norm.weight`` [d_idx]
      ``...self_attn.index_k_norm.weight`` [d_idx]
    These are ReplicatedLinear (no bias) + RMSNorm; the index branch is NOT tensor-parallel-sharded
    (it is tiny and shared/per-group). Call from the model's sparse-attention __init__ (or via the
    plugin's monkeypatch of HpuMiniMaxM3Attention.__init__). Idempotent."""
    if getattr(attn_module, "_msa_index_attached", False):
        return
    from vllm.model_executor.layers.linear import ReplicatedLinear

    d_idx = dims.sparse_index_dim
    n_q = dims.sparse_num_index_heads
    # disable_tp=True: the index branch is REPLICATED, not TP-sharded (4 index-Q heads + 1 shared
    # index-K head are tiny and shared across the GQA groups, not split across ranks). This also
    # makes the module constructible card-free (no TP group needed) for the parity harness.
    attn_module.index_q_proj = ReplicatedLinear(
        dims.hidden_size, n_q * d_idx, bias=False, quant_config=None, disable_tp=True,
        prefix=f"{getattr(attn_module, 'prefix', '')}.index_q_proj")
    attn_module.index_k_proj = ReplicatedLinear(
        dims.hidden_size, 1 * d_idx, bias=False, quant_config=None, disable_tp=True,
        prefix=f"{getattr(attn_module, 'prefix', '')}.index_k_proj")
    attn_module.index_q_norm = _IndexRMSNorm(d_idx, eps=rms_norm_eps)
    attn_module.index_k_norm = _IndexRMSNorm(d_idx, eps=rms_norm_eps)
    attn_module._msa_index_dims = dims
    attn_module._msa_index_attached = True


def _index_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The MAIN cos/sin the indexer reuses (MiniMaxM3VLRotaryEmbedding, default rope). positions [T]
    -> (cos, sin) [T, rotary_dim], fp32. inv_freq = 1/theta^(arange(0,rotary_dim,2)/rotary_dim);
    emb = cat((freqs,freqs)) so last dim == rotary_dim (==64). Matches modeling lines 300-326."""
    pos = positions.to(device).float()
    inv_freq = 1.0 / (float(theta) ** (torch.arange(0, rotary_dim, 2, device=device).float() / float(rotary_dim)))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_index_rope(q_idx: torch.Tensor, k_idx: torch.Tensor,
                      cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial RoPE on index q/k (modeling apply_rotary_pos_emb): rotate the first rotary_dim dims,
    pass the rest. q_idx [T,H_kv,D], k_idx [T,1,D], cos/sin [T,rotary_dim]; broadcast over heads.

    OFFICIAL clamp (MiniMaxM3VLIndexer.forward): the indexer slices cos/sin to the index head dim
    (``cos[..., :head_dim]``), so the effective rotary width is ``min(rotary_dim, D_idx)`` — a no-op
    for M3 (rotary_dim=64 < D_idx=128) but guards a config with rotary_dim > D_idx from over-rotating.
    Mirrors ref/msa_reference.apply_index_rope exactly (kept byte-identical so the wire == oracle)."""
    D = q_idx.shape[-1]
    rd = min(cos.shape[-1], D)
    cf = cos.to(torch.float32)[..., :rd].unsqueeze(1)
    sf = sin.to(torch.float32)[..., :rd].unsqueeze(1)

    def _rope(x):
        xf = x.to(torch.float32)
        x_rot, x_pass = xf[..., :rd], xf[..., rd:]
        return torch.cat([(x_rot * cf) + (_rotate_half(x_rot) * sf), x_pass], dim=-1)

    return _rope(q_idx), _rope(k_idx)


def compute_index_qk(hidden_states: torch.Tensor, attn_module: torch.nn.Module,
                     dims: MsaDims, positions: torch.Tensor | None = None,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden_states [T, hidden] -> (q_idx [T, H_kv, d_idx], k_idx [T, 1, d_idx]).

    OFFICIAL MiniMaxM3VLIndexer.forward order: proj -> reshape-to-heads -> Gemma QK-norm (x*(1+w),
    over d_idx) -> partial RoPE (rotary_dim=64, theta=5e6, SAME cos/sin as main) -> (dot upstream).
        q_idx = index_q_norm(reshape(hidden @ index_q_proj^T, [T, H_kv, d_idx]))
        k_idx = index_k_norm(reshape(hidden @ index_k_proj^T, [T, 1,    d_idx]))
        q_idx, k_idx = apply_rotary_pos_emb(q_idx, k_idx, cos[..,:128], sin[..,:128])  # rotates 64
    The norm is gated by MSA_APPLY_INDEX_NORM (default True). RoPE needs ``positions`` (the absolute
    query positions); if positions is None the rope step is skipped (legacy/no-rope path)."""
    d_idx = dims.sparse_index_dim
    n_q = dims.sparse_num_index_heads
    T = hidden_states.shape[0]
    q_flat, _ = attn_module.index_q_proj(hidden_states)        # [T, n_q*d_idx]
    k_flat, _ = attn_module.index_k_proj(hidden_states)        # [T, 1*d_idx]
    q_idx = q_flat.view(T, n_q, d_idx)
    k_idx = k_flat.view(T, 1, d_idx)
    if MSA_APPLY_INDEX_NORM and hasattr(attn_module, "index_q_norm"):
        q_idx = attn_module.index_q_norm(q_idx)
        k_idx = attn_module.index_k_norm(k_idx)
    if positions is not None:                                  # partial index RoPE (after the norm)
        cos, sin = _index_cos_sin(positions.view(-1)[:T], dims.rotary_dim, dims.rope_theta,
                                  hidden_states.device)
        q_idx, k_idx = _apply_index_rope(q_idx, k_idx, cos, sin)
    return q_idx, k_idx


# ==================================================================================================
# Pure-torch helpers that bracket the kernel ops (the oracle's tail, done UPSTREAM of the ops). These
# are byte-identical in spirit to ref/msa_reference.py's force-local / sort / -1-pad / block->token
# expansion; the kernel op returns RAW top-k blocks, so these run in the wire. CPU-testable.
# ==================================================================================================
def build_cmask(seqlen: int, abs_pos: torch.Tensor, block_size: int,
                device: torch.device) -> tuple[torch.Tensor, int, int]:
    """Causal mask for the indexer, key axis padded to n_blocks*block_size.

    Returns (cmask[Sq, Sk_pad] bool TRUE==drop, n_blocks, Sk_pad). A key at padded-tail column
    (>= seqlen) is non-causal for EVERY query (its 'position' exceeds all abs_pos), so it is TRUE
    -> -inf in the op -> a fully-padded block reduce-maxes to -inf -> never selected (kernel
    contract msa_block_indexer_backend_shared.h:408-413). Matches msa_index_scores causal+pad."""
    n_blocks = (seqlen + block_size - 1) // block_size
    sk_pad = n_blocks * block_size
    key_pos = torch.arange(sk_pad, device=device)
    cmask = key_pos[None, :] > abs_pos.to(device).long()[:, None]     # [Sq, Sk_pad], TRUE => drop
    return cmask, n_blocks, sk_pad


def force_local_sort_pad(block_ids_raw: torch.Tensor, abs_pos: torch.Tensor, dims: MsaDims,
                         n_blocks: int) -> torch.Tensor:
    """Take the op's RAW top-k blocks [Sq, kk] (ONE head-collapsed set per query) and apply the
    oracle's msa_block_select tail: force-include the local block(s) holding the query
    (sparse_local_block) and any sink blocks (sparse_init_block), drop any non-causal/future block,
    then ascending-sort + (-1) right-pad to sparse_topk_blocks. Returns [Sq, topk_blocks] int64 —
    the ONE shared selection per query (every GQA group gathers it).

    Mirrors ref/msa_reference.py::msa_block_select EXACTLY. The op returns its top-k blocks
    SCORE-DESCENDING (topk largest). A forced block must EVICT the lowest-SCORED op block to fit topk
    (the oracle's +inf-then-topk semantics) — NOT the highest-id one — so we reconstruct a per-block
    PRIORITY (forced = best, then op blocks by their rank position, then excluded), take topk by
    priority, then sort ascending. Reproduces ``forced ∪ (highest-scored non-forced, fill to topk)``
    from only the op's set+rank.

    Back-compat: a per-head [H_kv, Sq, kk] is collapsed (union over heads, min-rank) before the tail,
    but the corrected op already returns the one collapsed set.
    """
    if block_ids_raw.dim() == 3:                                    # legacy per-head -> collapse
        block_ids_raw = block_ids_raw.permute(1, 0, 2).reshape(block_ids_raw.shape[1], -1)
    assert block_ids_raw.dim() == 2, f"block_ids_raw must be [Sq,kk] (or [H,Sq,kk]), got {block_ids_raw.shape}"
    Sq, kk = block_ids_raw.shape
    dev = block_ids_raw.device
    topk = dims.sparse_topk_blocks
    bsz = dims.sparse_block_size
    qpos = abs_pos.to(dev).long()                                   # [Sq]
    raw = block_ids_raw.contiguous().long()                        # [Sq, kk]  score-desc

    # causal visibility: block b selectable iff b*bsz <= qpos (>=1 causal token).
    blk_first = torch.arange(n_blocks, device=dev) * bsz            # [n_blocks]
    visible = blk_first[None, :] <= qpos[:, None]                   # [Sq, n_blocks]

    # forced blocks: sink (init) + local window. AND with visible (never force a future block).
    forced = torch.zeros(Sq, n_blocks, dtype=torch.bool, device=dev)
    if dims.sparse_init_block > 0:
        forced[:, :min(dims.sparse_init_block, n_blocks)] = True
    if dims.sparse_local_block > 0:
        local_blk = torch.div(qpos, bsz, rounding_mode="floor")     # block holding query i
        rows = torch.arange(Sq, device=dev)
        for off in range(dims.sparse_local_block):
            b = (local_blk - off).clamp(min=0)
            forced[rows, b] = True
    forced &= visible                                               # [Sq, n_blocks]

    # Per-block PRIORITY (lower == kept first). Start everyone at +inf (excluded). The op's rank-j
    # block (causal only) gets priority j+1 (rank 0 best among op picks). Forced blocks get priority
    # 0 (above every op pick). Then topk-by-priority == forced ∪ top-scored, evicting the worst op
    # block when forced overflows — exactly the oracle's +inf/topk eviction order.
    BIG = float(n_blocks + kk + 8)
    prio = torch.full((Sq, n_blocks), BIG, device=dev)
    rank = torch.arange(kk, device=dev, dtype=torch.float32) + 1.0  # [kk] -> 1..kk
    valid_raw = (raw >= 0) & (raw < n_blocks)                       # drop the op's -inf/pad picks
    safe = torch.where(valid_raw, raw, torch.zeros_like(raw))      # [Sq, kk]
    vis_at_raw = torch.gather(visible, 1, safe)                    # block visible at this (q)?
    keep_raw = valid_raw & vis_at_raw                             # [Sq, kk]
    rank_b = rank.view(1, kk).expand(Sq, kk).clone()
    rank_b = torch.where(keep_raw, rank_b, torch.full_like(rank_b, BIG))
    # scatter-min the op-rank priorities into prio at each kept block id.
    prio.scatter_reduce_(1, safe, rank_b, reduce="amin", include_self=True)
    # forced blocks -> priority 0 (always above op picks).
    prio = torch.where(forced, torch.zeros_like(prio), prio)

    # take the topk LOWEST priorities; a slot still at BIG means "no real block" -> -1.
    sel_prio, sel_id = torch.topk(prio, k=min(topk, n_blocks), dim=-1, largest=False)  # [Sq,k]
    sel_id = torch.where(sel_prio >= BIG, torch.full_like(sel_id, -1), sel_id)

    # canonical: ascending block id, -1 right-padded to the tail.
    big_id = n_blocks + 1
    sort_key = torch.where(sel_id < 0, torch.full_like(sel_id, big_id), sel_id)
    order = torch.argsort(sort_key, dim=-1)
    sel_id = torch.gather(sel_id, -1, order)
    if sel_id.shape[-1] < topk:                                     # pad to fixed width
        pad = torch.full((Sq, topk - sel_id.shape[-1]), -1, dtype=sel_id.dtype, device=dev)
        sel_id = torch.cat([sel_id, pad], dim=-1)
    return sel_id.to(torch.int64)                                  # [Sq, topk]


def expand_blocks_to_tokens(block_idx: torch.Tensor, block_size: int, seqlen: int) -> torch.Tensor:
    """[Sq, topk] block ids (-1 pad) -> [Sq, topk*block_size] token ids (-1 for pad / token >=
    seqlen). ONE shared set per query (official head-collapse). Byte-identical to
    ref/msa_reference.py::msa_expand_blocks_to_tokens (which now also takes the one-set shape)."""
    dev = block_idx.device
    bi = block_idx.long()
    offs = torch.arange(block_size, device=dev)
    base = bi.unsqueeze(-1) * block_size + offs                     # [Sq, topk, block_size]
    valid_block = (bi >= 0).unsqueeze(-1)
    tok = torch.where(valid_block, base, torch.full_like(base, -1))
    tok = torch.where(tok < seqlen, tok, torch.full_like(tok, -1))  # OOB tail -> -1
    return tok.reshape(*block_idx.shape[:-1], block_idx.shape[-1] * block_size)


def causal_mask_token_idx(token_idx: torch.Tensor, abs_pos: torch.Tensor) -> torch.Tensor:
    """-1 out any selected token whose ABSOLUTE position exceeds its query's (within-block future).

    The local block (and any selected block straddling the diagonal) legitimately contains tokens at
    positions > the query position; the ORACLE consume (msa_gqa_gather_attention) masks those via
    ``kpos_r > qpos`` -> -inf. Our ``sparse_attn`` op masks ONLY pad/sentinel slots, so the wire must
    push that within-block causal cut INTO token_idx here: a token > abs_pos becomes -1 -> the op
    routes it to the sentinel row and pad-masks it == the oracle's -inf. token_idx [Sq, K] (one
    shared set); abs_pos [Sq]."""
    Sq = token_idx.shape[0]
    qpos = abs_pos.to(token_idx.device).long().view(Sq, 1)
    bad = (token_idx >= 0) & (token_idx > qpos)                     # within-range but non-causal
    return torch.where(bad, torch.full_like(token_idx, -1), token_idx)


# ==================================================================================================
# THE WIRE — installed via minimax_m3.set_msa_wire. Equals oracle msa_forward with the two kernel
# ops replacing the two torch hot loops (msa_index_scores+block topk -> block_indexer_topk;
# msa_gqa_gather_attention -> sparse_attn). PREFILL-FIRST: T==S, abs pos from `positions`.
# ==================================================================================================
def msa_wire(attn_module: torch.nn.Module, positions: torch.Tensor,
             q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """fn(attn_module, positions, q, k, v) -> attn_output [T, H_q*head_dim].

    q/k/v are flattened POST-rope/POST-qk-norm main projections from minimax_m3.py:417-420
    (q [T,H_q*Dh], k/v [T,H_kv*Dh]). The hidden states needed for the index branch are read from
    ``attn_module._msa_hidden`` (the model stashes them just before calling the wire — see the
    model patch in this module's docstring / platform_patch_msa). ``positions`` are the absolute
    query positions. PREFILL contract: T == S (no paged history). Decode/paged is a follow-up (§C).
    """
    dims: MsaDims = attn_module._msa_index_dims
    Hq, Hkv, Dh = dims.num_attention_heads, dims.num_key_value_heads, dims.head_dim
    G = dims.gqa_group
    bsz = dims.sparse_block_size
    dev = q.device

    T = q.shape[0]
    q3 = q.view(T, Hq, Dh)                       # [T, H_q, Dh]
    k3 = k.view(T, Hkv, Dh)                      # [S=T, H_kv, Dh]   (real GQA key)
    v3 = v.view(T, Hkv, Dh)                      # [S=T, H_kv, Dv=Dh]
    S = T                                        # PREFILL: keys == queries (abs order)

    abs_pos = positions.to(dev).long().view(-1)[:T]

    # ---- index branch (Gemma norm per MSA_APPLY_INDEX_NORM + partial RoPE; NO relu) -----------
    hidden = getattr(attn_module, "_msa_hidden", None)
    if hidden is None:
        raise RuntimeError(
            "[msa] wire needs attn_module._msa_hidden (the layer's hidden states for the index "
            "projections); the model must stash it before calling the wire (see msa_m3_wire "
            "docstring / the HpuMiniMaxM3Attention.forward patch).")
    q_idx, k_idx = compute_index_qk(hidden.view(T, dims.hidden_size), attn_module, dims,
                                    positions=abs_pos)             # [T,Hkv,D],[T,1,D] (Gemma-norm + rope)

    # ---- (1) cmask + (2) block-indexer-topk KERNEL OP (raw top-k blocks, HEAD-COLLAPSED) -------
    cmask, n_blocks, sk_pad = build_cmask(S, abs_pos, bsz, dev)     # [Sq, Sk_pad]
    # op wants head axis leading: q_idx -> [Hkv, Sq, D]; shared k_idx expanded to Hkv heads.
    q_op = q_idx.permute(1, 0, 2).contiguous()                     # [Hkv, Sq, D_idx]
    k_op = k_idx[:, 0, :].unsqueeze(0).expand(Hkv, -1, -1).contiguous()  # [Hkv, Sk, D_idx]
    # pad the index KEY rows to sk_pad with zeros (their cmask cols are TRUE -> -inf, so unused).
    if sk_pad != S:
        padk = torch.zeros(Hkv, sk_pad - S, dims.sparse_index_dim, dtype=k_op.dtype, device=dev)
        k_op = torch.cat([k_op, padk], dim=1)
    weights = torch.full((Hkv, S, 1), dims.index_scale, dtype=q_op.dtype, device=dev)  # 1.0 (raw)
    op_idx = MSA_OP_IMPLS["block_indexer_topk"]
    if op_idx is None:
        raise RuntimeError("[msa] block_indexer_topk op not bound (call bind_kernel_ops).")
    # OFFICIAL: scores.amax(-1).amax(dim=1) collapses the H_kv index heads to ONE block-score vector
    # per query BEFORE topk -> the op returns ONE shared set [Sq, kk] (see _cpu_block_indexer_topk /
    # the KERNEL IMPLICATION there: the compiled mode=1 needs a reduce_max over the head axis
    # inserted before its final topk to emit [Sq, kk] instead of the per-group [H_kv, Sq, kk]).
    block_ids_raw = op_idx(q_op, k_op, weights, cmask,
                           int(dims.sparse_topk_blocks), 1, int(bsz))   # [Sq, kk] int32 (ONE set)

    # ---- (3) force-local + ascending sort + (-1) pad  (oracle msa_block_select tail) ----------
    block_idx = force_local_sort_pad(block_ids_raw, abs_pos, dims, n_blocks)   # [Sq, topk]  ONE set

    # ---- (4) block -> token expansion  (oracle msa_expand_blocks_to_tokens) -------------------
    token_idx = expand_blocks_to_tokens(block_idx, bsz, seqlen=S)             # [Sq, topk*bsz]
    # within-block causal cut: the consume op masks ONLY pad slots, so push the oracle's
    # token_pos>query_pos -> -inf into token_idx as -1 (sentinel/pad). Without this the local block's
    # future tokens would leak (the one remaining consume-parity gap vs msa_gqa_gather_attention).
    token_idx = causal_mask_token_idx(token_idx, abs_pos)                     # [Sq, K]  ONE set
    K = token_idx.shape[-1]

    # ---- (5) consume KERNEL OP — SINGLE CALL over all heads (head-collapse simplification) ------
    # The official head-collapse means ONE shared token set per query (token_idx [T,K]), so EVERY
    # head/group gathers the SAME tokens. This is EXACTLY the compiled op's native contract
    # (msa_sparse_attn_backend_shared.h: token_idx is [T,K], one gid shared across all H heads). The
    # per-GQA-group 4-call loop is GONE: the 4 groups no longer have 4 different sets. We fan the 4
    # REAL kv-heads out to H_q via expand (NO 64-head physical inflation — the op gathers the GIVEN
    # K/V; an .expand view is a broadcast, not a 16x copy) and issue ONE op call.
    #
    #   *** This RESOLVES the prior 4-call-vs-1-call problem the kernel-schema note worried about:
    #   that note assumed per-group sets forced 4 calls and proposed widening token_idx to [T,H_kv,K].
    #   With the corrected head-collapse there is ONE set, so the EXISTING [T,K] op already gives the
    #   single call — NO kernel-schema change is needed for the consume. (The only on-card change is
    #   the indexer's head-collapse reduce in step 2 above.) The remaining card-free nicety would be
    #   relaxing the op's TORCH_CHECK(k.H==q.H) to k.H==H_kv so the gather runs on the 4 real heads
    #   and SDPA fans h<-h//G internally; until then we hand it the expanded view. ***
    op_attn = MSA_OP_IMPLS["sparse_attn"]
    if op_attn is None:
        raise RuntimeError("[msa] sparse_attn op not bound (call bind_kernel_ops).")
    q4 = q3.unsqueeze(2).contiguous()                             # [T, H_q, 1, Dh]
    # fan the 4 real kv-heads to H_q (head h <- kv-head h//G), expand-view (no physical 16x copy).
    k_h = k3.unsqueeze(2).expand(S, Hkv, G, Dh).reshape(S, Hq, Dh).contiguous()      # [S, H_q, Dh]
    v_h = v3.unsqueeze(2).expand(S, Hkv, G, v3.shape[-1]).reshape(S, Hq, v3.shape[-1]).contiguous()
    ti = token_idx.to(torch.int32).contiguous()                  # [T, K] (shared by all heads)
    out = op_attn(q4, k_h, v_h, ti, float(dims.main_scale), int(K))   # [T, H_q, Dv]  ONE call
    return out.reshape(T, Hq * v3.shape[-1])


# ==================================================================================================
# CPU reference impls of the two kernel ops (substituted in the parity harness via bind_kernel_ops).
# These are SMALL torch stand-ins for the COMPILED ops, matching their exact I/O contract from the
# kernel _shared.h headers. They let the WHOLE wire run on CPU so we prove the orchestration ==
# oracle. They are NOT the oracle (the oracle is ref/msa_reference.py); they are the op semantics.
# ==================================================================================================
def _cpu_block_indexer_topk(q, k, weights, cmask, index_topk, mode, block_size):
    """Torch stand-in for hpu_msa_ext::msa_block_indexer_topk (MSA mode=1), CORRECTED for the
    OFFICIAL head-collapse.

    q [H,Sq,D], k [H,Sk,D], weights [H,Sq,1], cmask [Sq,Sk] (TRUE=>drop). Returns the RAW top-k
    BLOCK ids [Sq,kk] int32 (kk=min(index_topk, n_blocks)) — ONE shared set per query.

    Node-chain (== MiniMaxM3VLIndexer.forward): batch_gemm q@k^T -> *weights(=1.0) ->
    masked_fill(cmask,-inf) -> reshape[H,Sq,nblk,B] -> amax(-1) (block-max) ->
    *** amax(dim=0) over the H index heads (HEAD-COLLAPSE, modeling line 580) *** -> topk(nblk).
    NO relu, NO head-sum. (DSA mode=0 unused by the wire.)

    >>> KERNEL IMPLICATION (on-card follow-up; do NOT hack the .so): the compiled mode=1 currently
    keeps the head axis through topk and returns [H,Sq,kk] (per-group sets — the now-WRONG premise).
    The fix is to INSERT one reduce_max over the head/batch axis (dim=0) BETWEEN node 6b (blkmax)
    and the final topk, so topk runs on [Sq,nblk] and the output is [Sq,kk]. This SIMPLIFIES the
    consume (ONE shared token set per query -> one gather, collapsing the 4-call loop to 1). <<<"""
    assert mode == 1, "wire only uses MSA mode=1"
    H, Sq, D = q.shape
    Sk = k.shape[1]
    assert Sk % block_size == 0, f"Sk={Sk} not a multiple of block_size={block_size}"
    n_blk = Sk // block_size
    s = torch.einsum("hqd,hkd->hqk", q.float(), k.float())          # [H,Sq,Sk]
    s = s * weights.float()                                          # broadcast [H,Sq,1] (=1.0)
    s = s.masked_fill(cmask[None, :, :], NEG_INF)                    # cmask broadcasts over H
    s = s.view(H, Sq, n_blk, block_size).amax(dim=-1)               # [H,Sq,n_blk] block-max-pool
    s = s.amax(dim=0)                                               # [Sq,n_blk]  HEAD-COLLAPSE (amax over H)
    kk = min(int(index_topk), n_blk)
    idx = torch.topk(s, k=kk, dim=-1).indices                       # [Sq,kk]  ONE shared set
    return idx.to(torch.int32)


def _cpu_sparse_attn(q, k, v, token_idx, scale, gather_width):
    """Torch stand-in for hpu_msa_ext::sparse_attn.

    q [T,H,1,Dh], k [S,H,Dh], v [S,H,Dv], token_idx [T,K] int32 (-1 pad), scale, gather_width==K.
    Returns out [T,H,Dv]. Reproduces the kernel: sentinel row S for -1/OOB, gather, pad-only mask,
    fp32 softmax, NaN guard. Causality is already in the selected token set (mask only pad slots) —
    matching msa_sparse_attn_backend_shared.h:37-49 / ref msa_gqa_gather_attention (consume oracle)."""
    T, H, L, Dh = q.shape
    assert L == 1
    S = k.shape[0]
    Dv = v.shape[-1]
    K = token_idx.shape[-1]
    assert int(gather_width) == K
    dev = q.device
    k_ext = torch.cat([k.float(), torch.zeros(1, H, Dh, device=dev)], dim=0)     # [S+1,H,Dh]
    v_ext = torch.cat([v.float(), torch.zeros(1, H, Dv, device=dev)], dim=0)     # [S+1,H,Dv]
    tok = token_idx.long()
    valid = (tok >= 0) & (tok < S)                                  # [T,K]
    gid = torch.where(valid, tok, torch.full_like(tok, S))          # sentinel row S
    out = torch.zeros(T, H, Dv, device=dev, dtype=torch.float32)
    qf = q.float().squeeze(2)                                       # [T,H,Dh]
    bias = torch.where(valid, torch.zeros_like(tok, dtype=torch.float32),
                       torch.full_like(tok, NEG_INF, dtype=torch.float32))       # [T,K] pad-only
    for h in range(H):
        kg = k_ext[gid, h, :]                                       # [T,K,Dh]
        vg = v_ext[gid, h, :]                                       # [T,K,Dv]
        s = torch.einsum("td,tkd->tk", qf[:, h, :], kg) * float(scale) + bias    # [T,K]
        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
        out[:, h, :] = torch.einsum("tk,tkd->td", p, vg)
    return out.to(q.dtype)


# ==================================================================================================
# MODEL PATCH — make minimax_m3.HpuMiniMaxM3Attention (1) build the index projections on sparse
# layers and (2) stash hidden_states so the wire can compute the index branch. minimax_m3.py's
# attention has the seam (MSA_WIRE call) but builds NO index projections and does NOT pass hidden
# states; this patch supplies both, out-of-tree (monkeypatch — the card-free dev copy at
# $HOME/AI/minimax-m3/vllm/minimax_m3.py is unchanged on disk; the installed serve is untouched).
#
# The index weights load with ZERO loader changes: the attached submodules are named
# ``index_q_proj`` / ``index_k_proj`` / ``index_q_norm`` / ``index_k_norm``, so the checkpoint names
# ``model.layers.N.self_attn.index_*`` resolve to them via AutoWeightsLoader's named_parameters walk
# (vLLM strips the ``language_model.`` multimodal prefix). No stacked-mapping entry needed.
# ==================================================================================================
def patch_model_attention(dims: MsaDims, *, rms_norm_eps: float = 1e-6) -> None:
    """Wrap HpuMiniMaxM3Attention.__init__ (attach index projections on sparse layers) and .forward
    (stash hidden_states before the wire). Idempotent. Call from install()."""
    m3 = _import_m3()
    Attn = m3.HpuMiniMaxM3Attention

    if not getattr(Attn.__init__, "_msa_index_wrapped", False):
        _orig_init = Attn.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # is_sparse is set on the module by the model (minimax_m3.py:367). Only sparse layers get
            # the index branch (dense layers 0..2 carry no index_* tensors in the checkpoint).
            if getattr(self, "is_sparse", False):
                add_index_projections(self, dims, rms_norm_eps=rms_norm_eps)

        __init__._msa_index_wrapped = True
        __init__._msa_orig = _orig_init
        Attn.__init__ = __init__

    if not getattr(Attn.forward, "_msa_hidden_wrapped", False):
        _orig_forward = Attn.forward

        def forward(self, positions, hidden_states):
            # Stash for the wire's index branch (read by msa_wire via attn_module._msa_hidden). Only
            # needed on sparse layers with the gate open; cheap reference assignment otherwise.
            if getattr(self, "is_sparse", False):
                self._msa_hidden = hidden_states
            return _orig_forward(self, positions, hidden_states)

        forward._msa_hidden_wrapped = True
        forward._msa_orig = _orig_forward
        Attn.forward = forward


def _import_m3():
    import importlib
    if _can_import("minimax_m3"):
        return importlib.import_module("minimax_m3")
    return importlib.import_module("vllm_gaudi.models.minimax_m3")


def install(config: Any) -> MsaDims:
    """Convenience: read dims from config, attach the index branch + hidden-stash to the model
    attention, and bind the wire into minimax_m3. Returns the dims.

    Does NOT flip the readiness gate (that is platform_patch_msa.maybe_flip_ready, called last by
    the plugin). With the wire installed but MSA_READY closed, the model STAYS in dense-fallback
    (set_msa_wire(..., full_attention_fallback=True))."""
    m3 = _import_m3()
    dims = MsaDims.from_config(config)
    patch_model_attention(dims, rms_norm_eps=float(_cfg_get(_resolve_text_config(config),
                                                            "rms_norm_eps", 1e-6)))
    # wire authored but gate CLOSED (full_attention_fallback=True) until on-card validation.
    m3.set_msa_wire(msa_wire, full_attention_fallback=True)
    # tell the platform patch the wire is present (so maybe_flip_ready can open the seam later).
    try:
        import importlib
        importlib.import_module("platform_patch_msa" if _can_import("platform_patch_msa")
                               else "hpu_msa_ext.platform_patch_msa").WIRE_INSTALLED = True
    except Exception:  # noqa: BLE001 — platform patch optional in a pure-wire smoke
        pass
    return dims


def _can_import(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None
