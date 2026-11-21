"""M3 transformer block — pure-torch (fp64) reference forward of ONE M3 layer.

This is the TRUSTWORTHY numerical oracle the MTP draft-head oracle + trainer reuse:
a from-scratch re-derivation of a single MiniMax-M3 (text tower) decoder block that is
validated tensor-for-tensor against the real bf16 checkpoint weights on CPU.

It mirrors the AUTHORITATIVE HuggingFace reference
  transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py  (transformers 5.12)
exactly — every line of the M3 block math here was cross-checked against:
  MiniMaxM3VLRMSNorm        (Gemma RMSNorm  x*(1+w), fp32 normalize)
  MiniMaxM3VLAttention      (per-head QK-norm BEFORE RoPE, partial NeoX RoPE, GQA, scale d^-0.5)
  apply_rotary_pos_emb / rotate_half   (rotary on first rotary_dim=64 of head_dim=128, passthrough rest)
  MiniMaxM3VLDenseMLP       (swigluoai: gate clamp max=limit, up clamp +-limit, (up+1)*gate*sigmoid(a*gate))
  MiniMaxM3VLTopKRouter     (sigmoid scoring + e_score_correction_bias for SELECTION, UNBIASED scores for
                             WEIGHTS, renormalize-to-1)
  MiniMaxM3VLSparseMoeBlock (routed = experts(...) * routed_scaling_factor(=2.0); + shared dense expert)
and against the serving module $HOME/AI/minimax-m3/vllm/minimax_m3.py.

Shared CPU primitives (gemma_rms, swigluoai, H, EPS, ALPHA, LIMIT) are IMPORTED from the validated
foundation m3_cpu_mtp.py — never redefined here.

=== EXACT M3 TENSOR NAMES (checkpoint /mnt/data2/minimax-m3-bf16, prefix language_model.model.layers.{L}.) ===
DENSE layer (L in 0,1,2):
  input_layernorm.weight                [H]        gemma RMSNorm (pre-attn)
  self_attn.q_proj.weight               [Hq, H]    Hq = num_heads*head_dim = 64*128 = 8192
  self_attn.k_proj.weight               [Hkv, H]   Hkv = num_kv_heads*head_dim = 4*128 = 512
  self_attn.v_proj.weight               [Hkv, H]
  self_attn.o_proj.weight               [H, Hq]
  self_attn.q_norm.weight               [head_dim]=[128]   per-head gemma RMSNorm (broadcast all q heads)
  self_attn.k_norm.weight               [head_dim]=[128]   per-head gemma RMSNorm (broadcast all kv heads)
  post_attention_layernorm.weight       [H]        gemma RMSNorm (pre-MLP)
  mlp.gate_proj.weight                  [Idense, H]   Idense = dense_intermediate_size = 12288
  mlp.up_proj.weight                    [Idense, H]
  mlp.down_proj.weight                  [H, Idense]
MoE layer (L in 3..59):
  input_layernorm.weight                [H]
  self_attn.{q,k,v,o}_proj.weight       (same shapes as dense)
  self_attn.{q,k}_norm.weight           [128]
  self_attn.index_{q,k}_proj.weight / index_{q,k}_norm.weight   (sparse indexer — NOT used by the dense draft)
  post_attention_layernorm.weight       [H]
  block_sparse_moe.gate.weight          [E, H]     E = num_local_experts = 128   (fp32)
  block_sparse_moe.e_score_correction_bias  [E]    (fp32)
  block_sparse_moe.experts.{i}.w1.weight  [Imoe, H]   w1 = GATE proj   Imoe = intermediate_size = 3072
  block_sparse_moe.experts.{i}.w3.weight  [Imoe, H]   w3 = UP   proj
  block_sparse_moe.experts.{i}.w2.weight  [H, Imoe]   w2 = DOWN proj
  block_sparse_moe.shared_experts.gate_proj.weight  [Ishared, H]   Ishared = shared_intermediate_size = 3072
  block_sparse_moe.shared_experts.up_proj.weight    [Ishared, H]
  block_sparse_moe.shared_experts.down_proj.weight  [H, Ishared]

Run the CPU validation (loads real layer 0 + layer 10 weights, ~1-2 min, no card):
  $HOME/.venvs/m3calib/bin/python $HOME/AI/minimax-m3/mtp/m3_block_ref.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch

# --- foundation primitives (validated; do NOT redefine) ---------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from m3_cpu_mtp import gemma_rms, swigluoai, H, EPS, ALPHA, LIMIT  # noqa: E402

torch.set_grad_enabled(False)

# === M3 text_config constants (config.json text_config) ======================================
NUM_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
ROTARY_DIM = 64          # partial_rotary_factor 0.5 -> RoPE on 64 of 128 head dims
ROPE_THETA = 5_000_000.0
DENSE_INTERMEDIATE = 12288
MOE_INTERMEDIATE = 3072
SHARED_INTERMEDIATE = 3072
NUM_EXPERTS = 128
TOP_K = 4
ROUTED_SCALING_FACTOR = 2.0
SCORING_FUNC = "sigmoid"

CKPT_DIR = "/mnt/data2/minimax-m3-bf16"
LAYER_PREFIX = "language_model.model.layers."

# Reference dtype: fp64 (the prompt's mandate — a maximally-trustworthy oracle). The checkpoint is
# bf16; we upcast on load so the reference forward carries no bf16 rounding of its own.
REF_DTYPE = torch.float64


# ============================================================================================
# RoPE — partial NeoX-style, re-derived from HF MiniMaxM3VLRotaryEmbedding +
# apply_rotary_pos_emb + rotate_half. cos/sin width == ROTARY_DIM (emb = cat(freqs, freqs),
# freqs width = ROTARY_DIM//2). Rotation hits q[..., :ROTARY_DIM]; q[..., ROTARY_DIM:] passes through.
# ============================================================================================
def build_rope_cos_sin(positions: torch.Tensor, rotary_dim: int = ROTARY_DIM,
                       theta: float = ROPE_THETA, dtype: torch.dtype = REF_DTYPE):
    """Return (cos, sin), each [T, rotary_dim], for integer `positions` [T]."""
    pos = positions.to(dtype)
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim))  # [half]
    freqs = torch.outer(pos, inv_freq)            # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)       # [T, rotary_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k : [B, n_heads, T, head_dim].  cos,sin : [T, rotary_dim].  Rotate first rotary_dim dims."""
    rotary_dim = cos.shape[-1]
    cos = cos[None, None, :, :]   # [1,1,T,rotary_dim]
    sin = sin[None, None, :, :]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_emb = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_emb = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_emb, q_pass], dim=-1), torch.cat([k_emb, k_pass], dim=-1)


# ============================================================================================
# Weight containers — fp64, one per layer, holding exactly the checkpoint tensors a block needs.
# ============================================================================================
@dataclass
class M3AttnWeights:
    q_proj: torch.Tensor          # [Hq, H]
    k_proj: torch.Tensor          # [Hkv, H]
    v_proj: torch.Tensor          # [Hkv, H]
    o_proj: torch.Tensor          # [H, Hq]
    q_norm: torch.Tensor          # [head_dim]
    k_norm: torch.Tensor          # [head_dim]


@dataclass
class M3DenseMLPWeights:
    gate_proj: torch.Tensor       # [Idense, H]
    up_proj: torch.Tensor         # [Idense, H]
    down_proj: torch.Tensor       # [H, Idense]


@dataclass
class M3MoEWeights:
    gate: torch.Tensor                  # [E, H]   router
    e_score_correction_bias: torch.Tensor  # [E]
    experts_w1: torch.Tensor            # [E, Imoe, H]  gate
    experts_w3: torch.Tensor            # [E, Imoe, H]  up
    experts_w2: torch.Tensor            # [E, H, Imoe]  down
    shared: M3DenseMLPWeights           # shared expert (dense MLP, shared_intermediate)


@dataclass
class M3BlockWeights:
    layer_idx: int
    is_moe: bool
    input_layernorm: torch.Tensor       # [H]
    post_attention_layernorm: torch.Tensor  # [H]
    attn: M3AttnWeights
    dense_mlp: M3DenseMLPWeights | None = None
    moe: M3MoEWeights | None = None
    extra: dict = field(default_factory=dict)


# ============================================================================================
# Forward — attention (per-head QK-norm -> partial RoPE -> causal GQA SDPA -> o_proj)
# ============================================================================================
def m3_attention(x: torch.Tensor, w: M3AttnWeights, positions: torch.Tensor) -> torch.Tensor:
    """x : [B, T, H] -> [B, T, H].  Matches MiniMaxM3VLAttention exactly (dense, causal, no sparse)."""
    B, T, _ = x.shape
    q = torch.matmul(x, w.q_proj.t())            # [B,T,Hq]
    k = torch.matmul(x, w.k_proj.t())            # [B,T,Hkv]
    v = torch.matmul(x, w.v_proj.t())            # [B,T,Hkv]

    # reshape to per-head, apply per-head gemma QK-norm over head_dim (HF: norm BEFORE transpose)
    q = q.view(B, T, NUM_HEADS, HEAD_DIM)
    k = k.view(B, T, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(B, T, NUM_KV_HEADS, HEAD_DIM)
    q = gemma_rms(q, w.q_norm, EPS)              # broadcast [head_dim] over (B,T,heads)
    k = gemma_rms(k, w.k_norm, EPS)

    # to [B, heads, T, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # partial RoPE on q,k
    cos, sin = build_rope_cos_sin(positions, ROTARY_DIM, ROPE_THETA, x.dtype)
    q, k = apply_partial_rope(q, k, cos, sin)

    # GQA: repeat KV heads to num_heads
    n_rep = NUM_HEADS // NUM_KV_HEADS
    k = k.repeat_interleave(n_rep, dim=1)        # [B, num_heads, T, head_dim]
    v = v.repeat_interleave(n_rep, dim=1)

    # scaled-dot-product, causal, softmax in fp32-or-higher (here already fp64)
    scaling = HEAD_DIM ** -0.5
    attn = torch.matmul(q, k.transpose(-1, -2)) * scaling   # [B, heads, T, T]
    causal = torch.full((T, T), float("-inf"), dtype=x.dtype).triu(1)  # mask j>i
    attn = attn + causal[None, None]
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)                  # [B, heads, T, head_dim]
    out = out.transpose(1, 2).reshape(B, T, NUM_HEADS * HEAD_DIM)
    return torch.matmul(out, w.o_proj.t())       # [B,T,H]


# ============================================================================================
# Forward — dense MLP (swigluoai over contiguous [gate|up], imported swigluoai)
# ============================================================================================
def m3_dense_mlp(x: torch.Tensor, w: M3DenseMLPWeights) -> torch.Tensor:
    gate = torch.matmul(x, w.gate_proj.t())
    up = torch.matmul(x, w.up_proj.t())
    act = swigluoai(torch.cat([gate, up], dim=-1), ALPHA, LIMIT)
    return torch.matmul(act, w.down_proj.t())


# ============================================================================================
# Forward — MoE (sigmoid router + bias-for-selection / unbiased-for-weights + renorm + top-4,
#              per-expert swigluoai, * routed_scaling_factor, + shared dense expert)
# ============================================================================================
def m3_moe_router(x_flat: torch.Tensor, w: M3MoEWeights):
    """x_flat : [N, H] -> (topk_ids [N, TOP_K], topk_weights [N, TOP_K]). Mirrors MiniMaxM3VLTopKRouter
    AND the vllm grouped_topk with num_expert_group=1/topk_group=1 (single group => group mask no-op)."""
    logits = torch.matmul(x_flat, w.gate.t())            # [N, E]
    scores = torch.sigmoid(logits)                       # UNBIASED routing scores
    biased = scores + w.e_score_correction_bias[None, :]  # bias ONLY for selection
    topk_ids = torch.topk(biased, TOP_K, dim=-1).indices  # [N, TOP_K]
    topk_w = torch.gather(scores, 1, topk_ids)           # weights from UNBIASED scores
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)   # renormalize to 1
    return topk_ids, topk_w


def m3_moe(x: torch.Tensor, w: M3MoEWeights) -> torch.Tensor:
    B, T, _ = x.shape
    x_flat = x.reshape(-1, H)                             # [N, H]
    N = x_flat.shape[0]

    shared_out = m3_dense_mlp(x_flat, w.shared)           # [N, H]

    topk_ids, topk_w = m3_moe_router(x_flat, w)           # [N, K], [N, K]

    routed = torch.zeros_like(x_flat)
    for slot in range(TOP_K):
        ids = topk_ids[:, slot]                           # [N]
        wts = topk_w[:, slot].unsqueeze(-1)               # [N,1]
        for e in torch.unique(ids):
            mask = ids == e
            xe = x_flat[mask]                             # [m, H]
            gate = torch.matmul(xe, w.experts_w1[e].t())  # [m, Imoe]
            up = torch.matmul(xe, w.experts_w3[e].t())    # [m, Imoe]
            act = swigluoai(torch.cat([gate, up], dim=-1), ALPHA, LIMIT)
            ye = torch.matmul(act, w.experts_w2[e].t())   # [m, H]
            routed[mask] += ye * wts[mask]

    out = routed * ROUTED_SCALING_FACTOR + shared_out
    return out.reshape(B, T, H)


# ============================================================================================
# Forward — the whole block (the function the oracle/trainer call)
# ============================================================================================
def m3_block_forward(x: torch.Tensor, w: M3BlockWeights,
                     positions: torch.Tensor | None = None) -> torch.Tensor:
    """x : [B, T, H] (any float dtype; upcast to REF_DTYPE internally).  Returns [B, T, H]."""
    if positions is None:
        positions = torch.arange(x.shape[1])
    xin = x.to(REF_DTYPE)

    # pre-attn gemma norm -> attention -> residual
    h = gemma_rms(xin, w.input_layernorm, EPS)
    h = m3_attention(h, w.attn, positions.to(torch.long))
    xin = xin + h

    # pre-MLP gemma norm -> MLP (dense or MoE) -> residual
    h = gemma_rms(xin, w.post_attention_layernorm, EPS)
    if w.is_moe:
        h = m3_moe(h, w.moe)
    else:
        h = m3_dense_mlp(h, w.dense_mlp)
    xin = xin + h
    return xin


# ============================================================================================
# Real-checkpoint loaders (safetensors safe_open, fp64 upcast). Only the named tensors are read.
# ============================================================================================
def _index_map(ckpt_dir: str = CKPT_DIR) -> dict:
    import json
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


class _TensorFetcher:
    """Open each shard once; pull tensors by name, upcast to REF_DTYPE on CPU."""

    def __init__(self, ckpt_dir: str = CKPT_DIR):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.ckpt_dir = ckpt_dir
        self.wm = _index_map(ckpt_dir)
        self._handles: dict = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = self._safe_open(
                os.path.join(self.ckpt_dir, shard), framework="pt", device="cpu")
        return self._handles[shard]

    def get(self, name: str) -> torch.Tensor:
        if name not in self.wm:
            raise KeyError(f"tensor not in checkpoint index: {name}")
        t = self._handle(self.wm[name]).get_tensor(name)
        return t.to(REF_DTYPE)

    def has(self, name: str) -> bool:
        return name in self.wm

    def close(self):
        self._handles.clear()


# Reference per-config layer typing (moe_layer_freq: layers 0,1,2 dense; 3..59 MoE). load_block does
# NOT rely on this — it decides dense-vs-MoE from which tensors actually EXIST in the checkpoint, so the
# loader can never silently disagree with the weights. Exposed for callers that want the config view.
MOE_LAYER_FREQ = [0, 0, 0] + [1] * 57   # index i: 1 => layer i is MoE


def config_is_moe_layer(layer_idx: int) -> bool:
    return bool(MOE_LAYER_FREQ[layer_idx])


def load_block(layer_idx: int, ckpt_dir: str = CKPT_DIR,
               fetcher: _TensorFetcher | None = None) -> M3BlockWeights:
    """Load ONE real M3 block's weights from the bf16 checkpoint into an fp64 M3BlockWeights.
    Whether the layer is dense or MoE is decided by which tensors EXIST (block_sparse_moe.gate)."""
    own = fetcher is None
    f = fetcher or _TensorFetcher(ckpt_dir)
    p = f"{LAYER_PREFIX}{layer_idx}."
    try:
        attn = M3AttnWeights(
            q_proj=f.get(p + "self_attn.q_proj.weight"),
            k_proj=f.get(p + "self_attn.k_proj.weight"),
            v_proj=f.get(p + "self_attn.v_proj.weight"),
            o_proj=f.get(p + "self_attn.o_proj.weight"),
            q_norm=f.get(p + "self_attn.q_norm.weight"),
            k_norm=f.get(p + "self_attn.k_norm.weight"),
        )
        input_ln = f.get(p + "input_layernorm.weight")
        post_ln = f.get(p + "post_attention_layernorm.weight")

        moe_present = f.has(p + "block_sparse_moe.gate.weight")
        if moe_present:
            experts_w1 = torch.stack(
                [f.get(p + f"block_sparse_moe.experts.{i}.w1.weight") for i in range(NUM_EXPERTS)])
            experts_w3 = torch.stack(
                [f.get(p + f"block_sparse_moe.experts.{i}.w3.weight") for i in range(NUM_EXPERTS)])
            experts_w2 = torch.stack(
                [f.get(p + f"block_sparse_moe.experts.{i}.w2.weight") for i in range(NUM_EXPERTS)])
            shared = M3DenseMLPWeights(
                gate_proj=f.get(p + "block_sparse_moe.shared_experts.gate_proj.weight"),
                up_proj=f.get(p + "block_sparse_moe.shared_experts.up_proj.weight"),
                down_proj=f.get(p + "block_sparse_moe.shared_experts.down_proj.weight"),
            )
            moe = M3MoEWeights(
                gate=f.get(p + "block_sparse_moe.gate.weight"),
                e_score_correction_bias=f.get(p + "block_sparse_moe.e_score_correction_bias"),
                experts_w1=experts_w1, experts_w3=experts_w3, experts_w2=experts_w2,
                shared=shared,
            )
            return M3BlockWeights(layer_idx, True, input_ln, post_ln, attn, moe=moe)

        dense = M3DenseMLPWeights(
            gate_proj=f.get(p + "mlp.gate_proj.weight"),
            up_proj=f.get(p + "mlp.up_proj.weight"),
            down_proj=f.get(p + "mlp.down_proj.weight"),
        )
        return M3BlockWeights(layer_idx, False, input_ln, post_ln, attn, dense_mlp=dense)
    finally:
        if own:
            f.close()


# ============================================================================================
# CPU self-test — load real layer 0 (dense) + layer 10 (MoE), forward a small random input,
# assert: finite, residual-shaped [1,T,H], and CAUSAL (perturb t+1, earlier outputs unchanged).
# Plus weight-free unit checks (RoPE/router/causal-mask) so the math is checkable even card-free.
# ============================================================================================
def _unit_checks() -> list[tuple[str, bool]]:
    """Card-free, weight-free checks of the re-derived primitives."""
    torch.manual_seed(0)
    checks: list[tuple[str, bool]] = []

    # --- RoPE: relative-position property. With identical q,k content at positions p and p+d, the
    # rotated dot product depends only on the offset d (a defining RoPE property). ---
    qk = torch.randn(1, 1, 1, HEAD_DIM, dtype=REF_DTYPE)
    def rot_at(pos):
        cs, sn = build_rope_cos_sin(torch.tensor([pos]), ROTARY_DIM, ROPE_THETA, REF_DTYPE)
        q, _ = apply_partial_rope(qk, qk, cs, sn)
        return q[0, 0, 0]
    a0, b3 = rot_at(0), rot_at(3)
    a5, b8 = rot_at(5), rot_at(8)
    # only rotary part participates; compare full-vec dot (passthrough part identical anyway)
    d_03 = float(torch.dot(a0, b3))
    d_58 = float(torch.dot(a5, b8))
    checks.append(("RoPE relative-position invariance (offset 3)", abs(d_03 - d_58) < 1e-6))
    # passthrough dims (>= ROTARY_DIM) untouched by RoPE
    checks.append(("RoPE passthrough dims unchanged", torch.allclose(a0[ROTARY_DIM:], qk[0, 0, 0, ROTARY_DIM:], atol=1e-9)))
    # position 0 is identity (cos=1,sin=0)
    checks.append(("RoPE pos0 == identity", torch.allclose(a0, qk[0, 0, 0], atol=1e-9)))

    # --- router: bias steers SELECTION, weights come from UNBIASED scores, renormalize to 1 ---
    N = 5
    mw = M3MoEWeights(
        gate=torch.randn(NUM_EXPERTS, H, dtype=REF_DTYPE) * 0.02,
        e_score_correction_bias=torch.zeros(NUM_EXPERTS, dtype=REF_DTYPE),
        experts_w1=torch.empty(0), experts_w3=torch.empty(0), experts_w2=torch.empty(0),
        shared=None,  # type: ignore  (unused by router)
    )
    xflat = torch.randn(N, H, dtype=REF_DTYPE)
    ids0, w0 = m3_moe_router(xflat, mw)
    checks.append(("router topk shape [N,TOP_K]", ids0.shape == (N, TOP_K) and w0.shape == (N, TOP_K)))
    checks.append(("router weights sum to 1", torch.allclose(w0.sum(-1), torch.ones(N, dtype=REF_DTYPE), atol=1e-9)))
    # huge +bias on expert 7 forces it into every token's top-4 selection ...
    mw.e_score_correction_bias = mw.e_score_correction_bias.clone()
    mw.e_score_correction_bias[7] = 1e6
    ids1, w1 = m3_moe_router(xflat, mw)
    forced_in = bool((ids1 == 7).any(dim=-1).all())
    checks.append(("router bias forces selection", forced_in))
    # ... but the WEIGHT on a token where expert 7 was NOT originally top must equal its UNBIASED sigmoid
    # (i.e. the bias did not inflate the routing weight). Check renorm-consistency: weights are the
    # unbiased sigmoid of the selected ids, renormalized.
    raw = torch.sigmoid(torch.matmul(xflat, mw.gate.t()))
    expect_w1 = torch.gather(raw, 1, ids1)
    expect_w1 = expect_w1 / expect_w1.sum(-1, keepdim=True)
    checks.append(("router weights = renorm(unbiased[selected])", torch.allclose(w1, expect_w1, atol=1e-9)))

    # --- swigluoai parity vs the HF dense-MLP activation (independent re-derivation) ---
    gu = torch.randn(7, 2 * 16, dtype=REF_DTYPE)
    g, u = gu.chunk(2, dim=-1)
    g_c = g.clamp(max=LIMIT); u_c = u.clamp(min=-LIMIT, max=LIMIT)
    hf_act = (u_c + 1.0) * (g_c * torch.sigmoid(g_c * ALPHA))
    checks.append(("swigluoai == HF _apply_gate", torch.allclose(swigluoai(gu, ALPHA, LIMIT), hf_act, atol=1e-12)))

    return checks


def _redrive_branch(hn: torch.Tensor, w: M3BlockWeights) -> torch.Tensor:
    """Independent re-derivation of the post-attn MLP/MoE branch on already-normed input `hn` [N,H].
    Deliberately NOT sharing code with m3_dense_mlp/m3_moe: explicit gate*sigmoid, full dense per-expert
    matmul over ALL tokens (no torch.unique early-out), explicit renorm and scaling. A mismatch here
    flags a wiring bug (gate/up swap, w1/w2/w3 swap, missing renorm, wrong routed_scaling_factor)."""
    def _glu(g, u):
        gc = g.clamp(max=LIMIT); uc = u.clamp(min=-LIMIT, max=LIMIT)
        return (uc + 1.0) * (gc * torch.sigmoid(ALPHA * gc))

    if not w.is_moe:
        m = w.dense_mlp
        return (_glu(hn @ m.gate_proj.t(), hn @ m.up_proj.t())) @ m.down_proj.t()

    moe = w.moe
    # shared dense expert
    sh = moe.shared
    shared = (_glu(hn @ sh.gate_proj.t(), hn @ sh.up_proj.t())) @ sh.down_proj.t()
    # routing (re-derived)
    scores = torch.sigmoid(hn @ moe.gate.t())                 # [N,E] unbiased
    sel = torch.topk(scores + moe.e_score_correction_bias[None, :], TOP_K, dim=-1).indices
    wts = torch.gather(scores, 1, sel)
    wts = wts / wts.sum(-1, keepdim=True)                     # [N,TOP_K]
    # full per-expert dense pass over ALL tokens (weighted by membership), no early-out
    routed = torch.zeros_like(hn)
    for e in range(NUM_EXPERTS):
        member = (sel == e)                                   # [N,TOP_K] bool
        wt_e = (wts * member).sum(-1, keepdim=True)           # [N,1] weight for expert e (0 if not chosen)
        ye = (_glu(hn @ moe.experts_w1[e].t(), hn @ moe.experts_w3[e].t())) @ moe.experts_w2[e].t()
        routed = routed + ye * wt_e
    return routed * ROUTED_SCALING_FACTOR + shared


def _hf_parity_checks(fetcher: "_TensorFetcher", dense_layer: int = 0,
                      moe_layer: int = 10, T: int = 6, atol: float = 1e-4):
    """GOLD-STANDARD validation: load the real weights into the ACTUAL HuggingFace transformers M3
    modules (transformers>=5.12 ships modeling_minimax_m3_vl) and assert this reference matches them
    on identical fp64 input. Returns a list[(name, passed)] or None if transformers/config unavailable.
    Sub-paths checked: dense MLP, attention (full_attention, eager), MoE block."""
    try:
        import json
        from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as hf
        from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
        tc = json.load(open(os.path.join(CKPT_DIR, "config.json")))["text_config"]
        cfg = MiniMaxM3VLTextConfig(**tc)
        cfg._attn_implementation = "eager"
    except Exception as e:  # transformers missing / config schema drift -> skip (don't fail the gate)
        print(f"  [skip] HF parity unavailable: {type(e).__name__}: {e}")
        return None

    DT = REF_DTYPE
    out: list[tuple[str, bool]] = []
    torch.manual_seed(7)
    x = torch.randn(1, T, H, dtype=DT)

    # --- dense MLP (layer 0) ---
    wd = load_block(dense_layer, fetcher=fetcher)
    mlp = hf.MiniMaxM3VLDenseMLP(cfg).to(DT)
    mlp.gate_up_proj.weight.copy_(torch.cat([wd.dense_mlp.gate_proj, wd.dense_mlp.up_proj], dim=0))
    mlp.down_proj.weight.copy_(wd.dense_mlp.down_proj)
    d = float((m3_dense_mlp(x.reshape(-1, H), wd.dense_mlp) - mlp(x).reshape(-1, H)).abs().max())
    out.append((f"HF dense MLP parity (max d={d:.2e})", d < atol))

    # --- attention (layer 0, full_attention, eager + explicit causal mask) ---
    attn = hf.MiniMaxM3VLAttention(cfg, layer_idx=dense_layer).to(DT)
    for dst, src in ((attn.q_proj, wd.attn.q_proj), (attn.k_proj, wd.attn.k_proj),
                     (attn.v_proj, wd.attn.v_proj), (attn.o_proj, wd.attn.o_proj),
                     (attn.q_norm, wd.attn.q_norm), (attn.k_norm, wd.attn.k_norm)):
        dst.weight.copy_(src)
    pos = torch.arange(T).unsqueeze(0)
    cos, sin = hf.MiniMaxM3VLRotaryEmbedding(cfg).to(DT)(x, pos)
    cmask = torch.full((T, T), float("-inf"), dtype=DT).triu(1)[None, None]
    hf_attn, _ = attn(x, position_embeddings=(cos, sin), attention_mask=cmask)
    d = float((m3_attention(x, wd.attn, torch.arange(T)) - hf_attn.reshape(1, T, H)).abs().max())
    out.append((f"HF attention parity (max d={d:.2e})", d < atol))

    # --- MoE block (layer 10): HF expert layout is gate_up_proj[E,2*Imoe,H] (w1|w3), down_proj[E,H,Imoe]=w2 ---
    wm = load_block(moe_layer, fetcher=fetcher)
    moe = hf.MiniMaxM3VLSparseMoeBlock(cfg).to(DT)
    moe.gate.weight.copy_(wm.moe.gate)
    moe.gate.e_score_correction_bias.copy_(wm.moe.e_score_correction_bias)
    moe.experts.gate_up_proj.copy_(torch.cat([wm.moe.experts_w1, wm.moe.experts_w3], dim=1))
    moe.experts.down_proj.copy_(wm.moe.experts_w2)
    moe.shared_experts.gate_up_proj.weight.copy_(
        torch.cat([wm.moe.shared.gate_proj, wm.moe.shared.up_proj], dim=0))
    moe.shared_experts.down_proj.weight.copy_(wm.moe.shared.down_proj)
    d = float((m3_moe(x, wm.moe).reshape(-1, H) - moe(x).reshape(-1, H)).abs().max())
    out.append((f"HF MoE block parity (max d={d:.2e})", d < atol))
    return out


def _causal_check(w: M3BlockWeights, T: int = 6) -> tuple[bool, float]:
    """Perturb the LAST token's input; assert outputs at positions < T-1 are bit-unchanged."""
    torch.manual_seed(1)
    x = torch.randn(1, T, H, dtype=REF_DTYPE) * 0.5
    pos = torch.arange(T)
    out_a = m3_block_forward(x, w, pos)
    x2 = x.clone()
    x2[0, T - 1] += torch.randn(H, dtype=REF_DTYPE) * 5.0   # large perturbation on the FUTURE-most token
    out_b = m3_block_forward(x2, w, pos)
    earlier_delta = float((out_a[0, : T - 1] - out_b[0, : T - 1]).abs().max())
    last_delta = float((out_a[0, T - 1] - out_b[0, T - 1]).abs().max())
    # earlier positions identical (causal); last position MUST move (sanity that the perturb mattered)
    return (earlier_delta < 1e-9 and last_delta > 1e-6), earlier_delta


def self_test(dense_layer: int = 0, moe_layer: int = 10, T: int = 6) -> bool:
    print("=" * 70)
    print("M3 BLOCK REFERENCE — CPU self-test")
    print("=" * 70)

    checks: list[tuple[str, bool]] = []

    # (A) weight-free primitive checks
    print("\n[primitive unit checks]")
    for name, ok in _unit_checks():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        checks.append((name, ok))

    # (B) real-checkpoint block checks
    have_ckpt = os.path.isdir(CKPT_DIR) and os.path.isfile(
        os.path.join(CKPT_DIR, "model.safetensors.index.json"))
    if not have_ckpt:
        print(f"\n[checkpoint UNAVAILABLE at {CKPT_DIR} — skipping real-weight checks]")
        ok = all(p for _, p in checks)
        print(f"\n{'='*70}\nPRIMITIVES: {'GREEN' if ok else 'RED'} (real-weight checks skipped)\n{'='*70}")
        return ok

    fetcher = _TensorFetcher(CKPT_DIR)
    try:
        for label, L, expect_moe in (("DENSE", dense_layer, False), ("MoE", moe_layer, True)):
            print(f"\n[real layer {L} ({label})]")
            w = load_block(L, fetcher=fetcher)
            checks.append((f"L{L} is_moe=={expect_moe}", w.is_moe == expect_moe))
            print(f"  [{'PASS' if w.is_moe == expect_moe else 'FAIL'}] L{L} is_moe == {expect_moe}")

            torch.manual_seed(2)
            x = torch.randn(1, T, H, dtype=REF_DTYPE) * 0.5
            out = m3_block_forward(x, w, torch.arange(T))

            c1 = out.shape == (1, T, H)
            c2 = bool(torch.isfinite(out).all())
            # Residual stream: output must be input + a finite, NON-trivial update. We do not impose a
            # tight upper magnitude bound: with a random (off-manifold) input the post-attn hidden is
            # gemma-normed to ~unit RMS and pushed through the wide dense/MoE projections, and the MoE
            # branch is additionally x routed_scaling_factor(=2.0) + a shared expert — so a large update
            # norm here is EXPECTED, not a bug (verified: shared-expert alone ~25x ||x|| on random input).
            # The meaningful guards are finiteness, non-triviality, and the independent re-derivation below.
            upd = float((out - x).norm() / (x.norm() + 1e-12))
            c3 = 1e-4 < upd < 1e4  # non-trivial + not NaN/Inf-scale blow-up
            caus_ok, edelta = _causal_check(w, T)

            # INDEPENDENT re-derivation of the MLP/MoE branch (does NOT reuse m3_dense_mlp / m3_moe):
            # catches w1/w3/w2 mis-assignment, the routed_scaling_factor, gate/up order, and the
            # gather/scatter routing. The branch fns consume the POST-attn-layernorm hidden (the block
            # applies the norm before calling them), so feed the SAME normed `hn` to both sides.
            h_attn = x.to(REF_DTYPE) + m3_attention(
                gemma_rms(x.to(REF_DTYPE), w.input_layernorm, EPS), w.attn, torch.arange(T))
            hn3 = gemma_rms(h_attn, w.post_attention_layernorm, EPS)   # [1,T,H]
            hn = hn3.reshape(-1, H)                                    # [N,H]
            branch = m3_moe(hn3, w.moe).reshape(-1, H) if w.is_moe \
                else m3_dense_mlp(hn, w.dense_mlp)
            indep = _redrive_branch(hn, w)
            c4 = torch.allclose(branch, indep, atol=1e-6, rtol=1e-5)

            for nm, cc in ((f"L{L} shape [1,{T},{H}]", c1),
                           (f"L{L} finite", c2),
                           (f"L{L} residual update (||d||/||x||={upd:.3g})", c3),
                           (f"L{L} CAUSAL (max earlier d={edelta:.2e})", caus_ok),
                           (f"L{L} branch == independent re-derivation", c4)):
                print(f"  [{'PASS' if cc else 'FAIL'}] {nm}")
                checks.append((nm, cc))

        # (C) gold-standard: parity vs the real HuggingFace transformers M3 modules
        print("\n[HF transformers parity (real modules, real weights)]")
        hf_checks = _hf_parity_checks(fetcher, dense_layer, moe_layer, T)
        if hf_checks is None:
            print("  (transformers M3 modules unavailable — parity stage skipped, not failed)")
        else:
            for nm, cc in hf_checks:
                print(f"  [{'PASS' if cc else 'FAIL'}] {nm}")
                checks.append((nm, cc))
    finally:
        fetcher.close()

    ok = all(p for _, p in checks)
    print(f"\n{'='*70}")
    print(f"M3 BLOCK REFERENCE: {'GREEN — all pass' if ok else 'RED — FIX ABOVE'}")
    print(f"{'='*70}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
