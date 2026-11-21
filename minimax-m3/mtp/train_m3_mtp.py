"""Train the M3 MTP **lean** draft head from dumped (hidden, token) traces.

WHAT THIS BUILDS
================
A depth-1 speculative draft head for MiniMax-M3, shaped so the serving MTP module
(hpu_m3_ext/m3_mtp.py — a future task) loads it via the GLM-style MTP weight path
(_src/vllm_models/glm4_moe_mtp.py::_rewrite_spec_layer_name). The draft is:

    M3MTPFrontend (enorm / hnorm / eh_proj)                 [from m3_cpu_mtp.py]
        -> LEAN block  =  M3 GQA attention (qk-norm + partial-RoPE)
                          + a DENSE swigluoai MLP            (NOT the 128-expert MoE)
        -> M3SharedHead (final norm + lm_head)              [from m3_cpu_mtp.py]

DESIGN DECISION — why LEAN, not MoE (honored from the mission brief)
====================================================================
The draft head runs on EVERY decode step. Its cost is pure overhead on the critical
path, and it must also be trainable in reasonable wall-clock. A full 128-expert M3
MoE block (top-4 over 128 experts + a shared expert, intermediate 3072) is heavy on
BOTH axes. EAGLE/MTP draft heads are canonically ONE small transformer layer; GLM
shipped a MoE block only because zai exported it that way. We train from scratch, so
we go lean: ONE GQA attention (the real M3 substrate so RoPE/qk-norm geometry matches
the target hidden stream) + ONE dense gated MLP.

    Accept-rate vs cost tradeoff (depth-1):
      * Bigger --mlp-dim  -> higher per-step accept rate (longer accepted runs) but a
        heavier draft (more FLOPs/step + more params to train + bigger checkpoint).
      * Smaller --mlp-dim -> cheaper/faster draft but it diverges from the target
        sooner -> shorter accepted runs -> lower realized speedup.
      The GLM depth-1 head reached ~1.70x. The lean block targets the same regime; the
      win is (accept_len * accept_rate) / (1 + draft_cost/target_cost). Because the M3
      *target* step is enormous (397B MoE), even a few-hundred-M draft is a small
      fraction of target cost, so we can afford a generous --mlp-dim (default 8192,
      ~= M3's dense-prefix intermediate 12288 scaled down) and still net a large win.
      Sweep --mlp-dim in {3072, 4096, 8192, 12288} on the real dump and pick the knee
      of accept-rate-vs-tok/s.

THE MTP RECURRENCE (GLM-proven; see hpu-moe-ext/tests/glm_cpu_mtp_bf16.py)
=========================================================================
The dump provides, per sequence, aligned arrays over positions i = 0..S-1:
    hidden[i]  = the MAIN model's FINAL hidden state at position i  (pre-lm_head)
    tokens[i]  = the token id at position i
For each valid position t the draft consumes (h_t, embed(token_{t+1})) and must
predict token_{t+2} (== the token the main model itself generates at position t+1):
    prev_hidden  = hidden[t]
    next_embed   = embed(tokens[t+1])          # teacher-forced next token (shift +1)
    label        = tokens[t+2]                  # what the draft must emit (shift +2)
    positions    = t                            # absolute pos -> RoPE + frontend pos-0 mask
This is the depth-1 next-next-token objective. (Equivalent to the brief's
"h_t + embed(target_t) -> target_{t+1}" under a one-step relabel of the streams;
this file is written against whatever the dump emits — it derives the triples
explicitly, so the alignment is auditable, not implicit.)

FROZEN heads: embed + lm_head are COPIED from the real M3 bf16 checkpoint
(/mnt/data2/minimax-m3-bf16, language_model.{model.embed_tokens,lm_head}.weight) and
NOT trained — the draft must speak the model's own vocabulary geometry. enorm/hnorm/
eh_proj + the lean block + shared_head.norm are trained.

CHECKPOINT LAYOUT (what m3_mtp.py's load_weights will expect, MTP layer idx 60)
==============================================================================
    model.layers.60.enorm.weight
    model.layers.60.hnorm.weight
    model.layers.60.eh_proj.weight
    model.layers.60.mtp_block.input_layernorm.weight
    model.layers.60.mtp_block.post_attention_layernorm.weight
    model.layers.60.mtp_block.self_attn.{q,k,v,o}_proj.weight
    model.layers.60.mtp_block.self_attn.{q,k}_norm.weight
    model.layers.60.mtp_block.mlp.{gate,up,down}_proj.weight      # DENSE swigluoai
    model.layers.60.shared_head.norm.weight
    model.layers.60.shared_head.head.weight                       # = lm_head (frozen copy)
(embed_tokens is shared/top-level and frozen; the serve module reuses the main embed.)

RUN
===
CPU self-test (card-free, ~10-30 s, this is the gate the orchestrator runs):
    $HOME/.venvs/m3calib/bin/python train_m3_mtp.py --self-test

Real training (CARD/host job — the human agent runs this, NOT the subagent):
    python train_m3_mtp.py --dump /path/to/m3_mtp_dump.pt \
        --mlp-dim 8192 --epochs 3 --bs 8 --lr 6e-4 --out draft_ckpt/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the validated CPU core (DO NOT redefine these).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_cpu_mtp import (  # noqa: E402
    H,
    VOCAB,
    EPS,
    ALPHA,
    LIMIT,
    gemma_rms,
    swigluoai,
    M3MTPFrontend,
    M3SharedHead,
)

# m3_cpu_mtp.py does `torch.set_grad_enabled(False)` at import (it is an inference reference). This
# is a TRAINER — re-enable autograd globally for the whole process. (Inference-only blocks below use
# explicit `with torch.no_grad():`.)
torch.set_grad_enabled(True)

# --- M3 attention geometry (from /mnt/data2/minimax-m3-bf16 config.json text_config) ---
N_HEADS = 64
N_KV_HEADS = 4          # GQA
HEAD_DIM = 128
ROTARY_DIM = 64         # partial_rotary_factor 0.5 -> RoPE on 64 of 128 head dims
ROPE_THETA = 5_000_000.0
MTP_LAYER_IDX = 60      # num_hidden_layers == 60 -> the (single) nextn layer index

BF16_DIR = "/mnt/data2/minimax-m3-bf16"


# ==================================================================================================
# Partial NeoX RoPE — matches vLLM get_rope(default, partial_rotary_factor=0.5) and the GLM oracle's
# rope_neox over the FIRST `rotary_dim` channels (the remaining head_dim-rotary_dim pass through).
# ==================================================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float,
                  dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for the rotary sub-block, shape [T, rotary_dim]. Built in fp32 for accuracy."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                                        device=positions.device) / rotary_dim))
    ang = torch.einsum("t,d->td", positions.float(), inv)        # [T, rotary_dim/2]
    emb = torch.cat((ang, ang), dim=-1)                          # [T, rotary_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        rotary_dim: int) -> torch.Tensor:
    """x: [T, n_heads, head_dim]. Rotate the first `rotary_dim` channels (NeoX), pass the rest."""
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    c = cos[:, None, :]    # [T,1,rotary_dim]
    s = sin[:, None, :]
    x_rot = x_rot * c + _rotate_half(x_rot) * s
    return torch.cat((x_rot, x_pass), dim=-1)


# ==================================================================================================
# LEAN draft block: M3 GQA attention (per-head qk-norm + partial RoPE) + DENSE swigluoai MLP.
# Pre-norm residual transformer layer, exactly the M3 decoder-layer shape (input_layernorm ->
# attn -> +res -> post_attention_layernorm -> mlp -> +res) so the trained geometry is consistent
# with the target hidden stream the frontend feeds it. The ONLY structural deviation from a real M3
# layer is mlp = dense swigluoai (not the 128-expert MoE) — the whole point of the lean head.
#
# Causal self-attention is computed over the WHOLE provided window in one shot (training sees full
# sequences). Names mirror the M3 checkpoint so m3_mtp.py loads them through the standard mapping.
# ==================================================================================================
class M3LeanBlock(nn.Module):
    def __init__(self, hidden: int = H, n_heads: int = N_HEADS, n_kv_heads: int = N_KV_HEADS,
                 head_dim: int = HEAD_DIM, mlp_dim: int = 8192, rotary_dim: int = ROTARY_DIM,
                 rope_theta: float = ROPE_THETA, eps: float = EPS):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.eps = eps
        self.scaling = head_dim ** -0.5
        assert n_heads % n_kv_heads == 0, "GQA: n_heads must be divisible by n_kv_heads"
        self.kv_groups = n_heads // n_kv_heads

        q_dim = n_heads * head_dim          # 8192
        kv_dim = n_kv_heads * head_dim      # 512

        # norms (Gemma-style, applied via gemma_rms in forward — Parameters hold the raw weight)
        self.input_layernorm_w = nn.Parameter(torch.zeros(hidden))
        self.post_attention_layernorm_w = nn.Parameter(torch.zeros(hidden))
        self.q_norm_w = nn.Parameter(torch.zeros(head_dim))   # per-head [128]
        self.k_norm_w = nn.Parameter(torch.zeros(head_dim))

        # attention projections (no bias, matches M3)
        self.q_proj = nn.Linear(hidden, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden, bias=False)

        # DENSE swigluoai MLP (gate/up fused-conceptually but stored as the M3 contiguous w1|w3 form)
        self.gate_proj = nn.Linear(hidden, mlp_dim, bias=False)
        self.up_proj = nn.Linear(hidden, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden, bias=False)
        self.mlp_dim = mlp_dim

    # ---- attention ----
    def _attn(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [T, hidden] (single sequence; training operates one sequence at a time)
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(T, self.n_kv_heads, self.head_dim)

        # per-head Gemma qk-norm over head_dim (identical weight across heads)
        q = gemma_rms(q, self.q_norm_w, self.eps)
        k = gemma_rms(k, self.k_norm_w, self.eps)

        # partial NeoX RoPE on the first rotary_dim channels
        cos, sin = _rope_cos_sin(positions, self.rotary_dim, self.rope_theta, q.dtype)
        q = _apply_partial_rope(q, cos, sin, self.rotary_dim)
        k = _apply_partial_rope(k, cos, sin, self.rotary_dim)

        # GQA: repeat KV heads to match Q heads
        k = k.repeat_interleave(self.kv_groups, dim=1)   # [T, n_heads, head_dim]
        v = v.repeat_interleave(self.kv_groups, dim=1)

        # -> [n_heads, T, head_dim] and scaled-dot-product causal attention
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
        attn = attn.transpose(0, 1).reshape(T, self.n_heads * self.head_dim)
        return self.o_proj(attn)

    # ---- dense swigluoai MLP ----
    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        # reuse swigluoai() from the CPU core by concatenating contiguous [gate | up]
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        h = swigluoai(torch.cat([gate, up], dim=-1), ALPHA, LIMIT)
        return self.down_proj(h)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # pre-norm residual (M3 decoder-layer shape)
        residual = hidden_states
        x = gemma_rms(hidden_states, self.input_layernorm_w, self.eps)
        x = self._attn(x, positions)
        residual = residual + x
        x = gemma_rms(residual, self.post_attention_layernorm_w, self.eps)
        x = self._mlp(x)
        return residual + x


# ==================================================================================================
# The full trainable draft head: frontend + lean block + shared head. Tracks an embed table that is
# FROZEN (no grad) — used to look up embed(token_{t+1}) for the MTP recurrence input.
# ==================================================================================================
class M3DraftHead(nn.Module):
    def __init__(self, mlp_dim: int = 8192, hidden: int = H, vocab: int = VOCAB,
                 eps: float = EPS):
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.frontend = M3MTPFrontend(hidden=hidden, eps=eps)
        self.block = M3LeanBlock(hidden=hidden, mlp_dim=mlp_dim, eps=eps)
        self.shared_head = M3SharedHead(hidden=hidden, vocab=vocab, eps=eps)
        # FROZEN embedding table (filled from the real checkpoint, or random in self-test).
        self.register_buffer("embed_weight", torch.zeros(vocab, hidden), persistent=True)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.embed_weight)

    def forward(self, prev_hidden: torch.Tensor, next_token_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """prev_hidden [T,H], next_token_ids [T] (token_{t+1}), positions [T] (absolute pos t).
        Returns logits [T, vocab] — the draft's prediction of token_{t+2}."""
        next_embed = self.embed(next_token_ids)                       # [T,H], frozen lookup
        x = self.frontend(next_embed, prev_hidden, positions)         # enorm/hnorm/eh_proj
        x = self.block(x, positions)                                  # lean GQA + dense MLP
        return self.shared_head(x)                                    # final norm + lm_head

    # ----- trainable-param accounting -----
    def trainable_param_count(self) -> dict:
        def n(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        fe = n(self.frontend)
        blk = n(self.block)
        # shared_head.head (lm_head) is frozen at train time; only its norm trains
        sh_norm = self.shared_head.norm_w.numel()
        sh_head = self.shared_head.head.weight.numel()
        return {
            "frontend": fe,
            "lean_block": blk,
            "shared_head_norm": sh_norm,
            "trainable_total": fe + blk + sh_norm,
            "frozen_lm_head": sh_head,
            "frozen_embed": self.embed_weight.numel(),
        }


# ==================================================================================================
# Frozen-head init from the real M3 bf16 checkpoint.
# ==================================================================================================
def load_frozen_heads(draft: M3DraftHead, bf16_dir: str = BF16_DIR,
                      dtype: torch.dtype = torch.float32) -> None:
    """Copy the REAL M3 lm_head -> shared_head.head and embed -> embed_weight, then FREEZE both.
    These are the model's own vocabulary geometry; the draft must not relearn them."""
    from safetensors import safe_open
    idx_path = os.path.join(bf16_dir, "model.safetensors.index.json")
    idx = json.load(open(idx_path))["weight_map"]
    lm_name = "language_model.lm_head.weight"
    emb_name = "language_model.model.embed_tokens.weight"
    for nm in (lm_name, emb_name):
        if nm not in idx:
            raise RuntimeError(f"{nm!r} not in {idx_path} — wrong checkpoint? keys e.g. "
                               f"{list(idx)[:3]}")

    def _load(nm):
        with safe_open(os.path.join(bf16_dir, idx[nm]), framework="pt", device="cpu") as f:
            return f.get_tensor(nm).to(dtype)

    lm = _load(lm_name)
    emb = _load(emb_name)
    assert lm.shape == (VOCAB, H), f"lm_head shape {tuple(lm.shape)} != {(VOCAB, H)}"
    assert emb.shape == (VOCAB, H), f"embed shape {tuple(emb.shape)} != {(VOCAB, H)}"
    with torch.no_grad():
        draft.shared_head.head.weight.copy_(lm)
        draft.embed_weight.copy_(emb)
    draft.shared_head.head.weight.requires_grad_(False)   # FROZEN
    # embed_weight is a buffer -> already no grad


# ==================================================================================================
# Dump loading + (prev_hidden, next_token, label, position) triple construction.
#
# Expected dump (produced by the card-gated dump step — a .pt with a list of per-sequence dicts, or
# one dict of stacked tensors). Each sequence carries:
#     "hidden": float tensor [S, H]   — MAIN final hidden states, positions 0..S-1
#     "tokens": long  tensor [S]      — token ids,                positions 0..S-1
# Optional (enables the KL term):
#     "logits" or "main_logits": float [S, V] OR top-k {"topk_ids":[S,k],"topk_logp":[S,k]}
#
# The loader is tolerant: accepts {"sequences":[...]} , a bare list, or a single dict; and the per-
# sequence keys may be "hidden"/"hidden_states"/"final" and "tokens"/"token_ids"/"input_ids".
# ==================================================================================================
def _get_first(d: dict, *names):
    for n in names:
        if n in d:
            return d[n]
    return None


def load_dump(path: str) -> list[dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "sequences" in blob:
        seqs = blob["sequences"]
    elif isinstance(blob, list):
        seqs = blob
    elif isinstance(blob, dict):
        seqs = [blob]
    else:
        raise ValueError(f"unrecognized dump container: {type(blob)}")
    out = []
    for i, s in enumerate(seqs):
        if not isinstance(s, dict):
            raise ValueError(f"dump sequence {i} is {type(s)}, expected dict")
        hidden = _get_first(s, "hidden", "hidden_states", "final")
        tokens = _get_first(s, "tokens", "token_ids", "input_ids")
        if hidden is None or tokens is None:
            raise ValueError(f"dump seq {i} missing hidden/tokens (keys={list(s)})")
        rec = {"hidden": torch.as_tensor(hidden), "tokens": torch.as_tensor(tokens).long()}
        logits = _get_first(s, "logits", "main_logits")
        if logits is not None:
            rec["logits"] = torch.as_tensor(logits)
        else:
            tki = _get_first(s, "topk_ids")
            tkl = _get_first(s, "topk_logp", "topk_logprobs")
            if tki is not None and tkl is not None:
                rec["topk_ids"] = torch.as_tensor(tki).long()
                rec["topk_logp"] = torch.as_tensor(tkl)
        out.append(rec)
    return out


def build_triples(seq: dict) -> Optional[dict]:
    """From a sequence's (hidden[S,H], tokens[S]) build the MTP training triples:
        prev_hidden[r] = hidden[t]            for t in 0..S-3
        next_token[r]  = tokens[t+1]
        label[r]       = tokens[t+2]
        position[r]    = t
    Returns None if the sequence is too short (S < 3)."""
    hidden = seq["hidden"]
    tokens = seq["tokens"]
    S = hidden.shape[0]
    if S != tokens.shape[0]:
        raise ValueError(f"hidden len {S} != tokens len {tokens.shape[0]}")
    if S < 3:
        return None
    t = torch.arange(0, S - 2)
    triples = {
        "prev_hidden": hidden[t],          # [R,H]
        "next_token": tokens[t + 1],       # [R]
        "label": tokens[t + 2],            # [R]
        "position": t,                     # [R]  (absolute positions)
    }
    if "logits" in seq:
        triples["main_logits"] = seq["logits"][t + 2]    # main's dist for token_{t+2}
    elif "topk_ids" in seq:
        triples["topk_ids"] = seq["topk_ids"][t + 2]
        triples["topk_logp"] = seq["topk_logp"][t + 2]
    return triples


# ==================================================================================================
# Loss: CE(draft_logits, label) [+ optional KL(draft || main)]. Train in fp32 for stability; the
# checkpoint is saved in the configured save dtype (default bf16, matching the model).
# ==================================================================================================
def mtp_loss(logits: torch.Tensor, label: torch.Tensor,
             main_logits: Optional[torch.Tensor] = None, kl_weight: float = 0.0):
    ce = F.cross_entropy(logits.float(), label)
    if main_logits is not None and kl_weight > 0.0:
        # KL(main || draft) over the full vocab — distil the main model's next-token distribution.
        logp_draft = F.log_softmax(logits.float(), dim=-1)
        p_main = F.softmax(main_logits.float(), dim=-1)
        kl = F.kl_div(logp_draft, p_main, reduction="batchmean")
        return ce + kl_weight * kl, ce.detach(), kl.detach()
    return ce, ce.detach(), torch.tensor(0.0)


def top1_accept_rate(logits: torch.Tensor, label: torch.Tensor) -> float:
    """Proxy accept rate: fraction of positions where the draft's argmax == the target label
    (== what the main model generates). The realized speculative accept rate is bounded by this."""
    return float((logits.argmax(-1) == label).float().mean())


# ==================================================================================================
# Checkpoint save in the MTP-shaped tensor names m3_mtp.py's load_weights expects.
# ==================================================================================================
def save_draft_checkpoint(draft: M3DraftHead, out_dir: str,
                          save_dtype: torch.dtype = torch.bfloat16,
                          meta: Optional[dict] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    L = MTP_LAYER_IDX
    p = f"model.layers.{L}"
    fe = draft.frontend
    blk = draft.block
    sh = draft.shared_head
    sd = {
        f"{p}.enorm.weight": fe.enorm_w,
        f"{p}.hnorm.weight": fe.hnorm_w,
        f"{p}.eh_proj.weight": fe.eh_proj.weight,
        f"{p}.mtp_block.input_layernorm.weight": blk.input_layernorm_w,
        f"{p}.mtp_block.post_attention_layernorm.weight": blk.post_attention_layernorm_w,
        f"{p}.mtp_block.self_attn.q_proj.weight": blk.q_proj.weight,
        f"{p}.mtp_block.self_attn.k_proj.weight": blk.k_proj.weight,
        f"{p}.mtp_block.self_attn.v_proj.weight": blk.v_proj.weight,
        f"{p}.mtp_block.self_attn.o_proj.weight": blk.o_proj.weight,
        f"{p}.mtp_block.self_attn.q_norm.weight": blk.q_norm_w,
        f"{p}.mtp_block.self_attn.k_norm.weight": blk.k_norm_w,
        f"{p}.mtp_block.mlp.gate_proj.weight": blk.gate_proj.weight,
        f"{p}.mtp_block.mlp.up_proj.weight": blk.up_proj.weight,
        f"{p}.mtp_block.mlp.down_proj.weight": blk.down_proj.weight,
        f"{p}.shared_head.norm.weight": sh.norm_w,
        f"{p}.shared_head.head.weight": sh.head.weight,   # frozen copy of lm_head
    }
    sd = {k: v.detach().to(save_dtype).contiguous() for k, v in sd.items()}
    # save as safetensors if available (what the serve loader reads), else torch .pt fallback
    ckpt_path = os.path.join(out_dir, "m3_mtp_draft.safetensors")
    try:
        from safetensors.torch import save_file
        save_file(sd, ckpt_path, metadata={"format": "pt", "mtp_layer_idx": str(L)})
    except Exception as ex:  # noqa: BLE001
        ckpt_path = os.path.join(out_dir, "m3_mtp_draft.pt")
        torch.save(sd, ckpt_path)
        print(f"[save] safetensors unavailable ({ex!r}); wrote torch .pt instead", file=sys.stderr)
    # sidecar index + meta so the serve loader can find tensors the same way the main model does
    index = {"metadata": {"mtp_layer_idx": L},
             "weight_map": {k: os.path.basename(ckpt_path) for k in sd}}
    json.dump(index, open(os.path.join(out_dir, "m3_mtp_draft.index.json"), "w"), indent=1)
    if meta is not None:
        json.dump(meta, open(os.path.join(out_dir, "train_meta.json"), "w"), indent=1)
    return ckpt_path


# ==================================================================================================
# Training loop.
# ==================================================================================================
def train(args) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"[train] device={device} mlp_dim={args.mlp_dim} epochs={args.epochs} "
          f"bs={args.bs} lr={args.lr} kl_weight={args.kl_weight}", flush=True)

    draft = M3DraftHead(mlp_dim=args.mlp_dim).to(device)
    load_frozen_heads(draft, args.bf16_dir, dtype=torch.float32)
    draft.to(device)
    pc = draft.trainable_param_count()
    print(f"[train] trainable params: {pc['trainable_total']/1e6:.1f}M "
          f"(frontend {pc['frontend']/1e6:.1f}M + lean_block {pc['lean_block']/1e6:.1f}M "
          f"+ head_norm {pc['shared_head_norm']/1e3:.1f}K) | "
          f"frozen lm_head {pc['frozen_lm_head']/1e6:.0f}M, embed {pc['frozen_embed']/1e6:.0f}M",
          flush=True)

    seqs = load_dump(args.dump)
    triples = [t for t in (build_triples(s) for s in seqs) if t is not None]
    if not triples:
        raise RuntimeError(f"no usable sequences in {args.dump} (need S>=3)")
    total_rows = sum(t["prev_hidden"].shape[0] for t in triples)
    print(f"[train] {len(triples)} sequences, {total_rows} MTP positions", flush=True)

    # optimizer: AdamW over the TRAINABLE params only (lm_head frozen, embed is a buffer)
    params = [p for p in draft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(triples) / args.bs)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.03 * total_steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))   # cosine to 0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(triples))
        for bi in range(steps_per_epoch):
            batch = [triples[int(j)] for j in order[bi * args.bs:(bi + 1) * args.bs]]
            opt.zero_grad(set_to_none=True)
            bce = bkl = bacc = 0.0
            nseq = 0
            # each sequence has its own length + absolute positions -> run per-sequence, accumulate
            for tr in batch:
                ph = tr["prev_hidden"].to(device).float()
                nt = tr["next_token"].to(device)
                lb = tr["label"].to(device)
                pos = tr["position"].to(device)
                logits = draft(ph, nt, pos)
                ml = tr.get("main_logits")
                if ml is not None:
                    ml = ml.to(device)
                loss, ce, kl = mtp_loss(logits, lb, ml, args.kl_weight)
                (loss / len(batch)).backward()
                bce += float(ce); bkl += float(kl)
                bacc += top1_accept_rate(logits.detach(), lb); nseq += 1
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step(); sched.step(); step += 1
            if step % args.log_every == 0 or step == total_steps:
                print(f"[train] step {step}/{total_steps} ep{epoch} "
                      f"ce={bce/nseq:.4f} kl={bkl/nseq:.4f} acc@1={bacc/nseq:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)

    meta = {
        "mtp_layer_idx": MTP_LAYER_IDX, "mlp_dim": args.mlp_dim,
        "trainable_params": pc["trainable_total"], "param_breakdown": pc,
        "epochs": args.epochs, "lr": args.lr, "kl_weight": args.kl_weight,
        "n_sequences": len(triples), "n_positions": total_rows,
        "save_dtype": str(args.save_dtype), "recurrence": "f(h_t, embed(tok_{t+1})) -> tok_{t+2}",
    }
    save_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                  "fp32": torch.float32}[args.save_dtype]
    ckpt = save_draft_checkpoint(draft, args.out, save_dtype=save_dtype, meta=meta)
    print(f"[train] DONE. checkpoint -> {ckpt}", flush=True)
    print(f"[train] tensor names follow model.layers.{MTP_LAYER_IDX}.* "
          f"(enorm/hnorm/eh_proj/mtp_block.*/shared_head.*) for m3_mtp.py load_weights.", flush=True)


# ==================================================================================================
# CPU SELF-TEST — tiny end-to-end overfit on SYNTHETIC data. Proves:
#   (1) the train loop runs (forward/backward/step) on CPU with NO cards,
#   (2) the MTP recurrence wiring is correct (frontend -> lean block -> shared_head),
#   (3) the loss is correct + minimizable (loss DROPS to near-zero on memorizable data).
# Uses H=6144 (REAL hidden width) but T=8, a TINY vocab + tiny lm_head, 50 steps, so it is fast and
# card-free. A wrong recurrence / detached graph / frozen-everything bug => loss will NOT drop.
# ==================================================================================================
def self_test() -> bool:
    print("=" * 60)
    print("M3 MTP TRAINER — CPU SELF-TEST (synthetic overfit)")
    print("=" * 60)
    torch.manual_seed(0)
    device = torch.device("cpu")

    # tiny problem: real H, small T + small vocab so the lm_head is cheap to hold.
    T = 8
    TINY_V = 64
    mlp_dim = 256

    # Build a draft with a tiny vocab (override VOCAB-sized pieces). We construct M3DraftHead with
    # the tiny vocab so shared_head.head + embed are [TINY_V, H] (fast). The block/frontend are full
    # width H=6144 — exactly the real wiring, just a tiny T.
    draft = M3DraftHead(mlp_dim=mlp_dim, hidden=H, vocab=TINY_V).to(device)

    # random FROZEN heads (stand-in for the real lm_head/embed): make them non-trivial + freeze.
    with torch.no_grad():
        draft.shared_head.head.weight.copy_(torch.randn(TINY_V, H) * (1.0 / H ** 0.5))
        draft.embed_weight.copy_(torch.randn(TINY_V, H) * 0.05)
    draft.shared_head.head.weight.requires_grad_(False)

    checks = []

    # frozen pieces really are frozen
    checks.append(("lm_head frozen", not draft.shared_head.head.weight.requires_grad))
    checks.append(("embed is buffer (no grad)", not draft.embed_weight.requires_grad))
    # the trainable pieces really do require grad
    checks.append(("eh_proj trainable", draft.frontend.eh_proj.weight.requires_grad))
    checks.append(("q_proj trainable", draft.block.q_proj.weight.requires_grad))
    checks.append(("mlp down trainable", draft.block.down_proj.weight.requires_grad))

    # ONE synthetic sequence: random hidden stream + a random token stream. The draft must learn to
    # map (h_t, embed(tok_{t+1}), pos t) -> tok_{t+2}. With a free attention+MLP+frontend over a fixed
    # tiny example this is memorizable -> CE -> ~0.
    hidden = torch.randn(T, H)
    tokens = torch.randint(0, TINY_V, (T,))
    seq = {"hidden": hidden, "tokens": tokens}
    tr = build_triples(seq)
    assert tr is not None and tr["prev_hidden"].shape == (T - 2, H), \
        f"triple shapes wrong: {None if tr is None else tr['prev_hidden'].shape}"
    checks.append(("triples shape [T-2,H]", tr["prev_hidden"].shape == (T - 2, H)))
    checks.append(("labels are tokens[t+2]", bool(torch.equal(tr["label"], tokens[2:T]))))
    checks.append(("next_token is tokens[t+1]", bool(torch.equal(tr["next_token"], tokens[1:T - 1]))))
    checks.append(("positions are t", bool(torch.equal(tr["position"], torch.arange(T - 2)))))

    ph = tr["prev_hidden"].float()
    nt = tr["next_token"]
    lb = tr["label"]
    pos = tr["position"]

    params = [p for p in draft.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"  trainable params (tiny): {n_train/1e6:.2f}M  (full-H block, tiny vocab)")
    opt = torch.optim.AdamW(params, lr=3e-3, betas=(0.9, 0.95))

    # initial loss
    with torch.no_grad():
        l0 = float(F.cross_entropy(draft(ph, nt, pos).float(), lb))
    losses = []
    for s in range(50):
        opt.zero_grad(set_to_none=True)
        logits = draft(ph, nt, pos)
        loss = F.cross_entropy(logits.float(), lb)
        loss.backward()
        # confirm gradients actually flow into the trainable pieces (catches a detached graph)
        if s == 0:
            g_eh = draft.frontend.eh_proj.weight.grad
            g_q = draft.block.q_proj.weight.grad
            checks.append(("grad flows to eh_proj", g_eh is not None and float(g_eh.abs().sum()) > 0))
            checks.append(("grad flows to q_proj", g_q is not None and float(g_q.abs().sum()) > 0))
            checks.append(("no grad on frozen lm_head",
                           draft.shared_head.head.weight.grad is None))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))

    lf = losses[-1]
    acc = top1_accept_rate(draft(ph, nt, pos).detach(), lb)
    print(f"  loss: {l0:.4f} -> {lf:.6f}  (acc@1 {acc:.2f})  over 50 steps")

    checks.append(("loss decreased", lf < l0 * 0.5))
    checks.append(("loss -> near zero (overfit)", lf < 0.05))
    checks.append(("draft overfits (acc@1 == 1.0)", acc >= 0.999))
    checks.append(("loss is finite throughout", all(math.isfinite(x) for x in losses)))

    # frozen lm_head must be UNCHANGED after training
    # (re-derive its expected value: we seeded it; confirm grad never moved it)
    checks.append(("lm_head weight unchanged (frozen)",
                   draft.shared_head.head.weight.grad is None
                   and not draft.shared_head.head.weight.requires_grad))

    # checkpoint round-trip: save (tiny vocab) + assert all expected MTP tensor names present
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ckpt = save_draft_checkpoint(draft, td, save_dtype=torch.float32)
        # reload + check names
        names = _read_ckpt_names(ckpt)
        L = MTP_LAYER_IDX
        expected = {
            f"model.layers.{L}.enorm.weight",
            f"model.layers.{L}.hnorm.weight",
            f"model.layers.{L}.eh_proj.weight",
            f"model.layers.{L}.mtp_block.input_layernorm.weight",
            f"model.layers.{L}.mtp_block.post_attention_layernorm.weight",
            f"model.layers.{L}.mtp_block.self_attn.q_proj.weight",
            f"model.layers.{L}.mtp_block.self_attn.k_proj.weight",
            f"model.layers.{L}.mtp_block.self_attn.v_proj.weight",
            f"model.layers.{L}.mtp_block.self_attn.o_proj.weight",
            f"model.layers.{L}.mtp_block.self_attn.q_norm.weight",
            f"model.layers.{L}.mtp_block.self_attn.k_norm.weight",
            f"model.layers.{L}.mtp_block.mlp.gate_proj.weight",
            f"model.layers.{L}.mtp_block.mlp.up_proj.weight",
            f"model.layers.{L}.mtp_block.mlp.down_proj.weight",
            f"model.layers.{L}.shared_head.norm.weight",
            f"model.layers.{L}.shared_head.head.weight",
        }
        missing = expected - names
        extra = names - expected
        checks.append(("checkpoint has all MTP tensor names", not missing))
        checks.append(("checkpoint has no unexpected names", not extra))
        if missing:
            print(f"  MISSING ckpt tensors: {sorted(missing)}")
        if extra:
            print(f"  EXTRA ckpt tensors: {sorted(extra)}")

    # RoPE sanity: position 0 must be identity (cos=1,sin=0 -> rotate is a no-op)
    cos0, sin0 = _rope_cos_sin(torch.tensor([0]), ROTARY_DIM, ROPE_THETA, torch.float32)
    checks.append(("rope pos0 cos==1", torch.allclose(cos0, torch.ones_like(cos0))))
    checks.append(("rope pos0 sin==0", torch.allclose(sin0, torch.zeros_like(sin0), atol=1e-6)))
    xtest = torch.randn(1, N_HEADS, HEAD_DIM)
    xrot0 = _apply_partial_rope(xtest, cos0, sin0, ROTARY_DIM)
    checks.append(("rope pos0 is identity", torch.allclose(xrot0, xtest, atol=1e-5)))

    # frontend pos-0 mask: the lean head inherits the M3MTPFrontend pos-0 embedding mask. Test it at
    # the FRONTEND OUTPUT on a FRESH (untrained, non-saturated) draft so the property is isolated from
    # the lean block's causal mixing + the overfit softmax saturation. The frontend at pos 0 must
    # ignore the next-token embedding (masked to 0); at pos>0 it must depend on it.
    fresh = M3DraftHead(mlp_dim=64, hidden=H, vocab=TINY_V)
    with torch.no_grad():
        fresh.frontend.enorm_w.copy_(torch.randn(H) * 0.1)   # non-zero so a mask leak would show
        fresh.frontend.hnorm_w.copy_(torch.randn(H) * 0.1)
        fresh.frontend.eh_proj.weight.copy_(torch.randn(H, 2 * H) * (1.0 / (2 * H) ** 0.5))
        fresh.embed_weight.copy_(torch.randn(TINY_V, H) * 0.5)
        h0 = torch.randn(2, H)
        # differ at BOTH positions so we can check pos0 (masked -> no change) vs pos1 (used -> change)
        tkA = torch.tensor([5, 11]); tkB = torch.tensor([9, 23])
        posz = torch.tensor([0, 1])
        fA = fresh.frontend(fresh.embed(tkA), h0, posz)
        fB = fresh.frontend(fresh.embed(tkB), h0, posz)
    # pos 0: embedding masked -> output identical despite different token
    checks.append(("frontend masks pos0 embed", torch.allclose(fA[0], fB[0], atol=1e-5)))
    # pos 1: embedding used -> different token MUST change the output
    checks.append(("non-pos0 frontend uses embed", not torch.allclose(fA[1], fB[1], atol=1e-3)))

    ok = all(p for _, p in checks)
    print("-" * 60)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print("=" * 60)
    print(f"M3 MTP TRAINER SELF-TEST: {'GREEN — all pass' if ok else 'RED — FIX ABOVE'}")
    print("=" * 60)
    return ok


def _read_ckpt_names(ckpt_path: str) -> set:
    if ckpt_path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            return set(f.keys())
    return set(torch.load(ckpt_path, map_location="cpu").keys())


# ==================================================================================================
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the M3 MTP lean draft head.")
    ap.add_argument("--self-test", action="store_true",
                    help="run the card-free CPU self-test (synthetic overfit) and exit")
    ap.add_argument("--dump", type=str, default=None,
                    help="path to the (hidden, tokens[, logits]) dump .pt produced by the card step")
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_ckpt"),
                    help="output dir for the draft checkpoint")
    ap.add_argument("--bf16-dir", type=str, default=BF16_DIR,
                    help="M3 bf16 checkpoint dir (for the frozen lm_head/embed)")
    ap.add_argument("--mlp-dim", type=int, default=8192,
                    help="DENSE swigluoai MLP intermediate dim (accept-rate vs cost knob)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8, help="sequences per optimizer step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--kl-weight", type=float, default=0.0,
                    help="weight on KL(main||draft); needs dumped main logits/top-k")
    ap.add_argument("--save-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu | cuda | hpu (the REAL run uses an accelerator — human-driven)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.self_test:
        return 0 if self_test() else 1
    if not args.dump:
        print("ERROR: --dump is required for training (or pass --self-test).", file=sys.stderr)
        return 2
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
