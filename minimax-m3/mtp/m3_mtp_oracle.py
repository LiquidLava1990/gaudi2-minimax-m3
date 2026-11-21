"""M3 LEAN MTP draft-head accept-rate ORACLE — the >=60% GO/NO-GO gate.

This is the gate that fires AFTER the human runs the card-gated dump->train cycle
and BEFORE any draft serve: it loads the trained LEAN draft checkpoint + a dumped
(hidden, target) shard, runs the draft forward in pure-torch fp64 on CPU, and
measures draft-vs-target greedy agreement. >=60% => MTP is worth serving.

It is the M3 analogue of hpu-moe-ext/tests/glm_cpu_mtp_oracle.py / glm_cpu_mtp_bf16.py,
adapted to the M3 LEAN draft (NOT a 128-expert MoE block):

  frontend (M3MTPFrontend: mask pos0 -> enorm/hnorm Gemma-RMSNorm -> eh_proj[2H->H])
    -> input_layernorm (Gemma)
    -> GQA attention: per-head Gemma qk_norm (head_dim) BEFORE rope, NEOX *partial*
       rope (rotary_dim=64 of head_dim=128, theta=5e6), 64 q heads / 4 kv heads
       (repeat_kv groups=16), causal softmax, o_proj
    -> residual add
    -> post_attention_layernorm (Gemma)
    -> DENSE swigluoai MLP (gate_up_proj H->2*inter, swigluoai, down_proj inter->H)
    -> residual add
    -> shared_head (M3SharedHead: final Gemma-RMSNorm -> lm_head)

DRAFT-SIZE TRADEOFF (why LEAN, not the 128-expert M3 MoE block):
  A draft head runs on EVERY decode step, so its cost is paid every token; and it
  must be trainable in reasonable wall-clock from scratch (we are NOT inheriting a
  shipped MTP module — M3's num_mtp_modules=7 refers to weights we do not have here).
  A full 128-expert top-4 MoE block (the GLM/DeepSeek MTP shape) is heavy on both
  axes: ~Nx the params to train and a routed gather every step at inference. EAGLE /
  canonical MTP heads are ONE small transformer layer; GLM only shipped a MoE MTP
  because zai exported it. We train from scratch, so we use a LEAN block: the real
  M3 attention substrate (so positional/causal structure matches the target exactly)
  + a single DENSE swigluoai MLP. Accept-rate vs cost: a lean head accepts somewhat
  less than a full-width MoE head per step, but its per-step latency is a small
  fraction, so at depth-1 the *throughput* win (accept_rate * cheap_draft) is higher
  and the wall-clock-to-trained is hours not days. `inter` is the one knob: bigger =
  higher accept ceiling, slower draft. Default mirrors M3's dense MLP width 12288;
  drop to ~3072-8192 if draft latency dominates (documented in __main__ banner).

The OUTER wiring (enorm/hnorm/eh_proj/shared_head + the tensor NAMES below) is kept
identical to what the serving MTP module expects, so the same trained checkpoint
loads via the MTP path with no renaming.

CHECKPOINT CONTRACT (what the trainer must write to draft_ckpt/, what this loads):
  draft_ckpt/draft.pt  (torch.save of a flat state_dict; fp32/bf16 both fine), keys:
    enorm.weight                          [H]
    hnorm.weight                          [H]
    eh_proj.weight                        [H, 2H]      (nn.Linear: y = x @ W^T)
    input_layernorm.weight                [H]
    self_attn.q_proj.weight               [nH*Dh, H]   (nH=64, Dh=128)
    self_attn.k_proj.weight               [nKV*Dh, H]  (nKV=4)
    self_attn.v_proj.weight               [nKV*Dh, H]
    self_attn.o_proj.weight               [H, nH*Dh]
    self_attn.q_norm.weight               [Dh]
    self_attn.k_norm.weight               [Dh]
    post_attention_layernorm.weight       [H]
    mlp.gate_up_proj.weight               [2*inter, H]
    mlp.down_proj.weight                  [inter, H]   -> applied as @ W^T (in=inter)
    shared_head.norm.weight               [H]
  The LM head (shared_head.head.weight [VOCAB,H]) and the token embedding
  ([VOCAB,H]) are TIED to the BASE model and loaded from the bf16 weights
  (--base, default /mnt/data2/minimax-m3-bf16) — the trainer need NOT (and should
  NOT) duplicate them. tie_word_embeddings=false so head != embed; both are read
  from the base safetensors here.
  (All Gemma norm weights are stored Gemma-style: the effective scale is (1 + w),
   so freshly-zero-init weights mean "identity * 1.0".)

DUMP SHARD CONTRACT (the (hidden, target) shard, mirrors the GLM oracle cache):
  a single torch.save dict with:
    'final'       [S, H]      pre-final-norm hidden states (== vllm 'hidden_states'
                              fed to the MTP block), one row per input token
    'ids'         [S] int     the input token ids that produced 'final'
  optional (preferred if present — exact teacher greedy incl. final-norm+lm_head):
    'logits_main' [S, VOCAB]  the BASE model's logits per position
  If 'logits_main' is absent the target greedy at t+1 falls back to ids[t+1]
  (teacher-forced) — correct for the served continuation, the GLM oracle pattern.

CPU SELF-TEST (no trained ckpt, no card, ~1s):
  $HOME/.venvs/m3calib/bin/python $HOME/AI/minimax-m3/mtp/m3_mtp_oracle.py
  Builds the oracle from RANDOM draft weights over SYNTHETIC (hidden, target) where
  the target is a KNOWN deterministic function of hidden, and asserts the agreement
  loop / argmax / off-by-one are correct: a draft whose logits ARE the synthetic
  rule scores 100% on matched data and ~0% on shuffled. Proves the gate's
  measurement logic before real weights exist.

REAL RUN (CPU only, AFTER the human's card-gated dump+train — no card needed here):
  $HOME/.venvs/m3calib/bin/python $HOME/AI/minimax-m3/mtp/m3_mtp_oracle.py \
      --ckpt $HOME/AI/minimax-m3/mtp/draft_ckpt \
      --shard $HOME/AI/minimax-m3/mtp/dump/shard0.pt \
      --base  /mnt/data2/minimax-m3-bf16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# --- foundation (validated; do NOT redefine) ---
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

torch.set_grad_enabled(False)
DT = torch.float64

# --- M3 text_config attention constants (config.json text_config) ---
N_HEADS = 64          # num_attention_heads
N_KV = 4              # num_key_value_heads (GQA)
HEAD_DIM = 128        # head_dim
ROTARY_DIM = 64       # rotary_dim (partial_rotary_factor 0.5 -> RoPE on 64 of 128)
ROPE_THETA = 5_000_000.0
KV_GROUPS = N_HEADS // N_KV            # 16
SCALING = HEAD_DIM ** -0.5
DENSE_INTER = 12288   # default LEAN MLP width (M3 dense_intermediate_size); knob


# ============================================================================
# LEAN draft block — pure-torch fp64, numerically mirrors the HF M3 reference
# (transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py): per-head Gemma
# qk_norm applied on the [.., n_heads, head_dim] view BEFORE rope, NEOX partial
# rope (rotate_half splits at midpoint; cos/sin = cat([freqs, freqs])).
# ============================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(positions: torch.Tensor, rotary_dim: int = ROTARY_DIM,
                  theta: float = ROPE_THETA):
    """NEOX cos/sin over the PARTIAL dim. inv_freq denominator uses the partial
    `rotary_dim` (HF: dim = head_dim*partial_rotary_factor), matching the M3 ref."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=DT) / rotary_dim))
    freqs = torch.einsum("s,d->sd", positions.to(DT), inv)   # [S, rotary_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                  # [S, rotary_dim]
    return emb.cos(), emb.sin()


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        rotary_dim: int = ROTARY_DIM) -> torch.Tensor:
    """x: [n_heads, S, head_dim]; cos/sin: [S, rotary_dim]. Rotate first rotary_dim
    dims (NEOX), pass the rest through — exactly HF apply_rotary_pos_emb."""
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    c = cos[None]                                            # [1, S, rotary_dim]
    s = sin[None]
    x_rot = (x_rot * c) + (_rotate_half(x_rot) * s)
    return torch.cat([x_rot, x_pass], dim=-1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[n_kv, S, Dh] -> [n_kv*n_rep, S, Dh] (repeat_interleave on the head axis)."""
    n_kv, S, Dh = x.shape
    if n_rep == 1:
        return x
    return x[:, None, :, :].expand(n_kv, n_rep, S, Dh).reshape(n_kv * n_rep, S, Dh)


class M3LeanDraftBlock(torch.nn.Module):
    """One M3-substrate transformer layer with a DENSE swigluoai MLP (the draft).

    Tensor NAMES match the serving MTP block (self_attn.{q,k,v,o}_proj,
    self_attn.{q,k}_norm, input_layernorm, post_attention_layernorm,
    mlp.gate_up_proj, mlp.down_proj) so a trained checkpoint loads unchanged."""

    def __init__(self, hidden: int = H, inter: int = DENSE_INTER, eps: float = EPS):
        super().__init__()
        self.hidden = hidden
        self.inter = inter
        self.eps = eps
        # norms (Gemma; stored as (1+w) scale -> zeros == identity)
        self.input_layernorm_w = torch.nn.Parameter(torch.zeros(hidden))
        self.post_attention_layernorm_w = torch.nn.Parameter(torch.zeros(hidden))
        # attention
        self.q_proj = torch.nn.Linear(hidden, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = torch.nn.Linear(hidden, N_KV * HEAD_DIM, bias=False)
        self.v_proj = torch.nn.Linear(hidden, N_KV * HEAD_DIM, bias=False)
        self.o_proj = torch.nn.Linear(N_HEADS * HEAD_DIM, hidden, bias=False)
        self.q_norm_w = torch.nn.Parameter(torch.zeros(HEAD_DIM))
        self.k_norm_w = torch.nn.Parameter(torch.zeros(HEAD_DIM))
        # dense swigluoai MLP
        self.gate_up_proj = torch.nn.Linear(hidden, 2 * inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)

    def _attn(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: [S, H] (already input_layernorm'd). Returns o_proj output [S, H]."""
        S = x.shape[0]
        q = self.q_proj(x).view(S, N_HEADS, HEAD_DIM)        # [S, nH, Dh]
        k = self.k_proj(x).view(S, N_KV, HEAD_DIM)
        v = self.v_proj(x).view(S, N_KV, HEAD_DIM)
        # per-head Gemma qk_norm over head_dim, BEFORE rope (HF order)
        q = gemma_rms(q, self.q_norm_w, self.eps)
        k = gemma_rms(k, self.k_norm_w, self.eps)
        q = q.transpose(0, 1)                                # [nH, S, Dh]
        k = k.transpose(0, 1)                                # [nKV, S, Dh]
        v = v.transpose(0, 1)
        cos, sin = _rope_cos_sin(positions)
        q = _apply_partial_rope(q, cos, sin)
        k = _apply_partial_rope(k, cos, sin)
        k = _repeat_kv(k, KV_GROUPS)                         # [nH, S, Dh]
        v = _repeat_kv(v, KV_GROUPS)
        mask = torch.full((S, S), float("-inf"), dtype=DT).triu(1)[None]
        aw = torch.softmax((q @ k.transpose(1, 2)) * SCALING + mask, dim=-1)
        out = (aw @ v).transpose(0, 1).reshape(S, N_HEADS * HEAD_DIM)  # [S, nH*Dh]
        return self.o_proj(out)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: [S, H] frontend output (eh_proj). Returns block output [S, H]."""
        n1 = gemma_rms(x, self.input_layernorm_w, self.eps)
        r2 = x + self._attn(n1, positions)
        n2 = gemma_rms(r2, self.post_attention_layernorm_w, self.eps)
        m = self.down_proj(swigluoai(self.gate_up_proj(n2), ALPHA, LIMIT))
        return r2 + m


class M3LeanDraft(torch.nn.Module):
    """Full draft: frontend (M3MTPFrontend) -> lean block -> shared_head.

    Forward consumes prev_hidden[t] + embed(token[t+1]) and emits draft logits for
    the token at t+1's *successor* (the depth-1 next-next token), per MTP semantics."""

    def __init__(self, hidden: int = H, vocab: int = VOCAB, inter: int = DENSE_INTER,
                 eps: float = EPS):
        super().__init__()
        self.frontend = M3MTPFrontend(hidden=hidden, eps=eps)
        self.block = M3LeanDraftBlock(hidden=hidden, inter=inter, eps=eps)
        self.shared_head = M3SharedHead(hidden=hidden, vocab=vocab, eps=eps)
        # embedding for embed(token[t+1]); tied to base, filled by load_base_tied
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)

    def to_fp64(self) -> "M3LeanDraft":
        return self.double()

    def forward(self, prev_hidden: torch.Tensor, ids_shift: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """prev_hidden: [S-1, H] == final[:-1]; ids_shift: [S-1] == ids[1:];
        positions: [S-1] original positions of the shifted tokens (1..S-1)."""
        embed_tnext = self.embed_tokens(ids_shift)           # [S-1, H]
        x = self.frontend(embed_tnext, prev_hidden, positions)
        h = self.block(x, positions)
        return self.shared_head(h)                           # [S-1, VOCAB]


# ============================================================================
# checkpoint + base-weight loading
# ============================================================================
# tolerant alias map: trainer keys (LHS, also the serving names) -> module attrs.
_CKPT_ALIASES = {
    "enorm.weight": "frontend.enorm_w",
    "hnorm.weight": "frontend.hnorm_w",
    "eh_proj.weight": "frontend.eh_proj.weight",
    "input_layernorm.weight": "block.input_layernorm_w",
    "post_attention_layernorm.weight": "block.post_attention_layernorm_w",
    "self_attn.q_proj.weight": "block.q_proj.weight",
    "self_attn.k_proj.weight": "block.k_proj.weight",
    "self_attn.v_proj.weight": "block.v_proj.weight",
    "self_attn.o_proj.weight": "block.o_proj.weight",
    "self_attn.q_norm.weight": "block.q_norm_w",
    "self_attn.k_norm.weight": "block.k_norm_w",
    "mlp.gate_up_proj.weight": "block.gate_up_proj.weight",
    "mlp.down_proj.weight": "block.down_proj.weight",
    "shared_head.norm.weight": "shared_head.norm_w",
}
# serving modules may prefix these with model.layers.<mtp_start>. or .mtp_block. —
# strip a leading "model.layers.<int>." and a "mtp_block." segment before matching.
import re as _re  # noqa: E402


def _canon_key(k: str) -> str:
    k = _re.sub(r"^model\.layers\.\d+\.", "", k)
    k = k.replace("mtp_block.", "")
    return k


def _module_param(model: torch.nn.Module, dotted: str) -> torch.Tensor:
    obj = model
    for part in dotted.split(".")[:-1]:
        obj = getattr(obj, part)
    return getattr(obj, dotted.split(".")[-1])


def _find_draft_state(ckpt_dir: str) -> str:
    for name in ("draft.pt", "draft_ckpt.pt", "pytorch_model.bin", "model.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p
    # else any single .pt
    pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if len(pts) == 1:
        return os.path.join(ckpt_dir, pts[0])
    raise FileNotFoundError(
        f"[m3-oracle] no draft state in {ckpt_dir} (looked for draft.pt / a single *.pt). "
        f"found: {os.listdir(ckpt_dir)}")


def load_draft_checkpoint(model: M3LeanDraft, ckpt_dir: str) -> None:
    """Load the trained LEAN draft weights into `model` (fp64). LOUD on any
    missing/extra/shape-mismatched tensor — a silently half-loaded draft would
    fake the gate."""
    path = _find_draft_state(ckpt_dir)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    loaded, missing, badshape = set(), [], []
    by_canon = {_canon_key(k): (k, v) for k, v in sd.items()}
    for want, attr in _CKPT_ALIASES.items():
        if want not in by_canon:
            missing.append(want)
            continue
        raw_k, v = by_canon[want]
        tgt = _module_param(model, attr)
        if tuple(v.shape) != tuple(tgt.shape):
            badshape.append(f"{want}: ckpt{tuple(v.shape)} != model{tuple(tgt.shape)}")
            continue
        tgt.copy_(v.to(DT))
        loaded.add(want)
    extra = [k for k in by_canon if k not in _CKPT_ALIASES
             and "shared_head.head" not in k and "embed_tokens" not in k]
    if missing or badshape:
        raise RuntimeError(
            f"[m3-oracle] draft checkpoint load FAILED ({path}).\n"
            f"  missing : {missing}\n  bad-shape: {badshape}\n"
            f"  (extra, ignored: {extra})\n"
            f"A partial draft would fake the accept-rate gate — refusing to proceed.")
    print(f"[m3-oracle] loaded {len(loaded)}/{len(_CKPT_ALIASES)} draft tensors from {path}"
          + (f"  (ignored extra: {extra})" if extra else ""), flush=True)


def _safetensors_index(base_dir: str):
    idx_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        return json.load(open(idx_path))["weight_map"]
    # single-file fallback
    one = os.path.join(base_dir, "model.safetensors")
    if os.path.isfile(one):
        from safetensors import safe_open
        with safe_open(one, framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"[m3-oracle] no safetensors index/file in {base_dir}")


def _load_base_tensor(base_dir: str, weight_map, name: str) -> torch.Tensor:
    from safetensors import safe_open
    if name not in weight_map:
        # tolerate the nested text-tower prefix (model.language_model. / text_model.)
        for pre in ("model.language_model.", "language_model.", "model.text_model.", "text_model."):
            if pre + name.split("model.", 1)[-1] in weight_map:
                name = pre + name.split("model.", 1)[-1]
                break
    if name not in weight_map:
        cand = [k for k in weight_map if k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight")]
        raise KeyError(f"[m3-oracle] base tensor {name!r} not in index; embed/head candidates: {cand}")
    with safe_open(os.path.join(base_dir, weight_map[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def load_base_tied(model: M3LeanDraft, base_dir: str) -> None:
    """Tie embed_tokens + shared_head.head to the BASE bf16 model (read-only)."""
    wmap = _safetensors_index(base_dir)
    emb = _load_base_tensor(base_dir, wmap, "model.embed_tokens.weight").to(DT)
    head = _load_base_tensor(base_dir, wmap, "lm_head.weight").to(DT)
    emb = torch.nan_to_num(emb)                              # mirror HPU_SANITIZE_EMBED
    if tuple(emb.shape) != (VOCAB, H):
        raise RuntimeError(f"[m3-oracle] base embed shape {tuple(emb.shape)} != {(VOCAB, H)}")
    if tuple(head.shape) != (VOCAB, H):
        raise RuntimeError(f"[m3-oracle] base lm_head shape {tuple(head.shape)} != {(VOCAB, H)}")
    model.embed_tokens.weight.copy_(emb)
    model.shared_head.head.weight.copy_(head)
    print(f"[m3-oracle] tied embed_tokens{tuple(emb.shape)} + lm_head{tuple(head.shape)} from base {base_dir}",
          flush=True)


# ============================================================================
# the gate: agreement on a dumped (hidden, target) shard
# ============================================================================
def _load_shard(shard_path: str):
    d = torch.load(shard_path, map_location="cpu", weights_only=False)
    if "final" not in d or "ids" not in d:
        raise KeyError(f"[m3-oracle] shard {shard_path} must have 'final' [S,H] and 'ids' [S]; got {list(d.keys())}")
    final = d["final"].to(DT)
    ids = torch.as_tensor(d["ids"]).long().view(-1)
    if final.shape[0] != ids.shape[0]:
        raise ValueError(f"[m3-oracle] shard rows mismatch: final {tuple(final.shape)} vs ids {tuple(ids.shape)}")
    logits_main = d.get("logits_main")
    if logits_main is not None:
        logits_main = logits_main.to(DT)
    prompt_len = int(d.get("prompt_len", 0))
    return final, ids, logits_main, prompt_len


def run_gate(model: M3LeanDraft, final: torch.Tensor, ids: torch.Tensor,
             logits_main: torch.Tensor | None, prompt_len: int = 0,
             decode=None, threshold: float = 0.60) -> float:
    """Print per-position ==/!= and the final ACCEPT RATE. Returns the rate [0,1].

    Row t = draft(prev_hidden=final[t], embed(ids[t+1])) at position t+1; target =
    the model's greedy at t+1 (argmax(logits_main[t+1]) if dumped, else ids[t+1])."""
    S = ids.shape[0]
    if S < 3:
        raise ValueError(f"[m3-oracle] need >=3 tokens to measure agreement, got S={S}")
    Sq = S - 1
    prev_hidden = final[:-1]                                 # final[0..S-2]
    ids_shift = ids[1:]                                      # ids[1..S-1]
    positions = torch.arange(1, S, dtype=torch.long)        # original positions
    draft_logits = model(prev_hidden, ids_shift, positions)  # [S-1, VOCAB], row r == position r+1

    agree = total = 0
    start = max(prompt_len - 1, 0)                           # skip prompt-internal positions
    # iterate t over input positions whose NEXT token is itself in-bounds
    for t in range(start, S - 2):
        # draft row for "predict token after ids[t+1]" lives at ids_shift index t
        d_tok = int(draft_logits[t].argmax())
        if logits_main is not None:
            tgt = int(logits_main[t + 1].argmax())          # model greedy after consuming ids[t+1]
        else:
            tgt = int(ids[t + 1 + 1]) if (t + 2) < S else int(ids[t + 1])  # teacher-forced next
        total += 1
        agree += int(d_tok == tgt)
        mark = "==" if d_tok == tgt else "!="
        if decode is not None:
            print(f"pos {t:3d} in={decode([int(ids[t + 1])])!r}: "
                  f"draft {decode([d_tok])!r} {mark} target {decode([tgt])!r}", flush=True)
        else:
            print(f"pos {t:3d}: draft {d_tok:6d} {mark} target {tgt:6d}", flush=True)
    rate = agree / max(total, 1)
    verdict = "MTP GO" if rate >= threshold else "MTP NO-GO"
    print(f"\nACCEPT RATE {agree}/{total} = {rate * 100:.0f}% — "
          f">={int(threshold * 100)}% => MTP GO  [{verdict}]", flush=True)
    return rate


def oracle_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M3 LEAN MTP accept-rate oracle (>=60% GO/NO-GO)")
    ap.add_argument("--ckpt", default="$HOME/AI/minimax-m3/mtp/draft_ckpt",
                    help="trained LEAN draft checkpoint dir (draft.pt)")
    ap.add_argument("--shard", required=True, help="dumped (hidden,target) shard .pt")
    ap.add_argument("--base", default="/mnt/data2/minimax-m3-bf16",
                    help="bf16 base model dir (for tied embed + lm_head)")
    ap.add_argument("--inter", type=int, default=DENSE_INTER,
                    help=f"LEAN MLP intermediate width (default {DENSE_INTER}; must match the trained ckpt)")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--tokenizer", default=None,
                    help="optional dir/name for a tokenizer to pretty-print tokens")
    args = ap.parse_args(argv)

    t0 = time.time()
    model = M3LeanDraft(inter=args.inter).to_fp64()
    load_base_tied(model, args.base)
    load_draft_checkpoint(model, args.ckpt)
    final, ids, logits_main, prompt_len = _load_shard(args.shard)
    decode = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        decode = tok.decode
    print(f"[m3-oracle] shard S={ids.shape[0]} prompt_len={prompt_len} "
          f"logits_main={'yes' if logits_main is not None else 'no(teacher-forced)'} "
          f"({time.time() - t0:.0f}s)", flush=True)
    rate = run_gate(model, final, ids, logits_main, prompt_len, decode, args.threshold)
    return 0 if rate >= args.threshold else 2


# ============================================================================
# CPU SELF-TEST — proves the MEASUREMENT logic (loop / argmax / off-by-one)
# with RANDOM draft weights + SYNTHETIC data, no card, no trained ckpt.
# ============================================================================
def _selftest() -> bool:
    torch.manual_seed(0)
    checks = []
    # SELF-TEST uses a SMALL vocab so the (VOCAB=200064 x H=6144) embed+head don't
    # blow up to ~20GB/model in fp64 — they're irrelevant to the measurement logic
    # under test (the synthetic drafts override forward / the real-vocab path is the
    # only place full VOCAB matters, exercised by the REAL run, not here).
    SVOCAB = 512

    # ---- 1. block / draft shapes + finiteness on random weights -------------
    model = M3LeanDraft(inter=2048, vocab=SVOCAB).to_fp64()  # small inter+vocab == fast
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.02)
    S = 16
    final = torch.randn(S, H, dtype=DT)
    ids = torch.randint(0, SVOCAB, (S,), dtype=torch.long)
    prev_hidden = final[:-1]
    positions = torch.arange(1, S, dtype=torch.long)
    out = model(prev_hidden, ids[1:], positions)
    checks.append(("draft output shape [S-1, vocab]", out.shape == (S - 1, SVOCAB)))
    checks.append(("draft output finite", bool(torch.isfinite(out).all())))

    # block output shape directly
    x = torch.randn(S, H, dtype=DT)
    blk = model.block(x, torch.arange(S, dtype=torch.long))
    checks.append(("lean block shape [S, H]", blk.shape == (S, H)))

    # ---- 2. partial-rope correctness ----------------------------------------
    # only the first ROTARY_DIM dims of each head are rotated; the rest pass through.
    cos, sin = _rope_cos_sin(torch.arange(S, dtype=torch.long))
    checks.append(("rope cos/sin shape [S, rotary_dim]", cos.shape == (S, ROTARY_DIM)))
    xh = torch.randn(N_HEADS, S, HEAD_DIM, dtype=DT)
    xr = _apply_partial_rope(xh, cos, sin)
    checks.append(("partial-rope passes through dims >= rotary_dim",
                   torch.allclose(xr[..., ROTARY_DIM:], xh[..., ROTARY_DIM:], atol=1e-9)))
    checks.append(("partial-rope DOES rotate dims < rotary_dim",
                   not torch.allclose(xr[..., :ROTARY_DIM], xh[..., :ROTARY_DIM], atol=1e-6)))
    # position 0 is the identity rotation (cos=1, sin=0) -> rotated == original there
    checks.append(("partial-rope identity at pos 0",
                   torch.allclose(xr[:, 0, :], xh[:, 0, :], atol=1e-9)))

    # ---- 3. repeat_kv expands the head axis correctly -----------------------
    kk = torch.randn(N_KV, S, HEAD_DIM, dtype=DT)
    rep = _repeat_kv(kk, KV_GROUPS)
    checks.append(("repeat_kv shape", rep.shape == (N_HEADS, S, HEAD_DIM)))
    # head h of the repeated tensor must equal kv-head (h // KV_GROUPS) of the source
    ok_rep = all(torch.allclose(rep[h], kk[h // KV_GROUPS]) for h in range(N_HEADS))
    checks.append(("repeat_kv interleave mapping (h -> h//groups)", ok_rep))

    # ---- 4. THE GATE MEASUREMENT LOGIC (the load-bearing self-test) ---------
    # Synthetic deterministic rule: target_next(t) := (ids[t+1] + 7) % VOCAB.
    # Build a "perfect draft" whose argmax at row t IS exactly that rule, by
    # monkey-patching the model.forward to emit one-hot logits at the rule token.
    # This isolates run_gate's loop/argmax/off-by-one from the (separately tested)
    # block math.
    SHIFT = 7
    Sg = 24
    ids_g = torch.randint(0, SVOCAB, (Sg,), dtype=torch.long)
    final_g = torch.randn(Sg, H, dtype=DT)

    class _PerfectDraft(M3LeanDraft):
        def forward(self, prev_hidden, ids_shift, positions):
            # rule: predict (ids_shift[t] + SHIFT) % SVOCAB at row t
            tgt = (ids_shift + SHIFT) % SVOCAB               # [S-1]
            logits = torch.full((ids_shift.shape[0], SVOCAB), -1e4, dtype=DT)
            logits[torch.arange(ids_shift.shape[0]), tgt] = 1e4
            return logits

    perfect = _PerfectDraft(inter=64, vocab=SVOCAB).to_fp64()
    # logits_main encoding the SAME rule => target greedy at t+1 == (ids[t+1]+SHIFT)
    logits_main_g = torch.full((Sg, SVOCAB), -1e4, dtype=DT)
    logits_main_g[torch.arange(Sg), (ids_g + SHIFT) % SVOCAB] = 1e4

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rate_match = run_gate(perfect, final_g, ids_g, logits_main_g, prompt_len=0, threshold=0.60)
    checks.append(("perfect draft on matching rule -> 100%", abs(rate_match - 1.0) < 1e-9))

    # shuffled target (break the rule) -> agreement collapses to chance ~0
    perm = torch.randperm(Sg)
    while bool((perm == torch.arange(Sg)).all()):
        perm = torch.randperm(Sg)
    logits_shuf = logits_main_g[perm].clone()
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        rate_shuf = run_gate(perfect, final_g, ids_g, logits_shuf, prompt_len=0, threshold=0.60)
    checks.append(("perfect draft on shuffled target -> ~0%", rate_shuf < 0.20))

    # ---- 5. teacher-forced fallback path (no logits_main) -------------------
    # If the rule is "next token is the literal next id" (SHIFT s.t. predicted ==
    # ids[t+2]), the teacher-forced branch (tgt = ids[t+2]) must score 100%.
    class _CopyNextDraft(M3LeanDraft):
        def __init__(self, ids_ref, **kw):
            super().__init__(**kw)
            self._ids_ref = ids_ref

        def forward(self, prev_hidden, ids_shift, positions):
            # row t corresponds to input position t+1; teacher target is ids[t+2].
            n = ids_shift.shape[0]
            logits = torch.full((n, SVOCAB), -1e4, dtype=DT)
            for t in range(n):
                nxt = int(self._ids_ref[t + 2]) if (t + 2) < self._ids_ref.shape[0] else int(self._ids_ref[t + 1])
                logits[t, nxt] = 1e4
            return logits

    copynext = _CopyNextDraft(ids_g, inter=64, vocab=SVOCAB).to_fp64()
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        rate_tf = run_gate(copynext, final_g, ids_g, None, prompt_len=0, threshold=0.60)
    checks.append(("teacher-forced fallback (copy ids[t+2]) -> 100%", abs(rate_tf - 1.0) < 1e-9))

    # ---- 6. off-by-one guard: a draft that predicts ids[t+1] (NOT its
    # successor) must score ~0 against the (t+2) teacher target ---------------
    class _OffByOneDraft(M3LeanDraft):
        def forward(self, prev_hidden, ids_shift, positions):
            n = ids_shift.shape[0]
            logits = torch.full((n, SVOCAB), -1e4, dtype=DT)
            logits[torch.arange(n), ids_shift] = 1e4        # predicts current, not next
            return logits

    offone = _OffByOneDraft(inter=64, vocab=SVOCAB).to_fp64()
    buf4 = io.StringIO()
    with contextlib.redirect_stdout(buf4):
        rate_off = run_gate(offone, final_g, ids_g, None, prompt_len=0, threshold=0.60)
    checks.append(("off-by-one draft (predicts current id) -> ~0%", rate_off < 0.20))

    # ---- 7. checkpoint alias canonicalization -------------------------------
    checks.append(("canon strips model.layers.<n>. + mtp_block.",
                   _canon_key("model.layers.60.mtp_block.self_attn.q_proj.weight")
                   == "self_attn.q_proj.weight"))
    checks.append(("canon leaves plain serving names",
                   _canon_key("eh_proj.weight") == "eh_proj.weight"))

    # ---- 8. round-trip a synthetic checkpoint through load_draft_checkpoint --
    import tempfile
    tmp = tempfile.mkdtemp()
    src = M3LeanDraft(inter=2048, vocab=SVOCAB).to_fp64()
    for p in src.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
    sd = {
        "enorm.weight": src.frontend.enorm_w.clone(),
        "hnorm.weight": src.frontend.hnorm_w.clone(),
        "eh_proj.weight": src.frontend.eh_proj.weight.clone(),
        "input_layernorm.weight": src.block.input_layernorm_w.clone(),
        "post_attention_layernorm.weight": src.block.post_attention_layernorm_w.clone(),
        "self_attn.q_proj.weight": src.block.q_proj.weight.clone(),
        "self_attn.k_proj.weight": src.block.k_proj.weight.clone(),
        "self_attn.v_proj.weight": src.block.v_proj.weight.clone(),
        "self_attn.o_proj.weight": src.block.o_proj.weight.clone(),
        "self_attn.q_norm.weight": src.block.q_norm_w.clone(),
        "self_attn.k_norm.weight": src.block.k_norm_w.clone(),
        "mlp.gate_up_proj.weight": src.block.gate_up_proj.weight.clone(),
        "mlp.down_proj.weight": src.block.down_proj.weight.clone(),
        "shared_head.norm.weight": src.shared_head.norm_w.clone(),
        # extras that must be ignored without error:
        "lm_head.weight": torch.zeros(2, 2),
        "model.embed_tokens.weight": torch.zeros(2, 2),
    }
    torch.save(sd, os.path.join(tmp, "draft.pt"))
    dst = M3LeanDraft(inter=2048, vocab=SVOCAB).to_fp64()
    buf5 = io.StringIO()
    with contextlib.redirect_stdout(buf5):
        load_draft_checkpoint(dst, tmp)
    rt_ok = (torch.allclose(dst.block.q_proj.weight, src.block.q_proj.weight)
             and torch.allclose(dst.frontend.eh_proj.weight, src.frontend.eh_proj.weight)
             and torch.allclose(dst.block.q_norm_w, src.block.q_norm_w)
             and torch.allclose(dst.shared_head.norm_w, src.shared_head.norm_w))
    checks.append(("checkpoint round-trip loads all draft tensors", rt_ok))

    # a missing tensor must RAISE (LOUD, not silent)
    bad = dict(sd)
    del bad["self_attn.o_proj.weight"]
    torch.save(bad, os.path.join(tmp, "draft.pt"))
    raised = False
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            load_draft_checkpoint(M3LeanDraft(inter=2048, vocab=SVOCAB).to_fp64(), tmp)
    except RuntimeError:
        raised = True
    checks.append(("missing-tensor checkpoint RAISES (loud)", raised))

    # ---- report -------------------------------------------------------------
    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\n{'=' * 60}\nM3 MTP ORACLE SELF-TEST: "
          f"{'GREEN — measurement logic verified' if ok else 'RED — FIX ABOVE'}\n{'=' * 60}")
    print("NOTE: this proves the accept-rate GATE (loop/argmax/off-by-one + ckpt I/O) "
          "with random weights + synthetic data.\n"
          "      The REAL >=60% gate needs the trained draft_ckpt/ + a dumped shard "
          "(card-gated dump+train first):\n"
          "      m3calib python m3_mtp_oracle.py --ckpt draft_ckpt --shard dump/shard0.pt "
          "--base /mnt/data2/minimax-m3-bf16")
    return ok


if __name__ == "__main__":
    # If --shard is given, run the REAL gate; otherwise run the CPU self-test.
    if any(a == "--shard" or a.startswith("--shard=") for a in sys.argv[1:]):
        sys.exit(oracle_main())
    sys.exit(0 if _selftest() else 1)
