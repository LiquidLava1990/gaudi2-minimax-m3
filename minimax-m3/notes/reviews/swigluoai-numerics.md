# MiniMax-M3 `swigluoai` activation — AUTHORITATIVE numerics

Angle: the exact formula + torch reference. Card-free.

## Source of truth
`transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py`
- `MiniMaxM3VLDenseMLP.forward` — lines **168-174**
- `MiniMaxM3VLExperts._apply_gate` — lines **209-215** (comment L210: "same as GPT OSS, but the weights are not interleaved")

Both code paths are **byte-for-byte identical**. Config: `swiglu_alpha=1.702`, `swiglu_limit=7.0` (config.json + configuration_minimax_m3_vl.py L113-114).

## CRITICAL TRAP (this is the blocker)
configuration_minimax_m3_vl.py **L127-129**:
> "Checkpoint declares `swigluoai`, but the gate is computed inline from swiglu_alpha/limit; hidden_act" → `self.hidden_act = "silu"`.

So `config.json` says `hidden_act="swigluoai"`, but transformers **overrides it to `"silu"`** and the MLP/Expert path **never calls `ACT2FN[hidden_act]`** — the gate is hand-written. Any fix that dispatches the MoE activation off `hidden_act` gets plain SiLU = WRONG/incoherent. The activation MUST be the inline swigluoai math below, regardless of the `hidden_act` string.

## Resolved questions (exact code)
1. **Clamp targets — asymmetric, like gpt-oss:** `gate = gate.clamp(max=limit)` (L171/212, no lower bound); `up = up.clamp(min=-limit, max=limit)` (L172/213, both bounds). NOT both to [-limit,+limit].
2. **GLU:** `glu = gate * sigmoid(gate * alpha)` (alpha INSIDE sigmoid). L173/214.
3. **+1:** YES — `(up + 1.0) * glu`. L174/215. (matches gpt-oss).
4. **Order:** chunk → clamp gate → clamp up → glu → `(up+1)*glu` → down_proj.

## Split / layout
`gate_up = gate_up_proj(x)`; `gate, up = gate_up.chunk(2, dim=-1)` — **gate = first half, up = second half**, NOT interleaved (L210). gate_up_proj out dim = `2 * intermediate`. down_proj applied to the `(up+1)*glu` result (intermediate → hidden).

## Torch reference (exact)
```python
import torch

def swigluoai(gate: torch.Tensor, up: torch.Tensor,
              alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """Exact MiniMax-M3 swigluoai. `gate`,`up` = the two halves of gate_up
    (gate=first, up=second), pre-clamp. Returns pre-down_proj activation."""
    gate = gate.clamp(max=limit)                 # upper only
    up   = up.clamp(min=-limit, max=limit)       # symmetric
    glu  = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu

# Fused form taking the concatenated projection:
def swigluoai_fused(gate_up, alpha=1.702, limit=7.0):
    gate, up = gate_up.chunk(2, dim=-1)
    return swigluoai(gate, up, alpha, limit)
```

## Elementwise op list
`clamp(max)` [gate] · `clamp(min,max)` [up] · `mul` (gate*alpha) · `sigmoid` · `mul` (gate*sigmoid) · `add` (up+1) · `mul` ((up+1)*glu). → **clamp, sigmoid, mul, add** only. No exp/tanh/gelu, no SiLU op.
