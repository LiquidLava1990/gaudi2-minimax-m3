"""M3 MTP draft-head — pure-torch CPU reference + wiring gate (no vllm / no HPU).

This is the CPU core shared by the serving module (hpu_m3_ext/m3_mtp.py), the
hidden-state dump, the trainer, and the accept-rate oracle. It mirrors the proven
GLM MTP shape (_src/vllm_models/glm4_moe_mtp.py) with M3's deltas:
  - Gemma RMSNorm  (text_config.use_gemma_norm=true)  ->  x_normed * (1 + w)
  - swigluoai activation (alpha=1.702, limit=7.0)      ->  the routed/shared expert act
  - eh_proj: Linear(2H -> H), cat order [enorm(embed_{t+1}), hnorm(prev_hidden)]
  - frontend masks position 0 (MTP does not predict from BOS)

Run the wiring gate (card-free, ~1s):
  $HOME/.venvs/m3calib/bin/python $HOME/AI/minimax-m3/mtp/m3_cpu_mtp.py
"""
import torch
import torch.nn as nn

torch.set_grad_enabled(False)

# --- M3 text_config constants (from /mnt/data2/minimax-m3-bf16/config.json) ---
H = 6144
VOCAB = 200064
EPS = 1e-6
ALPHA = 1.702
LIMIT = 7.0


def gemma_rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Gemma-style RMSNorm: normalize in fp32, scale by (1 + weight). M3 uses this
    (use_gemma_norm=true), NOT plain x*w like GLM/Llama."""
    d = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * (1.0 + w.float())).to(d)


def swigluoai(gate_up: torch.Tensor, alpha: float = ALPHA, limit: float = LIMIT) -> torch.Tensor:
    """gpt-oss swigluoai over a concatenated [gate | up] (contiguous halves):
        gate_c = clamp(gate, max=limit)              # upper bound only
        up_c   = clamp(up, -limit, +limit)           # both bounds
        out    = (up_c + 1) * (gate_c * sigmoid(alpha * gate_c))
    Matches m3_swigluoai_patch.swigluoai_tail / swigluoai-fused-kernel.md."""
    gate, up = gate_up.chunk(2, dim=-1)
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    glu = gate_c * torch.sigmoid(alpha * gate_c)
    return (up_c + 1.0) * glu


class M3MTPFrontend(nn.Module):
    """enorm / hnorm / eh_proj — the part that fuses the next-token embedding with
    the target's previous hidden state into the draft block's input."""

    def __init__(self, hidden: int = H, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.enorm_w = nn.Parameter(torch.zeros(hidden))
        self.hnorm_w = nn.Parameter(torch.zeros(hidden))
        self.eh_proj = nn.Linear(hidden * 2, hidden, bias=False)

    def forward(self, embed_tnext: torch.Tensor, prev_hidden: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        # mask position 0 (no MTP target there), then enorm the next-tok embedding
        e = torch.where(positions.unsqueeze(-1) == 0, torch.zeros_like(embed_tnext), embed_tnext)
        e = gemma_rms(e, self.enorm_w, self.eps)
        h = gemma_rms(prev_hidden, self.hnorm_w, self.eps)
        return self.eh_proj(torch.cat([e, h], dim=-1))


class M3SharedHead(nn.Module):
    """final norm + LM head -> logits (the draft's token prediction)."""

    def __init__(self, hidden: int = H, vocab: int = VOCAB, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.norm_w = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(gemma_rms(hidden_states, self.norm_w, self.eps))


# ============================ wiring gate ============================
def _ref_frontend(embed, prev, pos, ew, hw, ehW):
    """independent re-derivation of the spec (catches cat-order / norm-assignment /
    eh_proj-orientation / mask bugs by NOT reusing the module code path)."""
    e = embed.clone()
    e[pos == 0] = 0.0
    e = gemma_rms(e, ew)
    h = gemma_rms(prev, hw)
    return torch.cat([e, h], dim=-1) @ ehW.t()   # nn.Linear: y = x @ W^T


def _gate():
    torch.manual_seed(0)
    T = 12
    fe = M3MTPFrontend()
    # random (non-zero) weights so a wrong wiring can't accidentally pass
    fe.enorm_w.copy_(torch.randn(H) * 0.1)
    fe.hnorm_w.copy_(torch.randn(H) * 0.1)
    fe.eh_proj.weight.copy_(torch.randn(H, 2 * H) * (1.0 / (2 * H) ** 0.5))
    embed = torch.randn(T, H)
    prev = torch.randn(T, H)
    pos = torch.arange(T)

    out = fe(embed, prev, pos)
    ref = _ref_frontend(embed, prev, pos, fe.enorm_w, fe.hnorm_w, fe.eh_proj.weight)

    checks = []
    # 1. shape [T,H]
    checks.append(("frontend shape", out.shape == (T, H)))
    # 2. matches independent reference
    checks.append(("frontend == ref", torch.allclose(out, ref, atol=1e-6, rtol=1e-5)))
    # 3. cat ORDER matters (swap enorm<->hnorm must change the result)
    swapped = _ref_frontend(embed, prev, pos, fe.hnorm_w, fe.enorm_w, fe.eh_proj.weight)
    checks.append(("cat order is fixed (e,h != h,e)", not torch.allclose(out, swapped, atol=1e-4)))
    # 4. position-0 mask zeros the embedding contribution
    embed0 = embed.clone(); embed0[0] = 0.0
    out_masked0 = fe(embed0, prev, pos)
    checks.append(("pos0 embed masked", torch.allclose(out[0], out_masked0[0], atol=1e-6)))
    # 5. eh_proj orientation: input dim is 2H
    checks.append(("eh_proj in=2H out=H", fe.eh_proj.weight.shape == (H, 2 * H)))

    # shared head
    sh = M3SharedHead()
    sh.norm_w.copy_(torch.randn(H) * 0.1)
    sh.head.weight.copy_(torch.randn(VOCAB, H) * (1.0 / H ** 0.5))
    logits = sh(out)
    checks.append(("shared_head shape [T,vocab]", logits.shape == (T, VOCAB)))
    checks.append(("logits finite", bool(torch.isfinite(logits).all())))

    # swigluoai sanity: known values + clamp behavior
    gu = torch.tensor([[100.0, 100.0], [-100.0, -100.0], [0.0, 0.0]])  # [3, 2] -> gate,up each width 1
    sg = swigluoai(gu, ALPHA, LIMIT)
    # gate=+100 -> clamp 7 -> 7*sigmoid(1.702*7)~7 ; up=+100 -> clamp 7 -> (7+1)=8 ; ~56
    # gate=-100 -> clamp keeps -100 -> -100*sigmoid(-170)~0 ; up=-100 -> clamp -7 -> (-7+1)=-6 ; ~0
    # gate=0 -> 0*sigmoid(0)=0 -> out 0
    checks.append(("swigluoai gate clamp", abs(float(sg[0, 0]) - 8.0 * (7.0 * torch.sigmoid(torch.tensor(ALPHA * 7.0)))) < 1e-3))
    checks.append(("swigluoai neg gate ~0", abs(float(sg[1, 0])) < 1e-3))
    checks.append(("swigluoai zero -> 0", abs(float(sg[2, 0])) < 1e-6))

    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\n{'='*48}\nM3 MTP WIRING GATE: {'GREEN — all pass' if ok else 'RED — FIX ABOVE'}\n{'='*48}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _gate() else 1)
