#!/usr/bin/env python3
"""Card-free viability check for an fp8 GLM-5.1 MTP head (replacing the bf16 graft).

The int4 (AutoRound) export of MTP layer-78 is DEAD — int4 can't represent that layer
(constant junk under every wiring; PROJECT_FINDINGS §8). Question this answers WITHOUT a
card or the fabric: does fp8 (e4m3) represent layer-78 faithfully? If yes, the MTP head can
be fp8 (rule-compliant, ~1.25GB/card vs 2.5 bf16, faster draft) instead of bf16.

Method: load a representative sample of layer-78 weights from the bf16 original, quantize
to fp8_e4m3 with per-output-channel scales (the accurate, standard weight-quant), dequantize,
and measure relative error vs bf16. Low error (<~2-3%) => fp8 head viable. Compares against a
simulated int4-per-channel quant to show why int4 failed where fp8 won't.
"""
import glob
import json
import sys

import torch
from safetensors import safe_open

D = "/mnt/fastmodels/glm51-orig-l78"
LAYER = "model.layers.78."

FP8_E4M3_MAX = 448.0  # torch.float8_e4m3fn finite max


def per_channel_fp8(w: torch.Tensor):
    """Per-output-row (dim 0) symmetric fp8_e4m3 quant. Returns (deq, rel_err)."""
    w = w.to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / FP8_E4M3_MAX
    q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    deq = q.to(torch.float32) * scale
    rel = (deq - w).norm() / w.norm()
    return rel.item()


def per_channel_int4(w: torch.Tensor):
    """Per-output-row symmetric int4 (15 levels) — to illustrate the int4 failure mode."""
    w = w.to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    q = (w / scale).round().clamp(-7, 7)
    deq = q * scale
    rel = (deq - w).norm() / w.norm()
    return rel.item()


def main():
    idx = json.load(open(glob.glob(f"{D}/*.index.json")[0]))["weight_map"]
    l78 = [k for k in idx if k.startswith(LAYER)]
    # representative sample: attn projections + shared experts + a spread of routed experts
    attn = [k for k in l78 if ".self_attn." in k and k.endswith(".weight")
            and "norm" not in k and "indexer" not in k]
    shared = [k for k in l78 if ".shared_experts." in k]
    routed = sorted(k for k in l78 if ".mlp.experts." in k and k.endswith(".weight"))
    # 8 routed experts spread across the range, all 3 projections each
    pick_e = [routed[i] for i in range(0, len(routed), max(1, len(routed) // 24))][:24]
    sample = attn + shared + pick_e
    handles = {}

    def read(name):
        fn = idx[name]
        if fn not in handles:
            handles[fn] = safe_open(f"{D}/{fn}", framework="pt", device="cpu")
        return handles[fn].get_tensor(name)

    print(f"layer-78 total tensors={len(l78)}  sampling {len(sample)} weights "
          f"(attn={len(attn)} shared={len(shared)} routed={len(pick_e)})")
    print(f"{'tensor':<52}{'fp8 relerr':>12}{'int4 relerr':>12}")
    fp8_errs, int4_errs = [], []
    for name in sample:
        w = read(name)
        if w.dim() != 2:
            continue
        f8 = per_channel_fp8(w)
        i4 = per_channel_int4(w)
        fp8_errs.append(f8); int4_errs.append(i4)
        short = name.replace(LAYER, "").replace(".weight", "")
        print(f"{short:<52}{f8:>12.4f}{i4:>12.4f}")
    import statistics as st
    print("\n=== SUMMARY (relative error, lower=better) ===")
    print(f"  fp8_e4m3 per-channel : median={st.median(fp8_errs):.4f}  max={max(fp8_errs):.4f}")
    print(f"  int4    per-channel : median={st.median(int4_errs):.4f}  max={max(int4_errs):.4f}")
    ok = st.median(fp8_errs) < 0.03 and max(fp8_errs) < 0.06
    print(f"\n  fp8 MTP head VIABLE: {ok}  "
          f"(fp8 {'<<' if st.median(fp8_errs) < st.median(int4_errs) else '>='} int4 error)")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
