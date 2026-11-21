#!/usr/bin/env python3
# ==================================================================================================
# test_mme_fp8_cpu.py  —  CPU numeric REFERENCE for the M3_FP8_MME path.
#
# The real path calls torch.ops.hpu.fp8_gemm_v2 (a fp8 x fp8 -> bf16 MME GEMM). CPU torch has NO
# torch.ops.hpu namespace, so the *real* GEMM CANNOT run off-card (confirmed: `hasattr(torch.ops,
# 'hpu')` -> False on torch 2.12+cpu). What CPU torch DOES have is the float8_e4m3fn dtype, so this
# test EMULATES fp8_gemm_v2 in fp32 with the EXACT scale convention the native op uses, and measures
# the end-to-end relative error the MME path adds vs the current bf16-dequant route-b path.
#
# WHAT THE NATIVE OP DOES (mirrored here):
#   * weight already fp8 (per-OUTPUT-channel FORWARD scale s_w: real ~= w_fp8 * s_w).
#   * activation: dynamic_quant(x) = per-token amax FORWARD scale s_x (real ~= x_fp8 * s_x), cast to
#     e4m3 with 1/s_x. (ops.py:932-939)
#   * fp8_gemm_v2(A=x_fp8, B=w_fp8, A_scale_inv=s_x, B_scale_inv=s_w, out_dtype=bf16): accumulates the
#     fp8*fp8 products and applies the forward scales -> out[t,o] = sum_i (x_fp8[t,i]*s_x[t]) *
#     (w_fp8[i,o]*s_w[o]). We reproduce that as (x_fp8.f32 @ w_fp8.f32) * s_x * s_w.
#
# This is a REFERENCE (CPU) check of NUMERIC COHERENCE only — it does NOT prove the device kernel
# runs or the speedup. Those MUST be validated on-card (see the printed CARD-VALIDATION note).
#
#   $HOME/.venvs/m3calib/bin/python test_mme_fp8_cpu.py
# ==================================================================================================
import torch

torch.manual_seed(0)

# M3-ish small dims: H hidden, I intermediate (2I = gate||up), E local experts, top_k K, T=1 decode.
H, I, E, K, T = 256, 128, 8, 4, 1
twoI = 2 * I
ALPHA, LIMIT = 1.702, 7.0
FP8 = torch.float8_e4m3fn
# fnuz max (240) — the FP8_MAX the M3 stacks + dynamic_quant use (ops.py:30, int4_to_fp8_load:65).
FP8_MAX = float(torch.finfo(torch.float8_e4m3fnuz).max)


def swigluoai_tail(gate_up, alpha=ALPHA, limit=LIMIT):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


# ---- weight fp8 quant: per-OUTPUT-channel forward scale (EXACT quant_weight_fp8_per_channel) ------
def quant_w_fp8_oi(w_real_oi):
    """w_real_oi [out,in] -> (w_fp8_oi [out,in], s_w [out]) with real ~= w_fp8 * s_w (per out-chan)."""
    w = w_real_oi.float()
    amax = w.abs().amax(dim=1)                       # [out]
    s = (amax + 1e-8) / FP8_MAX                       # forward scale [out]
    w_fp8 = (w / s.unsqueeze(1)).to(FP8)              # cast to e4m3 grid
    return w_fp8, s.float()


# ---- activation fp8 quant: per-token forward scale (EXACT dynamic_quant) --------------------------
def quant_act_fp8(x):
    """x [T,in] -> (x_fp8 [T,in], s_x [T,1]) with real ~= x_fp8 * s_x (per token)."""
    s = (x.abs().max(dim=-1).values + 1e-8) / FP8_MAX
    s = s.unsqueeze(-1)                               # [T,1]
    x_fp8 = (x.float() / s).to(FP8)
    return x_fp8, s.float()


# ---- fp8 GEMM emulation: (x_fp8 @ w_fp8) then apply both forward scales, accumulate fp32 ----------
def fp8_gemm_emu(x_fp8, s_x, w_fp8_oi, s_w):
    """Emulate fp8_gemm_v2(A=x_fp8[T,in], B=w_fp8[out,in], trans_B=False, A_scale_inv=s_x,
    B_scale_inv=s_w) -> [T,out] fp32. out[t,o] = (sum_i x_fp8[t,i]*w_fp8[o,i]) * s_x[t] * s_w[o]."""
    acc = x_fp8.float() @ w_fp8_oi.float().t()       # [T,out] raw fp8*fp8 accumulate (fp32)
    return acc * s_x * s_w.unsqueeze(0)              # apply forward scales


# ---- current bf16-dequant route-b reference (what M3_FP8_MME replaces) ----------------------------
def bf16_dequant_oi(w_fp8_oi, s_w):
    # _dequant_fp8_slice_oi: (w_fp8.to(f32) * s.view(-1,1)).to(bf16) — [out,in]
    return (w_fp8_oi.to(torch.float32) * s_w.view(-1, 1)).to(torch.bfloat16)


def run():
    # build "ground-truth" real expert weights, then store them fp8 per-out-channel (what HBM holds).
    rel_errs, abs_norms = [], []
    for e in range(E):
        # plausible-magnitude real weights (post int4->bf16); the fp8 *weight* error is already baked
        # into BOTH paths (same w_fp8), so this test isolates the ADDED error of fp8-activation+fp8-GEMM.
        w13_real = torch.randn(twoI, H) * 0.05        # [out=2I, in=H]
        w2_real = torch.randn(H, I) * 0.05            # [out=H, in=I]
        w13_fp8, s13 = quant_w_fp8_oi(w13_real)
        w2_fp8, s2 = quant_w_fp8_oi(w2_real)

        for _ in range(16):
            x = (torch.randn(T, H) * 0.5).to(torch.bfloat16)   # decode activation

            # ---- REFERENCE: current bf16-dequant route-b (NOTRANSPOSE [out,in] form) ----
            w13_deq = bf16_dequant_oi(w13_fp8, s13)            # [2I,H] bf16
            w2_deq = bf16_dequant_oi(w2_fp8, s2)               # [H,I]  bf16
            gate_up_ref = (w13_deq @ x.t()).t()                # [T,2I] bf16
            h_ref = swigluoai_tail(gate_up_ref)                # [T,I]
            y_ref = (w2_deq @ h_ref.t()).t()                   # [T,H] bf16

            # ---- M3_FP8_MME: fp8 activation + fp8 GEMM (emulated), swigluoai tail bf16 ----
            x_fp8, s_x = quant_act_fp8(x)
            gate_up_mme = fp8_gemm_emu(x_fp8, s_x, w13_fp8, s13).to(torch.bfloat16)   # [T,2I]
            h_mme = swigluoai_tail(gate_up_mme)                # [T,I]
            h_fp8, s_h = quant_act_fp8(h_mme.to(torch.bfloat16))
            y_mme = fp8_gemm_emu(h_fp8, s_h, w2_fp8, s2).to(torch.bfloat16)           # [T,H]

            d = (y_ref.float() - y_mme.float())
            rel = d.norm() / (y_ref.float().norm() + 1e-8)
            rel_errs.append(rel.item())
            abs_norms.append(y_ref.float().norm().item())

    rel = torch.tensor(rel_errs)
    print("=" * 88)
    print("M3_FP8_MME CPU numeric REFERENCE (fp8_gemm_v2 EMULATED in fp32; real GEMM is on-card only)")
    print(f"  dims: H={H} I={I} 2I={twoI} E={E} K={K} T={T}  (fp8={FP8}, FP8_MAX(fnuz)={FP8_MAX})")
    print(f"  end-to-end relative error (||y_mme - y_bf16|| / ||y_bf16||), per (expert,token):")
    print(f"     mean = {rel.mean().item()*100:.3f}%   median = {rel.median().item()*100:.3f}%"
          f"   p95 = {rel.quantile(0.95).item()*100:.3f}%   max = {rel.max().item()*100:.3f}%")
    # COHERENCE GATE: fp8 activation adds a few % vs the bf16-activation path. The honest bar is a few
    # percent (NOT bit-identical). >~8% would be the per-tensor double-quant floor (incoherent); we are
    # per-channel weight + per-token activation, so we expect ~1-4%.
    bar = 0.08
    ok = rel.mean().item() < bar and rel.max().item() < (bar * 2)
    print(f"  coherence bar: mean < {bar*100:.0f}% AND max < {bar*200:.0f}%  ->  {'PASS' if ok else 'FAIL'}")
    print()
    print("  NOTE (off-card limit): this EMULATES the fp8*fp8 accumulate in fp32. The real Gaudi MME")
    print("  fp8_gemm_v2 accumulates in its own (bf16/f32) accumulator with HW e4m3fnuz mapping; the")
    print("  ADDED error magnitude is representative, but exact bits + the SPEEDUP are on-card only.")
    print("=" * 88)
    print("CARD-VALIDATION (MUST do on-card, cannot be done here):")
    print("  1) fp8_gemm_v2 actually executes (not a CPU-composite fallback) — check the LIVE log line")
    print("     'routed fp8 experts via NATIVE fp8_gemm_v2 MME GEMM'.")
    print("  2) coherence: same prompts, M3_FP8_MME=0 vs =1, outputs stay sensible (a few % logit drift).")
    print("  3) speedup: matmul device-time bucket shrinks (~2x on the two expert GEMMs) in the profile.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
