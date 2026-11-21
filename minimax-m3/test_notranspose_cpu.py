#!/usr/bin/env python3
# ==================================================================================================
# test_notranspose_cpu.py  —  CPU numeric A/B proof for M3_FP8_NOTRANSPOSE.
#
# Proves the no-transpose route-b form (consume CONTIGUOUS [out,in] weights, transpose only the tiny
# T==1 activation) is numerically identical to the current transposed-view form (consume [in,out]
# views, matmul forces a big-weight transpose on Gaudi). Runs entirely on CPU — no card, no vllm, no
# hpu. It reimplements the EXACT route-b math from m3_expert_runtime_patch.py for both forms:
#   dequant = (w_fp8.to(f32) * per_out_channel_scale).to(bf16)
#   swigluoai tail (gate.clamp(max=limit)*sigmoid(alpha*gate)) * (up.clamp(-limit,limit)+1)
#   GEMM + weighted combine over the K routed experts.
#
# OLD form (default, M3_FP8_NOTRANSPOSE off):
#   stacks are [in,out]: w13[e]=[H,2I], w2[e]=[I,H]; scale broadcasts over `in` (dim 0).
#   gate_up = x[T,H] @ w13_deq[H,2I];  y = h[T,I] @ w2_deq[I,H]
# NEW form (M3_FP8_NOTRANSPOSE on):
#   originals are [out,in]: w13_oi[e]=[2I,H], w2_oi[e]=[H,I]; scale broadcasts over `in` (dim 1).
#   gate_up = (w13_oi[2I,H] @ x.t()[H,T]).t();  y = (w2_oi[H,I] @ h.t()[I,T]).t()
#
# The [in,out] stack used by the OLD form is exactly the .transpose of the [out,in] original used by
# the NEW form, mirroring how m3_fp8_static_patch._build_fp8_stacks shares one storage.
#
#   python3 test_notranspose_cpu.py
# ==================================================================================================
import torch

torch.manual_seed(0)

# small dims (per the task): H=64, I=32 (2I=64), E=4 local experts, top_k K=2, T=1 decode token.
H, I, E, K, T = 64, 32, 4, 2, 1
twoI = 2 * I
ALPHA, LIMIT = 1.702, 7.0
FP8 = torch.float8_e4m3fn


def swigluoai_tail(gate_up, alpha, limit):
    # EXACT copy of m3_expert_runtime_patch._swigluoai_tail (contiguous halves: gate first, up second).
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


def dequant_io(w_fp8, s):
    # [in,out] dequant (= _dequant_fp8_slice): w_fp8 [in,out], s [1,out] -> broadcast over `in`.
    return (w_fp8.to(s.dtype) * s).to(torch.bfloat16)


def dequant_oi(w_fp8_oi, s):
    # [out,in] dequant (= _dequant_fp8_slice_oi): w_fp8_oi [out,in], s [out] -> reshape [out,1].
    return (w_fp8_oi.to(s.dtype) * s.view(-1, 1)).to(torch.bfloat16)


def make_fp8(shape):
    # random "fp8-ish" weights: a real fp8 tensor (round-trips bf16 random through float8_e4m3fn).
    return (torch.randn(shape) * 4.0).to(FP8)


def build_layer():
    # ORIGINAL contiguous [out,in] fp8 weights (what vLLM stores) + per-OUT-channel f32 scales.
    w13_oi = torch.stack([make_fp8((twoI, H)) for _ in range(E)])   # [E,2I,H]
    w2_oi = torch.stack([make_fp8((H, I)) for _ in range(E)])       # [E,H,I]
    s13 = (torch.rand(E, twoI).float() + 0.1) / 50.0                # [E,2I] per-out-channel
    s2 = (torch.rand(E, H).float() + 0.1) / 50.0                    # [E,H]  per-out-channel
    # the [in,out] stacks the OLD form gathers = transpose(1,2) of the originals (shared-storage style).
    w13_io = w13_oi.transpose(1, 2)                                 # [E,H,2I] view
    w2_io = w2_oi.transpose(1, 2)                                   # [E,I,H] view
    return w13_oi, w2_oi, w13_io, w2_io, s13, s2


def old_form(x, active, w13_io, w2_io, s13, s2, gate_mat):
    # gather K experts in [in,out], dequant, OLD GEMM. x [T,H] bf16.
    out = torch.zeros(T, H, dtype=x.dtype)
    w13p = w13_io.index_select(0, active)      # [K,H,2I]
    w2p = w2_io.index_select(0, active)        # [K,I,H]
    s13a = s13.index_select(0, active)         # [K,2I]
    s2a = s2.index_select(0, active)           # [K,H]
    for k in range(K):
        w13_deq = dequant_io(w13p[k], s13a[k].unsqueeze(0))   # [H,2I]
        w2_deq = dequant_io(w2p[k], s2a[k].unsqueeze(0))      # [I,H]
        gate_up = torch.matmul(x, w13_deq)                    # [T,2I]
        h = swigluoai_tail(gate_up, ALPHA, LIMIT)             # [T,I]
        y_k = torch.matmul(h, w2_deq)                         # [T,H]
        out = out + gate_mat[:, k:k + 1] * y_k
    return out


def new_form(x, active, w13_oi, w2_oi, s13, s2, gate_mat):
    # gather K experts in [out,in], dequant, NEW GEMM (transpose only x). x [T,H] bf16.
    out = torch.zeros(T, H, dtype=x.dtype)
    w13p = w13_oi.index_select(0, active)      # [K,2I,H]
    w2p = w2_oi.index_select(0, active)        # [K,H,I]
    s13a = s13.index_select(0, active)         # [K,2I]
    s2a = s2.index_select(0, active)           # [K,H]
    xt = x.t()                                 # [H,T]
    for k in range(K):
        w13_deq = dequant_oi(w13p[k], s13a[k])               # [2I,H]
        w2_deq = dequant_oi(w2p[k], s2a[k])                  # [H,I]
        gate_up = torch.matmul(w13_deq, xt).t()              # [T,2I]
        h = swigluoai_tail(gate_up, ALPHA, LIMIT)            # [T,I]
        y_k = torch.matmul(w2_deq, h.t()).t()                # [T,H]
        out = out + gate_mat[:, k:k + 1] * y_k
    return out


def main():
    w13_oi, w2_oi, w13_io, w2_io, s13, s2 = build_layer()

    # 1) dequant-level bit-faithfulness (the requirement-3 claim): for every expert, the [out,in]
    #    dequant transposed must equal the [in,out] dequant, element for element.
    max_deq_diff = 0.0
    for e in range(E):
        d_io = dequant_io(w13_io[e], s13[e].unsqueeze(0))    # [H,2I]
        d_oi = dequant_oi(w13_oi[e], s13[e])                 # [2I,H]
        max_deq_diff = max(max_deq_diff, (d_io - d_oi.t()).abs().max().item())
        d2_io = dequant_io(w2_io[e], s2[e].unsqueeze(0))     # [I,H]
        d2_oi = dequant_oi(w2_oi[e], s2[e])                  # [H,I]
        max_deq_diff = max(max_deq_diff, (d2_io - d2_oi.t()).abs().max().item())

    # 2) end-to-end route-b A/B over a few random tokens + random active-expert sets.
    max_e2e_diff = 0.0
    for trial in range(8):
        x = (torch.randn(T, H) * 0.5).to(torch.bfloat16)
        active = torch.randperm(E)[:K].to(torch.int64)
        gate_mat = (torch.rand(T, K) + 0.1).to(torch.bfloat16)
        a = old_form(x, active, w13_io, w2_io, s13, s2, gate_mat)
        b = new_form(x, active, w13_oi, w2_oi, s13, s2, gate_mat)
        max_e2e_diff = max(max_e2e_diff, (a.float() - b.float()).abs().max().item())

    print("=" * 80)
    print("M3_FP8_NOTRANSPOSE CPU A/B numeric test")
    print(f"  dims: H={H} I={I} 2I={twoI} E={E} K={K} T={T}  (fp8={FP8}, bf16 GEMM)")
    print(f"  max |dequant_io - dequant_oi.t()|       = {max_deq_diff:.3e}")
    print(f"  max |old_route_b - new_route_b| (e2e)   = {max_e2e_diff:.3e}")
    tol = 1e-3
    ok = (max_deq_diff < tol) and (max_e2e_diff < tol)
    print(f"  tolerance                                = {tol:.0e}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 80)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
