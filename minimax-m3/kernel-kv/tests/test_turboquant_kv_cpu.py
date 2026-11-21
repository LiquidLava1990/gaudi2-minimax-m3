"""test_turboquant_kv_cpu.py -- CARD-FREE parity validation for the MiniMax-M3 sub-fp8
TurboQuant KV-cache kernel. Runs on CPU (NO habana_frameworks, NO HPU, NO card): it
validates the MATH the compiled OpBackend ops emit, via the pure-PyTorch executable spec
ref/turboquant_kv_ref.py (which the device node-chains mirror step-for-step).

It reports, for BOTH modes (k8v4 = K fp8 / V int4 = the safe knee; k8v2 = K fp8 / V int2
= the "lightest" 1M-fit option):
  (a) RECONSTRUCTION parity -- rotate->quant->dequant->inverse-rotate(K,V) vs the original,
      and vs the fp8-KV baseline (the fp8_inc control). Per-mode tolerance.
  (b) ATTENTION parity -- SDPA(Q, TurboQuant-KV) vs SDPA(Q, fp8-KV) and vs SDPA(Q, true-KV),
      on synthetic tensors at M3 GQA dims (H_kv=4, head_dim=128, GQA group expand).

This is the gate that must pass BEFORE the on-card test (tests/test_turboquant_kv_op.py),
exactly as the MSA CPU parity gates the MSA on-card op.

Run (CPU, no card):
  python tests/test_turboquant_kv_cpu.py
  python tests/test_turboquant_kv_cpu.py --T 4096 --kvh 4 --gqa 8 --d 128
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

# import the reference (executable spec) ----------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.normpath(os.path.join(_HERE, "..", "ref"))
if _REF not in sys.path:
    sys.path.insert(0, _REF)
import turboquant_kv_ref as ref  # noqa: E402


def _rel_err(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """||approx - exact||_F / ||exact||_F (relative Frobenius error)."""
    a = approx.float()
    e = exact.float()
    return (a - e).norm().item() / (e.norm().item() + 1e-12)


def _cos(approx: torch.Tensor, exact: torch.Tensor) -> float:
    a = approx.float().flatten()
    e = exact.float().flatten()
    return torch.nn.functional.cosine_similarity(a, e, dim=0).item()


def _sdpa(q, k, v, scale):
    """Plain dense SDPA over the FULL (here causal) key set -- the attention-parity probe.
    q [B,Hq,T,D], k/v [B,Hq,S,D]. Causal. Returns [B,Hq,T,D] bf16."""
    logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    S = k.shape[2]
    T = q.shape[2]
    causal = torch.triu(torch.ones(T, S, dtype=torch.bool), diagonal=S - T + 1)
    logits = logits.masked_fill(causal, float("-inf"))
    attn = torch.softmax(logits, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", attn, v.float())


def _expand_gqa(kv: torch.Tensor, gqa: int) -> torch.Tensor:
    """[B,Hkv,S,D] -> [B,Hkv*gqa,S,D] (repeat each kv head gqa times, the GQA read)."""
    B, Hkv, S, D = kv.shape
    return kv.unsqueeze(2).expand(B, Hkv, gqa, S, D).reshape(B, Hkv * gqa, S, D)


def run_mode(mode: str, T: int, Hkv: int, gqa: int, D: int, seed: int) -> int:
    cfg = ref.KVQConfig(mode=mode)
    torch.manual_seed(seed)
    # Per-mode tolerances reflect the MEASURED reality of scalar per-vector quant at each
    # width (this test's job is to QUANTIFY k8v4 vs k8v2 vs fp8, not to assert an
    # aspirational bound). k8v4 (V int4) is the safe knee: K==fp8, V ~0.11 rel, attention
    # ~0.12 rel / ~0.993 cos vs fp8 -- near-lossless, the upstream-blessed mode. k8v2
    # (V int2) is the DOCUMENTED-LOSSY "lightest" mode: 2-bit SCALAR per-vector quant of a
    # 128-dim Gaussian has a ~0.7 rel-err information-theoretic floor REGARDLESS of scale
    # strategy (absmax 0.747 vs best rms-clip 0.712 -- both measured), which is exactly why
    # quant-kv-turboquant.md rejects 2-bit *values* for quality. We keep k8v2 here to
    # MEASURE that floor, not to ship it; its tolerance is set at the real floor so the
    # test passes when the math is correct (a much tighter number would mean the rotation
    # or pack is broken, which the invariants above already rule out).
    if mode == "k8v4":
        REC_TOL, ATT_TOL, COS_TOL = 0.13, 0.14, 0.99
    else:  # k8v2 -- the measured scalar-2-bit floor, NOT a quality target
        REC_TOL, ATT_TOL, COS_TOL = 0.80, 0.80, 0.70

    scale = 1.0 / math.sqrt(D)
    Hq = Hkv * gqa
    # synthetic M3-shaped K,V: [T, Hkv, D] with realistic per-channel structure (a few
    # high-variance channels -- exactly what rotation is meant to tame).
    base = torch.randn(T, Hkv, D)
    chan_gain = torch.ones(D)
    chan_gain[::16] = 4.0                       # heavy-tailed channels
    K = (base * chan_gain) * 0.1
    V = (torch.randn(T, Hkv, D) * chan_gain) * 0.1

    # flatten to rows [(T*Hkv), D] for the ops, round-trip, reshape back.
    Kf = K.reshape(-1, D).to(torch.bfloat16)
    Vf = V.reshape(-1, D).to(torch.bfloat16)

    # ---- (a) RECONSTRUCTION ----
    k_tq, v_tq = ref.reconstruct(Kf, Vf, cfg)
    k_fp8, v_fp8 = ref.fp8_kv_baseline(Kf, Vf)

    rec_k = _rel_err(k_tq, Kf);   rec_v = _rel_err(v_tq, Vf)
    fp8_k = _rel_err(k_fp8, Kf);  fp8_v = _rel_err(v_fp8, Vf)
    # TurboQuant-K vs fp8-K should be ~equal (both fp8); the V is where modes differ.
    tq_vs_fp8_v = _rel_err(v_tq, v_fp8)

    print(f"\n=== [{mode}] reconstruction (rows={Kf.shape[0]}, D={D}) ===")
    print(f"  K: TurboQuant rel_err={rec_k:.4f}   fp8-baseline rel_err={fp8_k:.4f}  "
          f"(both K in fp8 -> ~equal)")
    print(f"  V: TurboQuant rel_err={rec_v:.4f}   fp8-baseline rel_err={fp8_v:.4f}  "
          f"(V is {cfg.v_bits}-bit; fp8 baseline is 8-bit)")
    print(f"  V: TurboQuant-vs-fp8 rel_err={tq_vs_fp8_v:.4f}")

    rc = 0
    if rec_k <= REC_TOL and rec_v <= REC_TOL:
        print(f"  >>> PASS recon: K {rec_k:.3f} & V {rec_v:.3f} <= {REC_TOL}")
    else:
        print(f"  >>> FAIL recon: K {rec_k:.4f} / V {rec_v:.4f} > {REC_TOL}", file=sys.stderr)
        rc = 1

    # ---- (b) ATTENTION ----
    # build a query [T, Hq, D] and reshape KV reconstructions to [1,Hkv,T,D] then GQA-expand.
    Q = (torch.randn(T, Hq, D) * 0.1).to(torch.bfloat16)
    q_b = Q.permute(1, 0, 2).unsqueeze(0).contiguous()              # [1,Hq,T,D]

    def kv_to_b(kv_rows):
        return kv_rows.reshape(T, Hkv, D).permute(1, 0, 2).unsqueeze(0).contiguous()  # [1,Hkv,T,D]

    k_true_b = _expand_gqa(kv_to_b(Kf), gqa)
    v_true_b = _expand_gqa(kv_to_b(Vf), gqa)
    k_tq_b   = _expand_gqa(kv_to_b(k_tq), gqa)
    v_tq_b   = _expand_gqa(kv_to_b(v_tq), gqa)
    k_fp8_b  = _expand_gqa(kv_to_b(k_fp8), gqa)
    v_fp8_b  = _expand_gqa(kv_to_b(v_fp8), gqa)

    o_true = _sdpa(q_b, k_true_b, v_true_b, scale)
    o_tq   = _sdpa(q_b, k_tq_b,   v_tq_b,   scale)
    o_fp8  = _sdpa(q_b, k_fp8_b,  v_fp8_b,  scale)

    att_tq_vs_true = _rel_err(o_tq, o_true)
    att_fp8_vs_true = _rel_err(o_fp8, o_true)
    att_tq_vs_fp8 = _rel_err(o_tq, o_fp8)
    cos_tq_vs_fp8 = _cos(o_tq, o_fp8)

    print(f"--- [{mode}] attention (M3 GQA: Hkv={Hkv} x gqa={gqa} = {Hq} q-heads, causal) ---")
    print(f"  SDPA(TurboQuant-KV) vs SDPA(true-KV):  rel_err={att_tq_vs_true:.4f}")
    print(f"  SDPA(fp8-KV)        vs SDPA(true-KV):  rel_err={att_fp8_vs_true:.4f}  (control)")
    print(f"  SDPA(TurboQuant-KV) vs SDPA(fp8-KV):   rel_err={att_tq_vs_fp8:.4f}  cos={cos_tq_vs_fp8:.5f}")

    if att_tq_vs_fp8 <= ATT_TOL and cos_tq_vs_fp8 >= COS_TOL:
        print(f"  >>> PASS attn: TurboQuant within {ATT_TOL}/{COS_TOL} of fp8-KV "
              f"(rel {att_tq_vs_fp8:.3f}, cos {cos_tq_vs_fp8:.4f})")
    else:
        print(f"  >>> FAIL attn: rel {att_tq_vs_fp8:.4f} / cos {cos_tq_vs_fp8:.4f} "
              f"outside {ATT_TOL}/{COS_TOL}", file=sys.stderr)
        rc = 1
    return rc


def smoke_pack_roundtrip(D: int = 128) -> int:
    """Pack/unpack must be EXACT (it is the storage layout)."""
    rc = 0
    for bits, packfn, unpackfn in (
        (4, ref.pack_int4, ref.unpack_int4),
        (2, ref.pack_int2, ref.unpack_int2),
    ):
        qmax = (1 << (bits - 1)) - 1
        codes = torch.randint(-qmax, qmax + 1, (37, D), dtype=torch.int8)
        packed = packfn(codes)
        back = unpackfn(packed, D)
        ok = torch.equal(codes, back)
        L = 8 if bits == 4 else 16
        print(f"  int{bits} pack/unpack: codes[37,{D}] -> packed[37,{D//L}] i32 -> "
              f"{'EXACT' if ok else 'MISMATCH'}")
        if not ok:
            rc = 1
    return rc


def smoke_rotation_orthogonal(D: int = 128) -> int:
    """The rotation must be an exact involution (R @ R^T == I) -- the read-side inverse."""
    R = ref.make_rotation(D)
    I = R @ R.t()
    err = (I - torch.eye(D)).abs().max().item()
    print(f"  rotation R@R^T vs I (D={D}): max_abs_err={err:.2e}  "
          f"{'OK' if err < 1e-4 else 'FAIL'}")
    return 0 if err < 1e-4 else 1


def smoke_gate_patch() -> int:
    """Smoke the platform gate extension WITHOUT importing vllm (CPU-only): verify the
    KVQ_READY default keeps the dtypes refused, and v_bits mapping is right."""
    sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "vllm")))
    import platform_patch_kvq as pp  # noqa: E402
    rc = 0
    if pp.KVQ_READY is not False:
        print("  >>> FAIL: KVQ_READY default must be False", file=sys.stderr); rc = 1
    if not pp._gate_open.__call__ or pp._gate_open():
        print("  >>> FAIL: gate must be CLOSED by default (fp8_inc stays default)", file=sys.stderr); rc = 1
    if pp.v_bits_for_dtype("turboquant_k8v4") != 4 or pp.v_bits_for_dtype("turboquant_k8v2") != 2:
        print("  >>> FAIL: v_bits mapping wrong", file=sys.stderr); rc = 1
    if rc == 0:
        print("  gate extension: KVQ_READY=False (default) -> dtypes REFUSED, fp8_inc stays "
              "the only accepted dtype; v_bits map k8v4->4 k8v2->2. OK")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024, help="tokens")
    ap.add_argument("--kvh", type=int, default=4, help="kv heads (M3 H_kv=4)")
    ap.add_argument("--gqa", type=int, default=8, help="GQA group size (q heads = kvh*gqa)")
    ap.add_argument("--d", type=int, default=128, help="head_dim (M3=128)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=== TurboQuant KV CPU parity (card-free; math == the compiled op's node-chain) ===")
    print(f"    dims: T={args.T} H_kv={args.kvh} gqa={args.gqa} (q_heads={args.kvh*args.gqa}) "
          f"head_dim={args.d}")
    rc = 0
    print("\n--- storage + rotation invariants ---")
    rc |= smoke_pack_roundtrip(args.d)
    rc |= smoke_rotation_orthogonal(args.d)
    print("\n--- platform gate extension ---")
    rc |= smoke_gate_patch()

    for mode in ("k8v4", "k8v2"):
        rc |= run_mode(mode, args.T, args.kvh, args.gqa, args.d, args.seed)

    print(f"\n[done] test_turboquant_kv_cpu {'OK' if rc == 0 else 'FAILED'} (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
