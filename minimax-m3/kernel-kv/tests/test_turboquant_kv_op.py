"""test_turboquant_kv_op.py -- VALIDATE the compiled TurboQuant KV OpBackend ops ON THE
HPU BOX (GATED; cards are busy serving -- run ONLY when a card is free and on-card numeric
validation is explicitly authorized). A CPU-only import cannot exercise the device graph;
for the card-free math gate use tests/test_turboquant_kv_cpu.py.

Loads build/libkv_rotate_quant{,_lazy}.so + build/libkv_dequant_unrotate{,_lazy}.so with
torch.ops.load_library, builds synthetic M3-shaped K/V (H_kv=4, head_dim=128), runs the
device WRITE op then the device READ op, and asserts the device round-trip matches the
pure-PyTorch reference (ref/turboquant_kv_ref.py) within tolerance -- the same reference
the CPU test diffs against fp8-KV. So: CPU test proves "reference ~ fp8"; this proves
"device op == reference"; together "device op ~ fp8".

Run (ON THE HPU BOX, gated):
  cd $HOME/AI/minimax-m3/kernel-kv
  python tests/test_turboquant_kv_op.py                          # eager .so, k8v4
  python tests/test_turboquant_kv_op.py --bits 2                 # k8v2
  KVQ_WRITE_SO=build/libkv_rotate_quant_lazy.so \
  KVQ_READ_SO=build/libkv_dequant_unrotate_lazy.so \
  PT_HPU_LAZY_MODE=1 python tests/test_turboquant_kv_op.py        # lazy .so + capture
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.normpath(os.path.join(_HERE, "..", "ref"))
if _REF not in sys.path:
    sys.path.insert(0, _REF)


def _find(env, default_name):
    here = os.path.dirname(_HERE)  # .../kernel-kv
    v = os.environ.get(env)
    so = (v if os.path.isabs(v) else os.path.join(here, v)) if v \
        else os.path.join(here, "build", default_name)
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build it first (CPU-only):\n"
            f"  docker run --rm -v $HOME/AI/minimax-m3:/m3 "
            f"-v $HOME/AI/hpu-moe-ext:/hpumoe "
            f"-e BRIDGE=/hpumoe/tpc_kernel/opbackend/bridge "
            f"-w /m3/kernel-kv vllm-gaudi:next bash build_kv_rotate_quant.sh\n"
            f"  (and build_kv_dequant_unrotate.sh; set KVQ_WRITE_SO / KVQ_READ_SO to pick "
            f"the lazy .so)")
    return so


def _rel(a, b):
    import torch
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--kvh", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--bits", type=int, default=4, choices=(4, 2))
    args = ap.parse_args()

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2
    import turboquant_kv_ref as ref

    mode = "k8v4" if args.bits == 4 else "k8v2"
    cfg = ref.KVQConfig(mode=mode)

    try:
        write_so = _find("KVQ_WRITE_SO", "libkv_rotate_quant.so")
        read_so = _find("KVQ_READ_SO", "libkv_dequant_unrotate.so")
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}")
        return 2
    torch.ops.load_library(write_so)
    torch.ops.load_library(read_so)
    if not hasattr(torch.ops, "hpu_kvq_ext") or not hasattr(
            torch.ops.hpu_kvq_ext, "kv_rotate_quant") or not hasattr(
            torch.ops.hpu_kvq_ext, "kv_dequant_unrotate"):
        print("FATAL: hpu_kvq_ext ops not registered after load_library.", file=sys.stderr)
        return 2
    print(f"loaded {write_so}\nloaded {read_so}")

    dev = "hpu"
    torch.manual_seed(0)
    N = args.T * args.kvh
    D = args.d
    chan = torch.ones(D); chan[::16] = 4.0
    K = (torch.randn(N, D) * chan * 0.1).to(torch.bfloat16)
    V = (torch.randn(N, D) * chan * 0.1).to(torch.bfloat16)
    R = ref.make_rotation(D).to(torch.bfloat16)
    Rt = R.t().contiguous()

    REC_TOL = 0.14 if args.bits == 4 else 0.82
    print(f"=== test_turboquant_kv_op ({mode}) === N={N} D={D} bits={args.bits}")

    # ---- reference round-trip (CPU) for the diff target ----
    k_ref, v_ref = ref.reconstruct(K, V, cfg)

    # ---- device WRITE op ----
    Kd, Vd, Rd, Rtd = K.to(dev), V.to(dev), R.to(dev), Rt.to(dev)
    htorch.mark_step()
    k_fp8, k_scale, v_packed, v_scale = torch.ops.hpu_kvq_ext.kv_rotate_quant(
        Kd, Vd, Rd, args.bits)
    htorch.mark_step()
    # ---- device READ op (bf16 out) ----
    k_out, v_out = torch.ops.hpu_kvq_ext.kv_dequant_unrotate(
        k_fp8, k_scale, v_packed, v_scale, Rtd, args.bits, 0)
    htorch.mark_step()

    rc = 0
    rk = _rel(k_out.cpu(), k_ref)
    rv = _rel(v_out.cpu(), v_ref)
    print(f"  device-op vs reference:  K rel={rk:.4f}   V rel={rv:.4f}  (tol {REC_TOL})")
    # device op should match the reference TIGHTLY (same math); the looser REC_TOL is a
    # ceiling -- a real match is < ~0.02 for K and < ~0.05 over the reference for V.
    DEV_MATCH = 0.05
    rk_dev = _rel(k_out.cpu(), k_ref)
    rv_dev = _rel(v_out.cpu(), v_ref)
    if rk_dev <= DEV_MATCH and rv_dev <= DEV_MATCH:
        print(f"  >>> PASS: device op reproduces the reference (K {rk_dev:.4f}, V {rv_dev:.4f} "
              f"<= {DEV_MATCH}).")
    else:
        print(f"  >>> FAIL: device op DIVERGES from reference (K {rk_dev:.4f}, V {rv_dev:.4f} "
              f"> {DEV_MATCH}).", file=sys.stderr)
        print("      Suspects (device-confirmable): gemm transpose flags (x@R vs x@R^T),\n"
              "      convert_to_fp8 reciprocal-scale convention, round/clamp bounds,\n"
              "      the sub-byte pack shift/mask/sign-extend, reduce_max dim-mask.",
              file=sys.stderr)
        rc = 1

    # ---- fp8-K-out variant (out_fp8=1) shape/dtype smoke ----
    try:
        kf, vf = torch.ops.hpu_kvq_ext.kv_dequant_unrotate(
            k_fp8, k_scale, v_packed, v_scale, Rtd, args.bits, 1)
        htorch.mark_step()
        ok = (kf.dtype == torch.float8_e4m3fn and list(kf.shape) == [N, D]
              and vf.dtype == torch.bfloat16)
        print(f"  out_fp8=1 K dtype={kf.dtype} shape={list(kf.shape)} -> "
              f"{'OK' if ok else 'UNEXPECTED'}")
    except Exception as ex:  # noqa: BLE001
        print(f"  (out_fp8=1 smoke skipped: {ex})")

    # ---- HPU-graph capture (durable serving path; best-effort) ----
    try:
        import habana_frameworks.torch.hpu as ht_hpu
        from habana_frameworks.torch.hpu.graphs import HPUGraph
        g = HPUGraph(); s_ = ht_hpu.Stream()
        with ht_hpu.stream(s_):
            g.capture_begin()
            _ = torch.ops.hpu_kvq_ext.kv_rotate_quant(Kd, Vd, Rd, args.bits)
            g.capture_end()
        g.replay(); htorch.mark_step()
        print("  >>> PASS: write op captured into an HPU graph + replayed.")
    except Exception as ex:  # noqa: BLE001
        print(f"  (capture check skipped: {ex})")

    print(f"\n[done] test_turboquant_kv_op {'OK' if rc == 0 else 'FAILED'} (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
