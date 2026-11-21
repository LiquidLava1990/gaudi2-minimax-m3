"""Tests for the int4 -> fp8(E4M3) per-channel load-time transform (int4_to_fp8_load.py) and the
M3_EXPERT_RUNTIME switch (m3_expert_runtime_patch.py).

Two tiers:
  1. SYNTHETIC (always runs, CPU, no card, no checkpoint): the int4-pack -> int4->fp8-per-channel
     numerics, the per-channel scale shape, the int4 unpack faithfulness, and the runtime-switch
     wiring. This is the card-free proof the task asks for.
  2. END-TO-END (GATED): load the REAL int4 M3 checkpoint and convert one MoE layer's experts on a
     card. SKIPPED-LOUD unless the int4 checkpoint exists AND a card is free — neither is true while
     the download is mid-flight and the quant has not been run, so this tier prints exactly what
     remains gated and exits 0 (skip, not fail).

Runs as a plain script (matches the MSA test style; no pytest needed in the image):
    python tests/test_int4_to_fp8.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

import int4_to_fp8_load as i2f          # noqa: E402
import m3_expert_runtime_patch as switch  # noqa: E402

# The real int4 checkpoint the e2e tier is gated on (== quantize_m3_int4.py OUT_DIR).
INT4_CKPT = "/mnt/fastmodels/MiniMax-M3-int4-experts-AutoRound"

_fails = []


def _check(name: str, cond: bool, detail: str = ""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


# ==================================================================================================
def test_synthetic_numerics():
    print("-" * 88)
    print("TIER 1a — synthetic int4->fp8 per-channel numerics (CPU, card-free)")
    print("-" * 88)
    res = i2f.validate_synthetic(verbose=False)
    for name in ("w13(gate||up)", "w2(down)"):
        r = res[name]
        _check(f"{name}: int4 unpack == op dequant (bit-exact)",
               r["unpack_vs_op_dequant_maxabs"] == 0.0,
               f"maxabs={r['unpack_vs_op_dequant_maxabs']:.2e}")
        out = r["shape_in_out"][1]
        _check(f"{name}: per-channel scale is 1-D [out={out}]",
               r["scale_shape"] == (out,), f"got {r['scale_shape']}")
        _check(f"{name}: added err near fp8 floor (<=1.5x fresh-gaussian floor)",
               r["added_over_fp8_floor_ratio"] <= 1.5,
               f"ratio={r['added_over_fp8_floor_ratio']:.3f}x  added={r['added_rel_int4_vs_int4fp8']:.3e}")
        _check(f"{name}: added err < per-tensor double-quant floor 6.4e-2",
               r["added_rel_int4_vs_int4fp8"] < 6.4e-2,
               f"added={r['added_rel_int4_vs_int4fp8']:.3e}")
    s = res["_summary"]
    _check("overall synthetic verdict PASS", s["verdict"] == "PASS",
           f"worst added={s['worst_added_rel_int4_vs_int4fp8']:.3e}")


def test_perchannel_op_dequant_contract():
    """The native op's CHANNEL dequant is exactly weight_fp8 * scale_inv_fp8 (broadcast). Confirm our
    staged (weight, scale) reproduce the input bf16 within fp8 tolerance under THAT exact formula —
    this is what the op will compute (ops.py:1133-1135)."""
    print("-" * 88)
    print("TIER 1b — staged (weight, scale) obey MoeFP8Matmul.get_dequant_weight CHANNEL contract")
    print("-" * 88)
    torch.manual_seed(3)
    w_bf16 = (torch.randn(512, 1024) * 0.1).to(torch.bfloat16)   # [in, out], int4-dequant-like scale
    w_fp8, scale = i2f.quant_weight_fp8_per_channel(w_bf16)
    # EXACT op formula: real ~= weight.to(scale.dtype) * scale_inv_fp8  (scale [out] broadcasts on last dim)
    recon = (w_fp8.to(torch.float32) * scale.unsqueeze(0)).to(torch.bfloat16)
    rel = (recon.float() - w_bf16.float()).norm() / w_bf16.float().norm()
    _check("op-formula reconstruction within fp8 per-channel tol (<5e-2)", rel < 5e-2, f"rel={rel:.3e}")
    _check("scale dtype float32 (op casts weight to scale.dtype)", scale.dtype == torch.float32)
    _check("weight dtype is float8_e4m3fn (op operand)", w_fp8.dtype == torch.float8_e4m3fn)
    _check("weight keeps [in,out] orientation (same as the int4 op feeds the intrinsic)",
           tuple(w_fp8.shape) == (512, 1024), f"shape={tuple(w_fp8.shape)}")


def test_runtime_switch_wiring():
    print("-" * 88)
    print("TIER 1c — M3_EXPERT_RUNTIME switch wiring (card-free)")
    print("-" * 88)
    old = os.environ.get("M3_EXPERT_RUNTIME")
    try:
        os.environ.pop("M3_EXPERT_RUNTIME", None)
        _check("default (unset) resolves to int4", switch.expert_runtime() == "int4")
        os.environ["M3_EXPERT_RUNTIME"] = "fp8"
        _check("M3_EXPERT_RUNTIME=fp8 resolves to fp8", switch.expert_runtime() == "fp8")
        os.environ["M3_EXPERT_RUNTIME"] = "int4"
        _check("M3_EXPERT_RUNTIME=int4 resolves to int4", switch.expert_runtime() == "int4")
        os.environ["M3_EXPERT_RUNTIME"] = "bogus"
        raised = False
        try:
            switch.expert_runtime()
        except ValueError:
            raised = True
        _check("bad value raises (loud failure)", raised)
    finally:
        if old is None:
            os.environ.pop("M3_EXPERT_RUNTIME", None)
        else:
            os.environ["M3_EXPERT_RUNTIME"] = old


# ==================================================================================================
def test_e2e_real_checkpoint_gated():
    print("-" * 88)
    print("TIER 2 — END-TO-END on the REAL int4 checkpoint (GATED)")
    print("-" * 88)
    gates = []
    have_ckpt = os.path.isdir(INT4_CKPT) and bool(
        [f for f in (os.listdir(INT4_CKPT) if os.path.isdir(INT4_CKPT) else []) if f.endswith(".safetensors")])
    gates.append(("int4 checkpoint present", have_ckpt, INT4_CKPT))

    have_hpu = i2f._has_hpu_intrinsics()
    gates.append(("HPU intrinsics available (a free card)", have_hpu, "convert_from_uint4 / cast_to_fp8_v2"))

    vllm_active = False
    try:
        import subprocess
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=10)
        vllm_active = "vllm-active" in out.stdout.split()
    except Exception:
        pass
    gates.append(("vllm-active NOT running (cards free to take)", not vllm_active, "do not churn serving cards"))

    all_open = all(c for _, c, _ in gates)
    for name, ok, detail in gates:
        print(f"  [{'OPEN ' if ok else 'GATED'}] {name}" + (f"  ({detail})" if detail else ""))

    if not all_open:
        print("\n  -> TIER 2 SKIPPED (a gate is closed). This is EXPECTED while the bf16 download is")
        print("     incomplete and the int4 quant has not been run. What remains, in order:")
        print("       1. finish the bf16 download (/mnt/data2/minimax-m3-bf16, needs the index.json)")
        print("       2. run quant/quantize_m3_int4.py --run  -> writes", INT4_CKPT)
        print("       3. on a FREE card, serve with -e M3_EXPERT_RUNTIME=fp8 (this transform runs at load)")
        print("     The synthetic tiers above already prove the transform numerics + the switch wiring.")
        return  # SKIP, not a failure.

    # --- all gates open: actually convert one layer's experts on a card -----------------------------
    print("\n  All gates OPEN — running the real conversion on one synthetic-routed layer.")
    # (Intentionally minimal: build a tiny FusedMoE-like layer stub holding a FEW real experts read
    # from the checkpoint shards, run stage_layer_int4_to_fp8, and check the fp8 op produces finite
    # output within fp8 tolerance of the int4 op on a random token. Kept here as the on-card hook;
    # the heavy full-model serve check belongs to the serve smoke test, not this unit.)
    try:
        _run_oncard_onelayer()
        _check("on-card one-layer int4->fp8 convert + fp8-op forward finite & ~int4", True)
    except Exception as e:  # noqa: BLE001
        _check("on-card one-layer int4->fp8 convert", False, f"{e!r}")


def _run_oncard_onelayer():
    """Placeholder for the on-card path: when a card + checkpoint exist, load one MoE layer via the
    same hpu-moe-ext create_weights/process path, convert, and compare fp8-op vs int4-op output on a
    random token. Left as an explicit hook (raises NotImplementedError if reached without the full
    serve harness) so the gate logic above is the single source of truth for 'can we run e2e'."""
    raise NotImplementedError(
        "on-card e2e requires the full hpu-moe-ext layer-load harness; run via the serve smoke test "
        "(serve with M3_EXPERT_RUNTIME=fp8) once the checkpoint + a free card exist.")


# ==================================================================================================
if __name__ == "__main__":
    print("=" * 88)
    print("int4 -> fp8 transform + M3_EXPERT_RUNTIME switch — test suite")
    print("=" * 88)
    test_synthetic_numerics()
    test_perchannel_op_dequant_contract()
    test_runtime_switch_wiring()
    test_e2e_real_checkpoint_gated()
    print("=" * 88)
    if _fails:
        print(f"FAILED ({len(_fails)}): " + ", ".join(_fails))
        raise SystemExit(1)
    print("ALL CARD-FREE TESTS PASSED (TIER 2 gated/skipped as expected).")
    raise SystemExit(0)
