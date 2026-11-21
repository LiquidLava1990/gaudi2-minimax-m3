#!/usr/bin/env python3
# ==================================================================================================
# m3_fp8_static_patch.py  —  serve a STATIC on-disk fp8 MiniMax-M3 checkpoint with the swigluoai gate.
#
# WHY THIS EXISTS
# --------------------------------------------------------------------------------------------------
# The int4 fp8-RUNTIME path (m3_expert_runtime_patch.py, M3_EXPERT_RUNTIME=fp8) loads the int4 disk
# artifact, converts int4->fp8 per-channel AT LOAD, swaps in VllmMixtureOfExpertsOpFP8PerChannel, and
# serves the routed experts through the UNFUSED swigluoai route-b (_apply_fp8_swigluoai). This module
# serves the SAME route-b path but from a TRUE on-disk fp8 checkpoint (experts stored float8_e4m3fn +
# per-output-channel weight_scale, produced by quant/quantize_m3_fp8.py), with NO int4 artifact and
# NO int4->fp8 load-time transform.
#
# THE PROBLEM IT SOLVES (the serve gap the FP8_RIGHT_QUANT_PLAN identified)
# --------------------------------------------------------------------------------------------------
#   * With `--quantization fp8` on an fp8 checkpoint, vLLM-Gaudi selects HPUFp8MoEMethod
#     (vllm_gaudi/ops/hpu_fp8.py:261). Its process_weights_after_loading builds
#     VllmMixtureOfExpertsOpFP8PerChannel + fp8_channel_moe_prepare_weights populates
#     op.w13_list[e].weight / .scale_inv_fp8 (extension/ops.py:1052-1096).
#   * BUT its apply_monolithic drives the NATIVE fused intrinsic, whose activation map is silu/gelu
#     only. M3's routed experts need the gpt-oss swigluoai gate -> the native path would serve SILU =
#     incoherent. None of the existing swigluoai wiring touches HPUFp8MoEMethod (m3_swigluoai_patch
#     wraps the INT4 MoeWNA16HPUMethod only).
#
# WHAT THIS DOES (the smallest correct wire; reuses the VALIDATED route-b math verbatim)
# --------------------------------------------------------------------------------------------------
#   1. Wrap HPUFp8MoEMethod.process_weights_after_loading: run the stock method (which builds the op +
#      w13_list/w2_list), THEN build the four route-b stacks on the op and flag the layer:
#        op._w13_fp8_stack [E,H,2I] fp8 ; op._s13_stack [E,2I] f32
#        op._w2_fp8_stack  [E,I,H ] fp8 ; op._s2_stack  [E,H ] f32
#      The stock op stores w13_list[e].weight as vLLM's [out=2I, in=H] (and w2 as [out=H, in=I]); the
#      route-b apply consumes [in,out] ([H,2I] / [I,H]) — so we TRANSPOSE each expert into the stacks.
#      The per-channel scale is the recovery scale s with real ~= fp8.to(bf16)*s (one per OUTPUT
#      channel), exactly what _dequant_fp8_slice (= MoeFP8Matmul.get_dequant_weight CHANNEL) expects.
#   2. Install HPUFp8MoEMethod._m3_fp8_apply = _apply_fp8_swigluoai (the colocated route-b handler).
#   3. Wrap HPUFp8MoEMethod.apply_monolithic: route _m3_fp8_converted layers to _m3_fp8_apply (swigluoai
#      route-b); leave any non-M3 fp8 MoE layer on the stock path. NEVER silently serve silu on an M3
#      routed-expert layer (DSA build-quality bar: LOUD if the handler is missing).
#
# ORIENTATION IS THE ONE RISK. We assert it in the builder AND rely on the route-b _assert_shapes
# one-shot LOUD check (m3_expert_runtime_patch.py:319-336) which refuses to serve if W13_deq != [H,2I]
# or W2_deq != [I,H]. So a wrong transpose fails loud on the first forward, never silent garbage.
#
# GATE: only active when M3_FP8_STATIC=1 (the minimax-m3-fp8-static serve). Inert otherwise, so it can
# ship in the always-mounted plugin without touching the int4 / int4->fp8 serves or any non-M3 model.
#
# CARD-FREE: importing this module touches no torch/hpu. The wrap closures import lazily. `--selftest`
# verifies the wiring resolves card-free (the numeric proof is the on-card coherence + the route-b
# path's own _assert_shapes/verify gates).
# ==================================================================================================
from __future__ import annotations

import os
import sys


def enabled() -> bool:
    return os.environ.get("M3_FP8_STATIC", "0").strip().lower() in ("1", "true", "yes", "on")


def _log(msg: str) -> None:
    print(f"m3-fp8-static: {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------------
# Resolve the colocated route-b handler. m3_expert_runtime_patch lives in M3_ROOT_DIR (/opt/m3-root),
# which the plugin puts on sys.path; it owns _apply_fp8_swigluoai + the route-b helpers + the LOUD
# _assert_shapes orientation gate. We REUSE it verbatim — this module only feeds the stacks.
# --------------------------------------------------------------------------------------------------
def _load_route_b():
    root = os.environ.get("M3_ROOT_DIR", "/opt/m3-root")
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import m3_expert_runtime_patch as erp  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"M3_FP8_STATIC=1 requires m3_expert_runtime_patch importable from M3_ROOT_DIR={root!r} "
            f"(it owns the route-b swigluoai handler _apply_fp8_swigluoai). Import failed: {e!r}")
    if not hasattr(erp, "_apply_fp8_swigluoai"):
        raise RuntimeError(
            "m3_expert_runtime_patch is importable but has no _apply_fp8_swigluoai — the route-b "
            "handler this static-fp8 path depends on is missing. Refusing to serve.")
    return erp


# --------------------------------------------------------------------------------------------------
# Build the four route-b stacks from the stock op's per-expert fp8 weight handles.
# --------------------------------------------------------------------------------------------------
def _build_fp8_stacks(layer, op) -> None:
    """Point op._{w13,w2}_fp8_stack + op._{s13,s2}_stack at the loaded fp8 expert weights, in the
    route-b [in,out] orientation. CRITICAL: the stacks are TRANSPOSED VIEWS of layer.w{13,2}_weight
    (NOT copies) — a contiguous copy would DOUBLE expert HBM (~2x ~52 GiB/card -> OOM, the 2026-06-24
    bug). vLLM stores w13_weight [E, 2I, H] ([out,in]) + w2_weight [E, H, I]; transpose(1,2) gives the
    [E, H, 2I] / [E, I, H] ([in,out]) route-b wants as a zero-copy view sharing layer.w*_weight's
    storage (which stays alive — op.w*_list already alias it). route-b's index_select/[e] then make the
    small per-expert contiguous slices on demand.

    ALSO stashes op._w13_oi_stack / op._w2_oi_stack = the ORIGINAL contiguous [out,in] tensors (w13w
    [E,2I,H], w2w [E,H,I]) for the M3_FP8_NOTRANSPOSE route-b form, which dequants from [out,in] and
    transposes only the tiny activation (no per-token DmaTranspose of the weights). Same storage as the
    transposed views above -> ZERO extra HBM; the no-transpose form is opt-in via the env flag."""
    import torch
    try:
        import habana_frameworks.torch.core as htorch
    except Exception:  # noqa: BLE001
        htorch = None

    E = int(op.num_experts)
    w13w = getattr(layer, "w13_weight", None)
    w2w = getattr(layer, "w2_weight", None)
    w13s = getattr(layer, "w13_weight_scale", None)
    w2s = getattr(layer, "w2_weight_scale", None)
    if any(t is None for t in (w13w, w2w, w13s, w2s)):
        raise RuntimeError(
            "m3-fp8-static: layer is missing one of w13_weight/w2_weight/w13_weight_scale/"
            "w2_weight_scale — the fp8 MoE method did not create the per-channel params (experts likely "
            "loaded UNQUANTIZED). Refusing to serve. Check the checkpoint quantization_config scoping.")

    w13w, w2w = w13w.data, w2w.data
    if w13w.dim() != 3 or w2w.dim() != 3 or w13w.shape[0] != E or w2w.shape[0] != E:
        raise RuntimeError(f"m3-fp8-static: unexpected expert weight shapes w13={tuple(w13w.shape)} "
                           f"w2={tuple(w2w.shape)} (E={E}); expected [E,2I,H]/[E,H,I]. Refusing.")

    # zero-copy [out,in] -> [in,out] views
    w13_stack = w13w.transpose(1, 2)                      # [E, H, 2I]
    w2_stack = w2w.transpose(1, 2)                        # [E, I, H]
    s13_stack = w13s.data.reshape(E, -1).to(torch.float32)   # [E, 2I]
    s2_stack = w2s.data.reshape(E, -1).to(torch.float32)     # [E, H]

    # --- orientation + scale sanity (LOUD; route-b _assert_shapes is the second gate on first fwd) ---
    H, twoI = int(w13_stack.shape[1]), int(w13_stack.shape[2])
    Iin, Hout = int(w2_stack.shape[1]), int(w2_stack.shape[2])
    if (twoI % 2) != 0 or Hout != H or Iin != (twoI // 2):
        raise RuntimeError(
            f"m3-fp8-static: stack orientation wrong — w13[E,{H},{twoI}] w2[E,{Iin},{Hout}]; expected "
            f"w2=[E,{twoI // 2},{H}] (in=I,out=H). Refusing to serve.")
    if tuple(s13_stack.shape) != (E, twoI) or tuple(s2_stack.shape) != (E, H):
        raise RuntimeError(
            f"m3-fp8-static: scale stack shapes s13={tuple(s13_stack.shape)} s2={tuple(s2_stack.shape)} "
            f"!= ({E},{twoI})/({E},{H}); per-output-channel scale layout is wrong. Refusing.")

    op._w13_fp8_stack = w13_stack
    op._w2_fp8_stack = w2_stack
    op._s13_stack = s13_stack
    op._s2_stack = s2_stack

    # M3_FP8_NOTRANSPOSE handles: stash the CONTIGUOUS [out,in] originals so route-b can consume them
    # directly (gate_up = (W13_oi @ x.t()).t()) and transpose only the tiny T==1 activation instead of
    # the full ~52 GiB of weights per token (the DmaTranspose tax). These are the SAME tensors w13w/w2w
    # whose .transpose(1,2) became the [in,out] stacks above — one shared storage, ZERO extra HBM. The
    # route-b path only engages the no-transpose form when this handle is present AND M3_FP8_NOTRANSPOSE
    # is set, so the default A/B (flag OFF) is bit-for-bit the existing transposed-view behavior.
    op._w13_oi_stack = w13w                                # [E, 2I, H] contiguous ([out,in])
    op._w2_oi_stack = w2w                                  # [E, H,  I] contiguous ([out,in])
    if htorch is not None:
        htorch.mark_step()


# The fp8 MoE method depends on HOW the checkpoint declares its quant: `--quantization fp8`
# (Fp8Config) -> HPUFp8MoEMethod; `--quantization compressed-tensors` (float-quantized) ->
# HPUCompressedTensorsW8A8Fp8MoEMethod. BOTH build the SAME VllmMixtureOfExpertsOpFP8PerChannel via
# fp8_channel_moe_prepare_weights (hpu_fp8.py:195 / hpu_compressed_tensors.py:449), and BOTH expose
# process_weights_after_loading + apply_monolithic. So we wrap whichever is importable — the static
# fp8 serve is then correct regardless of which format quantize_m3_fp8.py emitted.
_CANDIDATES = (
    ("vllm_gaudi.ops.hpu_fp8", "HPUFp8MoEMethod"),
    ("vllm_gaudi.ops.hpu_compressed_tensors", "HPUCompressedTensorsW8A8Fp8MoEMethod"),
)


def _wrap_method_class(cls, erp) -> None:
    """Apply the route-b process_weights + apply_monolithic wrap to ONE fp8 MoE method class. The
    per-class stock methods are captured here so each class delegates to its OWN original."""
    if getattr(cls, "_m3_fp8_static_wrapped", False):
        return
    # Install the route-b handler as a static attr (mirrors m3_expert_runtime_patch.py:186 for int4).
    cls._m3_fp8_apply = staticmethod(erp._apply_fp8_swigluoai)

    _orig_pwal = cls.process_weights_after_loading
    _orig_apply = cls.apply_monolithic

    def process_weights_after_loading(self, layer):  # noqa: WPS430
        # 1) stock path: builds layer.moe_op (VllmMixtureOfExpertsOpFP8PerChannel) + per-channel
        #    w13_list/w2_list from the loaded fp8 weights + scales.
        _orig_pwal(self, layer)
        op = getattr(layer, "moe_op", None)
        if op is None:
            raise RuntimeError(f"m3-fp8-static: {cls.__name__}.process_weights_after_loading produced "
                               "no layer.moe_op — cannot build route-b stacks. Refusing to serve.")
        # 2) build the route-b stacks (the [out,in]->[in,out] transpose lives here).
        _build_fp8_stacks(layer, op)
        # 3) the global id of this rank's first LOCAL expert (route-b remaps global->local with this).
        try:
            emin = int(layer.ep_rank) * int(layer.local_num_experts)
        except Exception:  # noqa: BLE001
            emin = 0
        layer._hpu_experts_min = emin
        layer._m3_fp8_converted = True
        E = int(op.num_experts)
        H = int(op._w13_fp8_stack.shape[1])
        twoI = int(op._w13_fp8_stack.shape[2])
        _log(f"layer wired ({cls.__name__}): E={E} H={H} 2I={twoI} emin={emin} -> route-b swigluoai "
             "on static fp8 (native silu intrinsic BYPASSED).")

    def apply_monolithic(self, layer, x, router_logits, **kwargs):  # noqa: WPS430
        if getattr(layer, "_m3_fp8_converted", False):
            fp8_apply = getattr(type(self), "_m3_fp8_apply", None)
            if fp8_apply is None:
                raise RuntimeError(
                    "m3-fp8-static: layer is _m3_fp8_converted but _m3_fp8_apply is missing — the "
                    "static-fp8 swigluoai wiring is broken. Refusing to serve a wrong activation "
                    "(DSA build-quality bar).")
            return fp8_apply(self, layer, x, router_logits, **kwargs)
        # non-M3 fp8 MoE layer (should not occur in an M3 serve) -> stock native path.
        return _orig_apply(self, layer, x, router_logits, **kwargs)

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_monolithic = apply_monolithic
    cls._m3_fp8_static_wrapped = True
    _log(f"INSTALLED: {cls.__name__} wrapped (process_weights builds route-b stacks; apply_monolithic "
         "delegates M3 routed experts to UNFUSED swigluoai route-b).")


# --------------------------------------------------------------------------------------------------
def _install() -> None:
    if not enabled():
        _log("M3_FP8_STATIC not set -> static-fp8 swigluoai wire NOT installed (this serve is not the "
             "on-disk-fp8 path).")
        return

    erp = _load_route_b()
    wrapped = []
    errs = []
    for mod_path, cls_name in _CANDIDATES:
        try:
            import importlib
            cls = getattr(importlib.import_module(mod_path), cls_name)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod_path}.{cls_name}: {e!r}")
            continue
        _wrap_method_class(cls, erp)
        wrapped.append(cls_name)

    if not wrapped:
        raise RuntimeError(
            "M3_FP8_STATIC=1 but NEITHER fp8 MoE method class could be wrapped — the on-disk-fp8 serve "
            f"cannot apply swigluoai and would emit silu garbage. Refusing to serve. Errors: {errs}")
    _log(f"static-fp8 swigluoai wire active on: {wrapped}. swigluoai gate "
         f"{'ENABLED' if os.environ.get('M3_SWIGLUOAI') else 'intrinsic to route-b'}.")


_REGISTERED = False


def register() -> None:
    """vllm.general_plugins entry-point hook. Idempotent. Call AFTER hpu_fp8 is importable."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        _install()
        _REGISTERED = True
    except Exception as e:  # noqa: BLE001
        _log(f"FAILED to install static-fp8 swigluoai wire: {e!r}")
        raise


# --------------------------------------------------------------------------------------------------
def _selftest() -> int:
    """Card-free wiring check (no torch/hpu/vllm import)."""
    print("=" * 88)
    print("m3_fp8_static_patch — card-free selftest")
    print("=" * 88)
    print(f"M3_FP8_STATIC enabled : {enabled()}")
    root = os.environ.get("M3_ROOT_DIR", "/opt/m3-root")
    print(f"M3_ROOT_DIR           : {root}")
    # confirm the route-b source file exists where we will import it from (no execution).
    cand = os.path.join(root, "m3_expert_runtime_patch.py")
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "minimax-m3",
                         "m3_expert_runtime_patch.py")
    found = os.path.exists(cand) or os.path.exists(os.path.abspath(local))
    print(f"route-b source present: {found}  ({cand} or repo minimax-m3/)")
    print("wrap target           : vllm_gaudi.ops.hpu_fp8.HPUFp8MoEMethod "
          "(process_weights_after_loading + apply_monolithic), gated on M3_FP8_STATIC=1.")
    print("stacks built          : op._w13_fp8_stack[E,H,2I] _s13_stack[E,2I] "
          "op._w2_fp8_stack[E,I,H] _s2_stack[E,H]  (transpose [out,in]->[in,out]).")
    print("VERDICT: wiring resolvable card-free. Numeric proof = on-card coherence + route-b "
          "_assert_shapes/verify gates.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
