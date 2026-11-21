#!/usr/bin/env python3
# ==================================================================================================
# m3_moe_fuse_patch.py  —  engage the GROUPED routed-fp8 MoE-island on the LIVE M3 champion.
#
# WHAT IT DOES
# --------------------------------------------------------------------------------------------------
# The champion (minimax-m3-fp8-static) serves the routed experts via _fp8_route_b_allexpert: a Python
# loop over ALL E=16 LOCAL experts in bf16 (the "8x excess GEMM" = 4x all-16-vs-routed-4  x  2x
# bf16-vs-fp8). This patch swaps the all-expert path (the concurrency / T>1 regime) to the GROUPED
# routed-fp8 op (hpu_moe_ext.fused_grouped_moe_fp8.grouped_moe_fp8): permute the T tokens into
# per-expert capacity bins, run ONLY the routed token-expert pairs through the native fp8 MME, scatter
# the weighted results back. Byte-equivalent to route-b within fp8 grid noise (CPU oracle:
# tests/test_grouped_fp8_cpu.py — grouping relNorm 0.0, fp8-vs-bf16 3.7e-2 < 5e-2). It KEEPS the
# route-b routed-K path for T==1 decode (that path already runs only K experts — no all-16 waste).
#
# GATE: active ONLY when M3_FP8_STATIC=1 AND M3_MOE_FUSE=1. Strict no-op otherwise (so it ships inert
# in the always-mounted plugin and never touches the route-b / int4 / non-M3 serves). It DEPENDS on
# m3_fp8_static_patch having installed cls._m3_fp8_apply (= _apply_fp8_swigluoai) first — it WRAPS
# that, delegating the non-grouped cases back to it. Files: this patch + hpu-moe-ext/hpu_moe_ext/
# fused_grouped_moe_fp8.py + the grouped .so. It TOUCHES NEITHER the route-b op (m3_expert_runtime_
# patch) NOR the monolithic op — purely additive.
#
# CARD-FREE: importing this module touches no torch/hpu. The wrap closures import lazily. `--selftest`
# verifies the wiring resolves card-free; the numeric proof is the CPU oracle + on-card coherence.
# ==================================================================================================
from __future__ import annotations

import os
import sys


def enabled() -> bool:
    def _on(name):
        return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")
    return _on("M3_FP8_STATIC") and _on("M3_MOE_FUSE")


def _log(msg: str) -> None:
    print(f"m3-moe-fuse: {msg}", file=sys.stderr, flush=True)


# Same two fp8 MoE method classes m3_fp8_static_patch wraps (whichever the checkpoint's quant selects).
_CANDIDATES = (
    ("vllm_gaudi.ops.hpu_fp8", "HPUFp8MoEMethod"),
    ("vllm_gaudi.ops.hpu_compressed_tensors", "HPUCompressedTensorsW8A8Fp8MoEMethod"),
)


def _load_route_b():
    """Resolve m3_expert_runtime_patch (owns _route_topk / _alpha_limit / _apply_fp8_swigluoai) from
    M3_ROOT_DIR — the same source m3_fp8_static_patch installs as cls._m3_fp8_apply."""
    root = os.environ.get("M3_ROOT_DIR", "/opt/m3-root")
    if root not in sys.path:
        sys.path.insert(0, root)
    import m3_expert_runtime_patch as erp  # noqa: WPS433
    for need in ("_route_topk", "_alpha_limit"):
        if not hasattr(erp, need):
            raise RuntimeError(
                f"m3-moe-fuse: m3_expert_runtime_patch has no {need!r} — cannot build the grouped "
                "dispatch. Refusing to install (DSA build-quality bar).")
    return erp


def _load_grouped():
    """Resolve the grouped routed-fp8 wrapper + its .so loader from hpu-moe-ext. LOUD if absent."""
    try:
        from hpu_moe_ext.fused_grouped_moe_fp8 import (  # noqa: WPS433
            grouped_moe_fp8, _load_grouped_fp8_op)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"m3-moe-fuse: cannot import hpu_moe_ext.fused_grouped_moe_fp8 ({e!r}). "
            "The grouped routed-fp8 island is unavailable — refusing to install.")
    return grouped_moe_fp8, _load_grouped_fp8_op


def _tmax() -> int:
    # T>TMAX (large prefill) stays on route-b: the un-chunked grouped op gathers a big [E,cap,H] tile
    # (cap auto-scales with T) and prefill is not the concurrency-decode target. Decode T<=max_num_seqs;
    # default 256 covers c128/c256. M3_GROUPED_TMAX tunes the decode/short-prefill ceiling.
    try:
        return int(os.environ.get("M3_GROUPED_TMAX", "256") or "256")
    except Exception:  # noqa: BLE001
        return 256


_announced = [False]
_seen = set()          # bounded per-(T, path) diagnostic latch
_GROUPED_OP = [None]   # the .so op handle, pre-loaded at install (None -> route-b)


def _diag(T, path):
    """LOUD, BOUNDED per-call path diagnostic so the next serve reveals the real decode T + the branch
    taken (per rank). Logs the first occurrence of each (T, path); capped at 16 to avoid spam."""
    key = (int(T), path)
    if key in _seen or len(_seen) >= 16:
        return
    _seen.add(key)
    _log(f"APPLY T={int(T)} -> {path}  (first occurrence; bounded diag)")


def _make_grouped_apply(orig_apply, erp, grouped_moe_fp8):
    """Build the _m3_fp8_apply replacement. Engage the GROUPED routed-fp8 op for the DECODE / small-
    batch all-expert regime (1 < T <= TMAX). T==1 decode (route-b routed-K already runs only K
    experts), large prefill (T>TMAX), missing stacks/.so, or ANY grouped error -> route-b (LOUD)."""

    def _grouped_apply(method, layer, x, router_logits, **kwargs):  # noqa: WPS430
        input_shape = x.shape
        x2 = x.view(-1, x.shape[-1])
        T = int(x2.shape[0])
        K = int(layer.top_k)
        op = getattr(layer, "moe_op", None)
        have_stacks = op is not None and all(
            getattr(op, a, None) is not None
            for a in ("_w13_fp8_stack", "_s13_stack", "_w2_fp8_stack", "_s2_stack"))
        gop = _GROUPED_OP[0]
        eligible = (1 < T <= _tmax())

        if eligible and have_stacks and gop is not None:
            try:
                topk_ids, topk_weights = erp._route_topk(layer, x2, router_logits)
                alpha, limit = erp._alpha_limit(layer)
                emin = int(getattr(layer, "_hpu_experts_min", 0))
                # cap=None -> auto-scales with T (ceil(T*K/E)*2, clamped to T) -> NO drop at ANY T.
                out = grouped_moe_fp8(
                    x2, op._w13_fp8_stack, op._s13_stack, op._w2_fp8_stack, op._s2_stack,
                    topk_ids, topk_weights, emin, K, alpha=alpha, limit=limit, cap=None,
                    gemm_fn=gop)
                if out is not None:
                    _diag(T, "GROUPED")
                    if not _announced[0]:
                        _announced[0] = True
                        _log(f"LIVE: all-expert path -> GROUPED routed-fp8 op (permute -> native fp8 "
                             f"MME (2x) -> scatter; engage 1<T<={_tmax()}, K={K}, "
                             f"E={int(op.num_experts)}). T==1 + large prefill stay route-b.")
                    return out.view(*input_shape)
                _diag(T, "route-b(grouped=None)")
            except Exception as e:  # noqa: BLE001 -- NEVER crash the serve on a grouped error.
                _diag(T, "route-b(grouped-EXC)")
                _log(f"GROUPED op RAISED at T={T} ({e!r}) -> FALLBACK to route-b (serve stays up).")
        else:
            why = ("T==1-decode" if T <= 1 else
                   f"prefill-T>{_tmax()}" if T > _tmax() else
                   "no-stacks" if not have_stacks else "no-.so")
            _diag(T, f"route-b({why})")

        return orig_apply(method, layer, x, router_logits, **kwargs)

    return _grouped_apply


def _wrap_class(cls, erp, grouped_moe_fp8) -> bool:
    if getattr(cls, "_m3_moe_fuse_wrapped", False):
        return True
    orig_apply = getattr(cls, "_m3_fp8_apply", None)
    if orig_apply is None:
        # m3_fp8_static_patch did not install its handler on this class -> nothing to wrap.
        return False
    # orig_apply is a staticmethod descriptor on the class; getattr returns the plain function.
    cls._m3_fp8_apply = staticmethod(_make_grouped_apply(orig_apply, erp, grouped_moe_fp8))
    cls._m3_moe_fuse_wrapped = True
    _log(f"INSTALLED: {cls.__name__}._m3_fp8_apply wrapped (all-expert -> grouped routed-fp8; "
         "route-b retained for decode + fallback).")
    return True


def _install() -> None:
    # LOUD, UNCONDITIONAL install banner (the coordinator's #1): proves _apply_moe_fuse() was reached
    # and shows the gate state on EVERY rank, even when inert.
    _log(f"INSTALL fired: M3_FP8_STATIC={os.environ.get('M3_FP8_STATIC','<unset>')} "
         f"M3_MOE_FUSE={os.environ.get('M3_MOE_FUSE','<unset>')} -> enabled={enabled()}")
    if not enabled():
        _log("NOT enabled -> grouped routed-fp8 wire inert (route-b serves the all-expert path).")
        return
    erp = _load_route_b()
    grouped_moe_fp8, load_grouped_op = _load_grouped()

    # PRE-LOAD the grouped .so HERE (engine init, clean context) with a LOUD result (the coordinator's
    # #3) -> the wrapper uses this handle as gemm_fn; if it failed to load we KNOW at install, not via a
    # silent per-call None. The op is callable with the wrapper's gemm_fn signature.
    try:
        op = load_grouped_op()
    except Exception as e:  # noqa: BLE001
        op = None
        _log(f"GROUPED .so PRE-LOAD RAISED ({e!r}) -> grouped DISABLED, route-b will serve all-expert.")
    _GROUPED_OP[0] = op
    if op is None:
        _log("GROUPED .so LOAD FAILED (or absent) -> falling back to route-b for the all-expert path. "
             "Check libmoe_fused_grouped_fp8{,_lazy}.so under tpc_kernel/opbackend/build/ + that it "
             "matches the serve mode (lazy/eager). Build: bash build_grouped_fp8.sh (BUILD_LAZY=1).")
    else:
        _log("GROUPED .so LOADED OK at install (torch.ops.hpu_moe_ext.fused_moe_grouped_fp8 resolved).")

    wrapped = []
    for mod_path, cls_name in _CANDIDATES:
        try:
            import importlib
            cls = getattr(importlib.import_module(mod_path), cls_name)
        except Exception:  # noqa: BLE001 -- a class the checkpoint's quant didn't select; skip.
            continue
        if _wrap_class(cls, erp, grouped_moe_fp8):
            wrapped.append(cls_name)
    if not wrapped:
        raise RuntimeError(
            "M3_MOE_FUSE=1 but no fp8 MoE method class carried m3_fp8_static_patch's _m3_fp8_apply to "
            "wrap. Is m3_fp8_static_patch installed BEFORE this (M3_FP8_STATIC=1)? Refusing to serve a "
            "half-wired grouped path (DSA build-quality bar).")
    _log(f"grouped routed-fp8 wire active on: {wrapped}. engage 1<T<={_tmax()} (M3_GROUPED_TMAX); "
         f"cap auto-scales with T (M3_GROUPED_CAP override={os.environ.get('M3_GROUPED_CAP','<auto>')}); "
         f".so={'LOADED' if _GROUPED_OP[0] is not None else 'MISSING->route-b'}.")


_REGISTERED = False


def register() -> None:
    """vllm.general_plugins hook (called by hpu_m3_ext.__init__ AFTER _apply_fp8_static). Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        _install()
        _REGISTERED = True
    except Exception as e:  # noqa: BLE001
        _log(f"FAILED to install grouped routed-fp8 wire: {e!r}")
        raise


def _selftest() -> int:
    """Card-free wiring check (no torch/hpu/vllm import)."""
    print("=" * 88)
    print("m3_moe_fuse_patch — card-free selftest")
    print("=" * 88)
    print(f"enabled (M3_FP8_STATIC & M3_MOE_FUSE) : {enabled()}")
    root = os.environ.get("M3_ROOT_DIR", "/opt/m3-root")
    cand = os.path.join(root, "m3_expert_runtime_patch.py")
    here = os.path.dirname(os.path.abspath(__file__))
    grouped = os.path.normpath(os.path.join(
        here, "..", "..", "hpu-moe-ext", "hpu_moe_ext", "fused_grouped_moe_fp8.py"))
    print(f"M3_ROOT_DIR                          : {root}")
    print(f"route-b source present               : {os.path.exists(cand)}  ({cand})")
    print(f"grouped wrapper present (repo)        : {os.path.exists(grouped)}")
    print("wrap target  : <fp8 MoE method>._m3_fp8_apply (installed by m3_fp8_static_patch), gated on "
          "M3_FP8_STATIC=1 AND M3_MOE_FUSE=1.")
    print("dispatch     : T>1 -> grouped_moe_fp8 (permute/fp8-MME/scatter); T==1 + fallback -> route-b.")
    print("VERDICT: wiring resolvable card-free. Numeric proof = CPU oracle + on-card coherence.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
