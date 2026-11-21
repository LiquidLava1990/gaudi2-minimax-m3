"""Extend the vllm_gaudi platform KV-dtype gate so the MiniMax-M3 sub-fp8 TurboQuant
KV-cache dtypes ('turboquant_k8v4', 'turboquant_k8v2') become *acceptable* kv-cache
dtypes on HPU -- WITHOUT touching /opt or the installed plugins.

The stock gate (vllm_gaudi/platform.py:388, is_kv_cache_dtype_supported) returns
``kv_cache_dtype == "fp8_inc"`` -- fp8_inc is the ONLY accepted HPU KV dtype. This patch
monkeypatches that classmethod IN-PROCESS to ALSO accept the TurboQuant dtypes, gated
behind KVQ_READY (default False). It is the KV-dtype twin of hpu-dsa-ext's
platform_patch.apply() (which lifts the sparse-attention raise out-of-tree).

SAFETY (the PR #1301 / DSA_READY trap -- do NOT repeat it):
  * KVQ_READY defaults False. Until it is True, this patch is a NO-OP: the gate still
    returns ``== "fp8_inc"`` for everything, so fp8_inc remains the ONLY accepted dtype
    and the default M3 serve (M2_BASE_MAP.md §4 --kv-cache-dtype fp8_inc) is unchanged.
  * KVQ_READY flips True ONLY when HPU_KVQ_READY is truthy AND the two compiled ops
    (kv_rotate_quant + kv_dequant_unrotate) are loaded AND the cache-spec / attention
    wiring that actually stores+reads the TurboQuant KV is installed + on-card validated.
    Accepting the dtype at the gate WITHOUT that wiring would let vLLM allocate a KV pool
    in a dtype nothing knows how to write/read -> garbage. So the gate stays closed until
    the write/read path is proven, exactly as DSA_READY gates the sparse path.
  * Even with the dtype accepted, the model/runner seam must route KV writes through
    kv_rotate_quant and reads through kv_dequant_unrotate; that plumbing (the paged
    cache-spec extension to a packed layout) is the "hardest plumbing" item and is the
    real on-card gate (see RETURN notes / the design header). This patch ONLY lifts the
    dtype-acceptance gate; it does not itself wire the cache.

Out-of-tree: apply() saves the ORIGINAL classmethod, installs the wrapper, is idempotent.
A CPU-only import never imports vllm_gaudi (the import is inside apply()).
"""
from __future__ import annotations

import os

# The TurboQuant KV dtypes this package introduces. k8v4 == K fp8 / V int4 (the safe knee,
# default sub-fp8 mode); k8v2 == K fp8 / V int2 (the "lightest" 1M-fit mode, quality-risky
# per quant-kv-turboquant.md -- built + measured, never the default).
KVQ_DTYPES = ("turboquant_k8v4", "turboquant_k8v2")

# Set True ONLY when the compiled ops are loaded AND the TurboQuant KV write/read path is
# wired + validated on-card. Until then this patch is a NO-OP (fp8_inc stays the only
# accepted dtype). Mirrors platform_patch.DSA_READY / platform_patch_msa.MSA_READY.
KVQ_READY = False

# Exposed for the gate / tests (echoes whether the store+read wiring is installed). The
# dtype must NOT be accepted unless this is True (else vLLM allocates an unwritable pool).
WIRE_INSTALLED = False

_READY_ENV = "HPU_KVQ_READY"

# The original (or previously-chained) unbound is_kv_cache_dtype_supported captured at
# apply() time, so we chain through anything already installed.
_ORIG = None


def ready() -> bool:
    """True iff the operator opted into the (validated) live TurboQuant-KV flip via env."""
    return os.environ.get(_READY_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _gate_open() -> bool:
    """The dtype is accepted ONLY when the module gate is open AND the env opt-in is set
    AND the store/read wiring is installed -- all three. Defense in depth so a stray
    --kv-cache-dtype turboquant_k8v4 on a serve WITHOUT the kernels loaded still falls
    back to the stock 'fp8_inc only' refusal (loud, never silent garbage)."""
    return KVQ_READY and ready() and WIRE_INSTALLED


def _patched(cls, kv_cache_dtype, model_config) -> bool:
    """is_kv_cache_dtype_supported override: accept the TurboQuant dtypes IFF the gate is
    open; otherwise defer to the captured original (the stock 'fp8_inc only')."""
    if kv_cache_dtype in KVQ_DTYPES:
        if _gate_open():
            return True
        # Gate closed: refuse explicitly (do NOT silently accept an unwired dtype).
        return False
    return _ORIG(cls, kv_cache_dtype, model_config)


def apply() -> None:
    """Install the gate extension. Captures the CURRENT is_kv_cache_dtype_supported as the
    chain-through original, then binds our override. Idempotent. Out-of-tree (monkeypatch
    only; no /opt edit). Mirrors platform_patch.apply()."""
    global _ORIG
    from vllm_gaudi.platform import HpuPlatform

    cur = HpuPlatform.is_kv_cache_dtype_supported
    if getattr(cur, "_hpu_kvq_wired", False):
        return
    _ORIG = cur.__func__ if hasattr(cur, "__func__") else cur
    _patched._hpu_kvq_wired = True
    HpuPlatform.is_kv_cache_dtype_supported = classmethod(_patched)


def maybe_flip_ready() -> bool:
    """Flip KVQ_READY=True IFF HPU_KVQ_READY is truthy AND the write/read wiring is
    installed. Called LAST by the plugin register(), only after the compiled ops are
    loaded and the cache-spec/attention seam is bound. Until then (or without the env) the
    TurboQuant dtypes stay REFUSED and fp8_inc remains the only accepted dtype. Returns the
    resulting KVQ_READY value. Mirrors platform_patch_msa.maybe_flip_ready()."""
    global KVQ_READY
    if ready() and WIRE_INSTALLED:
        KVQ_READY = True
    return KVQ_READY


def v_bits_for_dtype(kv_cache_dtype: str) -> int:
    """Map an accepted TurboQuant dtype string to its V quant width (for the ops/runner)."""
    return {"turboquant_k8v4": 4, "turboquant_k8v2": 2}[kv_cache_dtype]
