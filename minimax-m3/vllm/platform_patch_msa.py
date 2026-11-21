"""Lift the vllm_gaudi platform.py hard-raise on sparse attention so MiniMax-M3 MSA
(GQA-sparse, NOT MLA) can route to the standard HPU attention backend.

This is the MSA twin of hpu-dsa-ext/hpu_dsa_ext/platform_patch.py. The DSA patch lifts the raise
for the ``use_sparse AND use_mla`` branch (MLA-sparse); M3 is ``use_sparse AND NOT use_mla``
(GQA-sparse), a DIFFERENT branch, so this patch handles exactly that case and leaves the MLA-sparse
branch to DSA (chaining through whatever ``get_attn_backend_cls`` is already installed, so the two
plugins co-exist; see ``apply``).

SAFETY (the PR #1301 trap — do NOT repeat it): until MSA_READY is True we KEEP refusing GQA-sparse
with a clear message, so an M3 model stays correctly BLOCKED rather than silently served DENSE
through a half-wired sparse path. Independently, minimax_m3.py's seam keeps MSA_FULL_ATTENTION_
FALLBACK=True so even if a sparse layer is reached it runs GENUINE dense GQA (coherent, logged
once) — never an empty-topk consume. Flip MSA_READY True ONLY after the wire (msa_m3_wire) +
the two compiled ops are parity-validated on-card.

Out-of-tree: ``apply()`` monkeypatches the imported ``HpuPlatform.get_attn_backend_cls`` in-process
(no edit to /opt). Mirrors the DSA pattern: save the ORIGINAL, gate behind ``_patched``, install.
"""
from __future__ import annotations

import os

# Set True ONLY when the full MSA path (wire + both compiled ops) is wired + validated on-card.
# Mirrors platform_patch.DSA_READY. Until then the GQA-sparse branch RAISES (never dense).
MSA_READY = False

# Mirror of platform_patch.FULL_ATTENTION_FALLBACK. When True (default), even if a sparse layer is
# reached (e.g. because the backend was somehow obtained), the MODEL seam (minimax_m3.py) runs
# genuine dense GQA. The plugin flips minimax_m3.MSA_FULL_ATTENTION_FALLBACK False only after the
# MSA consume is parity-validated; this flag documents/echoes that intent at the platform layer.
MSA_FULL_ATTENTION_FALLBACK = True

# Exposed for the gate / tests, like platform_patch.CONSUME_WIRED.
WIRE_INSTALLED = False

# The standard (non-MLA) HPU attention backend M3 GQA uses (platform.py default return). Resolved
# lazily in _patched so a CPU-only import of this module never imports vllm_gaudi.
_HPU_ATTN_BACKEND_V1 = "vllm_gaudi.v1.attention.backends.hpu_attn.HPUAttentionBackendV1"

# The original (or DSA-chained) unbound get_attn_backend_cls captured at apply() time.
_ORIG = None

_READY_ENV = "HPU_MSA_READY"


def ready() -> bool:
    """True iff the operator opted into the (validated) live MSA flip via env."""
    return os.environ.get(_READY_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _patched(cls, selected_backend, attn_selector_config, num_heads=None):
    """get_attn_backend_cls override: lift the sparse raise for the GQA-sparse (NOT MLA) branch
    behind MSA_READY; everything else chains to the captured original (which is DSA's _patched when
    both plugins are mounted, so MLA-sparse keeps working)."""
    use_sparse = getattr(attn_selector_config, "use_sparse", False)
    use_mla = getattr(attn_selector_config, "use_mla", False)
    if use_sparse and not use_mla:
        # This is the M3 MSA (GQA-sparse) case.
        if not MSA_READY:
            raise NotImplementedError(
                "[hpu-msa-ext] MSA (sparse GQA) selected but the HPU sparse path is not yet "
                "active (MSA_READY=False). Refusing to serve DENSE in place of sparse "
                "(as-designed mandate; the model seam runs genuine dense GQA in the meantime, "
                "but the platform must not hand back a backend that would run a half-wired sparse "
                "consume). Wire msa_m3_wire + validate the two compiled ops on-card first.")
        # Standard non-MLA GQA backend (the same one a dense GQA model gets); the real sparsity
        # lives in the MSA wire installed on the model attention, not in a special backend.
        return _resolve_backend(num_heads, cls, selected_backend, attn_selector_config)
    # MLA-sparse (DSA) / non-sparse / profiling -> the captured original (DSA-chained or stock).
    return _ORIG(cls, selected_backend, attn_selector_config, num_heads)


def _resolve_backend(num_heads, cls, selected_backend, attn_selector_config):
    """Return the standard HPU attention backend class string. We deliberately do NOT call the
    stock ``_ORIG`` here because the stock platform.py raises on use_sparse BEFORE it reaches the
    non-MLA return — so for a sparse-GQA config the only way to get the dense GQA backend is to
    name it. (The MSA wire on the model attention is what makes it actually sparse.)"""
    return _HPU_ATTN_BACKEND_V1


def apply() -> None:
    """Install the platform patch. Captures the CURRENT get_attn_backend_cls as the chain-through
    original (so if hpu-dsa-ext already patched it, MLA-sparse still routes through DSA), then binds
    our override. Idempotent. Out-of-tree (monkeypatch only; no /opt edit)."""
    global _ORIG
    from vllm_gaudi.platform import HpuPlatform

    cur = HpuPlatform.get_attn_backend_cls
    if getattr(cur, "_hpu_msa_wired", False):
        return
    # Capture the underlying function (classmethod -> __func__) so we can call it with cls.
    _ORIG = cur.__func__ if hasattr(cur, "__func__") else cur
    _patched._hpu_msa_wired = True
    HpuPlatform.get_attn_backend_cls = classmethod(_patched)


def maybe_flip_ready() -> bool:
    """Flip MSA_READY=True IFF HPU_MSA_READY is truthy AND the wire is installed. Called LAST by the
    plugin register(), only after msa_m3_wire is bound and the compiled ops loaded. Until then (or
    without the env) the GQA-sparse branch keeps refusing (NEVER dense). Also opens the model seam
    (minimax_m3.MSA_FULL_ATTENTION_FALLBACK=False) so sparse layers route through the validated wire.
    Returns the resulting MSA_READY value."""
    global MSA_READY, MSA_FULL_ATTENTION_FALLBACK
    if ready() and WIRE_INSTALLED:
        MSA_READY = True
        MSA_FULL_ATTENTION_FALLBACK = False
        # Open the model-side seam too (wire present + gate open == real sparsity).
        try:
            import importlib
            m3 = importlib.import_module("vllm_gaudi.models.minimax_m3")
            # set_msa_wire(fn, full_attention_fallback=False) opens the seam (fn already installed).
            if getattr(m3, "MSA_WIRE", None) is not None:
                m3.MSA_FULL_ATTENTION_FALLBACK = False
        except Exception:  # noqa: BLE001 — model not importable in a CPU smoke is fine
            pass
    return MSA_READY
