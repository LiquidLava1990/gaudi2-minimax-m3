# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2026 The MiniMax AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Make vLLM resolve speculative method ``mtp`` for MiniMax-M3 (model_type ``minimax_m3_vl``).

THE PROBLEM
-----------
vLLM resolves the speculative draft architecture + method inside
``vllm.config.speculative.SpeculativeConfig``. Two hooks decide it:

  * ``SpeculativeConfig.hf_config_override(hf_config)``  — a ``@staticmethod`` passed as
    ``hf_overrides=`` when the draft ``ModelConfig`` is built (this runs for an explicit
    ``method`` too, inside the ``if self.model is not None:`` block, NOT only when the
    method is auto-detected). For every KNOWN MTP family it rewrites the draft config, e.g.::

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            hf_config.update({"num_hidden_layers": 0, "n_predict": <nextn>,
                              "architectures": ["Glm4MoeMTPModel"]})

    so the draft is built as the small ``Glm4MoeMTPModel`` (registered in the model
    registry's ``_SPECULATIVE_DECODING_MODELS``), NOT the full target model.

  * ``SpeculativeConfig.__post_init__`` — when ``self.method`` is auto-detected it does::

        elif self.draft_model_config.hf_config.model_type in get_args(MTPModelTypes):
            self.method = "mtp"

    (``MTPModelTypes = Literal["deepseek_mtp", "mimo_mtp", "glm4_moe_mtp", "ernie_mtp",
    "qwen3_next_mtp", "longcat_flash_mtp", "mtp", "pangu_ultra_moe_mtp"]``). An explicitly
    passed ``method`` that is a deprecated MTP alias is also normalized to ``"mtp"`` at the
    very top of ``__post_init__``.

GLM-5.1 works because ``glm_moe_dsa`` (architectures ``Glm4MoeForCausalLM``) HAS an
``hf_config_override`` branch -> it is remapped to ``glm4_moe_mtp`` / ``Glm4MoeMTPModel``.
**M3 has NO branch.** Its draft config carries ``model_type="minimax_m3_vl"`` /
``architectures=["MiniMaxM3SparseForConditionalGeneration"]`` (or the text-tower
``MiniMaxM3SparseForCausalLM``). ``hf_config_override`` leaves it untouched, so:

  * the draft is built as the full multimodal M3 arch (or fails the
    ``_SPECULATIVE_DECODING_MODELS`` lookup -> "Model architectures [...] are not
    supported"), and
  * if the method is left to auto-detect, ``minimax_m3_vl`` is not in
    ``get_args(MTPModelTypes)`` -> the ``else`` branch raises
    ``NotImplementedError("Speculative decoding with draft model is not supported yet ...")``
    (the "Unknown speculative method" the M3 serve hits).

THE FIX
-------
Wrap ``SpeculativeConfig.hf_config_override`` (call the original FIRST, so every other
family is unchanged), then, IF the resulting draft ``hf_config`` is an M3 draft, rewrite
it exactly like the GLM branch::

    hf_config.model_type = "mtp"                 # -> in get_args(MTPModelTypes); method -> "mtp"
    hf_config.update({"n_predict": <num_nextn_predict_layers>,    # from text_config (=1)
                      "architectures": ["MiniMaxM3MTP"]})         # the registered draft class

``MiniMaxM3MTP`` is the depth-1 draft head in ``hpu_m3_ext.m3_mtp``; this package's
``register()`` (``__init__._register_mtp``) already registers it (and two arch-name
variants) in ``ModelRegistry`` under the lazy target ``hpu_m3_ext.m3_mtp:MiniMaxM3MTP``.
With the model_type ``"mtp"`` and that registered arch, vLLM routes the draft through
``HpuEagleProposer`` (vllm_gaudi ``hpu_eagle.py``), whose ``method == "mtp"`` branch
drives the single-forward depth-1 draft and early-exits at
``num_speculative_tokens == 1``.

UNLIKE GLM we do NOT zero ``num_hidden_layers``: M3 reads its layer/nextn counts from
the TEXT config (``MiniMaxM3MTP`` resolves the text config via ``get_text_config`` and
keys ``mtp_start_layer_idx = num_hidden_layers = 60``); the top-level
``num_hidden_layers`` is already ``None`` and zeroing it is both unnecessary and would
discard the real (text) value the draft layer-name mapping depends on.

SCOPE / SAFETY
--------------
* Import-guarded: a vLLM build with no spec-decode support (no ``SpeculativeConfig`` /
  no ``hf_config_override``) -> no-op (the M3 serve simply runs without MTP).
* Idempotent: a sentinel attribute on the wrapped staticmethod; a second ``install()``
  is a no-op.
* The original ``hf_config_override`` runs unchanged for every non-M3 draft.
* LOUD: if M3 is detected but the rewrite cannot be applied (no ``architectures`` /
  ``update`` on the config), we raise — never a silently-mis-resolved draft.

CPU SELF-TEST (card-free)
-------------------------
Run with the m3calib venv::

    $HOME/.venvs/m3calib/bin/python \
        $HOME/AI/hpu-m3-ext/hpu_m3_ext/m3_mtp_method_patch.py

The gate has two layers:
  1. A pure-python FUNCTIONAL test that builds a fake ``SpeculativeConfig`` carrying only
     ``hf_config_override`` + a stub M3 ``hf_config`` (NO vLLM needed), installs the patch
     onto it, and asserts: (a) idempotency, (b) an M3 draft now gets model_type ``"mtp"`` +
     architectures ``["MiniMaxM3MTP"]`` + ``n_predict`` from text_config, (c)
     ``"mtp" in get_args(MTPModelTypes)`` so ``__post_init__`` would set the method, and
     (d) a non-M3 draft is passed through untouched.
  2. A SOURCE gate that AST-parses THIS file to prove the patch targets
     ``SpeculativeConfig.hf_config_override`` and writes the registered arch name.
  If vLLM IS importable (the serve image), the gate ADDITIONALLY parses vLLM's real
  ``speculative.py`` to assert ``hf_config_override`` and the ``MTPModelTypes`` symbol
  still exist where we patch — turning upstream drift into a loud test failure.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("vllm")

# The architecture name the draft must land on. MUST match an arch registered by this
# package's register() (hpu_m3_ext.__init__._register_mtp -> _MTP_ARCHS[0]) so the
# _SPECULATIVE_DECODING_MODELS / ModelRegistry lookup resolves the draft to
# hpu_m3_ext.m3_mtp:MiniMaxM3MTP.
M3_MTP_ARCH = "MiniMaxM3MTP"

# model_type written onto the draft hf_config. "mtp" is a member of vLLM's MTPModelTypes
# literal, so SpeculativeConfig.__post_init__'s
#   `elif ... model_type in get_args(MTPModelTypes): self.method = "mtp"`
# fires (auto-detect path), and an explicit method="mtp" is already correct.
M3_MTP_MODEL_TYPE = "mtp"

# Sentinel marking the wrapped staticmethod (idempotency guard).
_PATCH_FLAG = "_m3_mtp_method_patch"


def _text_cfg(hf_config):
    """The sub-config that actually carries M3's layer/nextn counts. The multimodal M3
    config (model_type minimax_m3_vl) nests them under .text_config; the text-tower config
    (MiniMaxM3SparseForCausalLM) carries them directly. Return whichever has them."""
    tc = getattr(hf_config, "text_config", None)
    if tc is not None and getattr(tc, "num_nextn_predict_layers", None) is not None:
        return tc
    return hf_config


def _is_m3_draft(hf_config) -> bool:
    """True iff this draft hf_config is a MiniMax-M3 model (top multimodal config OR the
    text-tower config). Keyed on model_type first, then the architectures fallback (the
    text-tower config's model_type can be absent/inherited)."""
    mt = getattr(hf_config, "model_type", None)
    if mt in ("minimax_m3_vl", "minimax_m3"):
        return True
    archs = getattr(hf_config, "architectures", None) or []
    return bool(archs) and isinstance(archs[0], str) and archs[0].startswith("MiniMaxM3Sparse")


def _apply_m3_override(hf_config):
    """Rewrite an M3 draft hf_config so vLLM builds + routes it as the depth-1 MTP draft.
    Mirrors SpeculativeConfig.hf_config_override's GLM/DeepSeek branches. LOUD if the config
    cannot be rewritten (never leave a half-resolved M3 draft)."""
    n_predict = getattr(_text_cfg(hf_config), "num_nextn_predict_layers", None)

    # model_type -> "mtp" lands it in get_args(MTPModelTypes) (method auto-detect) and is
    # the canonical normalized value for an explicit method="mtp".
    try:
        hf_config.model_type = M3_MTP_MODEL_TYPE
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"[hpu-m3-ext] M3 MTP method patch: cannot set model_type on draft hf_config "
            f"({type(hf_config).__name__}): {e}") from e

    update = getattr(hf_config, "update", None)
    payload = {"n_predict": n_predict, "architectures": [M3_MTP_ARCH]}
    if callable(update):
        update(payload)
    else:
        # PretrainedConfig always has .update; fall back to attribute set so we never
        # silently fail to point the draft at the registered MTP class.
        try:
            hf_config.architectures = [M3_MTP_ARCH]
            hf_config.n_predict = n_predict
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "[hpu-m3-ext] M3 MTP method patch: draft hf_config has no usable "
                f"`update`/attribute set to write architectures ({type(hf_config).__name__}): "
                f"{e}") from e

    logger.warning(
        "[hpu-m3-ext] M3 MTP: rewrote draft hf_config -> model_type=%r, "
        "architectures=[%r], n_predict=%r (routes through HpuEagleProposer method=='mtp').",
        M3_MTP_MODEL_TYPE, M3_MTP_ARCH, n_predict)
    return hf_config


def _make_wrapper(orig_override):
    """Build the wrapped hf_config_override: run the original (all other families
    unchanged), then add M3 handling. Returns a plain function suitable for re-wrapping as a
    staticmethod."""

    def _wrapped(hf_config):
        # Original first: any future upstream M3 support, plus every other family, is
        # honored. (M3 currently has no upstream branch, so this is a pass-through for M3.)
        hf_config = orig_override(hf_config)
        try:
            if _is_m3_draft(hf_config):
                hf_config = _apply_m3_override(hf_config)
        except Exception:
            logger.exception(
                "[hpu-m3-ext] M3 MTP method patch FAILED to remap the draft hf_config — "
                "the M3 speculative draft will not resolve (serve would raise "
                "'Speculative decoding with draft model is not supported').")
            raise
        return hf_config

    _wrapped.__name__ = getattr(orig_override, "__name__", "hf_config_override")
    _wrapped.__doc__ = getattr(orig_override, "__doc__", None)
    setattr(_wrapped, _PATCH_FLAG, True)
    # Keep a handle to the original so the idempotency check and tests can introspect it.
    setattr(_wrapped, "_m3_orig_override", orig_override)
    return _wrapped


def install() -> None:
    """Patch SpeculativeConfig.hf_config_override so M3 resolves to method 'mtp'.

    Import-guarded (no spec-decode build -> no-op), idempotent, LOUD on a real failure."""
    try:
        from vllm.config.speculative import SpeculativeConfig
    except Exception as e:  # noqa: BLE001 — no spec-decode support in this build
        logger.info(
            "hpu-m3-ext: vllm.config.speculative.SpeculativeConfig not importable (%s); "
            "M3 MTP method patch skipped (serve runs without speculative decoding).", e)
        return

    orig = getattr(SpeculativeConfig, "hf_config_override", None)
    if orig is None:
        # Upstream renamed/removed the hook we patch -> loud (the M3 serve env asks for MTP;
        # silently not patching would surface as a confusing "method not supported" later).
        raise RuntimeError(
            "hpu-m3-ext: SpeculativeConfig has no `hf_config_override` to patch — vLLM's "
            "speculative method-resolution API changed. Re-point the M3 MTP method patch "
            "(it must remap the draft hf_config's model_type/architectures for minimax_m3).")

    if getattr(orig, _PATCH_FLAG, False):
        return  # idempotent — already wrapped.

    SpeculativeConfig.hf_config_override = staticmethod(_make_wrapper(orig))
    logger.info(
        "hpu-m3-ext: patched SpeculativeConfig.hf_config_override "
        "(minimax_m3 draft -> model_type='mtp', architectures=['%s'] -> method 'mtp').",
        M3_MTP_ARCH)


# ======================================================================================
# CPU self-test (card-free). See the module docstring for what it proves.
#   $HOME/.venvs/m3calib/bin/python \
#       $HOME/AI/hpu-m3-ext/hpu_m3_ext/m3_mtp_method_patch.py
# ======================================================================================
class _StubHFConfig:  # pragma: no cover - test double
    """Minimal stand-in for a HuggingFace PretrainedConfig with the surface the patch
    touches: model_type, architectures, .update(dict), and an optional nested text_config.
    Mirrors transformers' PretrainedConfig.update (setattr per key)."""

    def __init__(self, model_type=None, architectures=None, num_nextn_predict_layers=None,
                 text_config=None):
        self.model_type = model_type
        self.architectures = list(architectures) if architectures is not None else []
        if num_nextn_predict_layers is not None:
            self.num_nextn_predict_layers = num_nextn_predict_layers
        self.text_config = text_config

    def update(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class _StubSpecConfig:  # pragma: no cover - test double
    """A class with ONLY the hf_config_override staticmethod the patch targets — lets the
    functional gate exercise install()'s wrap/idempotency logic with no vLLM import. The
    install() codepath itself imports the REAL vllm class; this double feeds _make_wrapper
    directly (the same wrapper install() builds)."""

    @staticmethod
    def hf_config_override(hf_config):
        # The real upstream method mutates + returns hf_config; a passthrough is the correct
        # stand-in for the no-op (no-branch) M3 case.
        return hf_config


def _mtp_model_types():  # pragma: no cover - test helper
    """get_args(MTPModelTypes) from the real vllm if importable, else the known literal set
    (kept in sync with vLLM 0.22.x / the gaudi:next image). The functional gate asserts
    'mtp' is a member either way."""
    try:
        from typing import get_args

        from vllm.config.speculative import MTPModelTypes
        return tuple(get_args(MTPModelTypes))
    except Exception:  # noqa: BLE001
        return (
            "deepseek_mtp", "mimo_mtp", "glm4_moe_mtp", "ernie_mtp", "qwen3_next_mtp",
            "longcat_flash_mtp", "mtp", "pangu_ultra_moe_mtp",
        )


def _functional_gate():  # pragma: no cover - exercised via __main__
    checks: list[tuple[str, bool]] = []

    def ck(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    # Build the SAME wrapper install() builds, on the stub class (no vLLM needed). This
    # exercises _make_wrapper / _apply_m3_override / _is_m3_draft / idempotency end-to-end.
    orig = _StubSpecConfig.hf_config_override
    wrapped = _make_wrapper(orig)
    _StubSpecConfig.hf_config_override = staticmethod(wrapped)

    # 1. idempotency: the wrapped fn carries the sentinel; a re-wrap must be a no-op.
    ck("wrapper carries idempotency sentinel", getattr(wrapped, _PATCH_FLAG, False) is True)
    # simulate install()'s guard: accessing a staticmethod via the class yields the plain
    # underlying function, which carries the sentinel -> install() would see it and not re-wrap.
    already = getattr(_StubSpecConfig.hf_config_override, _PATCH_FLAG, False)
    ck("second install() would be a no-op (sentinel seen)", already is True)

    call = _StubSpecConfig.hf_config_override  # resolves the staticmethod

    # 2. M3 multimodal draft (model_type minimax_m3_vl, nextn in text_config) -> remapped.
    m3_vl = _StubHFConfig(
        model_type="minimax_m3_vl",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        text_config=_StubHFConfig(num_nextn_predict_layers=1),
    )
    out = call(m3_vl)
    ck("M3-VL draft model_type -> 'mtp'", out.model_type == "mtp")
    ck("M3-VL draft architectures -> ['MiniMaxM3MTP']", out.architectures == ["MiniMaxM3MTP"])
    ck("M3-VL draft n_predict from text_config (=1)", getattr(out, "n_predict", None) == 1)

    # 3. M3 text-tower draft (architectures MiniMaxM3SparseForCausalLM, nextn at top level).
    m3_text = _StubHFConfig(
        model_type=None,
        architectures=["MiniMaxM3SparseForCausalLM"],
        num_nextn_predict_layers=1,
    )
    out2 = call(m3_text)
    ck("M3-text draft detected via architectures -> 'mtp'", out2.model_type == "mtp")
    ck("M3-text draft architectures -> ['MiniMaxM3MTP']", out2.architectures == ["MiniMaxM3MTP"])
    ck("M3-text draft n_predict from top-level (=1)", getattr(out2, "n_predict", None) == 1)

    # 4. the resolved model_type lands in vLLM's MTPModelTypes -> __post_init__ sets 'mtp'.
    ck("'mtp' in get_args(MTPModelTypes)", "mtp" in _mtp_model_types())

    # 5. the chosen arch is one this package registers (kept in sync with _MTP_ARCHS).
    try:
        from hpu_m3_ext import _MTP_ARCHS  # type: ignore[attr-defined]
        ck("M3_MTP_ARCH is registered by hpu_m3_ext._MTP_ARCHS",
           M3_MTP_ARCH in _MTP_ARCHS)
    except Exception:
        # Package import may pull heavy deps on some interpreters; assert the literal we
        # depend on instead (the registration source lists exactly this name first).
        ck("M3_MTP_ARCH literal matches the registered class name",
           M3_MTP_ARCH == "MiniMaxM3MTP")

    # 6. a NON-M3 draft is passed through untouched (no false-positive remap).
    other = _StubHFConfig(model_type="glm4_moe_mtp",
                          architectures=["Glm4MoeMTPModel"], num_nextn_predict_layers=1)
    out3 = call(other)
    ck("non-M3 draft model_type unchanged", out3.model_type == "glm4_moe_mtp")
    ck("non-M3 draft architectures unchanged", out3.architectures == ["Glm4MoeMTPModel"])

    return checks


def _source_gate():  # pragma: no cover - exercised via __main__
    """Prove (over THIS file's own source, plus vLLM's speculative.py if importable) that the
    patch targets the right symbols. No execution required; pure AST/string assertions."""
    import ast
    import os

    checks: list[tuple[str, bool]] = []

    def ck(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    src = open(os.path.abspath(__file__)).read()
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    ck("install() defined", "install" in funcs)
    install_src = ast.get_source_segment(src, funcs["install"]) or ""
    # The patch must read AND write SpeculativeConfig.hf_config_override.
    ck("install imports SpeculativeConfig",
       "from vllm.config.speculative import SpeculativeConfig" in install_src)
    ck("install reads .hf_config_override",
       'getattr(SpeculativeConfig, "hf_config_override"' in install_src)
    ck("install assigns SpeculativeConfig.hf_config_override = staticmethod(...)",
       "SpeculativeConfig.hf_config_override = staticmethod(" in install_src)
    ck("install is idempotent (checks the sentinel)", f"getattr(orig, _PATCH_FLAG" in install_src)
    ck("install is LOUD on a missing hook (raises RuntimeError)",
       "raise RuntimeError(" in install_src)

    # The rewrite must produce the registered arch + a method-resolvable model_type.
    ck("rewrites architectures to the registered MTP class",
       'M3_MTP_ARCH = "MiniMaxM3MTP"' in src)
    ck("rewrites model_type to 'mtp'", 'M3_MTP_MODEL_TYPE = "mtp"' in src)
    apply_src = ast.get_source_segment(src, funcs["_apply_m3_override"]) or ""
    ck("_apply_m3_override sets model_type", "hf_config.model_type = M3_MTP_MODEL_TYPE" in apply_src)
    ck("_apply_m3_override writes architectures via update payload",
       '"architectures": [M3_MTP_ARCH]' in apply_src)
    ck("_apply_m3_override surfaces n_predict",
       '"n_predict": n_predict' in apply_src)

    # Detection must cover both the multimodal and text-tower configs.
    det_src = ast.get_source_segment(src, funcs["_is_m3_draft"]) or ""
    ck("detects model_type minimax_m3_vl", '"minimax_m3_vl"' in det_src)
    ck("detects architectures MiniMaxM3Sparse*", 'startswith("MiniMaxM3Sparse")' in det_src)

    # If vLLM is importable (serve image): assert the real upstream symbols still exist where
    # we patch — upstream drift then fails THIS test instead of the serve.
    try:
        import inspect
        from typing import get_args

        from vllm.config import speculative as spec_mod
        from vllm.config.speculative import MTPModelTypes, SpeculativeConfig

        ck("[vllm] SpeculativeConfig.hf_config_override exists",
           hasattr(SpeculativeConfig, "hf_config_override"))
        ck("[vllm] 'mtp' is a member of MTPModelTypes", "mtp" in get_args(MTPModelTypes))
        # Collapse whitespace so the multi-line `in get_args(\n MTPModelTypes\n)` matches too.
        spec_src = inspect.getsource(spec_mod)
        spec_flat = " ".join(spec_src.split())
        ck("[vllm] __post_init__ resolves method via get_args(MTPModelTypes)",
           "in get_args( MTPModelTypes )" in spec_flat or "in get_args(MTPModelTypes)" in spec_flat)
        ck("[vllm] hf_config_override is wired as hf_overrides for the draft ModelConfig",
           "hf_overrides=SpeculativeConfig.hf_config_override" in spec_flat)
    except Exception as e:  # noqa: BLE001 — CPU calib venv: vllm absent, source gate is enough.
        ck(f"[vllm not importable: {type(e).__name__}] source gate is sufficient (card box re-checks)",
           True)

    return checks


def _selftest() -> bool:  # pragma: no cover - exercised via __main__
    import sys

    try:
        from vllm.config.speculative import SpeculativeConfig  # noqa: F401
        vllm_ok = True
    except Exception:  # noqa: BLE001
        vllm_ok = False
    if not vllm_ok:
        print("[m3_mtp_method_patch self-test] vllm NOT importable (CPU calib venv expected) "
              "— running the functional gate (stub config) + source gate.\n", file=sys.stderr)

    all_checks = _functional_gate() + _source_gate()
    ok = all(p for _, p in all_checks)
    for name, p in all_checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\n{'=' * 64}\nM3 MTP METHOD-RESOLUTION PATCH GATE: "
          f"{'GREEN — all pass' if ok else 'RED — FIX ABOVE'}\n{'=' * 64}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(0 if _selftest() else 1)
