"""hpu-m3-ext — out-of-tree vLLM plugin that brings MiniMax-M3 up on vllm-gaudi.

register() is the ``vllm.general_plugins`` entry point. It runs in-process at engine init,
AFTER the stock vllm-gaudi plugins (entry-point names "01./02./03." sort earlier) and AFTER
hpu-moe-ext ("04."), and:

  1. Puts the M3 source dir (the workspace ``minimax-m3/vllm`` + its parent, mounted into the
     container) on ``sys.path`` so ``minimax_m3``, ``platform_patch_msa``,
     ``m3_expert_runtime_patch`` and ``int4_to_fp8_load`` import as top-level modules.
  2. Registers the config's top-level architecture
     ``MiniMaxM3SparseForConditionalGeneration`` -> ``minimax_m3:HpuMiniMaxM3ForConditionalGeneration``
     using ModelRegistry's lazy "module:Class" form (no import until model load -> avoids the
     CUDA-/HPU-init-in-fork trap, exactly how vllm registers external models).
  3. Applies the MSA platform patch (platform_patch_msa.apply()). With MSA_READY=False (the
     hard default) and HPU_MSA_READY unset, this is INERT for the dense-GQA phase: the M3
     attention builds STOCK GQA (use_sparse is never signaled to the platform), so the patched
     get_attn_backend_cls only ever chains to the original. The model's own seam keeps
     MSA_FULL_ATTENTION_FALLBACK=True -> sparse layers (3..59) run GENUINE dense GQA. Defensive
     install only; it cannot flip sparse on here.
  4. Calls m3_expert_runtime_patch.register(). With M3_EXPERT_RUNTIME=int4 (default) this is a
     NO-OP (the stock hpu-moe-ext int4->bf16 fused MoE path is exactly what we want). It only
     does work for M3_EXPERT_RUNTIME=fp8.

It edits NO files under /opt/vllm-gaudi (mirrors hpu-moe-ext / hpu-dsa-ext). An image rebuild
cannot clobber it. LOUD failure (raise) if any step it MUST do fails — never a silent wrong
serve (the DSA build-quality bar).
"""
from __future__ import annotations

import os
import sys

_REGISTERED = False

_ARCH = "MiniMaxM3SparseForConditionalGeneration"
_TARGET = "minimax_m3:HpuMiniMaxM3ForConditionalGeneration"

# M3 MTP (depth-1 speculative draft head). Registered LAZILY under the MTP draft-arch name(s) so the
# spec-decode path resolves `MiniMaxM3MTP` (hpu_m3_ext/m3_mtp.py) when a draft is requested. The class
# is co-located in THIS package, so the lazy "module:Class" target is the importable dotted path
# `hpu_m3_ext.m3_mtp:MiniMaxM3MTP` (NOT the top-level mounted `minimax_m3` used for the main model).
#
# ARCH NAME: vLLM picks the draft model class from the speculative draft config's `architectures`.
# Which exact name M3 lands on depends on how the serve's --speculative-config is written (see the
# minimax-m3.env "(2b) MTP" block, flagged << METHOD NAME NEEDS CONFIRMATION >>): a config that sets
# the draft model explicitly will carry `architectures: ["MiniMaxM3MTP"]`; the DeepSeek-style
# `{"method":"deepseek_mtp"}` path would instead make vLLM look up its own override arch. We register
# under SEVERAL plausible names (the class name + the two M3 ForCausalLM/ForConditionalGeneration
# arch-derived "...MTP" forms) so whichever the spec config resolves to finds this class. Registering
# the same lazy target under extra unused names is harmless (it is only resolved if requested).
_MTP_TARGET = "hpu_m3_ext.m3_mtp:MiniMaxM3MTP"
_MTP_ARCHS = (
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLMMTP",
    "MiniMaxM3SparseForConditionalGenerationMTP",
)

# Default mount points (set by the M3 serve env). M3_VLLM_DIR holds minimax_m3.py +
# platform_patch_msa.py; M3_ROOT_DIR (its parent) holds m3_expert_runtime_patch.py +
# int4_to_fp8_load.py. Both are added to sys.path so the lazy "minimax_m3:..." import resolves.
_DEFAULT_VLLM_DIR = "/opt/m3-vllm"
_DEFAULT_ROOT_DIR = "/opt/m3-root"


def _log(msg: str) -> None:
    print(f"hpu-m3-ext: {msg}", file=sys.stderr, flush=True)


def _ensure_on_path() -> str:
    """Put the M3 source dirs on sys.path. Returns the resolved vllm dir (where minimax_m3.py is).
    LOUD failure if minimax_m3.py is not found under it."""
    vllm_dir = os.environ.get("M3_VLLM_DIR", _DEFAULT_VLLM_DIR)
    root_dir = os.environ.get("M3_ROOT_DIR", _DEFAULT_ROOT_DIR)
    for d in (vllm_dir, root_dir):
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    model_py = os.path.join(vllm_dir, "minimax_m3.py")
    if not os.path.isfile(model_py):
        raise RuntimeError(
            f"hpu-m3-ext: minimax_m3.py not found at {model_py!r}. Set M3_VLLM_DIR to the mounted "
            f"minimax-m3/vllm dir (got M3_VLLM_DIR={vllm_dir!r}, M3_ROOT_DIR={root_dir!r}).")
    return vllm_dir


def _register_arch() -> None:
    from vllm import ModelRegistry
    # Lazy "module:Class" form — the module is NOT imported now (avoids HPU-init-in-fork). vLLM
    # imports it at model-load time, by which point M3_VLLM_DIR is on sys.path (we put it there
    # above, in this same process / before the registry resolves the string).
    ModelRegistry.register_model(_ARCH, _TARGET)
    _log(f"registered arch {_ARCH!r} -> {_TARGET!r} (lazy).")


def _register_mtp() -> None:
    # Register the M3 MTP draft head (hpu_m3_ext.m3_mtp:MiniMaxM3MTP) lazily under each plausible
    # draft-arch name (see _MTP_ARCHS rationale). Lazy target -> no import until a draft is loaded, so
    # importing this package's m3_mtp (which pulls vllm + the mounted minimax_m3) is deferred exactly
    # like the main-model arch. Idempotent within ModelRegistry; safe to register unused names.
    from vllm import ModelRegistry
    for arch in _MTP_ARCHS:
        ModelRegistry.register_model(arch, _MTP_TARGET)
    _log(f"registered MTP draft arch(s) {list(_MTP_ARCHS)!r} -> {_MTP_TARGET!r} (lazy, depth-1).")


def _apply_platform_patch() -> None:
    # Defensive in the dense phase (see module docstring item 3). Import the mounted module and
    # apply(). If it cannot import, that is a real wiring bug -> raise (loud).
    import platform_patch_msa as ppm
    ppm.apply()
    _log(f"platform_patch_msa applied (MSA_READY={ppm.MSA_READY}, "
         f"FULL_ATTENTION_FALLBACK={ppm.MSA_FULL_ATTENTION_FALLBACK}, "
         f"HPU_MSA_READY={os.environ.get('HPU_MSA_READY', '<unset>')}).")


def _apply_expert_runtime() -> None:
    import m3_expert_runtime_patch as mert
    mert.register()  # no-op at M3_EXPERT_RUNTIME=int4; wraps the fp8 op only for =fp8
    _log(f"m3_expert_runtime_patch registered (M3_EXPERT_RUNTIME="
         f"{os.environ.get('M3_EXPERT_RUNTIME', 'int4 (default)')}).")


def _apply_swigluoai() -> None:
    # PRIORITY-1 unfused swigluoai for the M3 routed int4 experts. NO-OP unless M3_SWIGLUOAI=1.
    # Gated by an EXPLICIT env (NOT the act string, which transformers overrides to "silu"). When on,
    # wraps hpu-moe-ext's MoeWNA16HPUMethod.apply_monolithic so the routed experts apply the EXACT
    # gpt-oss-style swigluoai gate instead of the silu intrinsic -> coherent M3. Co-located in this
    # package (hpu_m3_ext.m3_swigluoai_patch), so import relatively. LOUD on any failure to install.
    from . import m3_swigluoai_patch as msp
    msp.register()
    _log(f"m3_swigluoai_patch registered (M3_SWIGLUOAI="
         f"{os.environ.get('M3_SWIGLUOAI', '<unset> (off)')}, "
         f"installed={msp.enabled()}).")


def _apply_fp8_static() -> None:
    # STATIC on-disk fp8 path: wraps vllm_gaudi HPUFp8MoEMethod so the routed experts of a TRUE fp8
    # checkpoint (--quantization fp8) serve via the UNFUSED swigluoai route-b instead of the native
    # silu intrinsic. NO-OP unless M3_FP8_STATIC=1 (so it never touches the int4 / int4->fp8 serves
    # or any non-M3 fp8 model). Co-located in this package. LOUD on any failure to install.
    from . import m3_fp8_static_patch as mfs
    mfs.register()
    _log(f"m3_fp8_static_patch registered (M3_FP8_STATIC="
         f"{os.environ.get('M3_FP8_STATIC', '<unset> (off)')}, installed={mfs.enabled()}).")


def _apply_moe_fuse() -> None:
    # GROUPED routed-fp8 MoE island: wraps the fp8 method's _m3_fp8_apply (installed just above by
    # m3_fp8_static_patch) so the ALL-EXPERT (concurrency T>1) path serves via the grouped routed-fp8
    # op (permute -> native fp8 MME -> scatter) instead of the route-b all-16-expert bf16 loop. NO-OP
    # unless M3_FP8_STATIC=1 AND M3_MOE_FUSE=1. MUST run AFTER _apply_fp8_static (it wraps that handler).
    from . import m3_moe_fuse_patch as mmf
    mmf.register()
    _log(f"m3_moe_fuse_patch registered (M3_MOE_FUSE="
         f"{os.environ.get('M3_MOE_FUSE', '<unset> (off)')}, installed={mmf.enabled()}).")


def register() -> None:
    """vllm.general_plugins entry point. Idempotent. LOUD on any failure it must not swallow."""
    global _REGISTERED
    if _REGISTERED:
        return
    _ensure_on_path()
    _register_arch()
    _register_mtp()  # M3 depth-1 MTP draft head (hpu_m3_ext.m3_mtp:MiniMaxM3MTP) — lazy, multi-arch.
    # The two patches refine an already-imported stack; do them after arch registration. The
    # platform patch is defensive (dense phase) and the expert-runtime patch is a no-op at int4,
    # but a hard failure of EITHER means the serve env is mis-wired -> fail loudly, do not serve.
    _apply_platform_patch()
    _apply_expert_runtime()
    _apply_swigluoai()  # PRIORITY-1: unfused swigluoai for routed int4 experts (no-op unless M3_SWIGLUOAI=1)
    _apply_fp8_static()  # STATIC on-disk fp8 swigluoai wire (no-op unless M3_FP8_STATIC=1)
    _apply_moe_fuse()  # GROUPED routed-fp8 all-expert island (no-op unless M3_FP8_STATIC=1 & M3_MOE_FUSE=1)
    # NOTE: all_reduce mark_step removal lives in hpu-moe-ext _install_communicator_patch
    # (HPU_AR_NO_MARKSTEP=1 / HPU_TIME_AR=1) — the canonical universal place. Do not duplicate here.
    _REGISTERED = True
    _log("MiniMax-M3 wiring complete (dense-GQA fallback; int4 experts via hpu-moe-ext autodetect).")
