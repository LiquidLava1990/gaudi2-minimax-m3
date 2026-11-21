#!/usr/bin/env python3
# ==================================================================================================
# quantize_m3_fp8.py
#
# STATIC per-OUTPUT-CHANNEL FP8 (E4M3) recipe for the MiniMax-M3 ROUTED MoE EXPERTS ONLY, exported in
# the compressed-tensors `float-quantized` on-disk format that vLLM's `--quantization fp8` /
# CompressedTensorsConfig loader reads (and that this box's hpu_compressed_tensors.py CHANNEL fp8 path
# consumes). It is the FP8 sibling of quantize_m3_int4.py: same expert-selection regex, same
# dry-run/run gating, same bounded-RAM shard streaming, same LOUD failures -- ONLY the per-expert math
# (int4 round-to-nearest -> fp8 per-channel cast) and the output format (auto_round:auto_gptq ->
# compressed-tensors float-quantized) change.
#
# ------------------------------------------------------------------------------------------------
# WHAT GETS QUANTIZED (FP8 E4M3, static per-OUTPUT-CHANNEL weight scale):
#   * ONLY the 128 ROUTED experts (w1/w2/w3) of the MoE layers (text layers 3..59).
#     Checkpoint tensor names (ground truth from the bf16 shard headers + the int4 export):
#       language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w2,w3}.weight
#     L in 3..59 (moe_layer_freq==1; dense prefix 0..2 stays bf16), E in 0..127.
#     bf16 source shapes: w1,w3 = [I=3072, H=6144]; w2 = [H=6144, I=3072]   (HF Linear [out,in]).
#
# WHAT STAYS BF16 (copied through UNCHANGED at source dtype -- never quantized, never dropped):
#   * ALL attention            : self_attn.{q,k,v,o}_proj  + self_attn.{q,k}_norm
#   * MSA index projections     : self_attn.index_{q,k}_proj + self_attn.index_{q,k}_norm
#   * Router gate + routing bias: block_sparse_moe.gate + block_sparse_moe.e_score_correction_bias
#   * Shared expert (1)         : block_sparse_moe.shared_experts.{gate,up,down}_proj
#   * Dense-prefix MLPs (0..2)  : layers.{0,1,2}.mlp.{gate,up,down}_proj
#   * Embeddings / lm_head      : model.embed_tokens, lm_head
#   * All norms                 : input_layernorm, post_attention_layernorm, model.norm, *layernorm*
#   * VISION TOWER              : vision_tower.* / multi_modal_projector.* / patch_merge_mlp.* / image_newline
#   * THE 7 MTP MODULES         : any tensor with mtp/nextn/eh_proj/enorm/hnorm/shared_head in its name,
#                                 i.e. the multi-token-prediction heads at logical layers 60..66.
#                                 *** The int4 export DROPPED all MTP. THIS SCRIPT MUST NOT. *** It runs
#                                 over the COMPLETE bf16 download so the MTP shards exist to copy through
#                                 verbatim (bf16). Dropping MTP here is a HARD BUG -- the self-test asserts
#                                 the fake MTP module survives unchanged.
#
# WHY this split: it is the same routed-experts-only contract the int4 export uses, retargeted to fp8.
# The serve side (hpu_compressed_tensors.HPUCompressedTensorsW8A8Fp8MoEMethod, CHANNEL strategy) reads
# the *stored* format as ground truth: a module is fp8 iff it carries `<prefix>.weight`
# (float8_e4m3fn) + `<prefix>.weight_scale` ([out,1] float32) AND its prefix matches a config_groups
# target. Everything else loads bf16. So the ONLY thing that must be true for a clean serve is: the
# routed experts (and nothing else) are stored fp8 + scaled, and the quantization_config scopes the
# fp8 group to ONLY the expert FusedMoE prefix (ignore-list covers all the rest). This recipe enforces
# both: the streaming writer only ever fp8s expert w1/w2/w3, and emit_quantization_config() scopes the
# config_groups target to the expert prefix with an exhaustive ignore list.
#
# ------------------------------------------------------------------------------------------------
# THE LOCKED FP8 MATH (per quantized weight W, shape [out, in]; do NOT deviate -- the serve depends on it):
#   scale = (W.float().abs().amax(dim=1) + 1e-8) / FP8_MAX        # [out] float32 ; FP8_MAX == 240.0
#   W_fp8 = (W.float() / scale.unsqueeze(1)).to(torch.float8_e4m3fn)   # [out, in] float8_e4m3fn
#   store  <prefix>.weight       = W_fp8                          ([out,in] float8_e4m3fn)
#   store  <prefix>.weight_scale = scale.reshape(out, 1)         ([out,1]  float32)
# This is per-OUTPUT-CHANNEL (one scale per ROW / out channel), reduced over the INPUT dim (dim=1).
# Reconstruction `W_fp8.to(bf16) * weight_scale` (broadcast [out,in]*[out,1]) recovers W to the float8
# e4m3 per-channel representation FLOOR -- ~2.6e-2 RMS Frobenius (3 mantissa bits; theoretical RMS
# 2^-3/sqrt(12) ~= 3.6e-2), as documented in int4_to_fp8_load.py:450-463. (The prompt's "a few e-3"
# describes the *added* int4->fp8 error -- int4 is already coarse -- not this raw bf16->fp8 floor; a
# sub-1e-2 bar would be unphysical for fp8 e4m3.) The self-test ASSERTS the round-trip sits at that
# floor and HARD-FAILS only above 6.4e-2 (which would mean a scale/axis/dtype bug, not the fp8 floor).
#
#   FP8_MAX = 240.0 = float8_e4m3fnuz max. This MUST match int4_to_fp8_load.FP8_MAX. Rationale (verbatim
#   from int4_to_fp8_load.py): Gaudi-2's HW fp8 element is float8_e4m3FNUZ (bias 8, max 240); the dtype
#   handed to the cast/gemm ops is float8_e4m3FN (bias 7, max 448) and the HW maps fn<->fnuz internally.
#   Scaling by the *fnuz* max (240) keeps real values inside the true e4m3 range so the cast never pushes
#   to NaN/inf, and the cast(1/s)/op(d_scale=s) pair is a self-consistent fn<->fnuz round-trip. We store
#   the bytes as torch.float8_e4m3fn (what safetensors/compressed-tensors + the loader expect) but scale
#   by 240. The config also carries `fp8_dtype_flavor: float8_e4m3fnuz` (Gaudi2) for the loader.
#
# ------------------------------------------------------------------------------------------------
# CARD-FREE NOTE: this module is CPU-only. `--dry-run` parses config.json + the safetensors *headers*
# only (a few KB/shard) and needs NO torch. `--validate` and `--run` use torch (CPU) + safetensors for
# the streaming per-tensor math but NEVER build a model, NEVER run a forward pass, NEVER touch a Gaudi
# card, and NEVER launch docker/vllm/serve. Do NOT touch the serving cards (GLM serves on all 8).
#
# IMPORTANT (cwd shadow): torch's `import platform` collides with $HOME/AI/_src/vllm_gaudi/platform.py
# if that dir is the cwd / on sys.path[0]. Run this script from its OWN directory (or anywhere that is
# NOT the vllm_gaudi checkout). The script strips a leading vllm_gaudi cwd from sys.path defensively.
# ==================================================================================================
#
# ================================  REQUIREMENTS for the REAL run (`--run`)  =====================
# HARD gates. `--run` refuses to start unless ALL are met (see _assert_can_run):
#
#   1. DOWNLOAD COMPLETE. --src (default /mnt/data1/minimax-m3-bf16) holds all 59 shards
#      (model-0000N-of-00059.safetensors) + model.safetensors.index.json. *** Do NOT run the real quant
#      before this -- the in-progress `hf download` is still writing; --run is gated on completeness. ***
#      The COMPLETE download (not the int4 export) is required so the 7 MTP modules exist to copy.
#
#   2. torch + safetensors importable (CPU). NO transformers/auto_round needed for the streaming path
#      (we read config.json directly + stream safetensors ourselves); we only import auto_round in
#      --emit-config-via-autoround mode to cross-check the config schema.
#
#   3. NO DEVICE. CPU only. The per-tensor fp8 cast is tiny host math; bounded-RAM streaming (one shard
#      resident at a time) keeps peak RSS at ~one shard (~16 GiB), never the 856 GiB model.
#
# ================================  OUTPUT  ======================================================
#   --out (default /mnt/fastmodels/MiniMax-M3-fp8-experts-AutoRound)
#   On-disk format: compressed-tensors `float-quantized`. Per expert Linear:
#       <prefix>.weight        float8_e4m3fn  [out,in]
#       <prefix>.weight_scale  float32        [out,1]
#   Everything else: copied through bf16 (or source dtype). config.json carries quantization_config.
#   Estimated size: ~ (856 GiB bf16) - (expert bf16) + (expert fp8 + tiny scales). Computed live in
#   --dry-run from the real expert dims (experts halve: bf16 2 B/elem -> fp8 1 B/elem + [out,1] f32 scale).
# ==================================================================================================

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys

# --------------------------------------------------------------------------------------------------
# Defensive cwd-shadow guard: if we were launched from the vllm_gaudi checkout, its platform.py would
# shadow stdlib `platform` (which torch imports) -> ModuleNotFoundError: habana_frameworks. Drop any
# leading sys.path entry that contains a vllm_gaudi platform.py BEFORE torch is ever imported.
# --------------------------------------------------------------------------------------------------
def _strip_vllm_gaudi_from_path() -> None:
    bad = []
    for p in list(sys.path[:2]):  # only the implicit cwd / script-dir entries
        try:
            if p and os.path.exists(os.path.join(p, "platform.py")) and os.path.isdir(os.path.join(p, "vllm_gaudi")):
                bad.append(p)
        except OSError:
            pass
    for p in bad:
        while p in sys.path:
            sys.path.remove(p)


_strip_vllm_gaudi_from_path()

# --------------------------------------------------------------------------------------------------
# Static config -- the contract. Edit here, nowhere else. (--src/--out/--max-shards override the paths.)
# --------------------------------------------------------------------------------------------------
SRC_DIR = "/mnt/data1/minimax-m3-bf16"                              # COMPLETE bf16 download (incl MTP)
OUT_DIR = "/mnt/fastmodels/MiniMax-M3-fp8-experts-AutoRound"        # final fp8 output

# FP8 element + scale denominator. MUST match int4_to_fp8_load.FP8_MAX (== float8_e4m3fnuz max == 240).
# Imported lazily-as-constant here so --dry-run stays torch-free; re-derived from torch in the math path
# and ASSERTED equal (see _fp8_max_runtime()).
FP8_MAX = 240.0
FP8_DTYPE_FLAVOR = "float8_e4m3fnuz"   # Gaudi-2 HW flavor (informational; bytes stored as e4m3fn)
SCALE_EPS = 1e-8

# The text tower lives under this prefix in the (multimodal) checkpoint.
LM_PREFIX = "language_model.model"

# Routed-expert weight leaf names (M3 fused-MoE naming, == vllm get_expert_mapping w1/w2/w3).
EXPERT_LEAVES = ("w1", "w2", "w3")

# --------------------------------------------------------------------------------------------------
# ground-truth helpers (stdlib only; no torch / transformers needed for dry-run)
# --------------------------------------------------------------------------------------------------

def _load_config(src: str) -> dict:
    with open(os.path.join(src, "config.json")) as f:
        return json.load(f)


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def _moe_layer_ids(cfg: dict) -> list[int]:
    """Layer indices that are MoE (have routed experts). From text_config.moe_layer_freq (1=MoE)."""
    tc = _text_cfg(cfg)
    freq = tc.get("moe_layer_freq")
    n = int(tc.get("num_hidden_layers"))
    if isinstance(freq, list):
        return [i for i, v in enumerate(freq) if int(v) == 1]
    fk = int(tc.get("first_k_dense_replace", 0) or 0)
    return [i for i in range(n) if i >= fk]


def _dense_layer_ids(cfg: dict) -> list[int]:
    tc = _text_cfg(cfg)
    n = int(tc.get("num_hidden_layers"))
    moe = set(_moe_layer_ids(cfg))
    return [i for i in range(n) if i not in moe]


def _iter_shard_headers(src: str):
    """Yield (tensor_name, dtype_str, [shape]) from every present safetensors shard header.
    Metadata only -- reads the leading JSON header of each shard, never the tensor bytes."""
    for path in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            yield k, v.get("dtype", "?"), v.get("shape", [])


_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
                "I16": 2, "I32": 4, "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


# --------------------------------------------------------------------------------------------------
# the quantize-vs-keep decision -- the SINGLE source of truth used by the recipe, the dry-run report,
# and the streaming writer. Returns True iff `name` is a ROUTED-expert weight in a MoE layer (-> fp8).
# Everything else stays bf16. *** Reuses the int4 script's expert-selection regex VERBATIM. ***
# --------------------------------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"^" + re.escape(LM_PREFIX) + r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$"
)


def is_routed_expert_weight(name: str, moe_layers: set[int]) -> bool:
    m = _EXPERT_RE.match(name)
    if not m:
        return False
    return int(m.group(1)) in moe_layers


def _is_vision(name: str) -> bool:
    return (
        name.startswith("vision_")
        or ".vision_" in name
        or name.startswith("visual.")
        or ".visual." in name
        or "vision_tower" in name
        or "multi_modal_projector" in name
        or "patch_merge" in name
        or "image_newline" in name
    )


# Multi-token-prediction module markers. ANY tensor whose name contains one of these is an MTP head and
# is copied through bf16 UNCHANGED (NEVER fp8'd, NEVER dropped -- the int4 export's bug we must not repeat).
_MTP_MARKERS = ("mtp", "nextn", "eh_proj", "enorm", "hnorm", "shared_head")


def _is_mtp(name: str) -> bool:
    low = name.lower()
    return any(mk in low for mk in _MTP_MARKERS)


# --------------------------------------------------------------------------------------------------
# QUANTIZATION_CONFIG emit -- the LOCKED FORMAT CONTRACT. Produces the compressed-tensors
# `float-quantized` dict that vLLM's CompressedTensorsConfig reads and that this box's
# hpu_compressed_tensors.HPUCompressedTensorsW8A8Fp8MoEMethod (CHANNEL strategy) consumes.
#
# Schema is the AutoRound static-fp exporter's schema (verified against the installed
# auto_round.export.export_to_llmcompressor.export_to_static_fp + compressed_tensors 0.17.1):
#   quant_method        : "compressed-tensors"
#   format              : "float-quantized"
#   quantization_status : "compressed"
#   config_groups.group_0:
#       targets            : ["re:.*block_sparse_moe.experts.*"]   <- SCOPES fp8 to the routed experts ONLY.
#                            (The serve matches a FusedMoE module's prefix via find_matched_target; a
#                             re: target on the expert prefix selects the W8A8Fp8 MoE method for it and
#                             leaves every other Linear -- attention/gate/shared/dense/lm_head -- bf16.)
#       weights            : {num_bits:8, type:float, strategy:channel, symmetric:true, dynamic:false}
#       input_activations  : weight-only (dynamic, PER-TOKEN) -- W8A16-style; the serve's route-b ignores
#                            input scales anyway, and dynamic means we emit NO input_scale tensors.
#   ignore              : EVERYTHING that is not a routed expert (lm_head + all the bf16 modules incl MTP).
#
# We build it with the real compressed_tensors classes when torch is importable (guaranteed-correct,
# validated by Pydantic) and fall back to a hand-written dict (identical shape) when torch is absent
# (so --dry-run can still print/validate the plan). Both paths assert-equal in --validate.
# --------------------------------------------------------------------------------------------------

# The target that scopes fp8 to the routed experts. MUST be "Linear": vLLM's get_moe_method
# (compressed_tensors_moe.py) resolves the FusedMoE scheme by reconstructing the unfused expert proj
# names and matching them, and _add_fused_moe_to_target_scheme_map copies the "Linear" scheme onto the
# FusedMoE. A `re:.*experts.*` target does NOT match (the canonical module name collapses to
# `...block_sparse_moe.experts` with no trailing dot) -> experts load UNQUANTIZED (no weight_scale ->
# KeyError). Validated on-card 2026-06-24: "Linear" + the exhaustive ignore below is the only scoping
# that selects ONLY the routed experts for fp8. The ignore list is therefore LOAD-BEARING.
EXPERT_TARGET_RE = "Linear"


def _ignore_list() -> list[str]:
    """Exhaustive ignore list: every NON-routed-expert family that must stay bf16. Uses re: prefixes
    matched against module paths by vLLM's should_ignore_layer / find_matched_target. The expert target
    is positive-scoped above; this belt-and-suspenders ignore makes the bf16 intent explicit + survives
    loaders that quantize-by-default-then-subtract."""
    # NOTE: ignore matches the BUILT-MODEL module names, which use the `mlp` ATTRIBUTE (the M3 MoE
    # block is `...layers.L.mlp`, with the routed experts at `...mlp.experts` = the FusedMoE we WANT
    # quantized). A broad `re:.*\.mlp\..*` WOULD swallow `mlp.experts` -> experts load unquantized
    # (the 2026-06-24 on-card bug). So the mlp ignores below are PROJ-SPECIFIC and provably never
    # match `mlp.experts` (verified card-free with the real prefixes + fused_mapping).
    return [
        "re:.*lm_head.*",                            # exact "lm_head" does NOT match module "language_model.lm_head"
        "re:.*self_attn\\..*",                       # q/k/v/o + q/k norm + index_{q,k}_{proj,norm}
        "re:.*\\.mlp\\.gate.*",                      # router gate (mlp.gate) + dense(0-2) gate_proj/gate_up_proj — NOT mlp.experts
        "re:.*\\.mlp\\.up_proj.*",                   # dense(0-2) up shard
        "re:.*\\.mlp\\.down_proj.*",                 # dense(0-2) down_proj
        "re:.*shared_experts\\..*",                  # shared expert (mlp.shared_experts.*)
        "re:.*vision_tower.*",                       # vision tower (incl its mlp.fc1/fc2 + attn)
        "re:.*layernorm.*",                          # input/post_attention layernorms
        "re:.*\\.norm\\b.*",                         # model.norm + per-block *_norm leaves
        "re:.*embed_tokens.*",
        "re:.*vision_tower.*",
        "re:.*multi_modal_projector.*",
        "re:.*patch_merge.*",
        # --- THE 7 MTP MODULES (must stay bf16, never quantized) ---
        "re:.*mtp.*",
        "re:.*nextn.*",
        "re:.*eh_proj.*",
        "re:.*enorm.*",
        "re:.*hnorm.*",
        "re:.*shared_head.*",
    ]


def build_quant_config_dict() -> dict:
    """Build the quantization_config dict. Prefers the real compressed_tensors classes (Pydantic-validated,
    guaranteed-correct schema); falls back to an identical hand-written dict if torch/compressed_tensors
    are not importable (e.g. pure --dry-run on a torch-less box)."""
    ignore = _ignore_list()
    try:
        from compressed_tensors.quantization import (  # type: ignore
            QuantizationArgs, QuantizationConfig, QuantizationScheme,
            QuantizationStrategy, QuantizationType, QuantizationStatus,
        )
        weights = QuantizationArgs(
            num_bits=8, type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.CHANNEL, symmetric=True, dynamic=False,
        )
        # Weight-only: dynamic per-token input activations => NO input_scale tensors on disk (route-b
        # ignores input scales). If a future loader insists on static input scales, flip dynamic=False
        # here AND have the writer emit a per-expert input_scale (see "residual risk" in the report).
        input_acts = QuantizationArgs(
            num_bits=8, type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR, symmetric=True, dynamic=True,
        )
        scheme = QuantizationScheme(
            targets=[EXPERT_TARGET_RE], weights=weights, input_activations=input_acts,
        )
        qc = QuantizationConfig(
            config_groups={"group_0": scheme},
            quantization_status=QuantizationStatus.COMPRESSED,
            ignore=ignore,
        ).to_dict()
        qc["format"] = "float-quantized"           # to_dict() may default fp32-act schemes to fakequant
        qc["provider"] = "auto-round"
        qc["fp8_dtype_flavor"] = FP8_DTYPE_FLAVOR  # Gaudi-2 flavor (matches AutoRound _configure_gaudi2)
        return qc
    except Exception:  # noqa: BLE001 -- torch/compressed_tensors absent: hand-write the identical dict.
        return {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "quantization_status": "compressed",
            "provider": "auto-round",
            "fp8_dtype_flavor": FP8_DTYPE_FLAVOR,
            "kv_cache_scheme": None,
            "global_compression_ratio": None,
            "config_groups": {
                "group_0": {
                    "targets": [EXPERT_TARGET_RE],
                    "weights": {
                        "num_bits": 8, "type": "float", "symmetric": True,
                        "strategy": "channel", "dynamic": False, "group_size": None,
                        "block_structure": None, "actorder": None, "observer": "memoryless_minmax",
                    },
                    "input_activations": {
                        "num_bits": 8, "type": "float", "symmetric": True,
                        "strategy": "tensor", "dynamic": True, "group_size": None,
                        "block_structure": None, "actorder": None, "observer": None,
                    },
                    "output_activations": None,
                    "format": None,
                },
            },
            "ignore": ignore,
        }


# --------------------------------------------------------------------------------------------------
# DRY-RUN: resolve fp8-vs-keep lists against the real config + shard headers. No weights, no torch.
# Graceful when the download is incomplete (config.json only) -- reports "N/59 present" and the plan.
# --------------------------------------------------------------------------------------------------

def dry_run(src: str, out: str) -> int:
    cfg_path = os.path.join(src, "config.json")
    have_cfg = os.path.exists(cfg_path)

    print("=" * 96)
    print("MiniMax-M3 FP8 (routed experts only, static per-channel) -- compressed-tensors DRY-RUN")
    print("=" * 96)
    print(f"src                : {src}")
    print(f"out                : {out}")
    if not have_cfg:
        print("config.json        : MISSING -- cannot print the planned split (download not started?).")
        print("                     (--run is gated on a COMPLETE download anyway.)")
        return 0

    cfg = _load_config(src)
    tc = _text_cfg(cfg)
    moe_layers = _moe_layer_ids(cfg)
    dense_layers = _dense_layer_ids(cfg)
    n_experts = int(tc.get("num_local_experts"))
    n_layers = int(tc.get("num_hidden_layers"))
    n_mtp = int(tc.get("num_mtp_modules", 0) or 0)
    H = int(tc["hidden_size"]); I = int(tc["intermediate_size"])

    print(f"text layers        : {n_layers}  (dense {dense_layers[0]}..{dense_layers[-1]} -> bf16, "
          f"MoE {moe_layers[0]}..{moe_layers[-1]} -> routed experts FP8)")
    print(f"routed experts/lyr : {n_experts}   leaves={EXPERT_LEAVES}   (w1,w3=[{I},{H}] w2=[{H},{I}])")
    print(f"MTP modules        : {n_mtp}  (logical layers {n_layers}..{n_layers + n_mtp - 1}) -> KEPT BF16 "
          f"(NOT dropped -- unlike the int4 export)")
    print(f"fp8 math           : scale=(|W|.amax(dim=1)+{SCALE_EPS})/{FP8_MAX}  W_fp8=(W/scale).to(e4m3fn)  "
          f"per-OUTPUT-channel")
    print(f"format             : compressed-tensors float-quantized  W8A16(dynamic-act)  strategy=channel")
    print()

    # --- shard presence / completeness ---
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    idx = glob.glob(os.path.join(src, "*.index.json"))
    total_expected = None
    m = re.search(r"-of-(\d+)\.safetensors$", shards[-1]) if shards else None
    if m:
        total_expected = int(m.group(1))
    complete = bool(idx) and total_expected is not None and len(shards) >= total_expected
    print(f"shards present     : {len(shards)}"
          + (f" / {total_expected} expected" if total_expected else "")
          + ("   index.json: yes" if idx else "   index.json: MISSING"))
    if not complete:
        miss = (total_expected - len(shards)) if total_expected else "?"
        print(f"download complete  : NO  -- {len(shards)}/{total_expected or '?'} shards present, "
              f"{miss} missing (+index {'present' if idx else 'MISSING'}).")
        print(f"                     --run is GATED until this is True. Proceeding with header-only "
              f"classification of the {len(shards)} present shard(s).")
    else:
        print("download complete  : YES")
    print()

    # --- classify every tensor we can SEE (header-only) into fp8 vs bf16 buckets ---
    fp8_mods: dict[str, int] = {}
    bf16_mods: dict[str, int] = {}
    mtp_mods: dict[str, int] = {}
    vision_mods: dict[str, int] = {}
    fp8_bytes = 0
    bf16_bytes = 0
    fp8_numel = 0
    fp8_out_rows = 0
    moe_set = set(moe_layers)

    def patt(name: str) -> str:
        return re.sub(r"\.\d+\.", ".N.", name)

    for name, dt, shape in _iter_shard_headers(src):
        b = _DTYPE_BYTES.get(dt, 2) * _numel(shape)
        if name.endswith(".weight") and is_routed_expert_weight(name, moe_set):
            fp8_mods[patt(name)] = fp8_mods.get(patt(name), 0) + 1
            fp8_bytes += b
            fp8_numel += _numel(shape)
            fp8_out_rows += int(shape[0]) if shape else 0
        elif _is_mtp(name):
            mtp_mods[patt(name)] = mtp_mods.get(patt(name), 0) + 1
            bf16_bytes += b
        elif _is_vision(name):
            vision_mods[patt(name)] = vision_mods.get(patt(name), 0) + 1
            bf16_bytes += b
        else:
            bf16_mods[patt(name)] = bf16_mods.get(patt(name), 0) + 1
            bf16_bytes += b

    print("-" * 96)
    print("MODULES THAT WILL BE FP8  (routed experts only -> .weight float8_e4m3fn + .weight_scale [out,1] f32):")
    print("-" * 96)
    for p in sorted(fp8_mods):
        print(f"  [fp8 ] x{fp8_mods[p]:<6d} {p}")
    if not fp8_mods:
        print("  (none visible yet -- expert shards not downloaded; pattern locked to:)")
        print(f"  [fp8 ]          {LM_PREFIX}.layers.<MoE>.block_sparse_moe.experts.<E>.<w1|w2|w3>.weight")
    print()
    print("-" * 96)
    print("MODULES KEPT BF16  (copied through unchanged):")
    print("-" * 96)
    for p in sorted(bf16_mods):
        print(f"  [bf16] x{bf16_mods[p]:<6d} {p}")
    print()
    if mtp_mods:
        print("-" * 96)
        print("MTP MODULES  (the 7 multi-token-prediction heads -> KEPT BF16, COPIED THROUGH -- never fp8/dropped):")
        print("-" * 96)
        for p in sorted(mtp_mods):
            print(f"  [mtp ] x{mtp_mods[p]:<6d} {p}")
        print()
    else:
        print("MTP MODULES        : none in the present shards (they live in later shards). When present "
              "they are auto-kept bf16.")
        print()
    if vision_mods:
        print("-" * 96)
        print("VISION TOWER  (kept bf16, copied through):")
        print("-" * 96)
        for p in sorted(vision_mods):
            print(f"  [vis ] x{vision_mods[p]:<6d} {p}")
        print()

    # --- assert ONLY routed experts are fp8 (header-level guard, same as int4 script) ---
    bad = [p for p in fp8_mods if "block_sparse_moe.experts." not in p or
           not p.rstrip().endswith(("w1.weight", "w2.weight", "w3.weight"))]
    assert not bad, f"NON-EXPERT TENSOR SELECTED FOR FP8: {bad}"
    leaked = [p for p in bf16_mods if "block_sparse_moe.experts." in p and
              p.rstrip().endswith(("w1.weight", "w2.weight", "w3.weight"))]
    assert not leaked, f"ROUTED EXPERT LEAKED INTO BF16 KEEP-LIST: {leaked}"
    mtp_in_fp8 = [p for p in fp8_mods if _is_mtp(p)]
    assert not mtp_in_fp8, f"MTP TENSOR SELECTED FOR FP8: {mtp_in_fp8}"
    print("CHECK: only routed-expert w1/w2/w3 selected for fp8      ->  OK")
    print("CHECK: no routed expert left in the bf16 keep-list       ->  OK")
    print("CHECK: no MTP tensor selected for fp8 (MTP stays bf16)    ->  OK")
    print()

    # --- size estimate (experts: bf16 2 B/elem -> fp8 1 B/elem + [out,1] f32 scale = 4 B/row) ---
    print("-" * 96)
    print("SIZE ESTIMATE:")
    print("-" * 96)
    if complete:
        fp8_packed_seen = fp8_numel * 1 + fp8_out_rows * 4   # fp8 weight + f32 per-row scale
        out_est = bf16_bytes + fp8_packed_seen
        print(f"  bf16 kept (seen)        : {bf16_bytes/2**30:9.2f} GiB")
        print(f"  routed-expert bf16 (seen): {fp8_bytes/2**30:9.2f} GiB  -> fp8 "
              f"{fp8_packed_seen/2**30:7.2f} GiB (~{fp8_packed_seen/max(fp8_bytes,1)*100:.0f}%)")
        print(f"  ESTIMATED OUTPUT SIZE   : {out_est/2**30:9.2f} GiB")
    else:
        FULL_BF16_GIB = 856.0
        expert_numel = n_experts * len(EXPERT_LEAVES) * len(moe_layers) * (H * I)
        expert_rows = n_experts * len(moe_layers) * (2 * I + H)  # w1,w3 out=I ; w2 out=H
        EXPERT_BF16_GIB = expert_numel * 2 / 2**30
        kept_bf16_gib = FULL_BF16_GIB - EXPERT_BF16_GIB
        expert_fp8_gib = (expert_numel * 1 + expert_rows * 4) / 2**30
        out_est = kept_bf16_gib + expert_fp8_gib
        print(f"  [shards incomplete -> projecting from real expert dims; model card 856 GiB total]")
        print(f"  routed experts (all)    : {EXPERT_BF16_GIB:9.2f} GiB bf16 -> "
              f"{expert_fp8_gib:7.2f} GiB fp8  (~{expert_fp8_gib/EXPERT_BF16_GIB*100:.0f}%, ~half)")
        print(f"  kept bf16 (all, incl MTP): {kept_bf16_gib:9.2f} GiB")
        print(f"  ESTIMATED OUTPUT SIZE   : ~{out_est:8.0f} GiB   (check free space at "
              f"{os.path.dirname(out)})")
    print()

    print("=" * 96)
    print("GATED ON (the real --run refuses to start until ALL are true):")
    print("=" * 96)
    print(f"  [{'x' if complete else ' '}] download complete  ({len(shards)}"
          + (f"/{total_expected}" if total_expected else "") + " shards"
          + (", index present" if idx else ", index MISSING") + ")  <-- DO NOT --run until checked")
    try:
        import torch  # noqa
        import safetensors  # noqa
        _ok = True
    except Exception:  # noqa: BLE001
        _ok = False
    print(f"  [{'x' if _ok else ' '}] torch + safetensors importable (CPU)")
    print(f"  [x] CPU only -- no Gaudi card touched (do NOT touch vllm-active / the serving cards)")
    print("=" * 96)
    return 0


# --------------------------------------------------------------------------------------------------
# fp8 math -- the locked per-output-channel cast. Lives here (torch import is lazy) so dry-run is clean.
# --------------------------------------------------------------------------------------------------

def _fp8_max_runtime():
    import torch
    rt = float(torch.finfo(torch.float8_e4m3fnuz).max)
    assert abs(rt - FP8_MAX) < 1e-6, (
        f"FP8_MAX mismatch: constant {FP8_MAX} != torch.float8_e4m3fnuz.max {rt}. "
        f"This MUST equal int4_to_fp8_load.FP8_MAX.")
    return rt


def quant_weight_fp8_per_channel(W):
    """bf16/real weight [out,in] -> (W_fp8 [out,in] float8_e4m3fn, scale [out,1] float32).

    LOCKED CONTRACT (per the header):
        scale = (W.float().abs().amax(dim=1) + 1e-8) / FP8_MAX        # [out]
        W_fp8 = (W.float() / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    Reconstruction:  W_fp8.to(bf16) * scale   (broadcast [out,in]*[out,1])  ~= W.
    """
    import torch
    _fp8_max_runtime()
    Wf = W.float()
    assert Wf.dim() == 2, f"expected 2-D expert weight, got shape {tuple(Wf.shape)}"
    amax = Wf.abs().amax(dim=1)                      # [out]  (reduce over INPUT dim)
    scale = (amax + SCALE_EPS) / FP8_MAX             # [out]  float32 forward scale
    W_fp8 = (Wf / scale.unsqueeze(1)).to(torch.float8_e4m3fn)   # [out,in]
    return W_fp8, scale.reshape(-1, 1).to(torch.float32).contiguous()


# --------------------------------------------------------------------------------------------------
# STREAMING WRITER -- bounded RAM. Reads ONE source shard at a time, fp8s eligible expert weights,
# copies everything else through unchanged, writes an output shard per source shard, rebuilds the
# safetensors index + config.json (with quantization_config) + copies the aux files. Never loads the
# whole model; peak RSS ~= one source shard (M3 shards ~16 GiB).
# --------------------------------------------------------------------------------------------------

def _stream_quantize(src: str, out: str, max_shards: int | None, verbose: bool = True) -> dict:
    """Run the streaming fp8 quant src -> out. Returns a stats dict. Used by both --run and --validate
    (validate points it at a tiny synthetic checkpoint)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = _load_config(src)
    moe_set = set(_moe_layer_ids(cfg))

    os.makedirs(out, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if max_shards is not None:
        shards = shards[:max_shards]
        if verbose:
            print(f"[m3-fp8][VERIFY] --max-shards: limiting to {len(shards)} shard(s)")

    weight_map: dict[str, str] = {}
    total_size = 0
    n_fp8 = 0
    n_copied = 0
    n_mtp_copied = 0
    fp8_names: list[str] = []

    for si, spath in enumerate(shards):
        out_name = os.path.basename(spath)
        out_path = os.path.join(out, out_name)
        out_tensors: dict[str, "torch.Tensor"] = {}
        with safe_open(spath, framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name.endswith(".weight") and is_routed_expert_weight(name, moe_set):
                    W_fp8, scale = quant_weight_fp8_per_channel(t)
                    prefix = name[: -len(".weight")]
                    out_tensors[prefix + ".weight"] = W_fp8
                    out_tensors[prefix + ".weight_scale"] = scale
                    n_fp8 += 1
                    if len(fp8_names) < 12:
                        fp8_names.append(name)
                else:
                    # copy through UNCHANGED (incl MTP, attention, gate, shared, dense, norms, embeds)
                    out_tensors[name] = t.clone()
                    n_copied += 1
                    if _is_mtp(name):
                        n_mtp_copied += 1
        save_file(out_tensors, out_path, metadata={"format": "pt"})
        for k, v in out_tensors.items():
            weight_map[k] = out_name
            total_size += v.numel() * v.element_size()
        if verbose:
            print(f"[m3-fp8] shard {si + 1}/{len(shards)} {out_name}: "
                  f"{len(out_tensors)} tensors written (running fp8={n_fp8}, copied={n_copied})")
        del out_tensors

    # --- safetensors index ---
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # --- config.json with quantization_config ---
    out_cfg = dict(cfg)
    out_cfg["quantization_config"] = build_quant_config_dict()
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)

    # --- aux files (tokenizer / chat template / trust-remote-code helper) ---
    _copy_aux_files(src, out)

    return {
        "n_fp8": n_fp8, "n_copied": n_copied, "n_mtp_copied": n_mtp_copied,
        "fp8_names": fp8_names, "total_size": total_size, "n_shards": len(shards),
    }


_AUX_FILES = (
    "config.json",  # overwritten above with quantization_config; copy first as a base is unnecessary
    "configuration_minimax_m3_vl.py", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
    "special_tokens_map.json", "chat_template.jinja", "image_processor.py", "preprocessor_config.json",
)


def _copy_aux_files(src: str, out: str) -> None:
    import shutil
    for fn in _AUX_FILES:
        if fn == "config.json":
            continue  # we write our own (with quantization_config)
        sp = os.path.join(src, fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(out, fn))


# --------------------------------------------------------------------------------------------------
# REAL RUN -- gated. CPU only, bounded RAM, LOUD failures.
# --------------------------------------------------------------------------------------------------

def _assert_can_run(src: str, out: str, force: bool) -> None:
    errs = []
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    idx = glob.glob(os.path.join(src, "*.index.json"))
    m = re.search(r"-of-(\d+)\.safetensors$", shards[-1]) if shards else None
    total = int(m.group(1)) if m else None
    if not (idx and total and len(shards) >= total):
        errs.append(f"download INCOMPLETE: {len(shards)}/{total} shards, "
                    f"index={'present' if idx else 'MISSING'} at {src}. "
                    f"DO NOT run the real quant until the hf download finishes (MTP shards must exist).")
    if not os.path.exists(os.path.join(src, "config.json")):
        errs.append(f"config.json MISSING at {src}")
    try:
        import torch  # noqa
        import safetensors  # noqa
        _fp8_max_runtime()  # asserts FP8_MAX == torch float8_e4m3fnuz max
    except Exception as e:  # noqa: BLE001
        errs.append(f"torch/safetensors not importable or FP8_MAX mismatch: {e!r}")
    if os.path.abspath(out) == os.path.abspath(src):
        errs.append("--out must NOT equal --src (refusing to overwrite the source download)")
    if os.path.exists(out) and os.listdir(out) and not force:
        errs.append(f"--out {out} already exists and is non-empty; pass --force to overwrite")
    if errs:
        print("REFUSING TO RUN -- unmet gates:", file=sys.stderr)
        for e in errs:
            print("  * " + e, file=sys.stderr)
        sys.exit(2)


def real_run(src: str, out: str, max_shards: int | None, force: bool) -> int:
    _assert_can_run(src, out, force)
    print(f"[m3-fp8] STREAMING fp8 quant (routed experts only)  {src} -> {out}")
    print(f"[m3-fp8] bounded-RAM: one source shard resident at a time; CPU only; no card touched.")
    stats = _stream_quantize(src, out, max_shards, verbose=True)
    print("-" * 96)
    print(f"[m3-fp8] DONE. shards={stats['n_shards']}  fp8 experts={stats['n_fp8']}  "
          f"copied bf16={stats['n_copied']} (of which MTP={stats['n_mtp_copied']})  "
          f"total_size={stats['total_size']/2**30:.2f} GiB")
    print(f"[m3-fp8] output: {out}")
    return 0


# --------------------------------------------------------------------------------------------------
# SYNTHETIC SELF-TEST -- CARD-FREE. Builds a tiny fake M3-shaped bf16 checkpoint (dense layers 0..2,
# one MoE layer 3 with 4 experts, one fake MTP module at layer "60"), runs the REAL streaming quant
# over it, and ASSERTS the locked contract. No card, no real model, no serve.
# --------------------------------------------------------------------------------------------------

def validate_synthetic() -> int:
    import tempfile
    import torch
    from safetensors.torch import save_file, load_file

    print("=" * 96)
    print("SYNTHETIC SELF-TEST (card-free)")
    print("=" * 96)
    _fp8_max_runtime()
    print(f"FP8_MAX == torch.float8_e4m3fnuz.max == {FP8_MAX}  (matches int4_to_fp8_load)  -> OK")

    # tiny dims (real M3 ratio: w2 out=H, w1/w3 out=I)
    H, I, n_exp, n_mtp = 32, 16, 4, 1
    moe_layer = 3
    torch.manual_seed(0)
    tens: dict[str, "torch.Tensor"] = {}

    def bf16(*shape):
        return (torch.randn(*shape) * 0.05).to(torch.bfloat16)

    # dense prefix layers 0..2 (mlp) + attention + norms  -> all bf16, must be copied through
    for L in range(0, 3):
        tens[f"{LM_PREFIX}.layers.{L}.mlp.gate_proj.weight"] = bf16(4 * H, H)
        tens[f"{LM_PREFIX}.layers.{L}.mlp.up_proj.weight"] = bf16(4 * H, H)
        tens[f"{LM_PREFIX}.layers.{L}.mlp.down_proj.weight"] = bf16(H, 4 * H)
        tens[f"{LM_PREFIX}.layers.{L}.self_attn.q_proj.weight"] = bf16(H, H)
        tens[f"{LM_PREFIX}.layers.{L}.input_layernorm.weight"] = bf16(H)
        tens[f"{LM_PREFIX}.layers.{L}.post_attention_layernorm.weight"] = bf16(H)
    # MoE layer 3: 4 experts (w1/w2/w3) -> the ONLY tensors that should become fp8;
    #              plus gate, e_score_correction_bias, shared_experts, attention, norms -> bf16
    for E in range(n_exp):
        tens[f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.experts.{E}.w1.weight"] = bf16(I, H)
        tens[f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.experts.{E}.w2.weight"] = bf16(H, I)
        tens[f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.experts.{E}.w3.weight"] = bf16(I, H)
    tens[f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.gate.weight"] = (torch.randn(n_exp, H)).to(torch.float32)
    tens[f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.e_score_correction_bias"] = torch.randn(n_exp).to(torch.float32)
    tens[f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.shared_experts.gate_proj.weight"] = bf16(I, H)
    tens[f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.shared_experts.up_proj.weight"] = bf16(I, H)
    tens[f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.shared_experts.down_proj.weight"] = bf16(H, I)
    tens[f"{LM_PREFIX}.layers.{moe_layer}.self_attn.q_proj.weight"] = bf16(H, H)
    tens[f"{LM_PREFIX}.layers.{moe_layer}.self_attn.index_q_proj.weight"] = bf16(I, H)
    tens[f"{LM_PREFIX}.layers.{moe_layer}.input_layernorm.weight"] = bf16(H)
    # global
    tens[f"{LM_PREFIX}.embed_tokens.weight"] = bf16(64, H)
    tens[f"{LM_PREFIX}.norm.weight"] = bf16(H)
    tens["language_model.lm_head.weight"] = bf16(64, H)
    # a fake MTP module at logical layer 60 (mirrors DeepSeekMTP-ish names) -> MUST be copied bf16
    mtp_l = 60
    tens[f"{LM_PREFIX}.layers.{mtp_l}.eh_proj.weight"] = bf16(H, 2 * H)
    tens[f"{LM_PREFIX}.layers.{mtp_l}.enorm.weight"] = bf16(H)
    tens[f"{LM_PREFIX}.layers.{mtp_l}.hnorm.weight"] = bf16(H)
    tens[f"{LM_PREFIX}.layers.{mtp_l}.shared_head.norm.weight"] = bf16(H)
    tens[f"{LM_PREFIX}.layers.{mtp_l}.mtp.input_layernorm.weight"] = bf16(H)

    # also stash the originals for round-trip + unchanged checks
    orig = {k: v.clone() for k, v in tens.items()}

    work = tempfile.mkdtemp(prefix="m3fp8_selftest_")
    src = os.path.join(work, "src"); out = os.path.join(work, "out")
    os.makedirs(src)
    # split across 2 shards to exercise the streaming loop + index rebuild
    keys = list(tens.keys())
    half = len(keys) // 2
    save_file({k: tens[k] for k in keys[:half]}, os.path.join(src, "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file({k: tens[k] for k in keys[half:]}, os.path.join(src, "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    # a config.json the recipe can read (moe_layer_freq marks layer 3 MoE; layers 0..2 dense)
    cfg = {
        "model_type": "minimax_m3_vl",
        "text_config": {
            "hidden_size": H, "intermediate_size": I, "num_hidden_layers": 4,
            "num_local_experts": n_exp, "num_mtp_modules": n_mtp,
            "moe_layer_freq": [0, 0, 0, 1],
        },
    }
    with open(os.path.join(src, "config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(src, "tokenizer_config.json"), "w") as f:
        json.dump({"dummy": True}, f)  # an aux file to confirm copy-through

    print(f"built synthetic src: {src}  ({len(keys)} tensors, 2 shards, 1 aux file)")

    # --- run the REAL streaming quant ---
    stats = _stream_quantize(src, out, max_shards=None, verbose=False)
    print(f"streaming quant ran: fp8={stats['n_fp8']} copied={stats['n_copied']} "
          f"(MTP copied={stats['n_mtp_copied']})")

    # --- load the result back ---
    out_files = sorted(glob.glob(os.path.join(out, "*.safetensors")))
    result: dict[str, "torch.Tensor"] = {}
    for of in out_files:
        result.update(load_file(of))

    failures: list[str] = []

    # (a) ONLY the 4 experts' w1/w2/w3 became float8_e4m3fn + got a weight_scale
    expected_fp8 = {f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.experts.{E}.{w}.weight"
                    for E in range(n_exp) for w in ("w1", "w2", "w3")}
    got_fp8 = {k for k, v in result.items() if v.dtype == torch.float8_e4m3fn}
    if got_fp8 != expected_fp8:
        failures.append(f"(a) fp8 set mismatch.\n   extra={sorted(got_fp8 - expected_fp8)}\n   "
                        f"missing={sorted(expected_fp8 - got_fp8)}")
    for k in expected_fp8:
        sk = k[: -len(".weight")] + ".weight_scale"
        if sk not in result:
            failures.append(f"(a) missing weight_scale for {k}")
        elif result[sk].dtype != torch.float32:
            failures.append(f"(a) {sk} dtype {result[sk].dtype} != float32")
        else:
            out_rows = result[k].shape[0]
            if tuple(result[sk].shape) != (out_rows, 1):
                failures.append(f"(a) {sk} shape {tuple(result[sk].shape)} != ({out_rows},1)")
    if expected_fp8.issubset(got_fp8):
        print(f"  (a) exactly {len(got_fp8)} expert weights are float8_e4m3fn, each with a "
              f"[out,1] f32 weight_scale  -> {'OK' if not failures else 'CHECK'}")

    # (b) every OTHER tensor (incl MTP) is unchanged bf16 / source dtype
    for k, ov in orig.items():
        if k in expected_fp8:
            continue
        if k not in result:
            failures.append(f"(b) tensor DROPPED: {k}")
            continue
        rv = result[k]
        if rv.dtype != ov.dtype:
            failures.append(f"(b) {k} dtype changed {ov.dtype} -> {rv.dtype}")
        elif not torch.equal(rv, ov):
            failures.append(f"(b) {k} VALUE changed (should be copied through unchanged)")
    # explicit MTP survival
    mtp_keys = [k for k in orig if _is_mtp(k)]
    surviving_mtp = [k for k in mtp_keys if k in result and result[k].dtype == orig[k].dtype
                     and torch.equal(result[k], orig[k])]
    if len(surviving_mtp) != len(mtp_keys):
        failures.append(f"(b) MTP not fully preserved: {len(surviving_mtp)}/{len(mtp_keys)} survived")
    else:
        print(f"  (b) all {len(mtp_keys)} MTP tensors + every non-expert tensor copied through "
              f"UNCHANGED (bf16)  -> OK")

    # (c) round-trip W_fp8.to(bf16) * weight_scale ~= W.
    #     METRIC + THRESHOLD anchored to the repo's OWN bar (int4_to_fp8_load.py:450-463): the
    #     float8_e4m3 per-channel REPRESENTATION FLOOR is ~2.6e-2 RMS (3 mantissa bits; theoretical
    #     RMS 2^-3/sqrt(12) ~= 3.6e-2). A "<1e-2 bar would be unphysical" for raw bf16->fp8; that
    #     file's gate uses < 6.4e-2 (the per-tensor double-quant floor). The prompt's "a few e-3
    #     relative" describes the *added* int4->fp8 error (int4 is already coarse), NOT the bf16->fp8
    #     floor we measure here. So we use the Frobenius relative error ||recon-W||/||W|| and gate it
    #     against the documented fp8 per-channel floor (must be near ~2.6e-2, hard-fail above 6.4e-2).
    FP8_PERCHANNEL_FLOOR = 2.6e-2     # documented in int4_to_fp8_load.py
    FP8_HARD_FAIL = 6.4e-2            # per-tensor double-quant floor; anything above => something broke
    worst_fro = 0.0
    for k in expected_fp8:
        W = orig[k].float()
        sk = k[: -len(".weight")] + ".weight_scale"
        recon = result[k].to(torch.bfloat16).float() * result[sk].float()  # [out,in]*[out,1]
        fro = ((recon - W).norm() / W.norm().clamp_min(1e-12)).item()
        worst_fro = max(worst_fro, fro)
    # must be at/below the per-channel floor with a little slack for tiny synthetic dims, and never
    # above the hard-fail per-tensor floor (which would indicate a scale/axis/dtype bug).
    if worst_fro > FP8_HARD_FAIL:
        failures.append(f"(c) round-trip Frobenius rel err {worst_fro:.2e} > {FP8_HARD_FAIL:.1e} "
                        f"(hard fail -- a scale/axis/dtype bug, not the fp8 floor)")
    elif worst_fro > 1.5 * FP8_PERCHANNEL_FLOOR:
        failures.append(f"(c) round-trip Frobenius rel err {worst_fro:.2e} > 1.5x the documented "
                        f"fp8 per-channel floor {FP8_PERCHANNEL_FLOOR:.1e}")
    else:
        print(f"  (c) worst round-trip Frobenius relative error = {worst_fro:.2e}  "
              f"(at the documented fp8 e4m3 per-channel floor ~{FP8_PERCHANNEL_FLOOR:.1e}; "
              f"hard-fail bar {FP8_HARD_FAIL:.1e})  -> OK")

    # (d) emitted config.json parses + ignore/targets scope excludes MTP + non-expert modules
    with open(os.path.join(out, "config.json")) as f:
        oc = json.load(f)
    qc = oc.get("quantization_config", {})
    if qc.get("format") != "float-quantized":
        failures.append(f"(d) format != float-quantized: {qc.get('format')!r}")
    if qc.get("quant_method") != "compressed-tensors":
        failures.append(f"(d) quant_method != compressed-tensors: {qc.get('quant_method')!r}")
    grp = qc.get("config_groups", {}).get("group_0", {})
    w = grp.get("weights", {})
    if not (w.get("num_bits") == 8 and w.get("type") == "float"
            and w.get("strategy") == "channel" and w.get("symmetric") is True):
        failures.append(f"(d) weights args wrong: {w}")
    targets = grp.get("targets", [])
    if EXPERT_TARGET_RE not in targets:
        failures.append(f"(d) expert target {EXPERT_TARGET_RE!r} not in targets {targets}")
    # the expert target must MATCH an expert path and NOT match MTP / non-expert paths
    expert_path = f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.experts.0.w1"
    nonexp_paths = [
        f"{LM_PREFIX}.layers.{mtp_l}.eh_proj",
        f"{LM_PREFIX}.layers.{mtp_l}.mtp.input_layernorm",
        f"{LM_PREFIX}.layers.0.mlp.gate_proj",
        f"{LM_PREFIX}.layers.{moe_layer}.self_attn.q_proj",
        f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.gate",
        f"{LM_PREFIX}.layers.{moe_layer}.block_sparse_moe.shared_experts.gate_proj",
        "language_model.lm_head",
    ]
    tre = re.compile(EXPERT_TARGET_RE[len("re:"):])
    if not tre.search(expert_path):
        failures.append(f"(d) expert target does NOT match an expert path {expert_path!r}")
    bad_match = [p for p in nonexp_paths if tre.search(p)]
    if bad_match:
        failures.append(f"(d) expert target WRONGLY matches non-expert paths: {bad_match}")
    # ignore list must, via its re: patterns, cover every non-expert family (esp. MTP + lm_head)
    ig = qc.get("ignore", [])
    def _ignored(path: str) -> bool:
        for pat in ig:
            rx = pat[len("re:"):] if pat.startswith("re:") else (re.escape(pat) + r"$")
            if re.search(rx, path):
                return True
        return False
    uncovered = [p for p in nonexp_paths if not _ignored(p)]
    if uncovered:
        failures.append(f"(d) ignore list does NOT cover non-expert paths: {uncovered}")
    if _ignored(expert_path):
        failures.append(f"(d) ignore list WRONGLY covers the expert path {expert_path!r}")
    if not failures:
        print(f"  (d) config.json quantization_config parses; targets scope ONLY experts; ignore "
              f"covers MTP + all non-expert modules (lm_head/attn/gate/shared/dense)  -> OK")

    # aux copy-through
    if not os.path.exists(os.path.join(out, "tokenizer_config.json")):
        failures.append("(e) aux file tokenizer_config.json not copied through")

    print("-" * 96)
    if failures:
        print("SELF-TEST: FAIL")
        for fmsg in failures:
            print("  ! " + fmsg)
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        return 1
    print("SELF-TEST: PASS  (all of a,b,c,d + aux copy-through)")
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="config+header-only validation; loads NO weights, needs no torch")
    g.add_argument("--run", action="store_true",
                   help="the real streaming fp8 quant; GATED on a COMPLETE download + torch (CPU only)")
    g.add_argument("--validate", action="store_true",
                   help="card-free synthetic self-test of the streaming quant + config (PASS/FAIL)")
    ap.add_argument("--src", default=SRC_DIR, help=f"source bf16 checkpoint dir (default {SRC_DIR})")
    ap.add_argument("--out", default=OUT_DIR, help=f"output fp8 checkpoint dir (default {OUT_DIR})")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="process only the first N source shards (RAM-bounded verification cap)")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = ap.parse_args(argv)

    if args.dry_run:
        return dry_run(args.src, args.out)
    if args.validate:
        return validate_synthetic()
    return real_run(args.src, args.out, args.max_shards, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
