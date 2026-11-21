#!/usr/bin/env python3
# ==================================================================================================
# quantize_m3_int4.py
#
# AutoRound INT4 recipe for the MiniMax-M3 TEXT TOWER, matching this box's proven
# AutoRound -> hpu-moe-ext path (same contract used by INC4AI/GLM-5.1-int4-mixed-AutoRound
# and the Qwen3.5-397B GPTQ-Int4 checkpoints already serving here).
#
# WHAT GETS QUANTIZED (INT4, bits=4 group_size=128 sym, exported auto_round:auto_gptq):
#   * ONLY the 128 ROUTED experts (w1/w2/w3) of the MoE layers (text layers 3..59).
#     Checkpoint tensor names (ground truth from the shard headers):
#       language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w2,w3}.weight
#
# WHAT STAYS BF16 (ignored / kept full precision):
#   * ALL attention            : self_attn.{q,k,v,o}_proj  + self_attn.{q,k}_norm
#   * MSA index projections     : self_attn.index_{q,k}_proj + self_attn.index_{q,k}_norm
#   * Router gate + routing bias: block_sparse_moe.gate + block_sparse_moe.e_score_correction_bias
#   * Shared expert (1)         : block_sparse_moe.shared_experts.{gate,up,down}_proj
#   * Dense-prefix MLPs (0..2)  : layers.{0,1,2}.mlp.{gate,up,down}_proj
#   * Embeddings / lm_head      : model.embed_tokens, lm_head
#   * All norms                 : input_layernorm, post_attention_layernorm, model.norm
#   * VISION TOWER              : text-only quant -> excluded entirely (see VISION note below).
#
# WHY this split: it is exactly the GLM-5.1 "int4-mixed" contract that hpu-moe-ext autodetects.
# hpu-moe-ext treats the *stored* format as ground truth: it scans the safetensors shard headers
# for tensors ending in ".qweight" (auto_round_hpu.py:_load_quantized_prefixes) and only routes
# those modules through the int4 MoE kernel; everything without a .qweight is loaded bf16. So the
# ONLY thing that must be true for a clean serve is: the routed experts (and nothing else) are
# packed int4. This recipe enforces that via AutoRound's per-module `layer_config`.
#
# ------------------------------------------------------------------------------------------------
# CARD-FREE NOTE: importing this module and running `--dry-run` does NOT need torch / transformers /
# auto_round and does NOT load any weights. It parses config.json + the safetensors *headers* only
# (metadata, a few KB per shard). The real quant (`--run`) is GATED — see REQUIREMENTS below.
# ==================================================================================================
#
# ================================  REQUIREMENTS for the REAL run  ================================
# These are HARD gates. `--run` refuses to start unless they are met.
#
# *** MEMORY STRATEGY (the fix): the ~856 GiB bf16 model does NOT fit the ~448 GiB of free host RAM,
# so we do NOT load it. `--run` uses AutoRound 0.13.1's MODEL-FREE RTN path
# (auto_round.compressors.model_free): given the model PATH (string), it streams the safetensors
# shards ONE AT A TIME (1-deep prefetch + 1-deep async write), splits/quantizes eligible 2-D Linear
# weights per-tensor (round-to-nearest + per-group int scales), packs auto_round:auto_gptq, writes
# output shards, and calls clear_memory() after each shard. PEAK RSS plateaus at ~one shard's worth
# (M3 shards are ~14 GiB; measured peak ~17 GiB on a 2-shard run), NEVER 856 GiB. Verified: a 2-shard
# bounded run packed all of layer 25's experts to int4 with attention/gate/shared kept bf16. ***
#
# 1. DOWNLOAD COMPLETE. Source = /mnt/data2/minimax-m3-bf16 (856 GiB, 59 shards
#    model-00001..00059-of-00059 + model.safetensors.index.json). All 59 shards + index present.
#
# 2. transformers IMPORTABLE + config.json present.  (NOT a blocker anymore.)
#    The model-free RTN path does NOT run a forward pass and does NOT build an AutoModel, so it does
#    NOT need runnable M3 modeling code -- only config.json (read directly) + the safetensors. So
#    `models/minimax_m3_vl/` is NOT required (transformers 5.12.0 here happens to ship it, but the
#    path works without it). transformers must still import (AutoConfig/AutoTokenizer + auto_round IO).
#
# 3. auto_round >= 0.13.0 INSTALLED (the model-free streaming RTN path lives here). The CPU venv
#    $HOME/.venvs/m3calib has auto_round 0.13.1 + transformers 5.12.0 + torch 2.12 (CPU).
#
# 4. NO DEVICE NEEDED. model-free RTN is host-side (shard I/O + packing); it runs entirely on CPU and
#    needs NO Gaudi card. This is fully card-free -- do NOT touch vllm-active / the serving cards
#    (GLM serves on all 8). `--device hpu` is accepted but the per-tensor math still runs on CPU.
#
# ================================  OUTPUT  ================================
#   /mnt/fastmodels/MiniMax-M3-int4-experts-AutoRound   (924 GiB free there now)
#   Packed format: auto_round:auto_gptq  (same packing the Qwen/GLM int4 checkpoints use, the one
#   hpu-moe-ext's MoeWNA16HPUMethod._repack_weight + autoround_diag.py validate).
#
# ================================  RUNTIME (estimate)  ================================
#   iters=0  => AutoRound RTN mode (round-to-nearest + group scales, NO gradient tuning). This is
#   what the GLM-5.1 export used (config: "iters": 0). It is dominated by I/O + packing, not GPU
#   compute, so CPU and 1-card are comparable; the device mainly accelerates the calibration
#   forward pass (small at nsamples~128). Rough wall-clock on THIS box:
#       * weights to pack: 128 experts x 3 mats x 57 MoE layers = 21,888 expert matrices (~333 GiB
#         of the 856). Pack throughput is the bottleneck.
#       * CPU (RTN, nsamples=128, seqlen=2048):     ~6-10 h   (pack-bound; ~all of it is unpack/
#                                                              quantize/repack of 21.9k matrices)
#       * 1 Gaudi card (calib fwd on HPU, pack on host): ~4-7 h (fwd is faster; packing still host)
#   If you raise iters>0 (real AutoRound tuning) add HOURS-to-DAYS for a 200B+-class model -- NOT
#   recommended for the first pass; match GLM and keep iters=0.
#   Output size estimate is computed live in --dry-run from the real bf16 tensor sizes.
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
# Static config -- the contract. Edit here, nowhere else.
# --------------------------------------------------------------------------------------------------
SRC_DIR = "/mnt/data2/minimax-m3-bf16"
OUT_DIR = "/mnt/fastmodels/MiniMax-M3-int4-experts-AutoRound"

BITS = 4
GROUP_SIZE = 128
SYM = True
ITERS = 0                      # RTN -- matches the GLM-5.1 int4-mixed export ("iters": 0)
DATA_TYPE = "int"
PACKING_FORMAT = "auto_round:auto_gptq"

# The text tower lives under this prefix in the (multimodal) checkpoint.
LM_PREFIX = "language_model.model"

# Routed-expert weight leaf names (M3 fused-MoE naming, == vllm get_expert_mapping w1/w2/w3).
EXPERT_LEAVES = ("w1", "w2", "w3")

# Calibration: small representative code / agentic / long-context mix. AutoRound defaults otherwise.
NSAMPLES = 128
SEQLEN = 2048
# A compact, license-clean default calibration corpus. Swap/extend freely; keep it representative
# of the serving workload (code, tool-use/agentic transcripts, long context). These are loaded by
# auto_round's dataset plumbing (HF datasets) at run time, NOT at import/dry-run time.
CALIB_DATASET = "NeelNanda/pile-10k"   # generic fallback; see note below to point at a code/agent set

# --------------------------------------------------------------------------------------------------
# ground-truth helpers (stdlib only; no torch / transformers needed for dry-run)
# --------------------------------------------------------------------------------------------------

def _load_config() -> dict:
    with open(os.path.join(SRC_DIR, "config.json")) as f:
        return json.load(f)


def _text_cfg(cfg: dict) -> dict:
    tc = cfg.get("text_config", cfg)
    return tc


def _moe_layer_ids(cfg: dict) -> list[int]:
    """Layer indices that are MoE (have routed experts). From text_config.moe_layer_freq (1=MoE)."""
    tc = _text_cfg(cfg)
    freq = tc.get("moe_layer_freq")
    n = int(tc.get("num_hidden_layers"))
    if isinstance(freq, list):
        return [i for i, v in enumerate(freq) if int(v) == 1]
    # scalar freq + first_k_dense_replace fallback (GLM-style); M3 ships an explicit list.
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
# the quantize-vs-keep decision -- the SINGLE source of truth used by both the recipe and the
# dry-run report. Returns True iff `name` (a full checkpoint tensor name) is a ROUTED-expert weight
# in a MoE layer (i.e. should be INT4). Everything else stays bf16.
# --------------------------------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"^" + re.escape(LM_PREFIX) + r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$"
)


def is_routed_expert_weight(name: str, moe_layers: set[int]) -> bool:
    m = _EXPERT_RE.match(name)
    if not m:
        return False
    layer = int(m.group(1))
    return layer in moe_layers


def _is_vision(name: str) -> bool:
    # Vision tower tensors (text-only quant -> always skip). The text tower is under
    # `language_model.*`; vision params live under `vision_*`/`visual`/`model.vision_*`/projector.
    return (
        name.startswith("vision_")
        or ".vision_" in name
        or name.startswith("visual.")
        or ".visual." in name
        or "vision_tower" in name
        or "multi_modal_projector" in name
        or "image_newline" in name
    )


# --------------------------------------------------------------------------------------------------
# AutoRound layer_config builder. AutoRound quantizes by default; we pin every NON-routed-expert
# module to bits=16 (bf16, skipped). Using explicit per-module entries keyed by the REAL module
# names (sans `.weight`) is the most robust form and mirrors GLM-5.1's exhaustive `extra_config`.
# We ALSO pass regex fallbacks so any module we didn't enumerate defaults to bf16, and only the
# routed experts default to int4.
# --------------------------------------------------------------------------------------------------

def build_layer_config(cfg: dict) -> dict:
    """Return AutoRound `layer_config`: {module_name: {"bits":..,"group_size":..,"sym":..}}.
    Routed experts -> int4; everything else -> bits=16 (kept bf16)."""
    moe_layers = set(_moe_layer_ids(cfg))
    layer_config: dict[str, dict] = {}

    # Enumerate from the *stored* tensor names so the config is grounded in reality, not assumed
    # naming. (Falls back to a synthesized enumeration if shards are not all present yet -- the
    # full run is gated on the download anyway.)
    seen = 0
    for name, _dt, _shape in _iter_shard_headers(SRC_DIR):
        if not name.endswith(".weight"):
            # e_score_correction_bias etc.: not a .weight; never quantized. Pin bf16 by module.
            mod = name[: name.rfind(".")] if "." in name else name
            layer_config.setdefault(mod, {"bits": 16, "data_type": "float"})
            continue
        mod = name[: -len(".weight")]
        seen += 1
        if _is_vision(name):
            layer_config[mod] = {"bits": 16, "data_type": "float"}   # text-only: skip vision
        elif is_routed_expert_weight(name, moe_layers):
            layer_config[mod] = {"bits": BITS, "group_size": GROUP_SIZE, "sym": SYM,
                                 "data_type": DATA_TYPE}
        else:
            layer_config[mod] = {"bits": 16, "data_type": "float"}   # keep bf16
    return layer_config, seen


# --------------------------------------------------------------------------------------------------
# DRY-RUN: resolve quantize-vs-keep lists against the real config + shard headers. No weights loaded.
# --------------------------------------------------------------------------------------------------

def dry_run() -> int:
    cfg = _load_config()
    tc = _text_cfg(cfg)
    moe_layers = _moe_layer_ids(cfg)
    dense_layers = _dense_layer_ids(cfg)
    n_experts = int(tc.get("num_local_experts"))
    n_layers = int(tc.get("num_hidden_layers"))

    print("=" * 96)
    print("MiniMax-M3 INT4 (routed experts only) -- AutoRound recipe DRY-RUN")
    print("=" * 96)
    print(f"src                : {SRC_DIR}")
    print(f"out                : {OUT_DIR}")
    print(f"text layers        : {n_layers}  (dense {dense_layers[0]}..{dense_layers[-1]} -> bf16, "
          f"MoE {moe_layers[0]}..{moe_layers[-1]} -> routed experts int4)")
    print(f"routed experts/lyr : {n_experts}   leaves={EXPERT_LEAVES}")
    print(f"quant params       : bits={BITS} group_size={GROUP_SIZE} sym={SYM} iters={ITERS} "
          f"data_type={DATA_TYPE} packing={PACKING_FORMAT}")
    print()

    # --- shard presence / completeness ---
    shards = sorted(glob.glob(os.path.join(SRC_DIR, "*.safetensors")))
    idx = glob.glob(os.path.join(SRC_DIR, "*.index.json"))
    total_expected = None
    m = re.search(r"-of-(\d+)\.safetensors$", shards[-1]) if shards else None
    if m:
        total_expected = int(m.group(1))
    complete = bool(idx) and total_expected is not None and len(shards) >= total_expected
    print(f"shards present     : {len(shards)}"
          + (f" / {total_expected} expected" if total_expected else "")
          + ("   index.json: yes" if idx else "   index.json: MISSING"))
    print(f"download complete  : {complete}"
          + ("" if complete else "   <-- --run is GATED until this is True"))
    print()

    # --- classify every tensor we can SEE (header-only) into int4 vs bf16 buckets ---
    int4_mods: dict[str, int] = {}     # pattern -> count
    bf16_mods: dict[str, int] = {}
    vision_mods: dict[str, int] = {}
    int4_bytes = 0
    bf16_bytes = 0
    int4_numel = 0
    moe_set = set(moe_layers)

    def patt(name: str) -> str:
        return re.sub(r"\.\d+\.", ".N.", name)

    for name, dt, shape in _iter_shard_headers(SRC_DIR):
        b = _DTYPE_BYTES.get(dt, 2) * _numel(shape)
        if _is_vision(name):
            vision_mods[patt(name)] = vision_mods.get(patt(name), 0) + 1
            bf16_bytes += b
            continue
        if name.endswith(".weight") and is_routed_expert_weight(name, moe_set):
            int4_mods[patt(name)] = int4_mods.get(patt(name), 0) + 1
            int4_bytes += b
            int4_numel += _numel(shape)
        else:
            bf16_mods[patt(name)] = bf16_mods.get(patt(name), 0) + 1
            bf16_bytes += b

    print("-" * 96)
    print("MODULES THAT WILL BE INT4  (routed experts only):")
    print("-" * 96)
    for p in sorted(int4_mods):
        print(f"  [int4] x{int4_mods[p]:<6d} {p}")
    if not int4_mods:
        print("  (none visible yet -- expert shards not downloaded; pattern locked to:)")
        print(f"  [int4]          {LM_PREFIX}.layers.<MoE>.block_sparse_moe.experts.<E>.<w1|w2|w3>.weight")
    print()
    print("-" * 96)
    print("MODULES KEPT BF16  (ignored / full precision):")
    print("-" * 96)
    for p in sorted(bf16_mods):
        print(f"  [bf16] x{bf16_mods[p]:<6d} {p}")
    print()
    if vision_mods:
        print("-" * 96)
        print("VISION TOWER  (text-only quant -> EXCLUDED, kept bf16 / would be dropped by a "
              "text-only export):")
        print("-" * 96)
        for p in sorted(vision_mods):
            print(f"  [skip] x{vision_mods[p]:<6d} {p}")
        print()
    else:
        print("VISION TOWER       : no vision tensors in the present shards (text tower is under "
              "`language_model.*`).")
        print("                     If any vision_* tensors appear in later shards they are auto-"
              "skipped (kept bf16 / excluded).")
        print()

    # --- assert ONLY routed experts are int4 ---
    bad = [p for p in int4_mods if "block_sparse_moe.experts." not in p or
           not p.rstrip().endswith(("w1.weight", "w2.weight", "w3.weight"))]
    assert not bad, f"NON-EXPERT TENSOR SELECTED FOR INT4: {bad}"
    # also assert nothing in the keep-list is a routed expert
    leaked = [p for p in bf16_mods if "block_sparse_moe.experts." in p and
              p.rstrip().endswith(("w1.weight", "w2.weight", "w3.weight"))]
    assert not leaked, f"ROUTED EXPERT LEAKED INTO BF16 KEEP-LIST: {leaked}"
    print("CHECK: only routed-expert w1/w2/w3 selected for int4  ->  OK")
    print("CHECK: no routed expert left in the bf16 keep-list      ->  OK")
    print()

    # --- size estimate ---
    # int4 packed ~= weight/4 + scales(bf16, per group) + qzeros(int32 per group, but sym packs ~0).
    # group scales add ~ (numel/group_size)*2 bytes; sym -> qzeros are a constant fold (tiny).
    seen_bytes = int4_bytes + bf16_bytes
    # extrapolate to the FULL model if not all shards are present, by tensor-count ratio of experts.
    packed_int4 = int4_numel * 0.5                       # 4-bit weights
    scale_bytes = (int4_numel / GROUP_SIZE) * 2          # bf16 group scales
    int4_packed_total_seen = packed_int4 + scale_bytes

    print("-" * 96)
    print("SIZE ESTIMATE (from tensors visible in present shards):")
    print("-" * 96)
    print(f"  bf16 kept (seen)        : {bf16_bytes/2**30:9.2f} GiB")
    print(f"  routed-expert bf16 (seen): {int4_bytes/2**30:9.2f} GiB  -> packs to "
          f"{int4_packed_total_seen/2**30:7.2f} GiB int4 (~{int4_packed_total_seen/max(int4_bytes,1)*100:.0f}%)")
    print(f"  TOTAL bf16 seen         : {seen_bytes/2**30:9.2f} GiB")
    if complete:
        out_est = bf16_bytes + int4_packed_total_seen
        print(f"  ESTIMATED OUTPUT SIZE   : {out_est/2**30:9.2f} GiB  (bf16 kept + packed experts)")
    else:
        # Full-model projection from the model card (~856 GiB bf16 total). Compute from REAL
        # routed-expert dims (w1[I,H]+w3[I,H]+w2[H,I] = 3*H*I per expert), so the int4 ratio is
        # taken vs *numel* exactly like the "seen" path above (int4 weight = 0.5 B/elem; bf16 = 2).
        FULL_BF16_GIB = 856.0
        H = int(tc["hidden_size"]); I = int(tc["intermediate_size"])
        expert_numel = n_experts * len(EXPERT_LEAVES) * len(moe_layers) * (H * I)
        EXPERT_BF16_GIB = expert_numel * 2 / 2**30
        kept_bf16_gib = FULL_BF16_GIB - EXPERT_BF16_GIB
        expert_int4_gib = (expert_numel * 0.5 + (expert_numel / GROUP_SIZE) * 2) / 2**30  # 4-bit + bf16 scales
        out_est = kept_bf16_gib + expert_int4_gib
        print(f"  [shards incomplete -> projecting from real expert dims; model card 856 GiB total]")
        print(f"  routed experts (all)    : {EXPERT_BF16_GIB:9.2f} GiB bf16 -> "
              f"{expert_int4_gib:7.2f} GiB int4  (~{expert_int4_gib/EXPERT_BF16_GIB*100:.0f}%)")
        print(f"  kept bf16 (all)         : {kept_bf16_gib:9.2f} GiB")
        print(f"  ESTIMATED OUTPUT SIZE   : ~{out_est:8.0f} GiB   (924 GiB free at {os.path.dirname(OUT_DIR)})")
    print()

    # --- runtime estimate ---
    print("-" * 96)
    print("RUNTIME ESTIMATE (iters=0 / RTN, nsamples=%d seqlen=%d):" % (NSAMPLES, SEQLEN))
    print("-" * 96)
    n_expert_mats = n_experts * len(EXPERT_LEAVES) * len(moe_layers)
    print(f"  expert matrices to pack : {n_expert_mats:,}  ({n_experts} x {len(EXPERT_LEAVES)} x "
          f"{len(moe_layers)} MoE layers)")
    print(f"  CPU (451 GiB RAM)       : ~6-10 h   (pack/quantize bound)")
    print(f"  1 Gaudi card (idle)     : ~4-7 h    (calib fwd on HPU; packing still host-side)")
    print(f"  NOTE: keep iters=0 (matches GLM-5.1). iters>0 = hours-to-days for a 200B-class model.")
    print()

    # --- gating summary ---
    # Probe the live env so this summary reflects reality (the real --run gate is _assert_can_run).
    try:
        import transformers as _tf  # noqa
        _tf_ok = True
        _m3_present = os.path.isdir(os.path.join(os.path.dirname(_tf.__file__), "models", "minimax_m3_vl"))
        _tf_ver = _tf.__version__
    except Exception:  # noqa: BLE001
        _tf_ok = False; _m3_present = False; _tf_ver = "?"
    try:
        import auto_round as _ar  # noqa
        _ar_ver = getattr(_ar, "__version__", "?")
        from packaging.version import parse as _vp
        _ar_ok = _vp(_ar_ver) >= _vp("0.13.0")
    except Exception:  # noqa: BLE001
        _ar_ver = None; _ar_ok = False

    print("=" * 96)
    print("GATED ON (the real --run refuses to start until ALL are true):")
    print("=" * 96)
    print(f"  [{'x' if complete else ' '}] download complete  ({len(shards)}"
          + (f"/{total_expected}" if total_expected else "") + " shards"
          + (", index present" if idx else ", index MISSING") + ")")
    print(f"  [{'x' if _tf_ok else ' '}] transformers importable + config.json present  "
          f"(transformers {_tf_ver}; models/minimax_m3_vl present: {_m3_present} -- NOT required: the "
          f"model-free RTN path reads shards directly, no M3 forward pass)")
    print(f"  [{'x' if _ar_ok else ' '}] auto_round >= 0.13.0 for the MODEL-FREE streaming RTN path  "
          f"(found {_ar_ver})")
    print(f"  [~] a device: CPU is used (model-free is host-side shard I/O + packing, bounded RAM); "
          f"no Gaudi card needed -- do NOT touch vllm-active")
    print("=" * 96)
    return 0


# --------------------------------------------------------------------------------------------------
# REAL RUN -- gated. Imports torch/transformers/auto_round lazily so dry-run stays dependency-free.
# --------------------------------------------------------------------------------------------------

def _assert_can_run(device: str):
    errs = []
    # 1. download complete
    shards = sorted(glob.glob(os.path.join(SRC_DIR, "*.safetensors")))
    idx = glob.glob(os.path.join(SRC_DIR, "*.index.json"))
    m = re.search(r"-of-(\d+)\.safetensors$", shards[-1]) if shards else None
    total = int(m.group(1)) if m else None
    if not (idx and total and len(shards) >= total):
        errs.append(f"download INCOMPLETE: {len(shards)}/{total} shards, "
                    f"index={'present' if idx else 'MISSING'} at {SRC_DIR}")
    # 2. transformers importable + config.json present.
    #    NOTE: the model-free RTN path does NOT need runnable M3 modeling code (no forward pass, no
    #    AutoModel build) -- it reads the safetensors shards directly and needs only config.json (and
    #    optionally trust_remote_code config for ignore-layer rules). So models/minimax_m3_vl/ is no
    #    longer a HARD gate; we just report its presence. transformers must still be importable
    #    (AutoConfig/AutoTokenizer + the auto_round model-free I/O), and config.json must exist.
    try:
        import transformers  # noqa
        base = os.path.dirname(transformers.__file__)
        _has_m3 = os.path.isdir(os.path.join(base, "models", "minimax_m3_vl"))
        print(f"[m3-quant] transformers {transformers.__version__}; models/minimax_m3_vl present: "
              f"{_has_m3} (model-free path does not require it)", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        errs.append(f"transformers not importable: {e!r}")
    if not os.path.exists(os.path.join(SRC_DIR, "config.json")):
        errs.append(f"config.json MISSING at {SRC_DIR} (model-free needs it)")
    # 3. auto_round >= 0.13.0 (model-free streaming RTN lives here)
    try:
        import auto_round  # noqa
        from packaging.version import parse as _vparse
        if _vparse(getattr(auto_round, "__version__", "0")) < _vparse("0.13.0"):
            errs.append(f"auto_round {auto_round.__version__} too old; need >=0.13.0 for model-free RTN")
    except Exception as e:  # noqa: BLE001
        errs.append(f"auto_round not installed ({e!r}); pip install 'auto-round>=0.13.0'")
    # 4. device
    if device == "hpu":
        try:
            import habana_frameworks.torch.hpu as hthpu  # noqa
            if not hthpu.is_available():
                errs.append("device=hpu requested but no HPU available")
        except Exception as e:  # noqa: BLE001
            errs.append(f"device=hpu requested but habana_frameworks not importable: {e!r}")
    if errs:
        print("REFUSING TO RUN -- unmet gates:", file=sys.stderr)
        for e in errs:
            print("  * " + e, file=sys.stderr)
        sys.exit(2)


def real_run(device: str) -> int:
    _assert_can_run(device)

    import torch  # noqa
    from transformers import AutoConfig, AutoTokenizer
    from auto_round import AutoRound

    cfg = _load_config()
    layer_config, _seen = build_layer_config(cfg)

    # ----------------------------------------------------------------------------------------------
    # BOUNDED-MEMORY LOADING -- the fix.
    #
    # The OLD path did AutoModelForCausalLM.from_pretrained(..., device_map={"":"cpu"}) which
    # MATERIALIZES the full ~856 GiB bf16 model in host RAM -> OOM (only ~448 GiB free).
    #
    # AutoRound 0.13.1 ships a dedicated MODEL-FREE RTN path (auto_round.compressors.model_free)
    # that NEVER loads the whole model: given a *string path*, it streams the safetensors shards
    # ONE AT A TIME (1-deep prefetch + 1-deep async write), splits/quantizes eligible 2-D Linear
    # weights per-tensor (round-to-nearest + per-group int scales), packs auto_round:auto_gptq, and
    # writes output shards, calling clear_memory() after every shard. Peak RSS therefore plateaus at
    # roughly {current shard + 1 prefetched + 1 in-flight write} (~tens of GiB on M3's ~14 GiB
    # shards), NOT 856 GiB. This is exactly the RTN / iters=0 / int / auto_round:auto_gptq recipe we
    # already use -- model_free supports precisely W{2,4,8}A16 sym/asym int packed as auto_round:auto_gptq.
    #
    # Contract is preserved via `layer_config` + `scheme`:
    #   * scheme="W4A16"  -> default bits=4 group_size=128 sym=True data_type=int  (== BITS/GROUP_SIZE/SYM/DATA_TYPE)
    #   * layer_config (build_layer_config) pins EVERY non-routed-expert module to bits=16 -> kept FP.
    #     M3 stores experts per-expert & 2-D (layers.L.block_sparse_moe.experts.E.{w1,w2,w3}.weight),
    #     so split_fused_expert_tensors() is a no-op and our enumerated module names match 1:1.
    #   * quant_nontext_module=False (default) auto-ignores vision/visual/image/audio -> vision stays FP.
    #     The predefined skip set also keeps router .gate. and embeddings in FP.
    #
    # model_free is HOST-side I/O+pack bound; the `device` only picks where the tiny per-tensor RTN
    # math runs. We run it on CPU (matches the runtime estimate and stays fully card-free). If hpu is
    # requested we still keep the loading bounded and just note model_free is host-bound.
    # ----------------------------------------------------------------------------------------------
    assert BITS in (2, 4, 8), f"model-free supports bits in (2,4,8); BITS={BITS}"
    assert DATA_TYPE == "int", f"model-free supports data_type='int'; DATA_TYPE={DATA_TYPE!r}"
    scheme = f"W{BITS}A16"   # W4A16 -> bits=4 group_size=128 sym=True int (asserted-equal below)

    print(f"[m3-quant] loading config + tokenizer (trust_remote_code) from {SRC_DIR}")
    _hf_cfg = AutoConfig.from_pretrained(SRC_DIR, trust_remote_code=True)   # validates config loads
    tok = AutoTokenizer.from_pretrained(SRC_DIR, trust_remote_code=True)

    compute_device = "cpu"
    if device == "hpu":
        print("[m3-quant] NOTE: model_free RTN is host-side (shard I/O + packing); running its "
              "per-tensor math on CPU regardless of --device hpu (card-free, matches the estimate).")

    n_int4 = sum(1 for v in layer_config.values() if v.get("bits") == BITS)
    n_bf16 = sum(1 for v in layer_config.values() if v.get("bits") == 16)
    print(f"[m3-quant] contract: {n_int4} modules -> int4, {n_bf16} kept bf16")
    print(f"[m3-quant] strategy: AutoRound MODEL-FREE (shard-by-shard, BOUNDED RAM) -- scheme={scheme}, "
          f"group_size={GROUP_SIZE}, sym={SYM}, iters=0 (RTN), packing={PACKING_FORMAT}")

    # ----------------------------------------------------------------------------------------------
    # MODEL-FREE LAYER_CONFIG ADAPTATION (REQUIRED for correctness, not cosmetic).
    #
    # model_free resolves each tensor's scheme by matching its layer_name against the layer_config
    # keys *as regexes* (auto_round.utils.common.to_standard_regex wraps each key as ".*<key>.*"),
    # returning the FIRST matching entry in dict/insertion order; if none match it falls back to the
    # default scheme (here W4A16 int4). Passing build_layer_config()'s raw per-module dict straight
    # through is WRONG for two compounding reasons, both observed live (-> "Quantized layers (0)"):
    #
    #   (1) DIGIT-SUFFIX REWRITE. model_free._parse_layer_config appends "." to any key whose LAST
    #       char is a digit. Expert leaves end in w1/w2/w3 (last char a digit), so each expert key
    #       becomes "...experts.E.w1." -> regex "...w1\..*" which CANNOT match the tensor layer_name
    #       "...experts.E.w1" (".weight" already stripped, no trailing dot) -> experts miss their int4
    #       rule. Fix: re-key expert entries as "$"-anchored patterns "...experts.E.w1$"
    #         * last char "$" (not a digit) -> the ".-append" rewrite is skipped, and
    #         * to_standard_regex("...w1$") -> ".*...experts\.E\.w1$" matches "...w1" EXACTLY (and not
    #           a longer name). Verified against the installed auto_round 0.13.1 matcher.
    #
    #   (2) OVER-BROAD bf16 PREFIX. build_layer_config records the e_score_correction_bias parent as a
    #       BARE module key "...layers.L.block_sparse_moe" (bits=16). to_standard_regex makes that
    #       ".*...block_sparse_moe.*" -- which SWALLOWS every expert tensor ("...block_sparse_moe.
    #       experts.E.wN"). Since first-match wins, any such bf16 prefix sitting before the expert rule
    #       forces experts to bf16. Fix: (a) order the precise expert int4 keys FIRST, and (b) DROP any
    #       bf16 key whose regex would match an expert tensor (the bare block_sparse_moe prefix). That
    #       bare key is redundant anyway: e_score_correction_bias is a 1-D bias, not an eligible 2-D
    #       ".weight", so model_free leaves it untouched (full precision) regardless.
    #
    # build_layer_config() itself is left UNTOUCHED so --dry-run and its expert/keep assertions keep
    # operating on real module names. We derive the model-free view (ar_layer_config) here only.
    import re as _re
    from auto_round.utils.common import to_standard_regex as _to_std_re

    _EXPERT_PROBE = (f"{LM_PREFIX}.layers.{{L}}.block_sparse_moe.experts.0.w1")

    def _matches_expert(mod: str) -> bool:
        """True if `mod` (used as a model_free regex key) would match a routed-expert tensor name."""
        try:
            rx = _re.compile(_to_std_re(mod))
        except _re.error:
            return False
        # probe a representative expert layer_name across all MoE layers
        for L in _moe_layer_ids(cfg):
            if rx.search(_EXPERT_PROBE.format(L=L)):
                return True
        return False

    ar_layer_config: dict[str, dict] = {}
    rekeyed = 0
    for mod, spec in layer_config.items():               # experts FIRST (precise, $-anchored)
        if spec.get("bits") == BITS:
            ar_layer_config[mod + "$"] = spec
            rekeyed += 1
    dropped_bf16: list[str] = []
    for mod, spec in layer_config.items():               # then kept-bf16, minus expert-swallowers
        if spec.get("bits") == 16:
            if _matches_expert(mod):
                dropped_bf16.append(mod)
                continue
            ar_layer_config[mod] = spec
    assert rekeyed == n_int4, f"re-key mismatch: rekeyed {rekeyed} != {n_int4} int4 modules"
    print(f"[m3-quant] model-free layer_config: {rekeyed} expert int4 entries ($-anchored, ordered "
          f"first); dropped {len(dropped_bf16)} over-broad bf16 prefix key(s) that would swallow "
          f"experts: {dropped_bf16}")

    # TEMPORARY bounded-verification limiter: if M3_QUANT_MAX_SHARDS is set, process only the first N
    # shards so we can confirm RAM stays flat + a block actually packs, then stop. UNSET for the full run.
    _max_shards_env = os.environ.get("M3_QUANT_MAX_SHARDS")
    if _max_shards_env:
        _patch_model_free_max_shards(int(_max_shards_env))

    ar = AutoRound(
        model=SRC_DIR,                 # <-- STRING PATH: triggers the model-free streaming loader
        tokenizer=tok,
        scheme=scheme,                 # W4A16 == bits=4 group_size=128 sym=True data_type=int
        group_size=GROUP_SIZE,         # explicit (== scheme default 128); model-free reads it
        sym=SYM,
        iters=ITERS,                   # 0 -- RTN, matches GLM-5.1 export
        disable_opt_rtn=True,          # RTN (no opt) -- required precondition for the model-free route
        model_free=True,               # <-- THE FIX: bounded shard-by-shard quant, never loads 856 GiB
        layer_config=ar_layer_config,  # the contract: routed experts int4 ($-anchored), all else bits=16 -> kept FP
        device=compute_device,
        data_type=DATA_TYPE,
    )

    # Sanity: confirm we really got the bounded model-free compressor (not the full-load fallback).
    assert getattr(ar, "model_free", False) is True, (
        "AutoRound did NOT take the model-free route -- it would full-load the model and OOM. "
        "Check auto_round version / scheme / iters.")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[m3-quant] quantizing + exporting {PACKING_FORMAT} -> {OUT_DIR} (this is the long part) ...")
    ar.quantize_and_save(OUT_DIR, format=PACKING_FORMAT, inplace=True)

    # Ensure the exported config.json's quantization_config carries the hpu-moe-ext-detected contract
    # (block_name_to_quantize scope) and the trust_remote_code config helper is present.
    _finalize_config(cfg)
    print("[m3-quant] DONE.")
    return 0


def _patch_model_free_max_shards(n: int) -> None:
    """TEMPORARY verification aid: cap the model-free shard loop at the first `n` shards.

    Monkeypatches auto_round.compressors.model_free._list_weight_shards so the streaming loop
    processes only the first n shards, lets us watch RSS plateau + confirm a block packs, then stops.
    Activated ONLY when M3_QUANT_MAX_SHARDS is set; never used in the full production run.
    """
    import auto_round.compressors.model_free as _mf
    _orig = _mf._list_weight_shards

    def _capped(source_dir, *a, **kw):
        shards = _orig(source_dir, *a, **kw)
        capped = shards[:n]
        print(f"[m3-quant][VERIFY] M3_QUANT_MAX_SHARDS={n}: limiting {len(shards)} -> "
              f"{len(capped)} shards: {capped}")
        return capped

    _mf._list_weight_shards = _capped


def _finalize_config(src_cfg: dict):
    """Make sure the exported config.json's quantization_config matches the hpu-moe-ext-detected
    contract (quant_method=auto-round, packing auto_round:auto_gptq, bits/group_size/sym, and
    block_name_to_quantize scoped to the M3 text layers)."""
    out_cfg_path = os.path.join(OUT_DIR, "config.json")
    if not os.path.exists(out_cfg_path):
        return
    with open(out_cfg_path) as f:
        oc = json.load(f)
    qc = oc.get("quantization_config", {})
    qc.update({
        "quant_method": "auto-round",
        "packing_format": PACKING_FORMAT,
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "sym": SYM,
        "data_type": DATA_TYPE,
        "iters": ITERS,
        # scope gate -- M3 text tower nesting (cf. GLM used ['model.layers']).
        "block_name_to_quantize": [f"{LM_PREFIX}.layers"],
    })
    oc["quantization_config"] = qc
    with open(out_cfg_path, "w") as f:
        json.dump(oc, f, indent=2)
    # copy the config helper so AutoConfig(trust_remote_code) keeps working post-quant
    src_helper = os.path.join(SRC_DIR, "configuration_minimax_m3_vl.py")
    if os.path.exists(src_helper):
        import shutil
        shutil.copy2(src_helper, OUT_DIR)


# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="config+header-only validation; loads NO weights, needs no torch/auto_round")
    g.add_argument("--run", action="store_true",
                   help="the real quant; GATED on download+transformers(m3)+auto_round+free device")
    ap.add_argument("--device", choices=["cpu", "hpu"], default="cpu",
                    help="cpu (default; 451 GiB RAM) or hpu (1 idle Gaudi card)")
    args = ap.parse_args(argv)
    if args.dry_run:
        return dry_run()
    return real_run(args.device)


if __name__ == "__main__":
    raise SystemExit(main())
