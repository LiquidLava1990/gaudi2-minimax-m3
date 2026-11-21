#!/usr/bin/env python3
# ==================================================================================================
# dry_validate_m3_fp8_inc.py
#
# CARD-FREE dry-validate for the FP8(INC) recipe (calibrate_m3_fp8_inc.sh). Resolves the
# fp8-quantize-vs-keep-bf16 module lists against the REAL config.json + safetensors HEADERS
# (metadata only — NO weights loaded, no torch/INC/card needed), using the SAME blocklist-regex
# semantics INC itself uses at runtime (verified from the image:
#   neural_compressor/.../fp8_quant/_core/utils.py:should_quantize / is_re_match ->
#   a module is quantized iff its TYPE is a supported quantizable type (Linear / FusedMoE) AND its
#   name does NOT re.search-match ANY blocklist regex; allowlist-by-name is "not yet implemented").
#
# This is the fp8 analog of quantize_m3_int4.py's dry_run(): it asserts the contract
#   * FP8 (W8A8): the 128 routed experts (block_sparse_moe.experts w1/w2/w3), layers 3..59.
#   * KEEP BF16 : lm_head, embed_tokens, all norms, the router gate + e_score_correction_bias, AND
#                 the MSA index projections (self_attn.index_{q,k}_proj + index_{q,k}_norm).
# and prints an int4-vs-fp8 size comparison (~fp8 vs the ~285GB int4).
# ==================================================================================================
from __future__ import annotations

import glob
import json
import os
import re
import struct

SRC_DIR = os.environ.get("SRC_DIR", "/mnt/data2/minimax-m3-bf16")
FP8_SCOPE = os.environ.get("FP8_SCOPE", "experts")   # experts | broad
GROUP = 1  # fp8 per-tensor/per-channel scales are negligible vs the int4 per-group(128) scales

# ---- INC blocklist regexes (MUST match calibrate_m3_fp8_inc.sh exactly) --------------------------
BLOCK = {
    "experts": [
        r"lm_head", r"embed_tokens", r"self_attn",
        r"block_sparse_moe\.gate\b", r"e_score_correction_bias",
        r"shared_experts", r"\.mlp\.", r"layernorm", r"model\.norm",
    ],
    "broad": [
        r"lm_head", r"embed_tokens", r"self_attn\.index_",
        r"block_sparse_moe\.gate\b", r"e_score_correction_bias",
        r"layernorm", r"model\.norm",
    ],
}

# INC quantizes only these module TYPES. We map a checkpoint *.weight tensor to its would-be vLLM
# module type by name so the dry-validate matches runtime behavior (norms = RMSNorm -> never fp8;
# embeddings/lm_head -> VocabParallel/ParallelLMHead, kept bf16 via quant_config=None in the model
# AND blocklisted here). Routed experts are a FusedMoE (quantizable). Everything else Linear.
_NONLINEAR_SUFFIX = ("_norm.weight", "layernorm.weight", "norm.weight",
                     "e_score_correction_bias")

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
                "I16": 2, "I32": 4, "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

LM_PREFIX = "language_model.model"
_EXPERT_RE = re.compile(
    r"^" + re.escape(LM_PREFIX)
    + r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
_INDEX_RE = re.compile(r"self_attn\.index_")


def _load_config() -> dict:
    with open(os.path.join(SRC_DIR, "config.json")) as f:
        return json.load(f)


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def _iter_headers(src: str):
    for path in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            yield k, v.get("dtype", "?"), v.get("shape", [])


def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _patt(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


def _is_nonlinear(name: str) -> bool:
    return name.endswith(_NONLINEAR_SUFFIX) or name == LM_PREFIX + ".norm.weight" \
        or name.endswith("model.norm.weight")


def _blocked(name: str, pats) -> bool:
    return any(re.search(p, name) for p in pats)


def main() -> int:
    cfg = _load_config()
    tc = _text_cfg(cfg)
    pats = BLOCK[FP8_SCOPE]
    n_experts = int(tc["num_local_experts"])
    n_layers = int(tc["num_hidden_layers"])
    H = int(tc["hidden_size"])
    I = int(tc["intermediate_size"])
    moe_freq = tc.get("moe_layer_freq")
    moe_layers = [i for i, v in enumerate(moe_freq) if int(v) == 1] if isinstance(moe_freq, list) \
        else list(range(int(tc.get("first_k_dense_replace", 0) or 0), n_layers))

    print("=" * 98)
    print(f"MiniMax-M3 FP8(INC) W8A8 — calibrate-vs-keep DRY-VALIDATE   [scope={FP8_SCOPE}]")
    print("=" * 98)
    print(f"src                : {SRC_DIR}")
    print(f"text layers        : {n_layers}  (dense 0..{moe_layers[0]-1} / MoE {moe_layers[0]}..{moe_layers[-1]})")
    print(f"experts/layer      : {n_experts}  (top-{tc.get('num_experts_per_tok')}); H={H} I={I}")
    print(f"selection mech     : INC blocklist re.search (allowlist-by-name not implemented); fp8 iff "
          f"type=Linear/FusedMoE AND name NOT in blocklist")
    print(f"scale_method       : maxabs_hw  (g2 ±240 E4M3 fnuz grid; NEVER g3 ±448 -> clip; NEVER unit_scale)")
    print()

    # shard completeness (same gate as the int4 dry-run)
    shards = sorted(glob.glob(os.path.join(SRC_DIR, "*.safetensors")))
    idx = glob.glob(os.path.join(SRC_DIR, "*.index.json"))
    m = re.search(r"-of-(\d+)\.safetensors$", shards[-1]) if shards else None
    total = int(m.group(1)) if m else None
    complete = bool(idx) and total is not None and len(shards) >= total
    print(f"shards present     : {len(shards)}" + (f" / {total}" if total else "")
          + ("   index.json: yes" if idx else "   index.json: MISSING"))
    print(f"download complete  : {complete}" + ("" if complete else "   <-- --run is GATED until True"))
    print()

    fp8_mods: dict[str, int] = {}
    bf16_mods: dict[str, int] = {}
    fp8_numel = 0
    bf16_bytes = 0
    expert_bf16_bytes = 0

    for name, dt, shape in _iter_headers(SRC_DIR):
        b = _DTYPE_BYTES.get(dt, 2) * _numel(shape)
        nonlinear = _is_nonlinear(name)
        blocked = _blocked(name, pats)
        is_expert = bool(_EXPERT_RE.match(name))
        # INC fp8 only if a quantizable Linear/FusedMoE type AND not blocklisted.
        will_fp8 = (not nonlinear) and (not blocked)
        if will_fp8:
            fp8_mods[_patt(name)] = fp8_mods.get(_patt(name), 0) + 1
            fp8_numel += _numel(shape)
            if is_expert:
                expert_bf16_bytes += b
        else:
            bf16_mods[_patt(name)] = bf16_mods.get(_patt(name), 0) + 1
            bf16_bytes += b

    print("-" * 98)
    print(f"MODULES -> FP8 (W8A8):")
    print("-" * 98)
    for p in sorted(fp8_mods):
        print(f"  [fp8 ] x{fp8_mods[p]:<6d} {p}")
    if not fp8_mods:
        print("  (no quantizable modules visible yet — expert shards not all downloaded)")
    print()
    print("-" * 98)
    print(f"MODULES KEPT BF16:")
    print("-" * 98)
    for p in sorted(bf16_mods):
        print(f"  [bf16] x{bf16_mods[p]:<6d} {p}")
    print()

    # ---- CONTRACT ASSERTIONS -------------------------------------------------
    # 1) MSA index projections + norms MUST be bf16 in BOTH scopes (selection precision).
    idx_in_fp8 = [p for p in fp8_mods if _INDEX_RE.search(p)]
    assert not idx_in_fp8, f"MSA INDEX PROJECTION SELECTED FOR FP8 (selection precision!): {idx_in_fp8}"
    # 2) router gate, e_score bias, lm_head, embeddings, model.norm MUST be bf16.
    must_keep = [p for p in fp8_mods if re.search(
        r"block_sparse_moe\.gate\b|e_score_correction_bias|lm_head|embed_tokens|model\.norm", p)]
    assert not must_keep, f"KEEP-BF16 MODULE SELECTED FOR FP8: {must_keep}"
    # 3) the routed experts MUST be fp8 (they are the pivot target).
    leaked = [p for p in bf16_mods if "block_sparse_moe.experts." in p
              and p.rstrip().endswith(("w1.weight", "w2.weight", "w3.weight"))]
    assert not leaked, f"ROUTED EXPERT LEAKED INTO BF16 (should be fp8): {leaked}"
    # 4) experts-only scope: attention/shared/dense stay bf16 (the faithful int4 mirror).
    if FP8_SCOPE == "experts":
        attn_fp8 = [p for p in fp8_mods if "self_attn" in p]
        assert not attn_fp8, f"experts-only scope but attention selected for fp8: {attn_fp8}"
    print("CHECK: MSA index_{q,k}_proj/norm kept bf16 (selection precision)   -> OK")
    print("CHECK: router gate / e_score_bias / lm_head / embed / norm bf16    -> OK")
    print("CHECK: routed experts w1/w2/w3 selected for fp8                    -> OK")
    if FP8_SCOPE == "experts":
        print("CHECK: experts-only scope — attention/shared/dense kept bf16       -> OK")
    print()

    # ---- SIZE: fp8 vs the ~285GB int4 ---------------------------------------
    # fp8 W8A8 weight = 1 byte/elem + tiny per-tensor/channel scales (negligible). int4 = 0.5 B/elem
    # + bf16 per-group(128) scales. Project to the FULL model from real expert dims if incomplete.
    print("-" * 98)
    print("SIZE ESTIMATE  (fp8 vs the ~285 GiB int4 from quantize_m3_int4.py):")
    print("-" * 98)
    GIB = 2 ** 30
    # full-model expert footprint from real dims (w1[I,H]+w3[I,H]+w2[H,I] = 3*H*I per expert)
    expert_numel_full = n_experts * 3 * len(moe_layers) * (H * I)
    expert_bf16_full = expert_numel_full * 2 / GIB
    expert_fp8_full = expert_numel_full * 1 / GIB
    expert_int4_full = (expert_numel_full * 0.5 + (expert_numel_full / 128) * 2) / GIB
    FULL_BF16_GIB = 856.0
    kept_bf16_full = FULL_BF16_GIB - expert_bf16_full

    # Non-expert linear footprints from REAL config dims (q+k+v+o per layer; shared gate/up/down per
    # MoE layer; dense gate/up/down on the dense prefix). These move to fp8 ONLY in the broad scope.
    qo = int(tc["num_attention_heads"]) * int(tc["head_dim"])          # 8192
    kv = int(tc["num_key_value_heads"]) * int(tc["head_dim"])          # 512
    sI = int(tc.get("shared_intermediate_size", I))
    dI = int(tc.get("dense_intermediate_size", I))
    n_dense = n_layers - len(moe_layers)
    attn_bf16 = (qo * H + kv * H + kv * H + H * qo) * n_layers * 2 / GIB
    shared_bf16 = (sI * H * 2 + H * sI) * len(moe_layers) * 2 / GIB
    dense_bf16 = (dI * H * 2 + H * dI) * n_dense * 2 / GIB
    movable_bf16 = attn_bf16 + shared_bf16 + dense_bf16   # what broad converts to fp8

    if FP8_SCOPE == "experts":
        fp8_part = expert_fp8_full
        kept_part = kept_bf16_full
    else:
        fp8_part = expert_fp8_full + movable_bf16 * 0.5     # those linears halve to fp8
        kept_part = kept_bf16_full - movable_bf16           # rest stays bf16 (index/gate/norm/embed/head)

    out_fp8 = fp8_part + kept_part
    print(f"  routed experts (all): {expert_bf16_full:8.1f} GiB bf16 -> {expert_fp8_full:7.1f} GiB fp8 "
          f"(50%)   |  int4 was {expert_int4_full:6.1f} GiB (26%)")
    print(f"  kept bf16 (all)     : {kept_part:8.1f} GiB")
    print(f"  ESTIMATED FP8 OUTPUT: ~{out_fp8:7.0f} GiB     (vs int4 ~285 GiB; bf16 source ~856 GiB)")
    print(f"  NOTE: INC fp8 does NOT emit a re-packed checkpoint — it serves the bf16 weights and")
    print(f"        applies measured scales at load. The ~{out_fp8:.0f} GiB is the *served HBM* fp8")
    print(f"        weight footprint (what fits across 8x96GB), NOT a new on-disk model.")
    print()

    print("=" * 98)
    print("GATED ON (the real --run refuses until ALL true):")
    print("=" * 98)
    print(f"  [{'x' if complete else ' '}] download complete  ({len(shards)}"
          + (f"/{total}" if total else "") + (", index present" if idx else ", index MISSING") + ")")
    print(f"  [ ] transformers with models/minimax_m3_vl/  (build that ships M3-VL modeling)")
    print(f"  [ ] a FREE Gaudi card set (NOT vllm-active) — calibration needs the cards on-device")
    print(f"  [ ] #8 native-fp8-MoE verification: confirm the FusedMoE experts run fp8 ON DEVICE")
    print(f"      (VLLM_HPU_LOG_STEP_CPU_FALLBACKS=true must show NO cpu fallback for the experts)")
    print("=" * 98)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
