# MiniMax-M3 — STATIC fp8-on-disk checkpoint: produce + serve recipe (card-free analysis)

**Date:** 2026-06-23 · **Scope:** bf16 → static fp8 (E4M3) routed-experts-only checkpoint, on-disk,
no load-time dequant, MTP kept bf16. **Card-free reading/analysis only** — no docker/vllm/serve run,
no card touched, the `hf download` at `/mnt/data1/minimax-m3-bf16` left untouched.

---

## TL;DR VERDICT: **CONDITIONAL GO on the quant, NO-GO (today) on the serve as the tools stand.**

A static per-channel fp8 checkpoint of the routed experts **can be produced** from the bf16 source —
but **NOT** with the existing `quantize_m3_int4.py` model-free path (it hard-rejects fp8), and **NOT**
into the format the existing M3 serve stack autodetects. The on-disk fp8 serve path is **NOT wired**:
the whole M3 fp8 runtime (`m3_expert_runtime_patch.py` + `m3_swigluoai_patch.py`) is built around
hpu-moe-ext's int4 `MoeWNA16HPUMethod` and the **int4→fp8-at-load** transform. Serving genuinely-fp8
on-disk experts means going through upstream `--quantization fp8` (`HPUFp8MoEMethod`), onto which
**nothing wires M3's mandatory swigluoai gate** (the native fp8 op is silu-only). That is non-trivial
new serve wiring.

**The fallback the repo already implements and validates — keep int4 on disk (226 GiB) + int4→fp8
per-channel at load (`M3_EXPERT_RUNTIME=fp8`, `models/minimax-m3-fp8.env`) — gives the EXACT same
native-fp8-MME HBM execution with ZERO new code and a measured <few-e-3 added error.** Recommendation:
**do NOT build the static-fp8-on-disk artifact; serve the existing int4 checkpoint in fp8 runtime mode.**
Details, evidence, and the smallest-change path if static-on-disk is still wanted are below.

---

## 0. Ground truth (config + checkpoints)

- bf16 source config (`/mnt/data1/minimax-m3-bf16/config.json`, read-only): `text_config`
  `num_hidden_layers=60`, `num_local_experts=128`, `num_experts_per_tok=4`, `hidden_size=6144`,
  `intermediate_size=3072` (moe), `shared_intermediate_size=3072`, `dense_intermediate_size=12288`,
  `head_dim=128`, `num_attention_heads=64`, `num_key_value_heads=4`, `hidden_act=swigluoai`,
  `swiglu_limit=7.0`, **`num_mtp_modules=7`**, `num_nextn_predict_layers=1`,
  `moe_layer_freq` length 60 with **dense prefix = layers [0,1,2]** (so MoE = layers 3..59 = 57 layers).
  Top `architectures=['MiniMaxM3SparseForConditionalGeneration']`, `model_type=minimax_m3_vl`.
- **Download progress:** config.json present, **0 safetensors shards yet** at `/mnt/data1/minimax-m3-bf16`.
  ⚠️ `quantize_m3_int4.py:98` hard-codes `SRC_DIR="/mnt/data2/minimax-m3-bf16"` — the **real download is
  on `/mnt/data1`**. Any new fp8 script (or a `--src` flag) must point at `/mnt/data1/minimax-m3-bf16`.
- int4 checkpoint `/mnt/fastmodels/MiniMax-M3-int4-experts-AutoRound`: **exists, 226 GiB on disk**,
  67,200 tensors. Expert tensors `language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w2,w3}.{qweight,qzeros,scales}`
  (7296 each = 128×57). **NO MTP/nextn tensors** (confirmed: 0 `layers.60..66`, `eh_proj`, `enorm`, `nextn`,
  `mtp` — the int4 export DROPPED all 7 MTP modules, matching `bench/M3_MATRIX_REPORT.md:131-138`).
- Disk: `/mnt/fastmodels` 690 GiB used / 1.1 TiB free; `/mnt/data1` 3.4 TiB free.

---

## 1. THE QUANT — exact tool + scheme + layer_config

### 1a. Can `quantize_m3_int4.py` emit fp8 with a `--scheme`/`--fp8` flag? **NO.**

The script's whole load strategy is AutoRound's **model-free** path (`model_free=True`,
`quantize_m3_int4.py:616-628`), which is the only thing that keeps RAM bounded (~17 GiB peak vs 856 GiB).
**model-free cannot produce fp8.** Verified directly against the installed `auto_round 0.13.1`:

- `auto_round/compressors/model_free.py:25-37` docstring: *"Schemes that require special packing (FP8,
  MXFP4, NVFP4, GGUF, INT8_W8A8, BF16, FPW8A16, ...) are **not** supported in model-free mode and will
  raise."*
- `model_free._validate_supported_scheme` raises `ValueError` when `data_type != "int"`: *"Model-free
  mode only supports integer weight quantization (data_type='int') ... FP8 / MXFP / NVFP / GGUF / BF16
  schemes require the standard [path]."* (confirmed by running it).
- Supported preset list is integer-only: `W2A16, W2A16G32, W2A16G64, W4A16, W4A16_MIXED, W8A16`
  (`model_free.py:101`), `_SUPPORTED_INT_BITS=(2,4,8)` (`:108`).

So adding a `--scheme FP8_STATIC` to this script would hit the raise. **A sibling script is required.**

### 1b. What CAN produce a static per-channel fp8 export — the AutoRound STANDARD (non-model-free) path

AutoRound 0.13.1 ships two fp8 exporters that DO produce per-output-channel E4M3 with bf16 activations
(the W8A16 the native vllm_gaudi op wants — `act_bits>=16` ⇒ no activation calibration, `input_scale=None`):

- **auto_round-native fp8** (`export_to_autoround/export_to_fp8.py`): `FP8QLinear` stores
  `weight` (`torch.float8_e4m3fn`) + a `weight_scale` buffer; `pack_layer` (`:105-185`) computes
  `q_weight = weight / scale.unsqueeze(-1)` then `.to(float8_e4m3fn)`, and with `unsqueeze=True` reshapes
  `weight_scale` to `[out,1]` (**per-output-channel**). `fmt=e4m3` (`:224`). Format = auto_round-native.
- **llm-compressor / compressed-tensors fp8** (`export_to_llmcompressor/export_to_static_fp.py`): emits
  `format="float-quantized"`, `QuantizationType.FLOAT`, `num_bits=8`, `symmetric=True`. **CHANNEL** strategy
  for `group_size=-1` (`:206-212`). On Gaudi2 it auto-stamps `fp8_dtype_flavor="float8_e4m3fnuz"`
  (`:138-149`, matches the box's HW element — same fnuz convention as `int4_to_fp8_load.py:65`).
  **CAVEAT:** that exporter hardwires **static W8A8** — `input_activations=_construct_fp8_args()` with
  `dynamic=False` (`:227, 65-78`), requiring an `act_group_size==0` and producing per-tensor activation
  scales (needs calibration data flowing through every expert). That is NOT what the native vllm_gaudi
  per-channel op consumes (it does **dynamic** activation quant `dynamic_quant(x)` at runtime — see §3).
  → Prefer the **auto_round-native W8A16** form (weight-only, `act_bits=16`), not FP8_STATIC's W8A8.

The matching scheme is **`FPW8A16`** (`bits=8, group_size=0, sym=True, data_type='fp', act_bits=16`,
confirmed in `PRESET_SCHEMES`) for per-tensor, or a custom `QuantizationScheme(bits=8, group_size=-1,
data_type='fp', act_bits=16, sym=True)` for **per-output-channel** (group_size=-1) — which is the closest
match to `VllmMixtureOfExpertsOpFP8PerChannel`'s per-channel `[out]` scale (`int4_to_fp8_load.py:93-131`
documents that exact format).

### 1c. The RAM problem — the reason this is "conditional"

The standard path needs the model as an `nn.Module` (`AutoRound(model=<nn.Module or path>)` →
`from_pretrained`), NOT a string-streamed shard reader. AutoRound 0.13.1 has bounded-RAM knobs —
`low_cpu_mem_usage=True` (default, `compressors/base.py:161`) + an `OffloadManager` (`:228`) +
**immediate-saving / ShardWriter to meta device** (`export/utils.py is_immediate_saving_mode`,
`base.py:1258-1268`) — so a full-model fp8 export with offload is *plausible* without 856 GiB RAM.
BUT this is materially heavier and less proven than the model-free int4 path: it (a) needs runnable M3
modeling code (the multimodal `minimax_m3_vl` remote code), (b) disk-offloads shards, (c) is untested
on this box for a 426B model. **Also `compressed_tensors` is NOT installed in `$HOME/.venvs/m3calib`**
(import failed) — the llm-compressor exporter would need it added; the auto_round-native fp8 exporter does
not.

### 1d. Concrete CLI (the sibling script `quant/quantize_m3_fp8.py`, NOT yet written)

```bash
# NOTE: this is the STANDARD (offload) path — NOT the bounded model-free path. Validate RAM on a
# 2-shard cap first (mirror M3_QUANT_MAX_SHARDS discipline). compressed_tensors only needed for the
# llm-compressor format; the auto_round-native fp8 export does not need it.
$HOME/.venvs/m3calib/bin/python quant/quantize_m3_fp8.py --run --src /mnt/data1/minimax-m3-bf16 \
    --out /mnt/fastmodels/MiniMax-M3-fp8-experts-AutoRound
# under the hood:
#   AutoRound(model="/mnt/data1/minimax-m3-bf16", scheme=<W8A16-fp, group_size=-1 per-channel>,
#             iters=0, low_cpu_mem_usage=True, device_map="cpu", layer_config=<experts-only fp8>)
#   ar.quantize_and_save(out, format="auto_round")        # auto_round-native fp8 (fmt=e4m3, per-channel)
# layer_config: routed experts -> {bits:8, group_size:-1, data_type:'fp', act_bits:16, sym:True};
#               EVERYTHING ELSE -> {bits:16, data_type:'float'} (reuse build_layer_config()'s split verbatim,
#               only swapping the expert spec from int4 to fp8 — the regex selection logic is identical).
```

### 1e. Size + runtime estimate

- **Output size ~370–380 GiB.** Routed experts: 128×3×57×(6144×3072) ≈ 412B params ... actually the
  experts are 333 GiB bf16 (per `quantize_m3_int4.py:79`). fp8 = 1 byte/elem ⇒ experts ≈ **166 GiB**
  (vs int4's ~83 GiB) + per-channel bf16 scales (tiny) + ~210 GiB kept-bf16 (everything non-expert) =
  **~376 GiB** total. Fits `/mnt/fastmodels` (1.1 TiB free). (For comparison int4 on disk = 226 GiB.)
- **Runtime:** the standard offload path is slower than model-free (disk-offload churn). The int4
  model-free run was ~30 min–several h; budget **conservatively 3–8 h** for fp8 standard+offload, and
  gate it on a 2-shard RAM-bounded dry run first. (Cannot ground this tighter without running it.)

---

## 2. MTP PRESERVATION

**Mechanism:** AutoRound (both model-free and standard) only quantizes modules selected by `layer_config`
and copies every non-selected tensor through **at its source dtype = bf16**. `build_layer_config()`
(`quantize_m3_int4.py:219-244`) pins EVERY non-routed-expert module to `{bits:16, data_type:'float'}`,
including anything matching the MTP layers. So MTP tensors, IF PRESENT in the source, pass through bf16.

**⚠️ The real risk is the OPPOSITE of what was feared:** the int4 export **already dropped all MTP**
(§0: the int4 checkpoint has zero MTP tensors; `M3_MATRIX_REPORT.md:131-138` confirms the AutoRound int4
export contains no `eh_proj/enorm/nextn/mtp` and layers 60-66 are absent from the index). That happened
because `build_layer_config` enumerates from **whatever shards were present at build time**
(`:228-244` iterates `_iter_shard_headers(SRC_DIR)`), and the model-free copy-through only copies tensors
it iterates. **The fp8 export MUST run against the COMPLETE bf16 download** (all 59 shards present), so the
MTP shards are actually on disk to copy through. The int4 build's MTP loss is the cautionary precedent.

**MTP tensor names to verify (from config + the modeling code):** M3's 7 MTP modules live at logical
layers **60..66** under `model.layers.{60+i}.*` (the main model SKIPS them via
`get_spec_layer_idx_from_weight_name`, `minimax_m3.py:1172-1180`: `layer_idx=num_hidden_layers(60)`,
loop `range(num_mtp_modules=7)` → `model.layers.60..66`). In the multimodal export they are prefixed
`language_model.model.layers.60..66.*` (MTP blocks typically carry `eh_proj`, `enorm`/`hnorm`,
`shared_head`, plus a full transformer block). The exact leaf names can only be enumerated from the
**bf16 shard headers once they download** — do that as the verification.

**Exact verification (card-free, header-only):**
```bash
# after the fp8 export, count MTP-pattern tensors in OUTPUT shards == count in INPUT (bf16) shards:
for d in /mnt/data1/minimax-m3-bf16 /mnt/fastmodels/MiniMax-M3-fp8-experts-AutoRound; do
  echo "$d:"; python3 - "$d" <<'PY'
import sys,glob,os,struct,json,re
d=sys.argv[1]; n=0; dt=set()
for p in sorted(glob.glob(os.path.join(d,"*.safetensors"))):
    with open(p,"rb") as f: h=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(h))
    for k,v in hdr.items():
        if k=="__metadata__": continue
        if re.search(r"\.layers\.(6[0-6])\.|eh_proj|enorm|nextn|\.mtp\.|shared_head", k):
            n+=1; dt.add(v.get("dtype"))
print(f"  MTP-pattern tensors: {n}  dtypes seen: {sorted(dt)}")  # want EQUAL counts, output dtype BF16
PY
done
```
Pass condition: **input MTP count == output MTP count, and output dtype is BF16** (never F8_E4M3 for any
MTP tensor). If output count < input, the export dropped MTP (the int4 failure mode) → reject.

---

## 3. THE SERVE PATH (the crux) — can the EXISTING M3 stack load on-disk fp8?

### 3a. What the existing stack autodetects: **ONLY int4 `.qweight`.**

`hpu-moe-ext/hpu_moe_ext/auto_round_hpu.py` registers the `auto-round`/`auto_round` quant config and
routes a module to the int4 `MoeWNA16HPUMethod` **iff it owns a `.qweight`**
(`_load_quantized_prefixes`, `auto_round_hpu.py:182-229`: scans the index for keys ending `.qweight`).
A static fp8 checkpoint has **NO `.qweight`** (it has `.weight` fp8 + `.weight_scale`). So hpu-moe-ext
would treat every expert as **bf16 unquantized** → wrong (and the `.weight_scale` tensors would be
orphaned). **The current M3 autodetect path does NOT recognize on-disk fp8 experts — only int4.**

### 3b. The "right" upstream path for on-disk fp8 — `--quantization fp8` (`HPUFp8MoEMethod`), and why it's NOT wired for M3

`vllm_gaudi/ops/hpu_fp8.py` registers `HPUFp8MoEMethod` as the fp8 MoE method (`hpu_fp8.py:261`). On a
**channel** (default `VLLM_HPU_FORCE_CHANNEL_FP8=true`) fp8 checkpoint it builds the **native**
`VllmMixtureOfExpertsOpFP8PerChannel` (`hpu_fp8.py:195`), which feeds fp8 weights + fp8 activations
(dynamic-quantized at runtime, `dynamic_quant(x)`) + per-channel dequant scales straight into the native
`mixture_of_experts.fp8` intrinsic — **no dequant round-trip** (this is exactly the op `fp8-moe-gate.md`
validated as GO, evidence `extension/ops.py:1264-1298`). The on-disk layout it expects: per-expert fp8
`w13_weight`/`w2_weight` + `w13_weight_scale`/`w2_weight_scale` (per-channel `[out]`), built by
`fp8_channel_moe_prepare_weights` (`extension/ops.py:1023+`). vLLM's upstream `Fp8MoEMethod.create_weights`
materializes those param names from a compressed-tensors/fp8 checkpoint; detection is via
`Fp8Config.is_checkpoint_fp8_serialized` (from the checkpoint's `quantization_config`).

**Why this does NOT just work for M3 today (the gap):**
1. **swigluoai is silu-only on the native op.** M3's routed experts need the gpt-oss swigluoai gate
   (`hidden_act=swigluoai`, alpha=1.702, limit=7.0). The native fp8 intrinsic's activation map is
   silu/gelu/relu only (documented `m3_expert_runtime_patch.py:192-196`). The ENTIRE M3 swigluoai serve
   is wired onto **`MoeWNA16HPUMethod.apply_monolithic`** (`m3_swigluoai_patch.py:214, 433-449`) — the
   **int4** method. Nothing wraps `HPUFp8MoEMethod.apply` to inject swigluoai. Served via
   `--quantization fp8`, M3's experts would run **silu → incoherent** (the same class of bug that gave
   "token salad" before swigluoai was added, `minimax_m3.py:611-634`).
2. **The fp8 runtime handoff is int4-shaped.** `m3_expert_runtime_patch.py` (`_apply_fp8_swigluoai`,
   route-b) reads `op._w13_fp8_stack`/`_s13_stack` — stacks **built by the int4→fp8 staging**
   (`int4_to_fp8_load.stage_layer_int4_to_fp8`), not by `HPUFp8MoEMethod`. It assumes `layer.moe_op` was
   *swapped in by this patch*, keyed on `layer._m3_fp8_converted` (set only in the int4 wrap path,
   `m3_expert_runtime_patch.py:150`). An `HPUFp8MoEMethod`-built layer sets none of that.
3. **The M3 model class** builds `FusedMoE(quant_config=quant_config)` (`minimax_m3.py:493-509`); with
   `--quantization fp8` the quant_config IS fp8, so FusedMoE *would* pick `HPUFp8MoEMethod` — but then
   the gate (`ReplicatedLinear`, `minimax_m3.py:511-518`) and shared expert must also be handled, and
   the swigluoai gate is unwired (point 1).

### 3c. Smallest change to wire on-disk fp8 (if static-on-disk is mandated)

The user's intuition that "the tools exist" is **half-right**: the *transform numerics* and the
*swigluoai-on-fp8 apply* exist (`int4_to_fp8_load.py` + `_apply_fp8_swigluoai` route-b), but they are
plumbed for int4→fp8-at-load, not for a `HPUFp8MoEMethod`-built layer. Smallest wiring:
- **Add a `HPUFp8MoEMethod.apply` (or `apply_monolithic`) wrap** in a new sibling patch mirroring
  `m3_swigluoai_patch.py`: when the layer is an M3 routed-expert MoE, run `_apply_fp8_swigluoai`'s
  route-b math but reading from `HPUFp8MoEMethod`'s `layer.w13_weight`/`w2_weight` +
  `w13_weight_scale`/`w2_weight_scale` (build the `_w13_fp8_stack`/`_s13_stack` aliases from those in a
  `process_weights_after_loading` wrap, since `_dequant_fp8_slice` already does `weight.to(bf16)*scale`).
- The dequant/swigluoai/GEMM loop in `_apply_fp8_swigluoai` (`m3_expert_runtime_patch.py:270-432`) is
  **reusable as-is** — only the stack-source plumbing differs.
- Serve env mirrors `models/minimax-m3-fp8.env` but: `MODEL_PATH=MiniMax-M3-fp8-experts-AutoRound`,
  **add `--quantization fp8`**, **drop `M3_EXPERT_RUNTIME=fp8`** (no int4→fp8 transform needed), keep
  `M3_SWIGLUOAI=1`, `M3_FP8_SWIGLU_ROUTE=b`, `VLLM_HPU_FORCE_CHANNEL_FP8=true`, TP=8, EP, `fp8_inc` KV.
- **Risk:** the checkpoint's `quantization_config` must satisfy `Fp8Config.is_checkpoint_fp8_serialized`
  AND scope-ignore everything but experts (so attention/gate/shared/dense/lm_head/MTP stay bf16). The
  auto_round-native `fmt=e4m3` config may NOT be recognized by upstream `Fp8Config` (it expects
  `quant_method=fp8` / compressed-tensors `float-quantized`); the **llm-compressor `float-quantized`
  export (§1b) is the format upstream `--quantization fp8` reads**, so use THAT exporter (needs
  `compressed_tensors` installed in the venv) and set its `ignore` list to all non-expert modules.

This is **non-trivial new on-card-validated serve code** (a new method wrap + format-compat + an on-card
coherence/Jaccard gate), not a config flip.

---

## 4. MEASURED CONTEXT (int4 vs fp8 tok/s)

From `bench/M3_MATRIX_REPORT.md` (2026-06-15, the int4 serve, `vllm-gaudi:next`, TP8+EP, fp8_inc KV,
dense-GQA, max-model-len 8192):
- **int4 single-stream decode: 5.8–7.6 tok/s** (mns 16→64; warm bucketing). 8.3 on a warm recipe cache.
- **int4 aggregate: ~17.4–17.6 tok/s @ c≈30 (mns=64, the knee)**; mns doesn't scale M3 the way it did GLM.
- **The single-stream wall is structural host/sync-bound** (device util 14–19%), NOT the MoE compute form
  (batched bmm and ngram both REJECTED as regressions/crashes).
- **No fp8 numbers measured yet.** `fp8-moe-gate.md` is honest: fp8's payoff is at **concurrency/long-ctx**
  (removes the per-token int4→bf16 dequant that doesn't amortize at c≥30 because the routed-K mitigation
  gate `tokens*top_k≤K` turns OFF under load), NOT single-stream (still host-bound ~4–8 tok/s). The
  `minimax-m3-fp8.env` exists specifically to MEASURE int4 vs fp8 at c=10/30/50 — that A/B has not been
  recorded in the bench reports. **Expectation: fp8 ≈ int4 single-stream, fp8 > int4 aggregate at c≥30.**

So the decode-speed motivation for fp8 is concurrency, and **the int4-on-disk + fp8-at-load path delivers
the identical native-fp8-MME execution** — there is no measured tok/s advantage to static-on-disk fp8 over
load-time fp8 (both run the same op in HBM). Static-on-disk's only wins are (a) marginally faster load (no
int4→fp8 transform) and (b) a "true fp8" disk artifact; its costs are +150 GiB disk and the serve-wiring
gap above.

---

## 5. RISKS / GAPS + GO/NO-GO

| # | Risk / gap | Severity | Evidence |
|---|---|---|---|
| R1 | `quantize_m3_int4.py` cannot emit fp8 (model-free rejects `data_type!='int'`) | **blocker for "reuse"** | `model_free.py:25-37`, `_validate_supported_scheme` |
| R2 | Standard (non-model-free) fp8 export needs offload + runnable M3 code; RAM/feasibility UNVALIDATED on this box | high | `compressors/base.py:158-228, 1258-1268`; `compressed_tensors` not installed |
| R3 | hpu-moe-ext autodetects ONLY `.qweight` (int4); on-disk fp8 not recognized | **serve blocker** | `auto_round_hpu.py:182-229` |
| R4 | `--quantization fp8` path (`HPUFp8MoEMethod`) has NO swigluoai wiring for M3 → silu → incoherent | **serve blocker** | `m3_swigluoai_patch.py:214,433-449`; `m3_expert_runtime_patch.py:192-196` |
| R5 | FP8_STATIC llm-compressor export is W8A8-static (needs act calibration), not the W8A16 dynamic-act the native op wants | medium | `export_to_static_fp.py:65-78,219-228` |
| R6 | int4 export already DROPPED all 7 MTP — same enumeration logic could drop MTP from fp8 unless run on the COMPLETE bf16 download | medium | int4 index has 0 MTP; `M3_MATRIX_REPORT.md:131-138` |
| R7 | `SRC_DIR` in the int4 script points at `/mnt/data2`, real download is `/mnt/data1` | low (fixable) | `quantize_m3_int4.py:98` vs `df` |
| R8 | No measured fp8-vs-int4 tok/s exists; static-on-disk gives no HBM-execution advantage over load-time fp8 | informational | `fp8-moe-gate.md` "Speed honesty"; `M3_MATRIX_REPORT.md` |

**GO/NO-GO:**
- **Static-fp8-on-disk approach: NO-GO as a "reuse the existing tools" task.** It needs (a) a new
  standard-path quant script with offload (R1, R2), (b) the llm-compressor `float-quantized` export with
  an experts-only `ignore` (R5), AND (c) new on-card-validated serve wiring onto `HPUFp8MoEMethod` to
  inject swigluoai (R3, R4). That is a multi-step build with two on-card validation gates, contradicting
  "use the tools already here."
- **RECOMMENDED (the fallback, which is actually the repo's designed path): keep int4 on disk (226 GiB),
  serve in fp8 runtime via `models/minimax-m3-fp8.env` (`M3_EXPERT_RUNTIME=fp8`).** This runs the EXACT
  same `VllmMixtureOfExpertsOpFP8PerChannel`-class native fp8 MME in HBM (via `int4_to_fp8_load.py` +
  `_apply_fp8_swigluoai` route-b, both already validated card-free), keeps all swigluoai wiring intact,
  needs ZERO new code, and the only "cost" (the int4→fp8 added error) is measured at a few e-3 relative
  (`int4_to_fp8_load.validate_synthetic`, well under the per-tensor 6.4e-2 floor). MTP is moot for serve
  today regardless — vLLM 0.22 rejects the `mtp` method for `minimax_m3_vl` (`M3_MATRIX_REPORT.md:126-138`).
- **If static-on-disk is still mandated** (e.g. to shave load time or to hold a true-fp8 artifact for a
  future vLLM that registers M3 MTP): do it in this order, each gated — (1) write `quantize_m3_fp8.py`
  using the **llm-compressor `float-quantized`** exporter, experts-only fp8 + everything-else-`ignore`,
  `low_cpu_mem_usage` + immediate-save, validated on a 2-shard RAM cap first; install `compressed_tensors`
  in `m3calib`; run on the COMPLETE bf16 download (R6); (2) verify MTP pass-through (§2); (3) write the
  `HPUFp8MoEMethod` swigluoai wrap (§3c) and validate coherence + router Jaccard on-card before trusting.

---

## EXECUTIVE SUMMARY (the recipe)

1. **bf16→fp8 quant:** `quantize_m3_int4.py` CANNOT do it — AutoRound model-free hard-rejects fp8
   (`model_free.py:25-37`). A NEW sibling using the **standard (offload) path** + the **llm-compressor
   `float-quantized`** exporter (`export_to_static_fp.py`) is required.
2. **Scheme:** per-output-channel E4M3, **W8A16** (weight-only, `act_bits=16`, `group_size=-1`,
   `data_type='fp'`, `sym=True`, Gaudi2 `fp8_dtype_flavor=float8_e4m3fnuz`) — quantize ONLY the 128 routed
   experts (layers 3..59), `ignore` everything else (attn/index/gate/shared/dense/embed/lm_head/norms/
   vision/**MTP** → all bf16).
3. **Size/runtime:** ~376 GiB on disk (vs int4 226 GiB), ~3–8 h via the offload path (slower than int4
   model-free); gate on a 2-shard RAM-bounded dry run.
4. **MTP:** survives ONLY if run on the COMPLETE bf16 download (the int4 export dropped all 7 MTP); verify
   by counting `layers.60..66`/`eh_proj`/`enorm` tensors input==output and dtype==BF16 (§2 script).
5. **Serve — the crux:** the existing M3 stack canNOT load on-disk fp8: hpu-moe-ext autodetects only int4
   `.qweight` (`auto_round_hpu.py:182-229`); the right upstream path is `--quantization fp8`
   (`HPUFp8MoEMethod` → native `VllmMixtureOfExpertsOpFP8PerChannel`), but **nothing wires M3's swigluoai
   onto it** (the native op is silu-only; all swigluoai wiring is on the int4 `MoeWNA16HPUMethod`).
6. **Smallest fix:** a new patch wrapping `HPUFp8MoEMethod` to reuse `_apply_fp8_swigluoai` route-b
   (`m3_expert_runtime_patch.py:270-432`) reading `w13_weight`/`weight_scale` instead of the int4-staged
   stacks — non-trivial, needs on-card coherence + Jaccard validation.
7. **Measured context:** int4 = 5.8–7.6 tok/s single-stream, ~17.4 aggregate @c≈30; no fp8 tok/s recorded;
   fp8's win is concurrency, and the SAME native fp8 op runs in HBM whether the disk is int4 or fp8.
8. **Recommendation:** **NO-GO on static-fp8-on-disk** as a reuse task; **serve the existing int4
   checkpoint in fp8 runtime mode** (`minimax-m3-fp8.env`, `M3_EXPERT_RUNTIME=fp8`) — identical native
   fp8 HBM execution, zero new code, validated few-e-3 error, all swigluoai + MTP plumbing intact.
9. **Path fix if proceeding anyway:** install `compressed_tensors`; point at `/mnt/data1/minimax-m3-bf16`
   (not the script's `/mnt/data2`); run on the full download; validate RAM + MTP + on-card swigluoai.
10. **THE SINGLE BIGGEST RISK:** the serve is unwired — a static fp8 checkpoint would be silently served
    with **silu instead of swigluoai → incoherent output** (R4), because every swigluoai hook targets the
    int4 `MoeWNA16HPUMethod`, not `HPUFp8MoEMethod`; producing the fp8 artifact without first building +
    on-card-validating that swigluoai-on-`HPUFp8MoEMethod` wrap yields a checkpoint that loads but
    generates garbage.
