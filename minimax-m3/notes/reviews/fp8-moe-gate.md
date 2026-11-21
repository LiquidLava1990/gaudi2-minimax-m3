# FP8-MoE Decision Gate — MiniMax-M3 routed experts: int4 → FP8(INC)?

**Question this gate answers:** Does vllm-gaudi run INC fp8 MoE experts **natively** on the Gaudi 2 MME
(fp8 weights → fp8 GEMM, no dequant), or does it dequant fp8→bf16 first (like int4)? This decides whether
the fp8 pivot actually removes the measured ~98%-of-decode dequant bottleneck (PROJECT_FINDINGS §11a).

**Method:** CARD-FREE, code-only inspection (CPU container; no card / no vllm-active / no installed-plugin
touch). Evidence is file:line into the on-disk source that the running image is built from
(`$HOME/AI/_src/vllm_gaudi` == image `vllm-gaudi:next`, per PROJECT_FINDINGS §6).

---

## VERDICT: **GO** (native fp8 MoE serving path EXISTS on the stock stack; no kernel build needed)

> **vllm-gaudi has a wired, native fp8-MoE MME path for the routed experts.** The per-channel/per-tensor
> fp8 MoE op (`VllmMixtureOfExpertsOpFP8PerChannel`) passes **fp8 weights + fp8 activations + dequant scales
> straight into a native fp8 variant of the Habana `mixture_of_experts` intrinsic** — there is **NO**
> dequant-to-bf16 round-trip. This is the opposite of the int4 path (which is bf16-only and MUST dequant
> every expert). So pivoting M3's routed experts to FP8(INC) **does** remove the dequant tax that is the
> measured ~98%-of-decode cost. **GO — and do NOT build a kernel; the hpu-moe-ext fp8 scaffold is the wrong
> tool (it is an int4→fp8 double-quant op, see Q4).**

Confidence: **high** on "native fp8 MoE path exists & is wired" (direct code + open-source bridge op proof).
Medium on the absolute speedup magnitude (must be measured; the *mechanism* is removed, the *number* is
config/fabric-bound — see "Speed honesty" below).

---

## Q1 — Is there a native fp8-MoE MME serving path (stock vllm-gaudi OR hpu-moe-ext)? YES, in STOCK vllm-gaudi.

### The decisive code: two stock fp8 MoE ops, ONE native, ONE dequant

`vllm_gaudi/ops/hpu_fp8.py` registers `HPUFp8MoEMethod` as the MoE quant method (`fp8.Fp8MoEMethod =
HPUFp8MoEMethod`, hpu_fp8.py:261). At weight-load it picks ONE of two ops
(`HPUFp8MoEMethod.process_weights_after_loading`, hpu_fp8.py:176-210):

```
if self.block_quant and not VLLM_HPU_FORCE_CHANNEL_FP8:   # hpu_fp8.py:186
    layer.moe_op = VllmMixtureOfExpertsOpFP8(...)          #   BLOCK   → DEQUANT path
else:
    layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(...)#   CHANNEL → NATIVE  path
```

**`VLLM_HPU_FORCE_CHANNEL_FP8` defaults to `True`** (`envs.py:8`, `envs.py:25-26`). So even a block-scaled
fp8 checkpoint is force-converted to channel-wise fp8 at load and routed to the **native** op. Per-channel /
per-tensor (what INC produces) → native op unconditionally.

**NATIVE op — `VllmMixtureOfExpertsOpFP8PerChannel.forward` (extension/ops.py:1264-1319):**
- Weight lists are the **fp8 tensors themselves** — `self._cached_w13_views = tuple(w13_list[i].weight.squeeze())`
  where `.weight` is the stored fp8 weight (ops.py:1248-1251, 1279-1282). **No dequant loop.**
- Activation cast to fp8 once: `x_fp8, x_scale = dynamic_quant(x)` (ops.py:1285) — `dynamic_quant` →
  `cast_to_fp8_v2` (ops.py:932-939).
- Then the GEMM is the **native fp8 variant** of the intrinsic, fed fp8 weights + fp8 act + the dequant
  scales as op inputs (ops.py:1286-1298):
  ```
  torch.ops.hpu.mixture_of_experts(hidden_states=x_fp8, w12=<fp8 w13>, w3=<fp8 w2>,
      d_scale_hidden_states=x_scale, d_scale_w12=w13_weight_scale, d_scale_w3=w2_weight_scale, ...)
  ```
  The MME consumes fp8 operands and applies the scales internally → **no bf16 weight ever materialized, no
  HBM round-trip.** (Static-activation branch is the same, ops.py:1299-1317, with
  `d_scale_intermediate_hidden_states` for the down-proj.)

**DEQUANT op — `VllmMixtureOfExpertsOpFP8` (block, extension/ops.py:1160-1223), for contrast:** this one
DOES dequant — it loops `get_dequant_weight()` → `dequant_block_fp8_weight_naive` (fp8→bf16, ops.py:814-850)
for every expert, `mark_step()`, then calls **plain bf16** `mixture_of_experts` (ops.py:1187-1202). This is
the path you must AVOID, and the default env (`FORCE_CHANNEL_FP8=true`) already avoids it.

### Device-level confirmation: the native fp8 intrinsic is real, not a typo

The open-source Habana bridge vendored under hpu-moe-ext ships `mixture_of_experts` **fp8 op variants** as
first-class GUIDs (so the `d_scale_*` kwargs above bind a genuine native-fp8 recipe, not an emulation):
- `lazy_custom_ops.cpp:179` `mixture_of_experts_fp8_lazy(...)` — op_name `"mixture_of_experts.fp8"`
  (`:199`), inputs incl. `d_scale_hidden_states` (`:186`), `d_scale_w3` (`:190`).
- `lazy_custom_ops.cpp:234` `mixture_of_experts_fp8_fused_weights_lazy(...)` — op_name
  `"mixture_of_experts.fp8_fused_weights"` (`:252`), with `d_scale_w12`/`d_scale_w3` (`:242-243`).
- backend gen: `hpu_ops/backend/mixture_of_experts_gen.cpp` + `fp8_gemm_gen.cpp` (native fp8 MME GEMM GUID).

So the stock op is **bf16-only ONLY for the int4 WNA16 path** (which has no fp8 scale kwargs); for genuinely
fp8-stored weights the same intrinsic has a native fp8 entry point, and vllm-gaudi wires it.

### Net Q1: stock does NOT dequant fp8 (in the default channel-fp8 config). The hpu-moe-ext fp8 scaffold is irrelevant to this pivot (Q4).

---

## Q2 — Expected quality: clarify "fp8 ~6.3% floor" vs INC <1%

**The ~6.4e-2 (≈6.3%) figure is NOT the INC fp8 number. It is the int4→fp8 DOUBLE-QUANTIZATION floor of the
fused *int4* kernel's fp8-compute toggle — a different, irrelevant code path.**

- PROJECT_FINDINGS:154 (verbatim): *"INT4→fp8 is **double quantization** with a hard ~6.4e-2 relNorm floor
  (measured on-device … per-channel/per-group scaling does not break ~5e-2)."* §242 tags it
  *"relNorm floor (INT4→fp8 double-quant)"*. Root cause = the fn-vs-fnuz factor-of-2 + stacking int4 error
  under a 3-mantissa-bit fp8 (`fp8_routed_moe.py:96-107`, `FUSED_GENERALIZE_FP8_DESIGN.md:115-132`).
- That floor applies when you START from int4 weights and ADD fp8 compute (W4→bf16→fp8). M3's pivot does
  the opposite: it starts from the **bf16** checkpoint and quantizes ONCE to fp8 (bf16→fp8). **No int4 in
  the chain ⇒ the double-quant floor does not apply.**

**INC fp8-from-bf16 is single quantization → the guide's <1% bar.** Optimization Guide §1.1 / §2 (lines
83, 93-94): FP8(INC) E4M3 = *"~2× throughput, ~½ memory, <1% accuracy loss on 70B-class."* M3 should be
**easier** than 70B-class for the expert-only quant: experts are a fraction of the 426B (router/gate,
shared expert, attention, embeddings, lm_head stay bf16 by INC default — Guide §2 line 104), and the
per-token MoE error is the only thing quantized. Quality bar from quant-quality-eval.md:52: **Δ ≤ 1.0pp
(MMLU/GSM8K) vs the bf16 control, hard-fail >2pp** — generous for expert-only fp8.

**Mandatory quality discipline (or fp8 cliffs):** calibrate maxabs **on the G2 cards**
(`calibrate_model.sh -o g2` → `maxabs_quant_g2.json`); **NEVER `scale_method=unit_scale`** (Mixtral
PPL 11.34→96.65, Guide §"Myths"); fp8 element is **fnuz, ±240 not ±448** (the cast/gemm already handle
fn→fnuz internally, ops.py:30, but a wrong scale clips). Run the router-agreement Jaccard gate from
quant-quality-eval.md §2d.1 (top-1 ≥0.98, top-4 Jaccard ≥0.95) — the same parity discipline that gated DSA.

**Net Q2: expected quality is <1% (INC single-quant), NOT 6.3%. The 6.3% is the int4→fp8 double-quant
number for a path M3 does not use.**

---

## Q3 — Exact serve path that yields a servable fp8 M3, and does it keep the 7 MTP modules alive?

**M3 is bf16 on disk with NO upstream fp8 (or int4) checkpoint** (download is bf16, no `quantization_config`;
confirmed by the model-context sweep). So fp8 must be produced on-box via **INC calibrate → serve**. Two
equivalent serve shapes; the INC-dynamic one is the natural fit for a bf16 source.

### Path A (recommended) — INC dynamic quant (`--quantization inc` + `QUANT_CONFIG`)

How `inc` resolves: `oot_get_quantization_config("inc")` returns `_FakeINCConfig` (ops.py:1525-1532), whose
`get_quant_method` builds the MoE **unquantized** first — `UnquantizedFusedMoEMethod` → `VllmMixtureOfExpertsOp`
(bf16 `MoeMatmul`) (quant.py:27-32). **Intel Neural Compressor (`neural_compressor_pt`) then patches those MoE
modules at runtime** via the `get_dequant_weights_func` hook (ops.py:942-948, 982-985): MEASURE reads the bf16
weight to fit ranges; QUANTIZE **requantizes the bf16 weight to fp8 and updates the weight in place**, after
which the op runs the native fp8 MME (the hook contract is documented verbatim at ops.py:1143-1154:
*"In the quantization phase, INC requantizes the BF16 weight to FP8 and updates the weight"*). `QUANT_CONFIG`
also forces CPU weight-load to dodge bf16 OOM (platform.py:231-239) and is the master switch for fp8 KV.

Two-pass serve (mirrors Guide §2 + §4 "FP8 (INC)" recipe):
```
# 0) Build calibration set: code + agentic/tool-call + long docs (seqlen>=8192, nsamples>=256)
#    so all 128 experts are hit (quant-quality-eval.md §4). NOT generic pile-10k@2048.
# 1) MEASURE  — QUANT_CONFIG json mode=MEASURE → maxabs_measure_g2.json
# 2) QUANTIZE — QUANT_CONFIG json mode=QUANTIZE (maxabs_quant_g2.json), then serve:
PT_HPU_LAZY_MODE=1                 # INC fp8 REQUIRES lazy (Guide §3 / Part II §3; torch.compile unsupported for INC)
PT_HPU_ENABLE_LAZY_COLLECTIVES=true
PT_HPU_WEIGHT_SHARING=0            # free full-precision weights post-fp8 → avoid OOM (Guide 235B recipe, Part II §9)
VLLM_HPU_FORCE_CHANNEL_FP8=true    # DEFAULT — keeps the NATIVE per-channel op (do not flip to block)
QUANT_CONFIG=/path/maxabs_quant_g2.json
VLLM_DISABLE_MARK_SCALES_AS_CONST=true
vllm serve <m3-bf16> --quantization inc --kv-cache-dtype fp8_inc \
   --tensor-parallel-size 8 --enable-expert-parallel \
   --block-size 128 --max-num-seqs 64..128 --max-num-batched-tokens 16384 \
   --gpu-memory-utilization 0.9 --max-model-len <real need>
```

### Path B — pre-serialized fp8 checkpoint (`--quantization fp8`)

If you instead emit an fp8 safetensors checkpoint offline, `--quantization fp8` → `HPUFp8MoEMethod` →
(channel default) → the SAME native `VllmMixtureOfExpertsOpFP8PerChannel`. Identical native MME execution;
just shifts the quant from serve-time to export-time. Either is fine; Path A is simpler for a bf16 source.

### MTP (7 modules) — stays ALIVE under fp8. This is a *reason for fp8*, not a risk.

- M3 ships **7 MTP modules** (`num_mtp_modules=7`; loaded as a load-time SKIP at logical layers 60-66 via
  `get_spec_layer_idx_from_weight_name` over `range(num_mtp_modules)`, `minimax_m3.py:807-815`, skip at
  `:668-670`). They are **bf16** in the checkpoint.
- **INC/fp8 does not touch them the way int4 would.** The danger precedent is GLM, whose int4 AutoRound
  export quantized layer-78 to int4-EVERYWHERE with **no calibration data flowing through the MTP layer** →
  a DEAD draft head needing a painful bf16 graft (PROJECT_FINDINGS §8). With INC-fp8-experts on M3, the MTP
  modules are not in the routed-expert set the recipe quantizes; keep them bf16 (INC default skips
  non-targeted modules, same as lm_head/embeddings) → **all 7 MTP modules stay native bf16, no graft.**
  MTP delivered a proven **1.70× decode multiplier** on GLM (§12); choosing int4 for M3 risks killing it the
  GLM way, choosing fp8/bf16 preserves it (quant-holistic-optimum.md:76-83).
- Serve caveat: spec-decode (MTP-on) is incompatible with `--async-scheduling` on HPU; the platform already
  force-disables async when a speculative config is present (platform.py:244-247). Fine.

**Net Q3: serve via INC calibrate → `--quantization inc` (lazy + EP + `PT_HPU_WEIGHT_SHARING=0` + fp8_inc KV).
The native channel-fp8 MoE op is the default. All 7 MTP modules stay bf16-alive (no int4-dead-head).**

---

## Q4 — If native fp8-MoE were NOT wired: the build delta. (It IS wired, so this is the rejected branch.)

**Not needed — STOP. Do not build a kernel for this.** Recording why, so the question is closed:

- **The hpu-moe-ext fp8 scaffold is the WRONG tool for M3.** `moe_fused_expert_fp8_backend.cpp` (+
  `build_fp8.sh`, `test_fused_expert_fp8_op.py`, `FUSED_GENERALIZE_FP8_DESIGN.md`) and the zero-build
  `fp8_routed_moe.py` are an **int4→fp8** path: node 1 is `dequantize_4_bit_bf16(...)` from **int4** weights
  (FUSED_GENERALIZE_FP8_DESIGN.md:89), THEN bf16→fp8. That is the **double-quant** route with the
  documented ~6.4e-2 floor (lines 115-132) — it exists to give the *int4* kernel an fp8-compute toggle, the
  worst-of-both for a model that has a bf16 source. It takes int4-packed weights as input; M3's fp8 experts
  are not int4. Using it would re-introduce the very 6.3% floor Q2 rules out.
- **Build status of that scaffold (for the record):** `libmoe_fused_expert_fp8.so` exists (392KB, eager
  only — no `_lazy` build) but is **scaffold, not validated**: `BUILD_FP8_NODES=0` makes the `.so` fall back
  to the bf16 recipe; the real fp8 nodes are unbuilt/unvalidated on-device
  (FUSED_GENERALIZE_FP8_DESIGN.md:74-82). It is **not wired into any python serving path** (no
  `libmoe_fused_expert_fp8` import anywhere in `hpu_moe_ext/*.py`). Irrelevant either way.
- **The native path already exists in stock** (Q1) and is a stock Habana intrinsic — exactly the case
  PROJECT_FINDINGS Finding 5 says NOT to rebuild ("don't build a custom kernel to beat a well-optimized
  stock op; you inherit its optimization debt"). The OpBackend infra's payoff is kernels with NO stock
  equivalent (DSA/MSA), not this one.

If a *future* surprise showed the native op underperforming (it will not be a *correctness* gap — the op is
native), the lever is a **config/serve** A/B (max-num-seqs, moe_chunk, EP topology), NOT a kernel.

---

## Speed honesty (so this gate is not over-sold)

GO means the **dequant bottleneck mechanism is removed by construction** — fp8 experts hit a native fp8 MME,
not ~1M tiny int4→bf16 dequant kernels computing all 128 experts (the §11a pathology, which at the target
c≥30 is unavoidable for int4 because the routed-K mitigation gate `tokens*top_k≤K` turns OFF under
concurrency). fp8 also halves expert HBM traffic and runs the MME at ~2× (Guide §2; arXiv 2503.09975
~865 TFLOPS). **What GO does NOT promise:** a specific tok/s. Per PROJECT_FINDINGS §3/§11, single-stream
decode is host/collective/fabric-bound (the MoE op is a small slice at batch=1), and absolute throughput is
config- and **fabric**-bound (the box's intermittent scale-up link storm has repeatedly masked real numbers).
The honest framing: **fp8 makes the expert compute cheap and native — its payoff shows at the concurrency/
long-ctx regime M3 targets, and must be confirmed by measuring stock-fp8-MoE vs int4 at max-num-seqs 64/128
on a quiet fabric.** The decision to pivot does not depend on that number; the *mechanism* (no dequant) is
proven in code now.

---

## Evidence index (file:line)

| Claim | Evidence |
|---|---|
| fp8 MoE op selection (block→dequant vs channel→native) | `vllm_gaudi/ops/hpu_fp8.py:186-201` |
| `HPUFp8MoEMethod` is the registered fp8 MoE method | `hpu_fp8.py:261` |
| `VLLM_HPU_FORCE_CHANNEL_FP8` defaults True → native op | `vllm_gaudi/envs.py:8, 25-26` |
| **NATIVE fp8 MoE: fp8 weights+scales → fp8 intrinsic, NO dequant** | `extension/ops.py:1264-1298` (esp. 1279-1282, 1286-1298) |
| Native op static-act branch (down-proj scale) | `extension/ops.py:1299-1317` |
| Block fp8 op DOES dequant (the path to avoid; default-off) | `extension/ops.py:1160-1223` (loop 1187-1189 → bf16 moe 1193) |
| fp8→bf16 dequant helper (used only by block op) | `extension/ops.py:814-850` |
| Native fp8 intrinsic exists in open-source bridge | `…/bridge/habana_kernels/lazy_custom_ops.cpp:179, 199, 234, 252`; `…/hpu_ops/backend/mixture_of_experts_gen.cpp`, `fp8_gemm_gen.cpp` |
| INC resolves to placeholder → unquantized MoE, INC patches at runtime | `extension/ops.py:1525-1532`; `extension/quant.py:27-32` |
| INC requantizes bf16→fp8 in QUANTIZE phase (the hook) | `extension/ops.py:1143-1154`, `942-948`, `982-985` |
| INC forces CPU load (OOM dodge) | `platform.py:231-239` |
| 6.4e-2 floor = INT4→fp8 DOUBLE-quant (NOT INC) | `PROJECT_FINDINGS.md:154-155, 225, 242` |
| INC single-quant <1% bar | `Gaudi2_Optimization_Guide.md:83, 93-94, 104` |
| Quality thresholds for the M3 fp8/int4 build | `notes/reviews/quant-quality-eval.md:52, §2d.1 line 69` |
| hpu-moe-ext fp8 scaffold is int4→fp8 (dequant_4_bit first) | `…/opbackend/FUSED_GENERALIZE_FP8_DESIGN.md:89, 115-132`; `hpu_moe_ext/fp8_routed_moe.py:1-107` |
| fp8 scaffold is unbuilt/unvalidated/unwired | `FUSED_GENERALIZE_FP8_DESIGN.md:74-82`; `.so` exists but no python import |
| M3 = 128 experts, top-4, H=6144, I=3072, 60L, +1 shared, bf16, no quant_config | model-context sweep (`vllm/minimax_m3.py:204-247`; on-disk `config.json`) |
| 7 MTP modules, loaded as skip at L60-66 | `vllm/minimax_m3.py:807-815, 668-670` |
| MTP int4-death precedent (why fp8/bf16 keeps it alive) | `PROJECT_FINDINGS.md:§8`; `notes/reviews/quant-holistic-optimum.md:76-83` |
| spec-decode (MTP) ⊥ async-scheduling on HPU (auto-handled) | `platform.py:244-247` |
| int4 routed-K mitigation dies at concurrency (why int4 worse at c≥30) | `PROJECT_FINDINGS.md:§11a`; `quant-holistic-optimum.md:50-53` |

---

## One-line answer

**GO.** Stock vllm-gaudi runs INC/fp8 routed experts on a **native fp8 MME path** (per-channel op →
`mixture_of_experts.fp8`, fp8 weights+activations+scales, **no dequant** — ops.py:1286-1298, bridge
`lazy_custom_ops.cpp:179`); this removes the §11a dequant bottleneck. Expected quality **<1%** (INC
single-quant — the 6.3% figure is the unrelated int4→fp8 double-quant floor). Serve via **INC calibrate →
`--quantization inc`** (lazy + `--enable-expert-parallel` + `PT_HPU_WEIGHT_SHARING=0` + `fp8_inc` KV;
channel-fp8 is the default), which **keeps all 7 MTP modules bf16-alive** (no int4-dead-head graft).
**No kernel build** — the hpu-moe-ext fp8 scaffold is an int4→fp8 double-quant op, the wrong tool here.
