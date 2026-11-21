# MiniMax-M3 — Adversarial review: activation / compute precision (FP8 compute vs bf16-compute)

Lens: should the **MME compute** run in FP8 (E4M3) rather than just storing weights low-precision?
Target: fastest + minimal quality loss + **high concurrency (30+)** on 8× Gaudi 2, vllm-gaudi.
Sources: `$HOME/Desktop/Gaudi2_Optimization_Guide.md` §1–4 + calibration notes;
`$HOME/AI/Gaudi2_Kernel/PROJECT_FINDINGS.md` §2–4, §6, §11; M3 `config.json`.

---

## TL;DR VERDICT

**Two regimes, two answers — and the current "int4→bf16-dequant experts" plan is wrong for the throughput target:**

1. **Routed MoE experts → full INC W8A8-fp8 (fp8 weights + fp8 activations, native MME, NO dequant).**
   This is the #1 change. It is *both* faster than int4-dequant-to-bf16 AND faster than bf16, because
   it (a) feeds the MME its native fp8 path (~865 TFLOPS peak vs the bf16 MME) and (b) **deletes the
   int4→bf16 dequant round-trip that the project measured as the actual decode bottleneck** (PROJECT_FINDINGS §11:
   the on-disk Kineto trace attributes ~98% of decode device-time to ~1M tiny int4-dequant kernels computing
   experts — *not* collectives). fp8-storage experts skip that entirely.
2. **Attention (GQA + MSA) → keep bf16 compute** for bring-up; `fp8_sdpa` is a *later, measured* add-on,
   not a launch lever. KV cache stays `fp8_inc` (already the only accepted KV dtype, `platform.py:388`).

**Reject** the two tempting middles: (a) **W4A8** (int4 weights, fp8 acts) — you still pay int4→fp8
*double quantization* with a hard **~6.4e-2 relNorm floor** measured on-card (PROJECT_FINDINGS §4a/§6,
Finding 6), i.e. you keep the dequant cost AND add fp8 error for no compute that fp8-W8A8 doesn't already give;
(b) leaving experts **bf16** — fits arithmetic only barely and leaves the 2× MME on the table at concurrency.

---

## Why compute-precision matters here specifically (the M3 numbers)

| Fact (from config) | Value | Consequence for compute precision |
|---|---|---|
| Routed-expert params | **413 B** | 770 GiB bf16 / ~192 GiB int4 / **~206 GiB fp8** across 8×96 GB. The experts ARE the model; their GEMM precision *is* the throughput story. |
| Experts / layer, top-k | 128 / **4** | Sparse-4: at concurrency the active expert GEMMs are MME-bound → fp8 MME (2×) bites directly. |
| MoE layers | **57** of 60 | 57× per token of the expert GEMM whose precision we're choosing. |
| Attention | GQA 64q/4kv + MSA block-sparse | Attention FLOPs are a *small* fraction vs 413B experts → fp8-attn is a minor lever; fp8-experts is the dominant one. |

The plan's premise ("store int4, dequant to bf16 for the MME") is the **exact path PROJECT_FINDINGS §11
caught dominating the decode step** (≈98% device time in int4-dequant micro-kernels, not collectives —
this *corrects* the earlier "single-stream is 90% collective-bound" claim for the served sparse path).
A precision that **needs no dequant** removes that whole class of work. Only fp8-W8A8 (and bf16-storage)
qualify; int4-anything (W4A16 today, or W4A8) keeps the dequant.

---

## Throughput: what fp8 compute buys at 30+ concurrency

- **MME peak:** guide §2/§1 cites Gaudi 2 fp8 GEMM ceiling **~865 TFLOPS (~98% MFU)**; Llama-3.1-70B
  reference **decode ~387 TFLOPS @ batch-128**, prefill ~649 @1k. The bf16 MME is roughly half-rate, so
  the headline is **~2× MME throughput** on the expert GEMMs — realizable **only at concurrency** (batch
  fills the MME; the project proved single-stream is host/overhead-bound, so fp8 ≈ no single-stream win).
- **Concurrency is exactly the regime that pays.** PROJECT_FINDINGS §3 Finding 4: stock throughput scaled
  16→126, 64→486, **128→793 tok/s** and was still climbing — config-bound, not op-bound. Moving the expert
  GEMM to native-fp8 raises that ceiling further precisely where M3 is aimed (`--max-num-seqs 64→128`).
- **Memory → bigger batches too.** fp8 experts (~206 GiB) vs bf16 (770 GiB, doesn't replicate; barely fits
  TP8) frees HBM for KV/graph → larger `max-num-seqs` → the "saturate the batch" win (guide §4) compounds the
  raw 2×. Net realistic gain at c≥64: **meaningfully above the bf16-compute ceiling** (I won't quote a hard ×;
  it's the 2× MME *plus* batch headroom *minus* the residual collective/host floor — measure it, see below).

Quantified expectation to test: **fp8-W8A8 experts should beat the int4→bf16 plan by a wide margin at c≥32
(int4-dequant is the measured bottleneck) and beat bf16-compute by up to ~2× on the expert-GEMM-bound portion**,
converging toward the per-token collective/host floor at the top.

---

## Quality cost — and the MoE-FP8 landmine

- 70B-class INC fp8 loss is **<1%** (guide §1/§2); the risk lives in **MoE + uncalibrated**.
- **HARD RULE (guide §4, myth table): never uncalibrated FP8 on MoE.** `scale_method=unit_scale` took
  Mixtral PPL **11.34 → 96.65**. M3 is a *bigger, sparser* MoE (128 experts, top-4) → uncalibrated fp8 here
  is a quality cliff, not a <1% nick.
- Therefore fp8-W8A8 is **gated on INC calibration on the cards**:
  `vllm-hpu-extension/calibration/calibrate_model.sh -o g2` (MEASURE→QUANTIZE → `maxabs_quant_g2.json`),
  `--quantization inc`, `QUANT_CONFIG=…`. **Gaudi-2 specifics that bite (guide calibration notes):**
  ±240 E4M3 range (not ±448 — if any fp8 ckpt arrives OCP-scaled, set `VLLM_HPU_CONVERT_TO_FP8UZ=1`);
  HW pow2 scales {2⁻⁸,2⁻⁴,2⁰,2⁴} → validate `maxabs_hw` vs pow2; **`lm_head`/`embed` stay bf16**;
  calibrate on a representative code/agentic/long-ctx set (same corpus the int4 recipe names), β≈0.95–1.0.
  Post-calibration **gate on `lm-eval` MMLU/GSM8K** vs the bf16 baseline before flipping it on.
- Expected quality after calibration: **≈ bf16 to within ~1%** (MoE, well-calibrated). That is *better*
  quality than the project's int4 expert path (the int4-mixed AutoRound checkpoint already shows a
  long-context greedy drift / needle floor — PROJECT_FINDINGS §7). fp8-W8A8 is **higher quality AND faster**
  than int4-dequant here; the only cost is the calibration pass + ~206 vs 192 GiB HBM (negligible).

---

## Interaction with the weights-precision choice (this is the crux)

The activation choice is **not orthogonal** to weights — it *selects* the weight format:

- **bf16-compute** ⇒ weights bf16 or int4-with-dequant. Today's plan. Keeps the measured dequant bottleneck.
- **W4A8** ⇒ int4 weights, fp8 acts ⇒ int4→fp8 **double-quant, ~6.4e-2 floor**, still dequants. Worst of both.
- **fp8-W8A8 (recommended)** ⇒ weights stored fp8 ⇒ **single quantization, no dequant, native MME.** The
  activation decision (fp8) and the weight decision (fp8) collapse into one INC recipe.

So "store int4, compute bf16" vs "store fp8, compute fp8" is the real fork. For the **high-concurrency
throughput** mandate, fp8-W8A8 wins on both axes (speed: native MME + no dequant; quality: calibrated MoE
beats int4). int4 remains the **fit-only fallback** if a 206 GiB fp8 export can't be produced in time
(it needs an INC fp8 expert export; the existing `quantize_m3_int4.py` produces int4, so this is *new quant work*).

---

## Recommended config (activation/compute precision)

```
# experts: native fp8, calibrated — the throughput lever
--quantization inc
QUANT_CONFIG=/quant/m3_maxabs_quant_g2.json     # MEASURE→QUANTIZE on-card first; lm_head/embed bf16
--kv-cache-dtype fp8_inc                         # already the only accepted KV dtype
PT_HPU_LAZY_MODE=1                               # INC fp8 forces the lazy path (guide §3; not torch.compile)
PT_HPU_WEIGHT_SHARING=0                          # free full-precision weights post-fp8 (avoid OOM, guide #9)
VLLM_DISABLE_MARK_SCALES_AS_CONST=true           # faster fp8 warmup at scale_format=const
--tensor-parallel-size 8 --enable-expert-parallel --block-size 128
--max-num-seqs 128 --gpu-memory-utilization 0.9  # batch lever; fp8 HBM headroom enables it
# attention: bf16 SDPA for bring-up. fp8_sdpa = a SEPARATE later A/B, not a launch lever.
```

Caveats inherited from the box: fp8 ⇒ **lazy mode**, and the project's GLM lazy-decode-numerics bug
(`PT_HPU_LAZY_MODE=1` mis-computes GLM decode, PROJECT_FINDINGS §6c) is a **lazy-path risk M3 must re-verify**
(GLM is MLA; M3 is GQA + a *different* MoE path, so it may be clean — but greedy lazy-vs-eager byte-compare
is mandatory before trusting it). If lazy is broken for M3, fp8's throughput is gated until that's fixed —
which is itself the strongest reason to **measure fp8-W8A8 early**, not assume it.

---

## VERDICT (≤300-word return is the separate summary)

- **Compute precision:** **fp8-W8A8 (INC) for the routed experts**; **bf16 attention** (fp8_sdpa later);
  **fp8_inc KV**. Not bf16-compute, not W4A8.
- **Throughput:** ~2× expert-GEMM MME (865 TFLOPS ceiling) **+** deletes the measured int4-dequant
  bottleneck **+** HBM headroom for bigger batches → clears the bf16/int4 ceilings at c≥32–128.
- **Quality:** ≈ bf16 within ~1% **iff INC-calibrated** (MoE uncalibrated = Mixtral 11→97 PPL cliff);
  *higher* quality than the int4 expert path.
- **Confidence:** High that fp8-W8A8 > int4-dequant on both speed and quality at concurrency (directly
  supported by the §11 trace + §2 ceiling + the §4 MoE-calibration rule). Medium on the absolute multiple
  (lazy-path risk + residual collective floor) — must be measured, not assumed.
- **#1 change:** export the **experts to INC fp8 (W8A8), calibrated on-card**, and serve `--quantization inc`
  — replacing "int4 weights → bf16 dequant for the MME." This is new quant work (the current recipe makes int4),
  and it is the single biggest concurrency-throughput lever available without touching the fabric.
