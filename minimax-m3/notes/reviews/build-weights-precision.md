# Adversarial Review — Weights Precision (MiniMax-M3 on 8×Gaudi2)

**Lens:** re-derive optimal *weights* precision for FASTEST serve @ HIGH concurrency (30+), long ctx.
**Verdict: CHANGE → FP8(INC) routed experts, not INT4.** Confidence: **medium-high**.

---

## TL;DR
At the stated target (c≥30, ctx 32K/128K) **fp8 experts fit with full KV headroom**, so int4's only
advantage (extra KV room) does not pay off until *far* above c=30, while int4's dequant penalty hurts
*every* decode step. On this exact hardware the dequant tax is not theoretical — it is the measured
bottleneck (PROJECT_FINDINGS §11a). Build fp8 experts; keep int4 as the documented fit-only fallback.

## The numbers (M3 config: H=6144, I=3072, 60 L, 4 KV-heads×128, 128 experts, top-4 + 1 shared, GQA)

**Weight footprint** (usable ≈ 691 GiB @ util 0.9):
| Plan | Experts | Rest | Total | Fits? |
|---|---|---|---|---|
| **A — int4 exp + bf16 rest** (current) | 198 | 86 | **285 GiB** | yes, huge KV slack |
| **B — fp8 exp + bf16 rest** | 385 | 86 | **471 GiB** | **yes** |
| B2 — fp8 exp + fp8 rest | 385 | 47 | 432 GiB | yes |
| C — mixed (hot fp8 / cold int4) | — | — | ~331 GiB | yes |

**KV (GQA dense store; MSA *selects* top-k but still stores full K/V), fp8_inc = 60 KiB/tok →
1.88 GiB/seq @32K, 7.5 GiB/seq @128K.** Concurrent-seq capacity after ~20 GiB graph:

| Plan | c-cap @32K | c-cap @128K |
|---|---|---|
| int4 (285) | ~206 | ~51 |
| **fp8 (471)** | **~106** | **~26** |

→ **Crossover:** int4's KV edge only matters above **~c=100 @32K** or where you need **c>26 @128K**.
At c=30 both fit easily; the decision collapses to **decode speed**, where fp8 wins.

## Why fp8 is faster (the decisive, codebase-grounded point)
Gaudi2 MME is BF16/FP8-only. **int4 path = read 0.5 B/elem → dequant-to-bf16 (TPC tax) → bf16 MME
(432 TF ceiling).** **fp8 path = read 1 B/elem → native E4M3 MME (865 TF, 2×), no dequant.** This is
exactly what was measured: GLM int4-experts decode was *"~98% of device time = ~345k tiny ~13µs
int4-dequant kernels computing ALL experts/layer"* and the findings' own conclusion is verbatim
*"a dense/fp8 would be multiples faster — it avoids… ~1M tiny int4-dequant kernels"* (§11a). The guide
ranks FP8(INC) the **#1 throughput lever** (~2× tput, <1% loss); its 98-trick deep search put int4
**dead last, "fit-only, no native matmul."** At 428B/768 GiB we are **not** memory-forced into int4, so
the fit-only premise *inverts*: don't take the fallback when the fast option fits.

## Rough tok/s @c=30 (concurrent, MoE-bound)
int4 inherits the all-experts-dequant blowup → trends to GLM's ~tens-of-tok/s regime under load;
fp8 removes the per-step dequant kernels + doubles MME → realistically **~1.5–2× int4** at c=30,
larger at 32K (compute-dominated) than 128K (attention/KV-dominated). Exact crossover is config-bound;
**measure** stock-MoE fp8 vs int4 at max-num-seqs 64/128.

## Rejected alternatives
- **W4A8 / fp8-compute on int4:** still dequant-then-requant (double quant, §1b/F6: ~6.4e-2 relNorm
  floor) — accuracy loss *and* no speed win. No.
- **Mixed hot-fp8/cold-int4:** plausible HBM (~331 GiB) but needs per-expert routing-frequency split +
  two kernels; complexity unjustified when plain fp8 fits. Park.

## #1 highest-impact change
**Quantize routed experts to FP8(INC) (Plan B, 471 GiB) instead of INT4** — removes the dequant
bottleneck that is *the* measured decode cost on this stack, doubles MME, keeps full c=30 KV headroom at
128K, and aligns with the guide's #1 lever. Validate accuracy (lm-eval, calibrated maxabs, ±240 E4M3,
lm_head/embeds bf16). Keep the int4 recipe as the documented fit-only fallback.
