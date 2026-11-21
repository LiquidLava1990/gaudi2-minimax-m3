# swigluoai → working+fast M3 by morning — SEQUENCING / STRATEGY (agent 5 synthesis)

**Angle:** pick the path that gets a WORKING M3 tonight and a FAST M3 by morning. Card-free.
Reads: PROJECT_FINDINGS §11a/§13c (dequant tax + float-only MME), the sibling docs
(`swigluoai-numerics.md`, `fp8-moe-gate.md`, `giant-int4-strategy.md`, `giant-model-sizing.md`,
`w4a8-fused-kernel.md`), and the runnable code (`_src/vllm_gaudi/extension/ops.py`,
`hpu-moe-ext/hpu_moe_ext/moe_wna16_hpu.py`, `minimax-m3/vllm/minimax_m3.py`).

---

## THE REFRAME (this changes the whole question)

The prompt's premise — *"M3 int4 serving is one kernel-capability from working"* — is the
**wrong battle**, on three independently-sufficient grounds, all already proven on this box:

1. **M3 does NOT need int4.** M3 is **427–428B** → **fp8 ≈ 428 GB fits 768 GB comfortably**
   with full KV headroom (`giant-model-sizing.md:29-30,43`: M3 fp8 = "YES" / "still comfortable").
   int4 is a *memory* play for the 1T+ tier (Kimi/GLM-5.2); M3 is **Tier-1 (fp8)** by
   `giant-int4-strategy.md`'s own taxonomy.
2. **The int4 fused kernel is the wrong tool even where int4 IS needed.** Finding 2: the fused
   OpBackend MoE op is **slower than stock** (`~23 vs ~30` single / `~68 vs ~126` thrpt). Finding
   §11a: int4 decode is **~98% dequant tax** (float-only MME, `§13c`). Putting swigluoai into the
   int4 fused kernel inherits the dequant tax AND the on-card bring-up risk — for a model that
   shouldn't be int4 at all.
3. **swigluoai is not actually a "kernel capability" you must fab.** The activation lives in
   vllm's `SwigluOAIAndMul` CustomOp, applied **outside** the expert GEMM in every unfused path
   (`minimax_m3.py:64,324-339` dense MLP already uses it; `FusedMoE` carries
   `activation=hidden_act` + `swiglu_limit`). The *only* thing that hard-codes silu is the
   **fused** int4 graph (`moe_wna16_hpu.py:995,1062`: non-silu ⇒ returns None ⇒ **falls through to
   the stock op**) and the bridge `mixture_of_experts` intrinsic (which *does* expose a
   `scaled_swiglu` bool — `mixture_of_experts.h:63` — but the python wrapper doesn't plumb M3's
   alpha=1.702/limit=7.0). So swigluoai is a **wiring + small-activation** problem, not a
   fab-a-TPC-kernel problem.

**Net: do not chase option (B) [fused int4 + swigluoai mode] tonight.** It is the highest-risk,
lowest-payoff branch (overnight on-card bring-up, the FCD-geometry history, AND it loses to stock
even when it works). It is not on the critical path to either "working" or "fast."

---

## RECOMMENDED SEQUENCE (tonight → morning)

### TONIGHT — coherence-first on the path that is ALSO the fast path

This is the key efficiency: for M3 the **coherence-first** config and the **production-fast**
config are the *same family* (fp8 / native MME), so "working tonight" is not throwaway — it is the
first leg of "fast by morning."

- **Step 0 (now, card-free):** verify the swigluoai wire end-to-end on CPU against
  `swigluoai-numerics.md` — confirm `SwigluOAIAndMul(alpha=1.702, limit=7.0)` reproduces the HF
  inline math (asymmetric clamp: `gate.clamp(max=limit)`, `up.clamp(±limit)`, `glu =
  gate*σ(α·gate)`, `(up+1)·glu`). This is the gate for everything below. (≈30 min, no card.)

- **Step 1 — WORKING M3 tonight = bf16 experts, stock unfused MoE, swigluoai OUTSIDE the GEMM.**
  Serve M3 with **NO quantization** (bf16, `--quantization` unset): `FusedMoE` →
  `VllmMixtureOfExpertsOp` (bf16 `MoeMatmul`); swigluoai is applied by the vllm activation layer,
  silu is never invoked. This is **option (C) done right** — correct-by-construction, zero kernel
  risk, exercises the WHOLE stack (MSA dense-fallback, shared expert, MTP, dense-prefix). bf16 M3
  weights = ~856 GB which **overflows 768 GB**, so do Step 1 with **`--quantization inc`** (fp8,
  see Step 2) — i.e. the very first servable M3 is *already* the fp8 path. (If a strictly-bf16
  smoke is wanted, it must be a layer-subset / CPU parity, not a full TP8 serve.)

- **Step 2 — the SAME serve, made fast = FP8(INC), native fp8 MME, no dequant.** This is the
  production path and it removes the §11a dequant tax **by construction** (`fp8-moe-gate.md`:
  `VllmMixtureOfExpertsOpFP8PerChannel` → `mixture_of_experts.fp8`, fp8 weights+act+scales, no
  bf16 round-trip; `VLLM_HPU_FORCE_CHANNEL_FP8=true` default). swigluoai is STILL applied outside
  the GEMM, so fp8 is free of the silu hard-code too. INC calibrate (MEASURE→QUANTIZE on the G2
  cards) → `--quantization inc --kv-cache-dtype fp8_inc` (lazy + EP + `PT_HPU_WEIGHT_SHARING=0`).
  **Keeps all 7 MTP modules bf16-alive** (no int4-dead-head graft, contrast GLM §8).
  → A WORKING + BENCHMARKABLE M3 tonight, on the production-shaped config.

- **In parallel (no card contention, card-free):** run the INC calibration-set build (code +
  agentic/tool-call + long-doc, seqlen≥8192, nsamples≥256 so all 128 experts are hit —
  `quant-quality-eval.md §4`) and the swigluoai router-agreement / coherence harness so the
  MEASURE pass has data the moment the card frees.

### BY MORNING — optimize, do NOT rebuild

- **Warm + tune the fp8 serve:** full warmup, `--max-num-seqs 64→128`, MTP on (depth-1, the proven
  1.70× decode multiplier, `§12`). These are config levers, no kernel.
- **MSA:** keep `MSA_READY=False` (dense GQA fallback) for the first coherent+fast number; the MSA
  sparse op is the *separate* long-context lever (`build-msa-kernel.md`) — do not couple it to the
  swigluoai/precision bring-up tonight.
- **Quality gate before "done":** task-level eval Δ≤1.0pp vs bf16 control (`quant-quality-eval.md`),
  router top-1 ≥0.98 / top-4 Jaccard ≥0.95 — the same discipline that gated DSA. fp8 single-quant
  bar is <1% (NOT the 6.3% int4→fp8 double-quant floor, which is a different path M3 never touches).

### Option (A) [skip-activation + external swigluoai] — keep as the int4 FALLBACK only
If a future model *is* int4-mandatory (1T tier) and you want the int4 fused kernel's matmul fusion
*with* swigluoai, the cheap middle path is: emit the fused int4 graph **through the gate/up GEMM
only (skip its silu)**, return gate‖up, apply `SwigluOAIAndMul` in torch, then the down GEMM. That
is a smaller kernel delta than (B)'s in-graph swigluoai. **But it still inherits Finding 2 (loses to
stock) + the dequant tax**, so it is NOT tonight's path for M3 — it's a documented option for the
int4 tier, gated behind an on-card A/B vs stock. **For M3, fp8 dominates it on every axis.**

---

## #1 IMMEDIATE ACTION (now)

**Stand up the FP8(INC) M3 serve with swigluoai applied outside the GEMM — start the INC MEASURE
pass.** Concretely, the first command on the box is the INC calibrate→serve of M3 bf16 with
`--quantization inc --kv-cache-dtype fp8_inc` (config in `fp8-moe-gate.md` Path A), after the
30-min CPU swigluoai-parity check (Step 0). This single action delivers the WORKING, coherent,
benchmarkable M3 **and** is the production fast-path — no kernel build is on the critical path.

---

## "WORKING BY MORNING" — honest likelihood

**HIGH (~85%) for a WORKING + benchmarkable M3 by morning; MODERATE-HIGH (~70%) for
WORKING + a clean *fast* number.**

- *Working* is high because the path is **correct-by-construction stock code** (native fp8 MoE is
  wired and proven; swigluoai is an existing vllm CustomOp applied outside the GEMM; no on-card
  kernel bring-up — the thing that historically ate multiple overnight iterations is **removed from
  the critical path**). Residual risk is integration/serve plumbing (the M3 model file is new:
  weight-loader interleave for gate_up, INC calibrate run, MSA dense-fallback wiring) and a
  first-request eager-compile timeout — all known, all config/code, none novel-kernel.
- *Fast number* is gated by two things outside the software: **(a) the box's intermittent
  scale-up fabric storm** has repeatedly masked real tok/s (`§9/§12` — 2–5 tok/s under storm vs
  14 quiet); a clean window is needed for a trustworthy figure. **(b) INC MEASURE→QUANTIZE + full
  warmup on a 428B model costs real wall-clock** (GLM's full warmup was ~8 min and consumed a
  clean window). The *mechanism* of fast (native fp8 MME, no dequant, MTP 1.70×) is proven now;
  the *number* is fabric-gated, not software-gated.
- The thing that would BLOW the deadline is taking the prompt's premise literally and doing an
  overnight on-card fused-int4-swigluoai kernel bring-up (option B). **Avoiding that is the single
  biggest schedule decision.** Coherence-first on the fp8 path de-risks the night and the
  benchmark falls out of the same serve.
