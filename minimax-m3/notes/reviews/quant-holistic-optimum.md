# MiniMax-M3 on 8×Gaudi 2 — Holistic Quant/Speed Operating-Point Review

**Lens:** adversarial / devil's-advocate, whole-route challenge. **Date:** 2026-06-14.
**Question:** is `int4-experts + bf16-rest + fp8_inc-KV + MSA-sparse + TP8` the global optimum for
FASTEST + minimal-quality-loss + LONG-CONTEXT + HIGH-CONCURRENCY (30+), or is there a fundamentally
better operating point?

---

## VERDICT

**The route is NOT optimal. The single biggest strategic correction:**

> **Make M3 an FP8-everything model (INC E4M3 weights + fp8_inc KV + fp8 activations/compute,
> single-quant, no int4), not an int4-experts model. Then stack MTP (depth-1) + MSA. Drop int4 to a
> memory-fit fallback only.**

The team is **reflexively porting the GLM/Qwen int4 playbook to M3.** That playbook was never a
speed choice — it was a *memory-fit fallback forced by pre-quantized int4 checkpoints that already
existed on disk* (GLM-AWQ, Qwen-GPTQ). **M3 is different: the checkpoint is bf16 (no int4 export
exists), so precision is a FREE CHOICE** — and for the stated goals (high concurrency + long context
+ speed) that choice should be FP8, not int4.

---

## END-TO-END EVIDENCE

**1. M3 is 97% MoE experts — the whole question is "how do you run the experts."**
Analytical count from `config.json` (`text_config`): text tower = **426 B params, 413 B (97%) in the
57 MoE layers' 128 experts.** Attention+index = 6.6 B, embeddings = 2.5 B. Everything else is noise.

**2. HBM fit decides the menu (text tower only, /8 cards, 96 GB/card, ~30 GB/card reserved KV+graph):**
| Precision | GB total | GB/card | Fit? |
|---|---|---|---|
| bf16-everything | 852 | 107 | ❌ does not fit |
| **fp8-everything** | **426** | **53** | ✅ ~43 GB/card free for KV+graph |
| int4-experts+bf16-rest | 233 | 29 | ✅ most headroom (but pays the tax below) |

bf16 is off the table. The real contest is **fp8-everything vs int4-experts**, and fp8 fits with ample
KV room at 32K–256K context.

**3. The int4 dequant tax is real, measured, and INVERTS at concurrency — the killer point.**
Gaudi's only MoE intrinsic `torch.ops.hpu.mixture_of_experts` is **bf16-only**
(`KERNEL_ARCHITECTURE.md:29`). Every int4 MoE path is therefore forced to **dequant every expert →
bf16 → HBM → separate GEMM**. The on-disk Kineto trace verdict (`PROJECT_FINDINGS.md §11a`) on the
sibling GLM int4 path is unambiguous: decode step ≈193 ms (~5.2 tok/s), **~98% of device time in
~1M tiny (~13 µs) int4→bf16 dequant kernels computing ALL 256 experts/layer**, only ~5% collective.
The team's own conclusion, verbatim: *"a dense 120B fp8 would be multiples faster — it avoids all-256
blowup, ~1M tiny int4-dequant kernels, and EP all-to-alls."*
The int4 mitigation (`HPU_MOE_ROUTED_K`, dequant only the ≤K routed experts) **only fires at
single-stream**: the gate is `tokens*top_k ≤ K` (`SPEED_AND_ITERATION.md:98`). At the target
concurrency of 30+, `tokens*top_k ≫ K` → gate OFF → you eat the FULL all-expert dequant tax on every
step. **int4's one advantage is structurally unavailable in the regime we are optimizing for.**

**4. FP8 is native silicon, not a dequant emulation.** Gaudi 2 does **fp8 E4M3 GEMM at ~2× the bf16
MME rate** (`KERNEL_ARCHITECTURE.md §6`; arXiv 2503.09975: ~865 TFLOPS, ~98% MFU). FP8 weights are
read at half the bf16 bytes and fed straight to the MME — **no dequant round-trip, no all-256 tiny-
kernel blowup.** This is exactly the bandwidth+kernel-count pathology that makes int4 slow, removed by
construction. FP8(INC) is also the guide's **#1 ranked throughput lever** (`Guide §1.1`: ~2× tput,
~½ memory).

**5. Quality: FP8 is the BETTER quality point, not a worse one.** Decisive distinction the int4 route
obscures: INC FP8-from-bf16 is **single quantization** → **<1% loss on 70B-class** (`Guide §1.1/§2`,
native E4M3, calibratable on-card via `calibrate_model.sh -o g2`). int4 is *also* lossy, and any int4
attempt to recover MME throughput (the kernel's `fp8-compute` toggle) becomes **int4→fp8 DOUBLE
quantization with a hard measured 6.4e-2 relNorm floor** (`PROJECT_FINDINGS Finding 6`). So FP8
single-quant is simultaneously **faster AND higher-quality** than int4-with-fp8-compute. The only
caveat is MoE calibration discipline (never `unit_scale`: Mixtral PPL 11.34→96.65) — a solved,
documented procedure, and M3 being bf16 means INC MEASURE→QUANTIZE runs directly (the exact path GLM
*lacked*, which is the ONLY reason "stock FP8 MoE" was rejected for GLM — "no fp8 ckpt on disk", not
tested-and-failed).

**6. The real high-concurrency win is fp8-compute + MTP + KV-min — NOT shaving weight bits.**
Confirming the brief's hypothesis: the team is optimizing *weights-bits* (a memory-fit axis) when the
30+-concurrency throughput levers are (a) **fp8 native compute** (avoids the dequant tax, 2× MME),
(b) **MTP** — and here M3 has a structural gift GLM did not: **M3 ships SEVEN bf16 MTP modules**
(`num_mtp_modules=7`, checkpoint is bf16). GLM's MTP head was DEAD because its int4 export quantized
layer-78 to int4-everywhere with no calibration data — requiring a painful bf16 graft
(`PROJECT_FINDINGS §8`). **Choosing int4 for M3 risks killing MTP the same way; keeping M3 in
FP8/bf16 keeps all 7 MTP modules alive natively.** MTP delivered a proven **1.70× decode multiplier**
on-device for GLM (`§12`). (c) **fp8_inc KV** minimizes KV per seq → bigger batches/longer ctx; M3's
MSA already caps attention at ≤16 blocks×128 + 1 local, so KV pressure is low and fp8 KV stacks
cleanly.

**7. MSA sparse-attention is precision-orthogonal — it stays either way.** The MSA OpBackend kernel
work (`KERNEL_MAP.md`, `M2_BASE_MAP.md`) is correct and necessary regardless of weight precision; it
addresses the *attention/long-context* axis, not the *expert/throughput* axis. FP8 changes nothing
about it. This review does not touch MSA.

---

## RECOMMENDED GLOBAL OPERATING POINT

```
Weights:     FP8 (INC E4M3), single-quant from the bf16 checkpoint
             - MoE experts: INC fp8 + calibrated maxabs (calibrate_model.sh -o g2; NEVER unit_scale)
             - skip lm_head + embeddings from quant (INC default)
Activations: fp8 compute where the path supports it (native ~2× MME)
KV cache:    fp8_inc  (the only HPU-accepted KV dtype; ~2× KV headroom for concurrency/long-ctx)
Attention:   MSA block-sparse (top-16 blocks) via the OpBackend kernel — unchanged
Decode:      MTP depth-1 (num_speculative_tokens=1) — proven 1.70× multiplier, M3's 7 modules are
             bf16-live (no int4-dead-head; no graft needed)
Placement:   TP=8 + --enable-expert-parallel (426 GB fp8 / 53 GB-card REQUIRES TP8; cannot replicate)
Concurrency: --max-num-seqs 64→128, --max-num-batched-tokens 16384, --block-size 128
INT4:        FALLBACK ONLY — keep hpu-moe-ext autodetect wired (free), use it solely if an fp8
             export ever fails to fit or calibrate; it is NOT the speed/concurrency path.
```

**Why this beats the current route end-to-end:** at 30+ concurrency the current route runs the full
all-256-expert int4→bf16 dequant every step (routed-K gate is off) — the measured ~98%-of-device-time
pathology — while FP8 runs native fp8 GEMM at 2× MME with no dequant, fits in HBM with KV room to
spare, hits a *better* (<1%) quality point, and keeps MTP's 1.70× alive. It is faster, higher quality,
and more concurrent simultaneously.

---

## CONFIDENCE

**High (≈85%)** that FP8-everything is the correct strategic operating point for M3's goals, and that
defaulting to int4 is the wrong inherited reflex. The reasoning chain (97%-experts → HBM fit → bf16-
only intrinsic → measured int4 dequant tax → routed-K dies at concurrency → native fp8 2× MME → fp8
single-quant <1% → MTP alive) is each link sourced.

**The ONE gate this hangs on (must validate, do not assume):** whether vllm-gaudi runs **INC fp8 MoE
experts natively on the MME** (fp8 weights → fp8 GEMM, no re-dequant-to-bf16), or whether its only
*wired* MoE op is still the bf16-only `mixture_of_experts` — in which case even fp8 experts would
round-trip through bf16 and the advantage shrinks toward "fp8 KV + fp8 dense layers + MTP" only.
GLM's "stock FP8 MoE" was rejected merely for lack of an fp8 checkpoint (`SPEED_MASTER_PLAN.md:45`),
**never benchmarked**, so this is genuinely open. **First action: produce a small INC-fp8 M3 slice,
calibrate, and confirm fp8-native MoE execution + the <1% quality bar on-card BEFORE committing the
full route** — exactly the measure-first discipline the box already uses. If native fp8 MoE is NOT
wired, that is itself the highest-value engineering target (an fp8 analog of the existing OpBackend
MoE infra — the kernel scaffold `moe_fused_expert_fp8_backend.cpp` already exists), not another int4
optimization.

**Secondary risk (low):** TP8-only means no replica-throughput trick; mitigated because the model
cannot be replicated at any precision (53 GB fp8 / 29 GB int4 both leave a 426/233 GB model needing
all 8 cards) — so TP8 is forced regardless and is not a differentiator between routes.
