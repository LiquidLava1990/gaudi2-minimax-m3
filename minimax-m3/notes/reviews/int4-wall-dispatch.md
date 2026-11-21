# int4-wall-dispatch — a FASTER captured expert dispatch that avoids the broken gpt_swiglu op

Angle: can we get swigluoai through the FAST captured `mixture_of_experts` (plain silu/matmul)
op instead of the per-expert Python loop? Card-free static read of the runnable code.
Reads: `m3_swigluoai_patch.py`, `vllm_gaudi/extension/ops.py`
(`VllmMixtureOfExpertsOpWNA16` :1456 + the working silu path :1492 + the bias/swiglu schema
:681), `hpu_moe_ext/moe_wna16_hpu.py` (`_routed_moe` :837, `apply_monolithic` :1134), and the
sibling swigluoai-*.md.

---

## TL;DR — the loop already IS the captured fast dispatch; option (a) is blocked by the op API

The premise "the plain matmul/silu MoE op is captured + fast, so fuse the GEMMs there and do
swigluoai outside" **cannot be built on the stock op**, and the per-expert loop is **not** the
thing costing tok/s. Concretely, evaluating the three options:

### (a) Run the fast op to emit pre-activation gate&up, apply swigluoai outside — IMPOSSIBLE on the stock op
`torch.ops.hpu.mixture_of_experts` has **no pre-activation output mode**. It fuses
`gate_up GEMM → activation → down GEMM` entirely inside the intrinsic and returns ONLY the
final `[T,H]` (confirmed: ops.py :1492-1500 returns the single fused tensor;
`swigluoai-op-capability.md` §2 "the op always fuses act+down internally … no return
pre-activation gate&up output mode"; `activationModeMap={gelu,relu,silu}` — no identity/none).
You cannot ask it for the gate/up matmuls alone. The only way to "fuse the two GEMMs in the
fast captured op" is to let it ALSO run an activation — but the only activations it exposes are
gelu/relu/silu (wrong math) or the bias/swiglu overload `bias_fused_weights(alpha,limit)` —
which is **the broken op** (returns ≈0 / hangs the card on :next). So (a) reduces to either
the wrong activation or the broken one. **Dead end on :next.** (It would work on a lazy,
non-eager-blocklisted build via `bias_fused_weights` — that is the real unlock, but it is a
different image, not a dispatch trick.)

### (b) Is the loop iterating all 128 experts or only the 4 active? — ALREADY ACTIVE-ONLY at decode
This is the one thing worth confirming, and the answer is good: at single-token decode the M3
patch takes `_routed_swigluoai`, gated by `eligible = (… and (x2.shape[0]*K) <= K)` i.e.
**T==1** (m3_swigluoai_patch.py :434-435). That path is a byte-for-byte copy of the proven
stock `_routed_moe` (moe_wna16_hpu.py :837): it builds `present`→`topk(present,K)`→`active`
and `index_select(0, active)` so it dequants + GEMMs **only the K=top_k routed local experts**
(:228-241, :289-293), NOT all 128/local-16. So decode already does the minimal **4 matmul-pairs**
— it is the fast active-only dispatch the question asks for. The `_allexpert_swigluoai` 128/E
loop is ONLY hit on **prefill (T>1)** and as the AWQ fallback (:467-477), which is correct: the
stock op would FATAL there on `activation='swigluoai'` (mixture_of_experts_gen.cpp:799). No
optimization available in (b): decode is already active-only; prefill-all-expert is mandated by
the broken-op constraint and is not the single-stream bottleneck.

### (c) Minimal custom captured TPC swigluoai on the working op's output — REDUNDANT / wrong layer
The swigluoai tail is already pure captured elementwise (`swigluoai_tail`: clamp/sigmoid/mul/add,
:151-162) — it runs on TPC inside the same lazy graph as the matmuls (the benchmark confirms
**zero CPU fallback, no recompile after warmup**, M3_MATRIX_REPORT). There is nothing to "move
to a custom TPC op": a bespoke fused-swigluoai TPC kernel is the `swigluoai-fused-kernel.md`
path, explicitly NO-GO (≈2-4 on-card days, and the fused OpBackend op is *slower than stock*,
int4-speed-ceiling.md:115). The activation is ~7 elementwise nodes on `[T,I]` — at T=1 it is a
rounding error next to the two `[1,H]·[H,2I]` GEMMs.

---

## WHY there is no faster captured dispatch to be had here

The decode step is **host/sync-bound** (util 14-31%, ~90-116 W of 600 W), and at T=1 the routed
loop is exactly **4 already-graph-captured matmul-pairs + a tiny elementwise tail** — the SAME
4 expert GEMMs the fused intrinsic would issue, just unrolled. Collapsing them into one
`mixture_of_experts` call cannot remove host/sync latency (the all-reduce + sampler + attention
dominate the step), and the one form that *was* measured — batched `bmm` — REGRESSED
(5.0 vs 8.3 single-stream; TTFT 128 s) because stack/expand/reduce add host ops and the
all-expert prefill bmm explodes the recipe. The matmuls are already on the MME; the activation
is already on the TPC; both are already captured. **The loop is the captured fast dispatch.**

## The ACTUAL faster-dispatch levers (none is a dispatch trick on the silu op)

1. **fp8 experts** (`M3_EXPERT_RUNTIME=fp8`, `fp8-moe-gate.md` / `swigluoai-strategy.md`): native
   E4M3 MME, NO int4→bf16 dequant tax (decode is ~98% dequant, int4-speed-ceiling §11a), swigluoai
   still applied outside the GEMM so the silu hard-code never bites. M3 is 428B → fp8 ≈428 GB fits
   768 GB. **This is the real single-stream/throughput lever** — a precision change, not a dispatch
   change.
2. **lazy (non-eager) image + `bias_fused_weights(alpha=1.702, limit=7.0)`**: on a build where the
   device gpt_swiglu CGUID is NOT `frontend_blocklist:[eager]`, this is the genuine one-op fused
   swigluoai dispatch (option (a) done natively) — GLM-class. Requires the lazy container, not :next.

## Expected gain / feasibility / effort / confidence

- **From any captured-dispatch rework of the silu op: ~0 tok/s.** (a) impossible on :next,
  (b) already optimal at decode, (c) redundant. Effort to "improve the dispatch": none worth
  spending. **KEEP the per-expert loop.** Confidence **HIGH** (read verbatim from the op schema +
  the shipped routed path; matches the three measured rejections).
- **The gain lives one layer down:** fp8 experts (expected the GLM-class single-stream lift; the
  dequant tax is the dominant decode cost, removed by construction). Effort: INC calibrate + serve
  config, **no kernel** (separate effort, already scoped). Confidence **MEDIUM-HIGH** that fp8 — not
  any swigluoai dispatch change — is the only lever that moves the int4 single-stream wall.
