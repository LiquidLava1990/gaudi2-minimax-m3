# INT4 MoE Speed Ceiling on Gaudi 2 — "can int4 be made fast enough for giants?"

**Question:** For a giant int4 MoE (~1T total / ~32B active per token) at HIGH CONCURRENCY,
how fast can int4 decode get, what binds it, and how close can an int4-fused-fp8 (W4A8) path
get to native fp8? CARD-FREE analysis (no cards / no vllm-active touched).

**One-line verdict:** int4-at-concurrency is **compute-bound on the int4→bf16 dequant + the
bf16 MME (432 TF)**, NOT bandwidth/HBM, NOT host-sync. The realistic concurrency ceiling for a
hardened/fused int4 path is roughly **half of native fp8's** (fp8 ≈ 2× from the native E4M3 MME
+ no dequant). int4's mandatory-fit only wins for giants that **cannot** fit in fp8 (or need the
KV/offload envelope); when fp8 fits, fp8 is the clear win. The int4-fused-fp8 (W4A8) middle path
is a **trap**: double-quant (~6.4e-2 relNorm floor) AND it keeps the dequant in the chain.

---

## The load-bearing hardware fact (everything follows from this)

**Gaudi 2's MME (matrix engine) is FLOAT-ONLY: bf16 + fp8(E4M3/E5M2), NO integer matmul.**
Integers live on the TPC (elementwise), never on the GEMM engine. So **int4/6/8 ALL dequant to
bf16 (or fp8) before the GEMM** — this is not a software choice, it is the silicon. The whole int4
"penalty" is this mandatory dequant step plus the fact that, post-dequant, int4 runs the **bf16**
MME (432 TF ceiling) instead of the **native fp8** MME (865 TF). (Opt-Guide §IV.2; PROJECT_FINDINGS §13c.)

Per-element economics at the GEMM:
- **int4:** read 0.5 B/elem from HBM → TPC dequant-to-bf16 (the tax) → bf16 MME @ **432 TF**.
- **fp8:** read 1 B/elem → **native E4M3 MME @ 865 TF (~2×)**, **no dequant**.

So int4 *halves weight HBM traffic* but *pays a dequant tax every step* and *runs the slower
matmul*. int4 = a **memory play**; fp8 = **native speed**.

---

## (1) Realistic decode tok/s — single-stream vs aggregate@concurrency

**Single-stream (batch=1) is the WRONG regime to judge int4 by, and it does not matter for giants.**
At batch=1 the MoE compute is a tiny slice of the step; decode is host/collective/fabric-bound. On
this box the proven batch=1 ceiling is **~29–34 tok/s** regardless of the MoE kernel (the fused int4
op and the stock op both land there; the kernel is never the bottleneck at low batch). This is the
sync-bound floor measured at only ~8% of the HBM-BW floor — a structural host wall, not a quant
property. (PROJECT_FINDINGS §3 Finding 3; gaudi-perf-targets memory.)

**Aggregate@concurrency is the regime that matters, and here int4 and fp8 DIVERGE sharply.** Two
measured anchors on the box (Qwen-397B-int4, the giant-MoE proxy):

- **Stock bf16-path throughput SCALES with batch** (engine generation-throughput, warm):
  `max-num-seqs 16 → ~126 tok/s`, `64 → ~486`, `128 → ~793` — **still scaling at 128**, ceiling not
  yet hit. So the box is NOT throughput-starved; absolute aggregate is healthy and config-bound.
  (PROJECT_FINDINGS §3 Finding 4.)
- **BUT that is the stock dequant-all path, which at concurrency is MoE-COMPUTE-bound, not collective-
  bound.** The on-disk Kineto decode trace (the giant-int4 monolithic path) attributes the step
  directly: ProfilerStep ≈ **193 ms**, device **84% busy**, `hcclAllReduce` only **~9.8 ms (~5%)**,
  and **~98% of device time = the two int4→bf16 dequant MoE kernels** — ~345k tiny (~13 µs) launches
  per step computing **ALL 256 experts/layer**. (PROJECT_FINDINGS §11 / §11a.)

**The honest int4 concurrency number is a band, not a point** (fabric storms have repeatedly masked
clean readings on this box), but the SHAPE is unambiguous and measured: at high concurrency a giant
int4 MoE trends into the **tens-of-tok/s-aggregate, dequant-dominated** regime, whereas the same box
on a native-fp8 path is **~1.5–2× faster** (compute-dominated work halved + native MME doubled). The
ceiling is set by the int4→bf16 dequant + bf16 MME, and **fp8 removes both** by construction.

---

## (2) The BINDING bottleneck at concurrency

**INT4→BF16 DEQUANT COMPUTE + the bf16 MME — i.e. MME/TPC compute, NOT HBM bandwidth, NOT host sync.**

This is the single most important correction in the whole analysis, and it overturns the original
"~99 µs dequant-to-HBM bandwidth floor" premise that motivated the fused kernel:

- **It is NOT host-sync.** Host-sync (~30 tok/s) is the *batch=1* wall; it amortizes away under
  concurrency. The concurrency trace shows the device **84% busy** and all-reduce only **5%** — the
  device is doing work, not waiting on the host or the fabric.
- **It is NOT (purely) HBM bandwidth reading int4 weights.** The original premise — that int4's cost
  is the bf16-writeback HBM round-trip (~243 GB/rank/token, the "~99 µs floor") — **did not hold**.
  The fused SRAM-resident kernel *eliminated that round-trip by construction* (0 MB workspace, no bf16
  weight in DRAM) and **still lost to stock** in both regimes. The trace confirms the real cost is the
  **dequant kernels themselves** (compute: ~345k tiny TPC launches computing all 256 experts), and the
  device is compute-busy, not bandwidth-starved at the elementwise step. (PROJECT_FINDINGS §11 vs the
  fused-kernel verdict §3.) Reading the ~500 GB of int4 weights per token-batch IS large traffic, but
  at concurrency the *dequant + bf16-GEMM compute* is what saturates first on this silicon.
- **It is NOT the fp8 MME** — int4 never reaches the fp8 MME; it runs the *bf16* MME (432 TF, half of
  fp8's 865 TF). The fp8 MME being the *next* wall is exactly why fp8 is ~2× faster.

So the int4 binding bottleneck is **the mandatory dequant tax + the half-speed bf16 GEMM**, both
intrinsic to "no integer matmul on the MME." You cannot software your way past either with int4 weights.

---

## (3) The lever ranking to make int4 fast (and the honest ceiling of each)

Ranked by realized impact, with the hard-won verdicts attached:

1. **Routed-expert-only gather (dequant only the ~K routed experts, not all 256/layer).** This is the
   biggest *structural* int4 lever — it attacks the dequant term directly (the all-256 blowup is ~98%
   of the trace). At batch=1 it was the 5.6× arc (3.84→21.7 tok/s). **CRITICAL CAVEAT: the routed-K
   gate (`tokens*top_k ≤ K`) turns OFF under concurrency** — at c≥30 every expert is hit, so the
   mitigation *evaporates exactly in the regime that matters*. So routed-gather rescues single-stream
   but does NOT save int4 at high concurrency. (gaudi-single-stream-speed; fp8-moe-gate Q4/§Speed.)

2. **Native fp8 experts (the actual answer — abandon int4 for the compute).** Not an "int4 lever" but
   the real fix: vllm-gaudi already ships a **native fp8 MoE path** (`VllmMixtureOfExpertsOpFP8PerChannel`
   → `mixture_of_experts.fp8`, fp8 weights+act+scales straight into the MME, **no bf16 round-trip**,
   gated `VLLM_HPU_FORCE_CHANNEL_FP8`=True by default). This removes the dequant tax *and* doubles the
   MME. Estimated **~1.5–2× int4 at c=30**. No kernel to build. (fp8-moe-gate VERDICT; build-weights-precision.)

3. **Batch saturation + warm recipe cache + max-num-seqs 64–128.** The box scales 126→486→793 and is
   still climbing — much of the "int4 is slow" impression was config (max-num-seqs too low) + a
   validation-off-recipe-cache recompile artifact dragging benchmark *averages* below the warm *peak*.
   A production serve is warm-cache + pre-warm + high concurrency. Real, free, stackable — but it lifts
   *both* int4 and fp8; it does not close the int4↔fp8 gap. (PROJECT_FINDINGS §3 Finding 4.)

Levers that do NOT work (recorded so they aren't retried):
- **Fused SRAM-resident int4 kernel (DONE, validated, but NOT a speedup).** Correct, general (5/5
  families), stable, serves 397B end-to-end — a real engineering achievement — but **slower than the
  stock intrinsic in BOTH regimes** (single-stream ~23 vs ~30; throughput ~68 vs ~126 at the same
  config). The stock `mixture_of_experts` is a heavily-optimized batched/grouped intrinsic; the custom
  op's overhead (chunked sub-chains, OpBackend dispatch, larger graph) outweighs the SRAM-residency
  saving. **Do NOT enable HPU_MOE_FUSED_OP for speed.** Lesson: don't build a kernel to beat a
  well-optimized stock op — you inherit its optimization debt. (PROJECT_FINDINGS §3 Findings 2 & 5.)
- **Dequant-to-fp8 / W4A8 on int4 (overlap-or-not).** Double-quant (W4→bf16→fp8) with a hard
  **~6.4e-2 relNorm floor** AND it keeps the dequant in the chain → accuracy loss *and* no speed win.
  Worst-of-both for any model with a bf16 source. (PROJECT_FINDINGS §3 Finding 6; build-weights-precision.)
- **Overlap dequant↔MME.** The GC already bundles dequant into the consuming MME (SRAM-resident,
  proven) — there is no further overlap to win; the cost is the dequant *compute*, not a serialization
  gap. **MTP** is a decode multiplier (~1.70× proven on GLM) but it is orthogonal to int4-vs-fp8 and
  is checkpoint-fragile under int4-everywhere (int4 killed GLM's MTP head → bf16 graft; fp8 keeps MTP
  alive). (PROJECT_FINDINGS §8/§12; fp8-moe-gate Q3.)

---

## (4) How close can int4-fused-fp8 get to native-fp8 — and the giants verdict

**Two distinct paths must not be conflated:**

- **"int4-fused-fp8" (W4A8: int4 on disk → dequant → fp8 compute):** does NOT close on native fp8.
  It still pays the dequant tax (int4 is still in the chain) and it eats the **~6.4e-2 double-quant
  relNorm floor**. It buys a smaller disk artifact and nothing else. **Gap = large + a quality
  penalty; reject.** (fp8-moe-gate Q4; build-weights-precision "Rejected alternatives".)

- **Native fp8 (bf16→fp8 ONCE, via INC):** single-quant, **<1% accuracy** (the 6.3% figure is the
  *int4→fp8 double-quant*, a different path), and it hits the native E4M3 MME with no dequant. This is
  the real target and the real win — but it is reached by **abandoning int4 for the compute**, not by
  fusing int4 better.

**Can int4 be made fast enough for giants?** Honest answer:

- **If the giant FITS in fp8 → NO, don't make int4 fast; use fp8.** The gap is not closeable with int4
  weights (the bf16 MME is half-speed and the dequant tax is mandatory). int4's *only* edge is the
  bigger KV/concurrency-reach envelope (e.g. ~150K vs ~66K ctx at c≥30 for M3), which only pays *far*
  above the target concurrency. Taking int4 when fp8 fits eats the dequant tax every decode step for
  no benefit. **fp8 is the clear win.** (build-weights-precision crossover; Opt-Guide §IV.3.)

- **If the giant does NOT fit in fp8 (truly mandatory-fit) → int4 is the only option, and "fast enough"
  means accepting ~half native-fp8 throughput.** That is the real ceiling: int4-at-concurrency lands
  around **0.5–0.67× native-fp8** (fp8's 2× MME + no-dequant advantage), and the fused kernel does NOT
  recover it (it loses to stock). The honest framing for a ~1T/32B-active giant that won't fit fp8:
  serve it int4 (it's the only way it runs at all), drive concurrency hard with the stock op + warm
  cache + max-num-seqs 64–128, and accept that you are paying the dequant tax — int4 is the *enabler*
  (mandatory fit, and the only envelope that supports 1M KV-offload), not the *speed* path.

**Bottom line:** int4's value for giants is **fit, not speed.** When fp8 fits, fp8 wins decisively
(~1.5–2×). When it doesn't, int4 is the only runner and tops out at roughly half native-fp8 — a gap
that is *intrinsic to Gaudi 2 having no integer matmul*, and one that no fused/overlap/W4A8 software
lever has been able to close on this box.

---

## Evidence index

| Claim | Source |
|---|---|
| MME is float-only (bf16+fp8), no integer matmul; int4 all dequant to bf16 | Opt-Guide §IV.2; PROJECT_FINDINGS §13c |
| Per-elem: int4 0.5 B + dequant + bf16 MME (432 TF) vs fp8 1 B + native (865 TF, 2×) | Opt-Guide §IV.2 (lines 685-687); build-weights-precision |
| Kineto: 193 ms/step, 84% device busy, all-reduce 5%, 98% in dequant, ~345k tiny launches, ALL 256 experts | PROJECT_FINDINGS §11/§11a |
| Batch=1 ceiling ~29-34 (sync-bound, ~8% of BW floor); MoE op never the bottleneck at low batch | PROJECT_FINDINGS §3 Finding 3; gaudi-perf-targets |
| Throughput scales 126→486→793 (max-num-seqs 16/64/128), still climbing | PROJECT_FINDINGS §3 Finding 4 |
| Fused SRAM-resident int4 kernel correct+general+stable but SLOWER than stock both regimes (23 vs 30; 68 vs 126) | PROJECT_FINDINGS §3 Findings 2/5; fused-moe-kernel-validated |
| "~99 µs dequant-to-HBM floor" premise did NOT hold (fused removed round-trip, still lost) | PROJECT_FINDINGS §3 |
| Routed-K gate turns OFF at concurrency → int4 worse at c≥30 | fp8-moe-gate §Speed; PROJECT_FINDINGS §11a |
| Native fp8 MoE path exists in stock (no dequant); ~1.5-2× int4 at c=30 | fp8-moe-gate VERDICT; build-weights-precision |
| W4A8 / int4→fp8 = double-quant ~6.4e-2 floor, no speed win | PROJECT_FINDINGS §3 Finding 6; fp8-moe-gate Q4 |
| int4 KV-envelope edge (~150K vs ~66K @ c≥30) only pays far above target conc; int4 mandatory only for 1M-offload | Opt-Guide §IV.3/§IV.5; PROJECT_FINDINGS §13d |
