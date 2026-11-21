# The INT4 Single-Stream Wall — Honest Strategy + Ceiling (the synthesizer, 4/4)

**Question:** M3 int4 sits at ~6–8 tok/s single-stream, 14–31% util, host-sync-bound, MoE captured,
the three obvious MoE-compute levers (native swiglu op, batched bmm, ngram) all dead on `:next`. Is
int4 single-stream MAXED on this box? Is fp8 the real lever for M3 (and does faster compute even help
a host-bound wall)? Is there ANY int4 path past the wall on ONE node for the int4-mandatory giants
(Kimi K2.7 1T)? **CARD-FREE strategic synthesis of the four deliberation angles.** Date 2026-06-15.

Sources: `gaudi-perf-targets` + `gaudi-single-stream-speed` memories; PROJECT_FINDINGS §3/§7/§8/§11/§12/§13;
Opt-Guide Part IV (§IV.2/3/8); siblings `int4-speed-ceiling.md`, `fp8-moe-gate.md`, `w4a8-fused-kernel.md`,
`giant-int4-strategy.md`, `giant-model-sizing.md`, `M3_MATRIX_REPORT.md`.

---

## VERDICT (one line)

**int4 single-stream IS maxed on this box (~6–8 tok/s for M3's all-512-expert path; ~29–34 the
absolute MoE-agnostic sync cap). The wall is host/collective-sync, NOT MoE compute — so faster
compute (fp8, fused kernels) cannot move single-stream; it pays at throughput/concurrency only. For
M3: PROCEED to fp8 experts, but for the right reason (concurrency/long-ctx + keeps MTP alive), not a
single-stream miracle. The ONLY proven single-stream multiplier on this box is MTP (1.70×), and int4
KILLS it — which is itself the decisive argument for fp8 on M3. For Kimi (int4-mandatory, can't fit
fp8): there is NO int4 path past the host-sync wall on one node — honest ceiling ~8–17 tok/s single +
throughput@concurrency; "snappy + fast" needs a 2nd node (for fp8 fit) or accepts int4-is-fit-not-speed.**

Confidence: **high** on the wall mechanism + that the 3 sibling levers are dead (all source/measured);
**high** that fp8 removes the dequant mechanism (code-proven, fp8-moe-gate); **medium** on exact fp8
concurrency magnitude (config/fabric-bound — must measure stock-fp8 vs int4 at mns 64/128, quiet fabric).

---

## The load-bearing fact: WHY faster compute can't help single-stream

PROJECT_FINDINGS Finding 3 is the hinge: **at batch=1 the MoE compute is ~0.16 ms of a ~127 ms decode
step (~90% host/collective overhead).** The proven sync-bound ceiling is **~29–34 tok/s on ANY MoE op**
— the stock intrinsic, the validated fused int4 kernel, and a hypothetical fp8 op all land there because
the step is dominated by per-token collective barriers + host input-prep + per-layer mark_step, not the
GEMM. M3 measures *worse* (~6–8) than GLM/Qwen (~29–34) for two structural reasons, both confirmed in
`M3_MATRIX_REPORT`: (1) the M3 int4 path dequants **ALL 512 experts/layer/token** (the routed-K gate is
not engaged the same way), and (2) a serialized 2K prefill. Util 14–19% + 90–116 W of 600 W is the
**signature of a host-starved device**, not a compute-bound one.

**Therefore: any lever that only makes the 0.16 ms expert compute cheaper or more native — fp8 MME (2×),
W4A8-fused, a better loop — buys ≈0 single-stream.** This is the single most important synthesis result
and it dissolves the question's tension: the wall is real, structural, and orthogonal to MoE precision.

---

## The four angles, and their likely outcomes

| Angle | Lever | Single-stream | Throughput@conc | Verdict |
|---|---|---|---|---|
| **1. fp8 experts (M3)** | native E4M3 MME, no dequant | **≈0 gain** (host-bound) | **~1.5–2× int4** | **DO IT — for concurrency + MTP, not single-stream** |
| **2. int4-mandatory giant (Kimi)** | stay int4 (only fit) | ~8–17 (host-bound floor) | tens-of-tok/s, dequant-dominated | **No int4 path past the wall; fit-not-speed** |
| **3a. captured-dispatch / fused** | W4A8-fused, fused int4 op | ≈0 (and loses to stock) | inherits fused-op debt | **REJECT for speed (Finding 2/5)** |
| **3b. speculation (MTP/ngram)** | 1.70× decode multiplier | **the ONLY real lever** | orthogonal | **DEAD under int4; ALIVE under fp8** |
| **3c. host-cost cut** | governor=perf, warm cache, NUMA | a few % each | a few % | **already applied; not a wall-mover** |

**(1) fp8 experts — the honest win for M3, but understood correctly.** fp8-moe-gate is a code-proven
**GO**: stock vllm-gaudi's `VllmMixtureOfExpertsOpFP8PerChannel` feeds fp8 weights+act+scales straight
into the native `mixture_of_experts.fp8` MME — **no bf16 round-trip, no kernel to build**, quality <1%
(INC single-quant; the 6.3% figure is the unrelated int4→fp8 double-quant). It removes the §11a dequant
**mechanism**. But on a host-sync-bound *single-stream* step, removing the 0.16 ms expert cost is
invisible. **fp8's real payoff is (a) throughput@concurrency ~1.5–2× int4 at c=30, the regime M3 actually
targets, and (b) it keeps the 7 MTP modules bf16-alive — int4 kills them.** That second point is the
decisive one: MTP is the *only* proven single-stream multiplier on this box (1.70×, §12), so fp8 is the
only path that even *reopens* a single-stream lever. M3 at 428B/768 GB fits fp8 (~471 GiB) with c=30 KV
headroom to ~66K — it is NOT memory-forced into int4, so taking int4 eats the dequant tax for no benefit.

**(2) Kimi K2.7 (1T, int4-MANDATORY) — the honest giant ceiling.** Kimi fp8 ≈1000 GB does NOT fit 768 GB;
int4 ~610 GB is the *only* fit (`giant-model-sizing`). So fp8 — the only *fast* path — is off the table,
and with it MTP (int4 export drops MTP tensors, as M3 confirmed empirically). Single-stream is still
host-sync-bound → **honest ceiling ~8–17 tok/s single-stream, with NO int4 software lever to beat it on
one node** (routed-K rescues batch=1 but evaporates at c≥30; fused/W4A8 lose to stock; ngram breaks the
int4 MoE plumbing). The real Kimi answer is **throughput@concurrency + warm cache + mns 64–128 + sparse
attention/MLA for KV** — int4 is the *enabler* (mandatory fit, and the only envelope for 1M KV-offload),
**not the speed path**. "Snappy AND fast" for a 1T model needs a **2nd node** (→1536 GB lets it run in
fp8, escaping the dequant tax), per `giant-int4-strategy`: on one node, for 1T+, fast-and-lean are
mutually exclusive by the FLOAT-only-MME fact. (Caveat: scale-out is RoCE, not NVSwitch — a 2nd node buys
fp8-fit + capacity, NOT single-stream, and PP is unsupported in vllm-gaudi today; measure before buying.)

**(3a) captured-dispatch / fused (W4A8, fused int4 op) — REJECT for speed.** Finding 2 is the hard-won
result: the fused-expert OpBackend op is *slower than the stock intrinsic in both regimes* (single-stream
~23 vs ~30; throughput ~68 vs ~126). W4A8-fused adds 8 per-forward cast/amax nodes on top → inherits the
debt AND a ~6.4e-2 double-quant floor. It is a **1T-class memory-fit enabler only** (lets an int4-only
giant touch the fp8 MME at all), never a single-stream win. Don't build it for M3 (fp8 fits → load-time
transform + stock native op is strictly better).

**(3b) speculation — DEAD under int4, the whole reason to move to fp8.** M3 int4 empirically: `deepseek_mtp`
→ vLLM 0.22 raises `Unsupported speculative method: 'mtp'` AND the int4 export has **zero MTP tensors**;
`ngram` wires in but **crashes decode warmup** (`index 16 out of bounds`, 16 = local experts — the
multi-token draft shape breaks int4 MoE routing). So on int4 there is no spec-decode. Under fp8/bf16 the
7 MTP modules stay alive (1.70× proven on GLM). **This is the single-stream lever the wall otherwise
forbids, and only fp8 unlocks it.**

**(3c) host-cost cut — already harvested, not a wall-mover.** governor=performance (cpu-performance.service),
warm recipe cache, lazy+graphs, NUMA pin — each is a few % on a host-bound step and most are already in
the current ~30 (Qwen) / ~6–8 (M3) numbers. Worth keeping; **not worth one on-card attempt as a wall-breaker.**

---

## Is one sibling angle worth one on-card attempt BEFORE pivoting to fp8? NO.

Every single-stream lever for **int4** is exhausted or dead: MoE-compute levers don't touch a host-bound
step (Finding 3, re-confirmed by the 3 M3 rejections); spec-decode is checkpoint-dead under int4; host-cost
is harvested; fused/W4A8 lose to stock. **The decisive next move is not another int4 single-stream probe —
it is the fp8 pivot**, whose value is concurrency + reviving MTP, not single-stream. (If you want one
*measurement*, it is **stock-fp8-MoE vs int4 at mns 64/128 on a quiet fabric** — to bank the ~1.5–2×
throughput number — NOT another single-stream attempt.)

---

## BOTTOM LINE (decisive)

- **#1 action NOW:** stop spending single-stream effort on int4 — **pivot M3 to fp8 experts via INC**
  (`--quantization inc` + native channel-fp8 op, no kernel build, <1% quality, MTP stays bf16-alive).
  Measure it for **throughput@concurrency (mns 64/128) and MTP-on single-stream**, not bare batch=1.
- **Realistic int4 single-stream ceiling on this box:** **~6–8 tok/s for M3's all-512-expert path;
  ~29–34 the absolute MoE-agnostic sync cap.** It is **MAXED** — the wall is host/collective-sync, not
  MoE compute, so no int4 software lever moves it (the 3 sibling angles confirm this from three directions).
- **Proceed to int4→fp8 for M3? YES — decisively**, but for concurrency + reviving MTP (the only proven
  ~1.70× single-stream multiplier, which int4 kills), NOT for a bare single-stream gain. fp8 fits M3 with
  KV headroom, so int4 is leaving capability + spec-decode on the table.
- **Honest giant (Kimi K2.7 1T) ceiling:** **~8–17 tok/s single-stream + throughput@concurrency on ONE
  node — and that is the floor int4 cannot beat there** (fp8 doesn't fit → no native MME, no MTP). int4 is
  **fit, not speed**. A 2nd node (→fp8 fit) is the only lever for "fast" 1T; on one node, fast-and-lean are
  mutually exclusive by the FLOAT-only-MME fact. Measure scale-out cost first (RoCE, single-stream-neutral).
