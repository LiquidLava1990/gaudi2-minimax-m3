# Maximize Kimi-K2.7 int4 + Session Execution-Order Synthesis (4/4, synthesizer)

**Question (user goal):** "maximize int4" for Kimi-K2.7-Code (1T int4, DeepSeek-V3/MLA/silu,
`./serve.sh kimi-k2.7`, ~74 GB/card). Kimi is **int4-ONLY** — fp8 (~1 TB) does NOT fit 768 GB,
so it CANNOT use the int4→fp8-at-load trick (`int4_to_fp8_load.py`) that gave M3 ~3.6–5×
concurrency (no HBM room to expand int4→fp8). **Date 2026-06-15. CARD-FREE.**

Sources read: sibling reviews `w4a8-fused-kernel.md`, `int4-speed-ceiling.md`,
`giant-moe-serving.md`, `giant-int4-strategy.md`, `int4-wall-strategy.md` (the M3 4/4 synth),
`int4-wall-speculation.md`, `build-msa-kernel.md`; `moe_fused_expert_fp8_backend.cpp` (the W4A8
scaffold); `kimi-k2.7.env`; verified on-disk: checkpoint 555 G / 64 shards present,
`/mnt/fastmodels/kimi-k2.7-code` mountpoint exists, `$HOME/AI/serve.sh` ready.

---

## VERDICT (one line)

**Do NOT build W4A8-fused for Kimi. It does not give Kimi an "fp8-compute concurrency win" — that
analogy to M3 is FALSE: M3's 3.6–5× came from the native stock fp8 MoE op consuming
load-time-converted *resident* fp8 (no per-forward dequant); W4A8-fused is a different, slower
animal — a custom fused op that (Finding 2) already loses to the stock intrinsic, then adds 8
per-forward cast/amax nodes AND a ~6.4e-2 double-quant floor. It is a memory-fit *curiosity* (lets
int4-only weights touch the fp8 MME), never a serving win on one node. The practical answer is
(2): serve Kimi int4-bf16 stock (the proven CT-WNA16 HPU path), measure it, accept it as
"fit-not-speed," and a 2nd node is the only real lever for fp8-class speed at 1T. FIRST measure the
int4 baseline before any kernel talk.**

Confidence: **high** on the W4A8≠M3-win mechanism (the M3 win is *stock-op-on-resident-fp8*, which
1T cannot host; code + 4 concordant reviews); **high** Kimi serves as-is (checkpoint + path
verified on-disk); **medium** on absolute Kimi tok/s (extrapolated from GLM, fabric-gated).

---

## (1) Is W4A8-fused worth building for Kimi? **NO.**

The user's hope: "int4 footprint + fp8 compute → M3-like concurrency win." Three strikes, each
independently fatal, all already established and re-confirmed here:

1. **The M3 win is NOT what W4A8 does.** M3's 3.6–5× concurrency came from `int4_to_fp8_load.py`:
   convert int4→fp8 ONCE at load, hold **resident fp8 in HBM**, run the **stock native**
   `VllmMixtureOfExpertsOpFP8PerChannel` — zero per-forward dequant, <1% loss. **Kimi cannot do this
   at all** (fp8 resident ≈ 1 TB > 768 GB). W4A8-fused is the *opposite*: int4 stays resident,
   dequant→bf16→fp8 happens **every forward** inside a *custom* fused op. Same MME, totally different
   (and worse) overhead profile. Equating them is the core error to refute.

2. **It inherits a proven LOSS, then adds cost.** The fused-expert OpBackend op is **slower than the
   stock `mixture_of_experts` intrinsic in both regimes** (single ~23 vs ~30; throughput ~68 vs ~126;
   PROJECT_FINDINGS Finding 2/5 — the project's hardest-won result). W4A8 = that same op + 8 extra
   per-forward abs/reduce/cast/`fp8_gemm`-swap nodes. The fp8 MME's 2× must first repay the fused-op
   debt AND the new cast/amax tax before it nets positive — and even then:

3. **Single-stream is host/collective-bound → MME 2× buys ≈0.** At batch=1 MoE compute is ~0.16 ms of
   a ~127 ms step (~90% collective+host). Halving the fp8 GEMM is invisible. So the *only* possible
   W4A8 payoff is throughput@concurrency — exactly where strike 2 says it loses to stock. Plus a HARD
   **~6.4e-2 relNorm double-quant floor** (vs stock int4-bf16's clean dequant), which scale
   granularity does NOT fix (measured: per-channel/per-token → 6.28e-2; the binding limit is fp8's 3
   mantissa bits + the intermediate-h requant). **Net: accuracy give-up for a speed that isn't there.**

Its ONLY defensible value is as a "touch-the-fp8-MME-at-all" memory enabler for a 1T int4-only model
— and even that is gated behind an unmeasured on-card W4A8-vs-int4→bf16 A/B that, per strikes 2–3, is
unlikely to win. **The ~3–6 on-device kernel-days + FCD/SIF bring-up risk are not justified.** Park
the scaffold; it is correctly written and LOUD-failure-gated if ever revisited, but it is not the move.

> **The honest way to "maximize int4" for Kimi is NOT a precision trick — it is throughput
> amortization:** stock CT-WNA16 op (do NOT enable `HPU_MOE_FUSED_OP`), `--max-num-seqs` 16→64→128,
> warm recipe cache + pre-warm, `--block-size 128`, fp8_inc KV (MLA KV is tiny → high conc fits),
> `--enable-expert-parallel` (384 experts sharded once). int4 = *fit*; speed is clawed back by
> concurrency + MLA-compact KV, never by the GEMM precision (Gaudi-2 MME is FLOAT-only; int4 always
> dequants to the half-rate bf16 MME). No MTP exists (`num_nextn_predict_layers=0`) → no spec-decode
> lever either; don't chase it.

## (2) Kimi serve + measure plan (DO THIS FIRST — establishes the baseline)

Kimi **serves as-is today** (verified card-free): in-image `KimiK25ForConditionalGeneration` →
stock deepseek_v2 MLA + the HPU CT-WNA16 override; checkpoint 555 G/64 shards on /mnt/data2,
mountpoint present, `serve.sh` ready. No new wiring, no `--quantization`, no custom class.

```
./serve.sh kimi-k2.7      # vllm-gaudi:next, TP8+EP, fp8_inc KV, 8k ctx / mns 16 (env is correct as-shipped)
```
Then, on a **quiet fabric** (OAM-reseat first if the scale-up link storm is active — it has crushed
prior readings to 2–5 tok/s):
1. **Coherence + single-stream** at 8k/mns1 — expect **~5–10 tok/s** (GLM-class all-MoE int4 analog,
   not Qwen). This is the floor; it will NOT be interactive-snappy — that's the structural int4 wall.
2. **Throughput sweep** mns 16→64→128 at 8k, then 32k — bank the aggregate (~150–300 tok/s @ c32–64
   expected; KV pool ~103 GiB caps conc at longer ctx). **This is the number that matters for Kimi.**
3. Record vs the M3 fp8 numbers so the "int4-fit-not-speed vs fp8-fast" gap is concrete for the
   2nd-node decision. Raise ctx/conc only after coherence holds.

**Answer to the user's "maximize int4":** maximized = stock op + concurrency + warm cache + fp8_inc
MLA-KV at the highest mns the KV pool allows. There is no int4 precision lever beyond this on one node.

## (3) SYNTHESIZED execution order — rest of the session

Two threads (item 4 = M3 MSA long-context; item 5 = maximize Kimi). Order by leverage × card-cost,
front-loading the cheap measurement that de-risks everything downstream:

1. **NOW — Serve + measure Kimi int4 baseline** (§2 above). Cheapest, answers item 5 directly, and
   gives the apples-to-apples int4-vs-fp8 datum the 2nd-node call needs. ~1 card session.
2. **THEN — Bank the M3 fp8 concurrency win as the reference.** M3's int4→fp8-at-load (the real 3.6–5×)
   is the proven "fast giant" template; confirm/record stock-fp8 vs int4 at mns 64/128 (the
   `int4-wall-strategy` 4/4 recommended measurement). This is also the *control* that proves Kimi's
   int4 ceiling is fit-forced, not a config miss.
3. **THEN — M3 MSA (item 4), in the build-msa-kernel order:** (a) re-pin `index_q/k_norm` vs official
   modeling (correctness gate — `index-norm-reconfirm` flagged corruption; gates all speed work),
   (b) validate the two bf16 MSA GC graphs on-card + flip `MSA_READY`, (c) land the single-call consume
   (collapse the 4×/layer loop, gather on 4 real KV heads — the #1 decode-throughput lever), (d) fp8
   index logits + `masked_batch_gemm` for prefill TTFT at 128K–1M. THEN the MSA-aware KV-offload
   (`offload-design-SYNTHESIS`) as phase-2 for 1M.
4. **DEFER / DO-NOT-BUILD — W4A8-fused for Kimi** (§1). Park the scaffold. Revisit only if a 2nd node
   is off the table AND step-1 measurement somehow shows Kimi concurrency is fp8-GEMM-bound (it won't
   be — it's dequant+host-bound). The real "fast Kimi" lever is a **2nd node** (→1536 GB lets 1T run in
   fp8, escaping the dequant tax) — but measure RoCE scale-out cost first; it buys capacity/fp8-fit, NOT
   single-stream, and vllm-gaudi has no PP today.

**One-line ordering rationale:** measure-cheap-first (Kimi baseline → M3 fp8 control), then spend cards
on the MSA long-context build (the only item with a real un-shipped speed lever left), and explicitly
NOT on W4A8 (refuted) — because for a 1T int4-only model on one node, fast-and-lean are mutually
exclusive by the FLOAT-only-MME fact; the lever for "fast 1T" is hardware (2nd node), not a kernel.
