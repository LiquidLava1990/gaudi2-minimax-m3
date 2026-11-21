# Adversarial review — weight-quant method (minimal quality loss)

**Scope:** MiniMax-M3 (428B MoE, 60 layers, 128 routed experts/layer top-4, 1 shared expert,
dense MLPs L0-2), served via vllm-gaudi + hpu-moe-ext on 8×Gaudi2. Lens: is AutoRound int4 RTN
(iters=0) the best *weight* quant, vs iters>0 / GPTQ / AWQ / **mixed precision** (GLM is
"int4-mixed")?

## VERDICT: **KEEP RTN-int4 on routed experts.** Do NOT adopt GLM's int8 attention/shared-expert
## mixed-precision. One worthwhile add: keep `w2` (down_proj) of the *first/last* MoE layers bf16.

Confidence: **HIGH** on keep-RTN + don't-int8-the-rest; **MEDIUM** on the w2 edge-layer tweak.

---

## The decisive constraint: on Gaudi 2, weight-bits ≠ speed

`Gaudi2_Optimization_Guide.md:431-435` and the hpu-moe-ext dispatch (`auto_round_hpu.py:323-336`)
agree: **INT4 has no native matmul on Gaudi 2 — int4/int8 weights dequant to bf16 at compute.**
hpu-moe-ext routes *every quantized LINEAR* (attention, shared-expert, indexer) through
`GPTQDequantToBF16LinearMethod` (int4-linear convert "compiles pathologically slowly… engine-death
on lazy compile at 397B"). **Only the routed MoE experts run on the fast int4 WNA16 op.**

Consequences for "fastest + minimal quality loss":
- Weight-bits on *non-expert* modules is a pure **memory-fit lever**, not a throughput knob. Lower
  bits there buy you nothing in speed and cost quality + warmup risk.
- The experts are the only place int4 both saves memory *and* is the supported fast path. RTN-int4
  there is correct.

## Memory economics (M3 `--dry-run`, ground truth)

| Bucket | bf16 | int4 |
|---|---|---|
| Routed experts (all) | 769.5 GiB | **198.4 GiB** |
| Everything else (kept bf16) | 86.5 GiB | — |
| **Total** | | **~285 GiB** |

8×96 = 768 GiB HBM. Output ~285 GiB ⇒ **~480 GiB free** for KV cache + graphs + concurrency.
**There is no memory pressure forcing attention/shared-expert below bf16.** GLM went int8 there
because GLM is 744B and *had* to fit; M3 does not. Quantizing them would trade real quality
(see below) for memory you don't need and speed you won't get.

## What "int4-mixed" actually is (decoded from the GLM config + loader)

GLM-5.1-int4-mixed-AutoRound `extra_config`, per-module:
- **int4**: routed experts (default, never listed).
- **int8**: all attention/MLA (`q_a/q_b/kv_a/kv_b/o_proj`), indexer `wk`/`wq_b`, **and the shared
  expert** (`mlp.shared_experts.{gate,up,down}_proj`, every layer).
- **bf16**: dense MLPs L0-2, **router `gate`**, indexer `weights_proj`.

So "mixed" = *down-grade some bf16 to int8 to fit 744B*, **not** *up-grade experts above int4*.
The M3 recipe already keeps attention + shared-expert + router + dense MLPs at **bf16** — i.e. M3 is
**strictly higher-precision than GLM** on every sensitive module, while still int4 on the bulk
experts. Copying GLM's int8 would be a quality *regression*, for no fit benefit. Confirmed: hpu-moe-ext
trusts *stored format* as ground truth (`_load_quantized_prefixes`, shard headers), and would happily
serve either; the choice is ours.

## RTN (iters=0) vs AutoRound iters>0

- AutoRound docs: RTN = "slight accuracy drop at W4" vs tuned; tuning needs ~200 steps. The file's
  own runtime note: iters>0 = **hours-to-days** for a 200B+ model.
- The drop lands *only* on the int4 experts (everything else is bf16). With top-4-of-128 routing,
  per-expert error is heavily averaged at the token level, and bf16 attention/shared-expert/router
  carry the load-bearing signal — so the RTN gap is small in practice.
- **Quality-per-hour: RTN wins decisively for the first pass.** iters>0 is a *later* optional A/B if
  an eval shows a real gap, not a default. (alg_ext / auto-round-light are cheap middle options if so.)

## GPTQ vs AWQ vs AutoRound

- All export to the **same auto_gptq int4 packing** hpu-moe-ext consumes — method choice is a
  *calibration-algorithm* choice, not a runtime-format change.
- 2026 consensus: AWQ ≈ +1-2% over GPTQ at W4 (protects ~1% salient channels); AutoRound is
  competitive-to-best at W4 and *already the proven path on this box* (GLM + Qwen-397B serve through
  it). No reason to switch tooling. If anything, AutoRound's **AutoScheme** can auto-assign mixed
  bits — but per above we don't want lower bits, and we don't need higher (no fit pressure).

## Which modules genuinely deserve > int4 (SOTA 2026)

QuantMoE-Bench, MxMoE (ICML'25), GEMQ, EAQuant, Dynamic-Expert-Quant all converge:
1. **Shared expert** — all-token traffic ⇒ highest precision. ✅ M3 already bf16.
2. **Router `gate`** — int4 misroutes ⇒ math/code/reasoning loss. ✅ M3 already bf16.
3. **`down_proj` (w2)** — outlier-heavy (down-proj variance/outliers >> other linears; max/median
   ratios up to ~100×) ⇒ worst int4 victim. M3 quantizes expert `w2` to int4 like w1/w3. **This is
   the one place RTN-int4 leaves measurable quality on the table.**
4. **First/last layers** — most sensitive; embeddings/L0-2 already bf16.

## #1 CHANGE (only material one)

**Keep the expert `w2` (down_proj) of the first and last MoE layers (e.g. L3-5 and L57-59) at
bf16**, via a `layer_config` regex on those experts' `w2`. Rationale: down-proj outliers + edge-layer
sensitivity are the documented worst case for W4; the cost is tiny (a handful of the 21,888 expert
matrices → a few GiB of the 480 GiB free HBM) and it does **not** touch the fast-path for the other
99% of experts. If you want it dead-simple and measurable: ship pure RTN-int4 first, run lm-eval
(GSM8K/MMLU/code) vs the bf16 control, and only add the w2-edge carve-out if the eval shows a gap.

**Do NOT** (anti-recommendations): don't int8 attention/shared-expert (GLM-style) — no fit need, no
speed gain, real quality cost + slow warmup; don't raise iters>0 for pass 1; don't switch to
GPTQ/AWQ tooling (same packing, proven AutoRound path). Keep `group_size=128 sym` (matches the
served checkpoints; sym int4 is the packing the WNA16 op + repack cache validate).

### Calibration-data note
Default `NeelNanda/pile-10k`, nsamples=128 is fine for **RTN** (RTN barely uses calibration — it's
round-to-nearest + group scales; calib mainly sets ranges). It matters far more if you ever go
iters>0 — *then* switch CALIB_DATASET to a code/agentic/long-context mix matching the serving
workload (the file's TODO is correct). For the RTN pass, leave it.
