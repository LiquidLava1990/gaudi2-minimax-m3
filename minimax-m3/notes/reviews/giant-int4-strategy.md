# Giant int4 Models on 8×Gaudi 2 (768 GB) — The Reusable "Run Any Giant Fast + Lean" Strategy

**Lens:** strategist + devil's-advocate, end-to-end. **Date:** 2026-06-15. **CARD-FREE.**
**Thesis under test (user):** *"fp8 will be too big for the giants (Kimi K2.7, GLM 5.2, the 1T
wave ship int4 because of size), and there HAS to be a way int4 makes more sense."* **Working
answer under test (user):** **W4A8-fused** — int4 weights in HBM (the only thing that fits),
**dequant int4→fp8 fused in-kernel → native fp8 MME** = lean AND fast; one dynamic OpBackend
kernel covers int4-W4A8 + fp8 + future; + MTP for decode.

---

## VERDICT (one line)

**The thesis's *premise* is right — int4 is the only thing that fits the true giants, and the box
genuinely needs int4 as the load-bearing default for 1T-class. But the proposed *mechanism*
(W4A8-fused: int4→fp8 dequant fused in a custom kernel) is REFUTED on this exact hardware: it is
DOUBLE-quantization with a measured hard ~6.4e-2 relNorm floor, it does NOT hit "fp8 MME speed,"
and the #1 thing to build for it is ALREADY BUILT — vllm-gaudi ships a native fp8 MoE MME path.
The correct giant-model default is a TWO-TIER recipe keyed on fit, not one fused kernel:**

> **Tier 1 (fits in fp8 — ≲600B-class on 768 GB): FP8(INC) experts, native fp8 MME, no int4 in
> the chain.** This is "fast" done right (native E4M3 GEMM, ~2× bf16, <1% loss, MTP stays alive).
> **Tier 2 (too big for fp8 — the 1T+ wave): int4 weights in HBM + stock bf16-dequant MoE +
> MSA/DSA sparse + MTP.** Here int4 is the *only fit*, you EAT the dequant tax (it is structural,
> not removable by fusion), and you win back speed via sparsity + spec-decode + concurrency, NOT
> via a W4A8 fused kernel. int4 is a MEMORY play; you do not get to also call it fast.**

Confidence: **high** on the precision mechanism (direct code + measured on-card floors); **high**
on the fit arithmetic; **medium** on exact tok/s (config/fabric-bound, must measure on quiet
fabric).

---

## (1) Is int4-W4A8-fused the right DEFAULT for giant models? NO — but int4-in-HBM is, for the giants that don't fit fp8.

Three precisions, one load-bearing hardware fact, decided per the M3 adversarial pass + PROJECT_FINDINGS:

**THE FACT (everything hangs on this): Gaudi 2's MME is FLOAT-ONLY — bf16 + fp8(E4M3/E5M2), NO
integer matmul.** Integers live on the TPC (elementwise), never the GEMM engine
(`KERNEL_ARCHITECTURE.md:29`; Guide Part IV §IV.2). Therefore **int4/int6/int8 ALL dequant to
bf16 (or fp8) before the GEMM.** This is not theoretical — it is the measured §11a decode
pathology: on the sibling GLM int4 path, decode ≈193 ms (~5.2 tok/s), **~98% of device time in
~1M tiny (~13 µs) int4→bf16 dequant kernels computing ALL 256 experts/layer**, ~5% collective.

Now grade each option:

| Option | Fits the giants? | Fast? | Verdict |
|---|---|---|---|
| **fp8 (native MME)** | NO for 1T+ (too big); YES ≲600B | YES — native E4M3, ~865 TF, ~2× bf16, no dequant | **Tier-1 default when it fits** |
| **int4 → bf16 (stock WNA16)** | YES — smallest footprint, fits everything | SLOW — the full dequant tax above | **Tier-2 default (fit-forced) — the honest 1T path** |
| **int4 → fp8 fused (the user's W4A8)** | YES (same int4 HBM footprint) | **NO** — see below | **REFUTED on this HW** |

**Why W4A8-fused (int4→fp8 dequant-in-kernel) is the wrong mechanism — three independent strikes:**

1. **It is DOUBLE-quantization with a hard, measured floor.** Starting *from* int4 and adding fp8
   compute is W4→bf16→fp8: two lossy steps stacked. Measured on-device **~6.4e-2 relNorm floor**
   (CPU-emulation-matched; per-channel/per-group scaling does NOT break ~5e-2; only W2-in-bf16
   claws to 5.1%) — PROJECT_FINDINGS Finding 6 / §154. That is ~6% expert error vs FP8(INC)
   single-quant's **<1%**. You pay accuracy for a speed you don't get.

2. **The "fused int4→fp8 = fp8 MME speed" premise does not survive contact with the stock op.**
   This is the central hard-won lesson of the entire kernel project: a custom fused int4 MoE
   OpBackend kernel was *built, validated (~4e-3), general (5/5 families), and serves 397B
   end-to-end* — and it is **SLOWER than the stock `mixture_of_experts` intrinsic in BOTH regimes**
   (single-stream ~23 vs ~30; throughput ~68 vs ~126 at the same config). Fusing the dequant→MME
   does not beat the well-optimized stock op; you inherit its optimization debt as your bar
   (Finding 2/5). A W4A8-*fused* variant is the same trap wearing an fp8 hat — and worse, it adds
   the double-quant floor on top.

3. **The kernel you'd build to do it already exists as scaffold and is the wrong tool.** hpu-moe-ext
   ships exactly an int4→fp8 path (`moe_fused_expert_fp8_backend.cpp`, `fp8_routed_moe.py`: node 1
   is `dequantize_4_bit_bf16` from int4, THEN bf16→fp8). It is unbuilt/unvalidated/unwired *and it
   is the 6.4e-2-floor path by construction* (`FUSED_GENERALIZE_FP8_DESIGN.md:74-132`). Finishing
   it would re-introduce the very floor we reject.

**Where the user is RIGHT:** the *intuition* that "int4 is the thing that fits and there must be a
way to make it make sense" is correct for the true giants — int4 IS the only fit for 1T+, and it
IS the only weights envelope that fits **1M-context KV-offload** (the resident MSA index-K ~230 GB
+ double-buffer fits the int4 ~340 GB free envelope, NOT fp8's ~150 GB → fp8×offload@1M OOMs;
offload-design-SYNTHESIS.md, Guide §IV.5). So int4 has TWO non-negotiable homes: (a) models too
big for fp8, (b) the 1M-offload config at any size. The error is only in *how you make it fast*:
not by fusing int4→fp8 (double-quant, no win), but by accepting the dequant tax and beating it with
**sparse attention + MTP + concurrency** — the orthogonal levers that don't touch the GEMM precision.

---

## (2) The reusable pipeline (download → quantize model-free → serve fast+lean)

A single recipe with a precision *branch* decided by one fit check. Each stage is already built or
validated on this box:

```
                         ┌─ Tier 1: FITS in fp8 (≲600B on 768 GB)
download bf16 ──┬─ FIT? ─┤
(or int4 ckpt)  │        └─ Tier 2: TOO BIG for fp8 (1T+) │ OR │ need 1M-offload @ any size
                │
                └─ if already int4 on disk (GLM-AWQ/Qwen-GPTQ/Kimi-int4): skip quant, go Tier 2
```

**Stage A — Quantize, model-free, RAM-bounded (the giant-model enabler).** The bf16 giant does NOT
fit host RAM — *do not load it*. **AutoRound model-free RTN** (`ModelFreeCompressor`, `auto_round
>=0.13`): give it the model PATH string; it streams safetensors shards ONE at a time (1-deep
prefetch + async write), per-tensor RTN, packs `auto_round:auto_gptq`, `clear_memory()` per shard →
**peak RSS ~17 GiB (one shard), never the full 856 GiB.** No forward pass, no AutoModel, **no
runnable modeling code — only config.json + shards** (so a not-yet-in-vLLM giant is quantizable
day-one). Contract = **routed experts int4 / everything else bf16**; `iters=0` RTN. Two model-free
`layer_config` traps (both surface as "Quantized layers (0)"): keys ending in a digit get "."
appended → expert `w1/w2/w3` need a `$`-anchor; a bare `block_sparse_moe` bf16 prefix regex-swallows
experts → order expert int4 keys FIRST + drop swallowers. Script pattern:
`minimax-m3/quant/quantize_m3_int4.py` (`--dry-run` is stdlib-only, card-free). **For Tier 1, the
fp8 conversion is NOT this step — it's INC at serve time (Stage C).**

**Stage B — Dispatch from ground truth, not labels.** hpu-moe-ext autodetects quant by scanning
shard headers for `.qweight` prefixes — **stored format is ground truth, NOT the often-mislabeled
`extra_config`** (Qwen-397B tags int4 shared-expert projections as `bits:16`; trusting the label →
word-salad). Architecture-agnostic; correct for any int4-mixed export. This is why "download any
int4 giant and serve" works without per-model surgery.

**Stage C — Serve, by tier:**

- **Tier 1 (fp8, the fast path):** `--quantization inc` + `QUANT_CONFIG` (two-pass MEASURE→QUANTIZE;
  INC requantizes bf16→fp8 **in place**, then the **native** `VllmMixtureOfExpertsOpFP8PerChannel`
  → `mixture_of_experts.fp8` intrinsic — fp8 weights+activations+scales straight into the MME, **NO
  bf16 round-trip**, gated `VLLM_HPU_FORCE_CHANNEL_FP8=true` (default). Env: `PT_HPU_LAZY_MODE=1`
  (INC requires lazy), `PT_HPU_ENABLE_LAZY_COLLECTIVES=true`, `PT_HPU_WEIGHT_SHARING=0` (free
  full-precision post-fp8 → dodge OOM), `--kv-cache-dtype fp8_inc`, TP=8 `--enable-expert-parallel`.
  Calibrate maxabs on-card (`calibrate_model.sh -o g2`); NEVER `unit_scale` (Mixtral PPL
  11.34→96.65); E4M3 is fnuz ±240 not ±448. **MTP depth-1 stays alive** (modules stay bf16, not in
  the quantized expert set → no graft) = proven **1.70× decode multiplier**.

- **Tier 2 (int4, the fit-forced path):** stock **bf16-dequant WNA16** MoE (the `.qweight` autodetect
  path) — do NOT enable the fused int4 MoE op for speed (it loses to stock). Accept the dequant tax;
  win speed back with: **MSA/DSA block-sparse attention** (the OpBackend kernel family — caps decode
  attention at ~16 blocks×128, the long-ctx enabler, no stock equivalent so it's a real win), **MTP**
  (graft bf16 draft if the int4 export killed it, per GLM §8), **concurrency** (`--max-num-seqs`
  64→128; the routed-K single-stream mitigation dies at c≥30 so int4 leans entirely on throughput
  amortization), `--block-size 128`, fp8_inc KV, recipe cache. **1M:** int4-in-HBM × MSA-aware
  KV-offload (index-K HBM-resident exact; only main K/V tiered host-RAM→NVMe with step-ahead
  prefetch; consume op reads HBM unchanged, no new TPC kernel).

**Shared across both tiers (the durable infra):** the **OpBackend sparse-attention kernel family**
(DSA token/MLA + MSA block/GQA via one `mode` flag + a `*Dims` config reader — a sibling attention
shape drops in as config + ≤2 nodes), the `.qweight`-ground-truth dispatch, AutoRound model-free
quant, the `*_READY` never-silent-fallback safety gate, recipe-cache + warm-first DP serve.

---

## (3) The ONE thing to BUILD — it is NOT the W4A8-fused MoE kernel

**Do NOT build the W4A8-fused MoE kernel.** Strikes 1–3 above: it's double-quant (6.4e-2 floor),
the fused approach already lost to the stock op in both regimes, and its scaffold already exists and
is the wrong tool. Building it spends weeks to land *slower AND less accurate* than what ships today.
The native fp8 MoE MME path — the thing W4A8 was trying to reach — **is already wired in stock
vllm-gaudi** (no build).

**The actual highest-value build, IF a measured run proves the box is HBM-KV-bound at the target:
sub-fp8 KV as an OpBackend op + a platform-gate extension.** Today only `fp8_inc` passes the HPU
`is_kv_cache_dtype_supported` gate; TurboQuant/Kitty/KIVI shipped upstream **CUDA-only, zero HPU**.
Build it the DSA/MSA way (OpBackend op + `KVQ_READY` gate). Method when built: **TurboQuant `k8v4`**
(fp8 K + 4-bit V, Walsh-Hadamard, calibration-free) — the near-lossless knee (V-err 0.11, NIAH
0.997); reject 2-bit (garbage, V-err 0.75) and all 3-bit-VALUE modes. **Honest framing: this is a
MEMORY play (~1.3–2× concurrent long seqs over fp8_inc), NOT faster/better** — it moves the
concurrency×context knee, the thing that actually binds the giants at scale. Build only on a
falsifiable HBM-KV-bound trigger, ~2 kernel-weeks.

**Runner-up build (cheap, do regardless): the model-free quant + `.qweight`-dispatch + tier-select
serve harness as ONE push-button pipeline** — so "new 1T int4 giant drops" → download → dry-run fit
check → auto-pick tier → calibrate-or-skip → serve, with zero per-model code. Most of the pieces
exist (`dp_serve.sh` auto-sizing, the quant script, the autodetect); the work is the glue + the
tier-decision fit check. This is what makes the recipe *reusable* rather than a per-model project.

---

## (4) Honest limits — where even int4 doesn't fit / doesn't serve fast, and the 2nd-node question

**Usable HBM: 8×96 = 768 GB raw → ~691 GiB at util 0.9 → ~620 GiB after ~69 GiB graph reserve.**
The weights envelope (leaving KV room) is realistically **~470 GiB fp8 / ~285 GiB int4** for a
~430B-active-class model; the hard ceiling for weights-only at int4 is ~0.5 B/param.

**int4 weights-only fit ceiling ≈ a ~1.0–1.1 T-param dense-equivalent** (0.5 B/param → ~550 GiB,
leaving ~70 GiB for KV+graph — tight, low concurrency, short context). Concretely:

| Model class | bf16 | fp8 | int4 | On 768 GB? |
|---|---|---|---|---|
| ~400–430B (M3, Qwen-397B) | ~850 GiB ❌ | ~470 GiB ✅ | ~285 GiB ✅ | **fp8 (Tier 1) with KV room** |
| ~600–750B (GLM-5.1 744B) | ❌ | ~370–470 GiB ✅(tight) | ~233–285 GiB ✅ | **fp8 borderline / int4 comfortable** |
| **~1 T (Kimi K2.7-class, GLM 5.2 1T-wave)** | ❌ | **~500–600 GiB — does NOT fit with KV** ❌ | **~285–340 GiB ✅** | **int4 ONLY (Tier 2), modest KV** |
| **~1.5–2 T+ (true frontier)** | ❌ | ❌ | **~430–550+ GiB — fits weights, starves KV** | **int4 fits weights but NOT at serving concurrency/context** |

**So the honest wall is not "does int4 fit" — int4 fits weights up to ~1 T comfortably and ~1.5–2 T
barely. The wall is KV + concurrency + context.** A 1 T int4 model leaves only ~280–330 GiB for KV;
at the M3 budget (~74 KB/tok incl. the un-quantizable bf16 MSA index cache) that's the concurrency×
context knee around **conc 30 × ctx ≈ 100–130K**, and a 1.5 T+ model leaves so little KV that you're
forced to **low concurrency or short context** — exactly the memory-triangle "pick 2 of {fp8 speed,
128K+ ctx, 30+ conc}" (Guide §IV.3), pushed harder: at 1 T+ you're picking **1 of 2** because fp8
is off the table (forced int4 = the slow GEMM tax on every step) AND KV is starved.

**Does the box need a 2nd node for the true 1T+ models?** **Verdict, nuanced:**

- **For 1T int4 weights + serviceable serving (≤128K, modest concurrency): NO — one node fits,**
  but you're permanently in Tier 2 (the dequant tax, no fp8) and at the lower-left of the triangle.
  It *serves*; it is not *fast* in the fp8 sense and never can be (the GEMM tax is structural).
- **For 1T+ at fp8 speed (to escape the dequant tax), OR 1.5–2 T+ at any usable concurrency/context,
  OR 1 T at high concurrency + long context simultaneously: YES — a 2nd node is the only lever.**
  A second 8×Gaudi 2 (→ 1536 GB) is what lets a ~1 T model run in **fp8** (native MME speed, escape
  the tax) instead of fit-forced int4, and is what restores KV headroom for 30+ conc at 128K+. That
  is the real argument for the 2nd node: not "to fit" (int4 fits 1 T on one node), but **"to run the
  giants FAST instead of merely lean,"** which is precisely the user's own goal — and on one node,
  for the 1T+ wave, fast and lean are mutually exclusive by the FLOAT-only-MME fact.
- **Inter-node caveat (must measure before committing):** Gaudi 2 scale-out is RoCE, not NVSwitch;
  a 16-card TP or PP split pays cross-node collective latency on a single-stream-already-sync-bound
  decode (the box is already at ~8% of the BW floor at batch=1). The 2nd node buys *capacity and
  fp8-fit*, and throughput@concurrency; it will NOT lift single-stream tok/s and may hurt it. And
  pipeline-parallel is **not supported** in vllm-gaudi today (planned only) — so a 2nd node is a
  TP-16 or external-replication play, both with open scale-out-fabric risk on this hardware (and
  this box's intermittent scale-up link storm is a standing reliability tax even at TP-8).

---

## BOTTOM LINE

- **Is int4-W4A8-fused the giant-model default? NO.** The premise (int4 is the only fit for the
  giants) is correct; the mechanism (fuse int4→fp8 for "fp8 MME speed") is refuted: it's
  double-quant (~6.4e-2 floor), the fused approach already lost to the stock op in both regimes, and
  the native fp8 MME path it chases is already shipped — no kernel to build.
- **The reusable recipe is TWO-TIER, keyed on fit:** Tier 1 (fits fp8, ≲600B) = **FP8(INC) native
  MME experts + fp8_inc KV + MTP** — fast done right. Tier 2 (1T+, or any 1M-offload) = **int4 weights
  + stock bf16-dequant MoE + MSA/DSA sparse + MTP + concurrency** — int4 as the honest memory play,
  speed clawed back by sparsity/spec/batch, NOT by fusion. Shared: AutoRound model-free quant
  (RAM-bounded, no modeling code), `.qweight`-ground-truth dispatch, the dynamic OpBackend
  sparse-attention family, never-silent-fallback gates.
- **#1 thing to build:** NOT the W4A8 kernel. Build **sub-fp8 KV (TurboQuant k8v4) as an OpBackend op
  + gate extension** — but only on a measured HBM-KV-bound trigger, and knowing it's a memory/
  concurrency play, not a speed one. Cheaper runner-up: wire the quant→dispatch→tier-select serve
  glue into one push-button pipeline (what actually makes "run any giant" reusable).
- **Honest hardware limit:** one node (768 GB) fits int4 weights to ~1 T comfortably and ~1.5–2 T
  barely — but KV/concurrency/context starve above ~1 T, and fp8 (the only *fast* path) is off the
  table above ~600B. **The 2nd node is not needed to FIT the 1T wave in int4; it is needed to run it
  FAST (fp8) and at real concurrency/long-context — and on one node, for 1T+, fast-and-lean are
  mutually exclusive by the FLOAT-only-MME fact.** Measure scale-out collective cost first; it helps
  capacity/concurrency, not single-stream.
