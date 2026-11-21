# Giant-MoE serving on 8×Gaudi2 — a ~1T int4 MoE (Kimi-K2.7-class)

**Question:** Does a ~1T-param int4 MoE (≈500 GB raw weights, ~32B active/token) fit and serve
usefully on our 8×HL-225H (768 GB nominal HBM)? **Date:** 2026-06-15. **Method:** CARD-FREE
arithmetic + extrapolation from the box's measured GLM-5.1 (744B/40B-active int4) and
Qwen3.5-397B/A17B int4 numbers. Did NOT touch cards / `vllm-active`.

**Verdict up front (see §6):** **YES it fits, NO it is not viable at int4.** A 1T int4-mixed MoE
*loads* on 768 GB with a usable KV envelope, but it serves in **GLM's slow regime** — int4
forces the all-experts dequant tax (the measured ~98%-of-decode cost on this stack), and at 1T
that base is *worse* than GLM's. Realistic single-stream **~5–10 tok/s**, aggregate at
concurrency **~150–300 tok/s** — far below the Qwen-397B numbers, because Qwen's throughput
came from a path int4-at-this-scale can't take. The model that *would* serve well (fp8 experts,
~2× native MME, no dequant) **does not fit at 1T** (~1 TB). So: technically possible, practically
not worth it. **Position: serve ≤~600B at fp8, not ~1T at int4.**

---

## 0. Inputs (Kimi-K2.7-class, ~1T int4 MoE)

| Quantity | Value | Source / note |
|---|---|---|
| Total params | ~1.0 T | brief |
| Active params / token | ~32 B (~3.2%) | brief; K2-class sparsity |
| Raw int4 weights | ~500 GB | 1T × 0.5 B/elem |
| Usable HBM | **~691 GiB** | 768 GB nominal × **0.9** util (NOT 768 — the box's real ceiling) |
| Graph reserve | ~69 GiB | `VLLM_GRAPH_RESERVED_MEM=0.1` of usable |
| MME precision | **float-only (bf16+fp8)** | Guide §IV.2 — int4 ALWAYS dequants to bf16 before GEMM |

The brief's "500 GB / 8 = 62 GB/card → 34 GB/card for KV+graph" is the right shape but uses 768;
against the box's real **691 GiB usable** the headroom is tighter (see §1).

---

## 1. FIT — does ~1T int4 fit on 691 GiB usable? (YES, narrowly)

**Weights (int4-mixed, the proven `hpu-moe-ext` contract = routed experts int4, everything else
bf16):**
- routed-expert int4 ≈ 470 GB × **1.10** (group-128 scales/zp) ≈ **~517 GB**
- non-expert bf16 (attn, shared expert, dense, router/gate, embed, untied lm_head) for a 1T model
  ≈ **~35–45 GB** (a 1T model has a large vocab + wide hidden; conservative 40 GB)
- **total serve weights ≈ ~557 GB ≈ ~519 GiB** → **~64.9 GiB/card at TP8.**

(Pure-bf16 would be ~2 TB — TP8-only and *still* OOM at >768 GB; **int4 is mandatory just to load.**
fp8 experts would be ~1.0 TB total — **does not fit**, the decisive viability fact, §4/§6.)

```
usable 691 GiB − weights 519 GiB − graph 69 GiB  =  KV pool ≈ 103 GiB   (~12.9 GiB/card)
```

**It fits, but the KV pool is small (~103 GiB vs M3-int4's ~337 GiB).** This is the whole story:
at 1T the weights eat the box, leaving GLM-class (not M3-class) KV room.

### The context × concurrency envelope (fp8_inc KV)

KV/token depends on the (unknown, K2-class) attention. K2 uses **MLA** (DeepSeek-style), whose KV
is *tiny* — a single compressed latent (~576 elems/tok/layer), not full GQA. Assume MLA, ~61
layers, fp8_inc ≈ **~70 KiB/tok** (MLA latent + rope key). Then KV pool 103 GiB ≈ **~1.54 M
tokens** total budget:

| Context / seq | KV / seq (fp8_inc, MLA) | **Max concurrent seqs** |
|---|---|---|
| 8 K  | 0.56 GiB | **~184** |
| 32 K | 2.24 GiB | **~46** |
| 64 K | 4.48 GiB | **~23** |
| 128 K| 8.95 GiB | **~11** |

Concurrency-first (budget ÷ conc): **conc 30 → ~50 K ctx · conc 64 → ~24 K · conc 128 → ~12 K.**

> If K2.7 is **GQA** not MLA, KV is ~3–4× heavier and these halve again (conc 30 → ~16–24 K). MLA
> is the optimistic case and the likely one for a K2-class model. **Envelope: comfortable at
> 8–32 K / 30+ conc; 128 K only at low conc (~11).** The binding cap is KV pool size, which is
> small *because the int4 weights are huge* — not because KV/tok is large.

---

## 2. SERVE CONFIG (TP8 + expert-parallel, int4)

Forced layout: **TP8 × DP1.** A 1T model is ~65 GiB/card *already* — it does **not** fit one card
at any quant, so the entire replicate-don't-TP / `dp_serve.sh` regime is **N/A** (cf. M3-int4
which also lands `dp1_tp8`; the guide auto-sizer picks TP8 for anything >~376 GB). No DP contest
exists here. TP8 also holds the experts **once** with `--enable-expert-parallel` (≈ experts/8
per card); without EP the experts shard like dense layers → slower + graph-compile failures.

```bash
PT_HPU_LAZY_MODE=1                 # int4 WNA16 MoE + lazy capture path
PT_HPU_ENABLE_LAZY_COLLECTIVES=true   # MANDATORY for the TP8 all-reduce path
PT_HPU_WEIGHT_SHARING=0            # free full-precision staging post-quant → avoid load OOM
VLLM_EXPONENTIAL_BUCKETING=true
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600   # first-request compiles on a 1T model are long
PT_HPU_RECIPE_CACHE_CONFIG=<local-dir>,true,16384   # then ,false, — wipe on any plugin/.so change

vllm serve <kimi-k2.7-int4> \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization <autoround/gptq_hpu>        # int4 experts via hpu-moe-ext autodetect
  --kv-cache-dtype fp8_inc \                  # the ONLY KV-compression the platform gate allows
  --block-size 128 \
  --max-num-seqs 64 \                         # sweep 32→64→128; KV pool caps it (~46 @32K)
  --max-num-batched-tokens 16384 \            # guards the PR#1346 >2K-prefill graph-skip (−29%)
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768                       # cap to real need; KV pool is the limit
```

- **`--max-num-seqs`: 64**, sweep to 128 *only if* ctx is small (≤16 K) — unlike M3, the KV pool
  here can't afford c128 at 32 K (~46 seq ceiling). Set `VLLM_DECODE_BS_BUCKET_MAX=max_num_seqs`.
- **fp8 KV is mandatory**, not optional — the int4 weights leave too little KV pool for bf16 KV.
- **MTP risk:** if the int4 export quantized the K2 MTP/draft layer int4-everywhere with no
  calibration, it ships a **dead draft head** (the exact GLM §8 failure) → spec-decode 0% until a
  bf16 graft. fp8 would keep MTP alive; int4 likely needs the graft. Budget for it.

---

## 3. THROUGHPUT (extrapolated from the box's GLM/Qwen MoE numbers)

The honest anchor is **which measured model this resembles**, and it resembles **GLM, not Qwen**:

| Box-measured model | Quant / active | Single-stream | Aggregate (warm) | Why it's the anchor it is |
|---|---|---|---|---|
| Qwen3.5-397B / A17B | int4, **45 DeltaNet + 15 full-attn** | ~30 tok/s | **486 @c64, 793 @c128** | bounded per-layer compute; hybrid attn — the GOOD case |
| **GLM-5.1 / 744B / 40B-act** | **int4, all-256-expert dense MoE** | **~14 (warm), ~23 w/MTP** | **GLM regime (tens)** | all-experts int4 dequant blowup, collective+host bound |
| **→ 1T int4 / 32B-act** | int4, ~K2-class | **~5–10** | **~150–300** | **closest to GLM, scaled up** |

**Why GLM, not Qwen, is the analog:** the §11a trace showed GLM's int4 decode is **~98% device
time in ~345k tiny ~13µs int4→bf16 dequant kernels computing all experts/layer** — and the
routed-K mitigation (`tokens*top_k ≤ K`) that helps single-stream **turns OFF under concurrency**,
so at c≥30 int4 *must* dequant the full expert set every step. A 1T model has **more experts and
2× the active params** (32B vs GLM's effective 40B is comparable, but the total expert pool to
dequant-on-miss is larger), so its dequant tax is **≥ GLM's**. Qwen's 486–793 tok/s came from a
**bounded-compute hybrid-attn** path that an all-MoE int4 giant cannot replicate.

**Estimate (quiet fabric, warm cache, c=32–64, 32 K ctx):**
- **Single-stream: ~5–10 tok/s** (GLM's 14 was 744B; 1T scales the dequant + collective work up).
- **Aggregate @ c=32–64: ~150–300 tok/s.** int4 does NOT scale to Qwen's 486–793 because each
  decode step is dequant-bound, not batch-saturation-bound — adding sequences adds dequant work
  roughly linearly rather than amortizing a fixed MME.
- **An fp8 version (if it fit) would be multiples faster** (native MME, no dequant, ~2× — Guide
  §IV.2, fp8-moe-gate GO). It doesn't fit at 1T → that speed is unreachable here.

> Numbers are **MEDIUM confidence** (extrapolation, not measured) and **fabric-gated** — the box's
> intermittent scale-up link storm (gaudi-fabric-defect-rma) has repeatedly crushed real readings
> to 2–5 tok/s. A clean bench needs the OAM reseat first.

---

## 4. BINDING CONSTRAINT

Two stacked, and they fight each other:

1. **HBM-for-weights is the FIT constraint** (decides yes/no, and *forces* int4). A 1T model in
   the precision that serves fast (fp8, ~1 TB) **does not fit**; only int4 (~519 GiB) fits,
   leaving a small KV pool. So HBM dictates the *worst-performing* precision.
2. **Dequant + collective/host overhead is the SPEED constraint** (decides useful-or-not). Given
   int4 is forced, decode is the §11a all-experts dequant tax (compute-bound at concurrency) on
   top of per-token TP8 all-reduce + host overhead (the GLM single-stream floor). **This is the
   binding throughput constraint, and it is a direct consequence of constraint 1.**

KV is **not** the primary binding cap (MLA KV is tiny; the pool is small but ~32 K/30-conc fits).
Fabric is an *availability* gate (must be quiet to even bench), not the steady-state ceiling.
**Net: HBM forces int4 → int4 forces the dequant tax → the dequant tax is the wall.**

---

## 5. Sensitivity

| Weights (int4-mixed) | KV pool | conc @32K | maxctx @c30 | Fit? |
|---|---|---|---|---|
| 500 GiB (optimistic, lean bf16 rest) | ~122 GiB | ~54 | ~58 K | ✅ |
| **519 GiB (central)** | **~103 GiB** | **~46** | **~50 K** | ✅ |
| 545 GiB (heavy bf16 rest / larger vocab) | ~78 GiB | ~35 | ~38 K | ✅ tight |
| 560 GiB (+ MTP draft resident) | ~63 GiB | ~28 | ~31 K | ✅ very tight |

Fit is **robust** across ±30 GiB of weight estimate (always loads). The **viability verdict does
not move**: every row is GLM's slow int4 regime; none unlocks fp8 (all >> the ~150 GiB free fp8
would need). The knob that would change the answer — fp8 experts — is locked out by the 1 TB
footprint at every plausible weight estimate.

---

## 6. VERDICT

**(1) Fit:** **YES**, narrowly. ~1T int4-mixed ≈ **519 GiB weights** loads on 691 GiB usable,
leaving a **~103 GiB KV pool**. Envelope (MLA KV, fp8_inc): **comfortable at 8–32 K context /
30+ concurrency** (conc 30 → ~50 K ctx; ~46 seqs @32 K); **128 K only at low conc (~11)**. fp8
experts (~1 TB) **do NOT fit** — int4 is mandatory just to load.

**(2) Serve config:** **TP8 × DP1 + `--enable-expert-parallel`** (no DP option exists at 1T),
int4 experts via `hpu-moe-ext`, `--kv-cache-dtype fp8_inc`, `--block-size 128`,
`--max-num-seqs 64` (sweep, KV-capped), `--max-num-batched-tokens 16384`, lazy + lazy-collectives
+ `PT_HPU_WEIGHT_SHARING=0`, recipe cache local.

**(3) Throughput:** anchored to **GLM (the all-MoE int4 analog), not Qwen.** Realistic
**single-stream ~5–10 tok/s**, **aggregate ~150–300 tok/s @ c32–64** (quiet fabric, warm) —
int4's dequant tax prevents the Qwen-style 486–793 batch-saturation scaling. MEDIUM confidence,
fabric-gated.

**(4) Binding constraint:** **HBM-for-weights forces int4 → int4's all-experts dequant tax
(§11a) is the throughput wall** (compute-bound at concurrency, atop TP8 collective+host overhead).
Not KV, not fabric-steady-state.

**(5) VIABLE? — NO, not at int4 (technically loads, practically not worth it).** It *fits and
runs coherently*, but serves in GLM's slow regime: ~tens of tok/s aggregate, single-digit
single-stream — useless for interactive use, weak even for batch. The model that *would* serve
well on this box (fp8 experts, native ~865 TF MME, no dequant — the proven #1 lever) is **too big
at 1T** (~1 TB > 768 GB). **The box's sweet spot for a giant MoE is ~≤600B at fp8** (fits ~1 TB→
under budget, native MME, ~2× int4, MTP-alive), **not ~1T at int4.** A 1T int4 K2.7 is the wrong
shape for this hardware: it's the one precision Gaudi 2 has no native matmul for, forced by a
weight footprint that crowds out both KV and the fast precision. **Recommendation: don't serve
1T-int4 here for production; if the 1T weights are a hard requirement, accept ~150–300 tok/s
aggregate batch-only and benchmark GLM-style on a quiet fabric first to confirm the floor.**

**Confidence:** HIGH on fit arithmetic + the int4-forced-dequant-wall mechanism (config-derived +
matches GLM's measured §11a pathology); MEDIUM on absolute tok/s (extrapolated, fabric-gated) and
on KV/tok (depends on whether K2.7 is MLA — assumed — vs GQA).
