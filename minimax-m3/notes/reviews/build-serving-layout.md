# Adversarial review — MiniMax-M3 serving layout (parallelism + batch saturation)

Lens: is **TP8 + expert-parallel** the right placement for MiniMax-M3 (428B MoE, int4-experts)
at 30+ concurrency / long context, or does a **DP layout** beat it on aggregate throughput?

Reviewed 2026-06-14 against: `Gaudi2_Optimization_Guide.md` (§3/§4/§DP, `dp_serve.sh`,
"replicate-don't-TP", Part II ranked), `Gaudi2_Kernel/PROJECT_FINDINGS.md` (§3, §11, §12),
`models/glm51ar-dsa.env` + `glm51ar-lazy.env`, M3 `config.json`, on-disk checkpoint.

---

## TL;DR — **KEEP TP8 + `--enable-expert-parallel`.** (high confidence)

A DP layout cannot beat it because **the only DP layout that would help does not fit**, and the
one that fits (TP4×DP2) wins nothing on aggregate concurrency while doubling EP all-to-all
fan-out and losing in-engine expert-sharing. The headroom is real but is spent better on KV
(concurrency/context), not on a second replica.

---

## 1. Fit math (the decisive constraint)

**Architecture (config.json, verified):** H=6144, L=60 (3 dense + 57 MoE), Hq=64, **Hkv=4**,
dh=128, 128 routed experts, **top-4**, Imoe=3072, +1 shared expert (3072), dense MLP 12288,
vocab 200064, untied lm_head. **~426B params, ~22.6B active/token (5.3%).**
Serve target = AutoRound **int4 routed experts only**, everything else bf16
(`quant/quantize_m3_int4.py`) — the proven `hpu-moe-ext` contract. On disk now: bf16 download
**in progress, 16/59 shards (~226G of ~852G); no int4 export exists yet.**

Footprint of the int4-mixed serve weights (routed-int4 ×1.10 for group-128 scales/zp + bf16 rest):
- routed-int4: **~227 GB**  ·  bf16 (attn/index/shared/dense/gate/embed): **~26 GB**
- **total serve weights ≈ 253 GB (~236 GiB).**  (All-bf16 would be ~794 GiB — TP8-only.)

| Layout | weights/card | KV+graph headroom/card | Fits 96 GiB? |
|---|---|---|---|
| **TP8 ×DP1 (current)** | **29.5 GiB** | **66.5 GiB** | ✅ |
| TP4 ×DP2 | 59.0 GiB | 37.0 GiB | ✅ |
| TP2 ×DP4 | **118.0 GiB** | **−22 GiB** | ❌ **impossible** |
| TP1 ×DP8 | 236 GiB | −140 GiB | ❌ impossible |

**The "replicate-don't-TP / does ~285GB fit at TP2→DP4×TP2?" hypothesis is FALSE:** 428B int4
needs ≥118 GiB/card at TP2 — over the card. M3 does not fit below **TP4**. The whole `dp8_tp1`
("model fits one card") regime that makes replication win for a 20–35B model never applies to a
428B model; `dp_serve.sh` auto-sizing would itself pick `dp1_tp8` here (cf. GLM 376–392 GB →
dp1_tp8 in the guide). So the only real contest is **TP8×DP1 vs TP4×DP2**.

## 2. TP8×DP1 vs TP4×DP2 — aggregate concurrency is a wash, TP8 wins on the tie-breakers

KV/seq is GQA-tiny (Hkv=4): fp8_inc, all 60 L, ctx 32k ≈ **2.01 GB/seq** total. KV-bound ceiling:

| Layout | free/card | KV/seq/card @32k | ~seqs/instance | ×DP | **aggregate seqs** |
|---|---|---|---|---|---|
| **TP8×DP1** | 66.5 GiB | 480 MiB | ~121 | ×1 | **~121** |
| TP4×DP2 | 37.0 GiB | 480 MiB | ~58 | ×2 | **~117** |

Essentially identical aggregate concurrency. The tie-breakers all favor TP8:

- **GQA KV-replication tax cuts the OTHER way than usual.** Hkv=4 < 8, so at **TP8 the KV cache
  replicates ×2** (4 heads can't shard 8 ways → 1 head/rank, duplicated). That is the one genuine
  TP8 demerit and it is already priced into the 480 MiB/card above — yet aggregate concurrency
  still ties, because TP4×DP2 must hold the **full weight set twice** (two replicas), spending its
  headroom on duplicated experts instead of KV.
- **Expert-sharing.** TP8×DP1 with `--enable-expert-parallel` holds the 128 experts **once**
  (16/card). TP4×DP2 duplicates all experts (32/card ×2 copies) — strictly more HBM for no
  throughput, the same tradeoff the guide flags for external replication (§DP "duplicates all MoE
  experts N×"). At 5.3% active this is pure waste.
- **In-engine DP is DEAD** (Findings/guide §DP — lazy-gated, CI-only). TP4×DP2 must be **external
  `dp_serve.sh`** (2 containers + nginx `least_conn` + sticky router). That is real operational
  surface (warm-first cache hand-off, per-replica recipe dirs) for a layout that doesn't win.

**Counter-argument considered:** at 5.3% active, decode is collective+host-bound (PROJECT_FINDINGS
§11/§12: GLM reverts to per-token all-reduce+host overhead once MoE work is small), and TP8 pays an
8-way all-reduce + EP all-to-all every layer vs TP4's 4-way. This is the strongest case for TP4 and
is why this is "high" not "very-high" confidence. But: (a) the all-reduce is ~5% of the step even on
GLM's worst trace; (b) M3's MoE is **not** GLM's all-256-expert int4 blowup — top-4 routed with a
stock `FusedMoE`/`mixture_of_experts` path keeps per-layer compute bounded; (c) splitting into DP2
does **not** reduce per-instance collective cost (each instance is still TP4 doing the same per-token
all-reduce) — it only halves the *participant count per reduce*, a sublinear latency effect that the
guide rates below the expert-duplication + ops cost. Net: **measure it, but launch TP8.**

## 3. The flags (validated for 30+ concurrency / long context)

Current serve (TP8 + EP, `--block-size 128`, fp8_inc KV, warmup/HPU-graphs ON) is **correct**.
Tune for saturation:

- **`--max-num-seqs`: raise to 128, sweep 64→128→192.** This is the #1 throughput dial and the
  highest-impact change (below). GLM proved near-linear scaling 16→64→128 (126→486→793 tok/s,
  §3/§4d) on the *same box* — M3's int4 MoE is the same engine path. KV is cheap (480 MiB/seq/card
  @32k), so 128 is comfortably KV-affordable. Set **`VLLM_DECODE_BS_BUCKET_MAX = max_num_seqs`** so
  warmup captures full batches.
- **`--max-num-batched-tokens ≥ 16384`** — guards the PR#1346 graph-skip regression that ran >2K
  prefills eager (−29% on a TP8 FP8 run); grep startup for the graph-skip `info_once` line.
- **Exponential bucketing ON** (default `VLLM_BUCKETING_STRATEGY=exp`) — keep; do not hand-tune
  linear knobs unless a profile shows one shape dominating.
- **`--async-scheduling`** — keep (already in the draft env); it is the live V1 host-overhead lever.
  Do NOT port CUDA `--num-scheduler-steps` (no-op). Note: platform auto-disables async with
  spec-decode, so MTP-on and async are mutually exclusive (phase-1 MTP is off → fine).
- **`VLLM_GRAPH_RESERVED_MEM`: keep ~0.1, then read the usable/graph/KV split the engine prints and
  lower toward more KV** only until graph capture nears OOM. TP8 has 66 GiB/card — push it to KV.
- **`PT_HPU_RECIPE_CACHE_CONFIG` per layout, local disk, own dir** — first run delete=True then
  delete=False (FP8 504→34s). Wipe on any plugin/.so change (the documented stale-recipe trap).
- **Sticky prefix router** (`vllm-project/router`, consistent-hash) — **only relevant if you ever run
  DP**. At TP8×DP1 there is **one engine**, so just `--enable-prefix-caching` *iff* MSA allows it.
  ⚠️ Contiguous-PA auto-disables prefix caching, and the M3 MSA plan sets `--no-enable-prefix-caching`
  (block-selection assumes no cross-request reuse) — so the sticky router is **N/A for this build**.
- **`--max-model-len`**: cap to real need (32768 phase-1). Full 1M needs the MSA sparse kernel
  (`hpu-msa-ext`, not yet built); dense GQA at 1M is intractable. Bound `VLLM_DECODE_BLOCK_BUCKET_MAX`.

## 4. Per-layout summary (HBM/card + aggregate concurrency)

| Layout | weights/card | KV headroom/card | agg. concurrency @32k | verdict |
|---|---|---|---|---|
| **TP8×DP1 (KEEP)** | 29.5 GiB | ~57 GiB (post-graph) | ~121 | ✅ holds experts once; most KV headroom; one engine |
| TP4×DP2 | 59.0 GiB | ~28 GiB | ~117 | ties concurrency, ×2 expert HBM, external-DP ops, no win |
| TP2×DP4 / TP1×DP8 | 118 / 236 GiB | — | — | ❌ do not fit |

---

## VERDICT

**KEEP TP8 + `--enable-expert-parallel`.** It is the only layout that both fits with large KV
headroom and holds the 128 experts a single time; the sole DP alternative that fits (TP4×DP2)
ties on aggregate concurrency while doubling expert HBM and adding dead-DP operational surface,
and the DP layout the brief hoped for (TP2×DP4) is **physically impossible** at 428B (118 GiB/card).

**Confidence: high.** Fit math is hard (config-derived, cross-checked vs GLM/Qwen on-disk int4).
The one open empirical question — whether halving TP collective fan-out (TP4) outruns expert
duplication at this 5.3%-active point — is worth a single A/B once the int4 export exists, but the
expected value favors TP8.

**#1 highest-impact change: raise `--max-num-seqs` 64 → 128 (sweep to ~192) with
`VLLM_DECODE_BS_BUCKET_MAX=max_num_seqs`.** Placement is already optimal; the box's proven
throughput scaling (GLM 64→128 = 486→793 tok/s on this exact engine) lives in batch saturation,
and KV at TP8 is cheap enough (~480 MiB/seq/card @32k) to afford it. Aggregate throughput at 30+
concurrency is gated by steady-state batch fullness, not by the TP/DP split.

### Caveats / dependencies
- **No int4 export yet** (bf16 download 16/59 shards). All-bf16 (~794 GiB) is **TP8-only** — DP is
  moot until the int4 quant lands. Numbers above assume the int4-mixed serve weights.
- **MSA sparse kernel not built** (`hpu-msa-ext` absent); phase-1 is **dense GQA**, so long context
  is capped by dense-attention cost, not by this layout choice. 1M context is an MSA deliverable.
- Re-confirm vs a quiet fabric: the box's recent throughput numbers were storm-degraded
  (PROJECT_FINDINGS §9/§12); the OAM reseat (gaudi-fabric-defect-rma) gates any clean TP8 bench.
