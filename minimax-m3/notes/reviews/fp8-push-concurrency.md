# fp8 M3 — pushing the concurrency ceiling (review)

**Date:** 2026-06-15 · CARD-FREE analysis · context: M3 int4→fp8 serve hit **agg 39.9/63.4/90 tok/s @ c=10/30/50, STILL
CLIMBING at c=50**, coherent, single-stream ~4.6 (host-sync-maxed). Sources: `minimax-m3-fp8.env` (mns 64, util 0.90,
max-model-len 8192, MSA-OFF dense GQA, fp8_inc KV), `m3_expert_runtime_patch.py` (route-b UNFUSED per-expert swigluoai
loop), `int4_to_fp8_load.py`, Guide §IV.2/§IV.3, `M3_MATRIX_REPORT.md` (int4 control).

## The numbers that decide it (per-card budget, 8×96 GiB)

KV/token, **dense-GQA / MSA-OFF** (the served config — no index-K surcharge): `2·n_kv(4)·head_dim(128)·layers(60)·1B
(fp8) = ` **60.00 KB/tok exactly** (config-confirmed). Because **n_kv_heads=4 < TP=8, KV is REPLICATED on every card**
→ the per-card KV pool is the binding limit, not the 8-card sum.

| util | usable/card | fp8 weights | graph (0.1) | **KV pool/card** | seqs @ ml=8192 | seqs @ REAL 2304-tok |
|---|---|---|---|---|---|---|
| 0.90 (env) | 86.4 G | ~59 G | 8.6 G | **18.8 G** | **40** | ~143 |
| 0.94 | 90.2 G | ~59 G | 9.0 G | **22.2 G** | **47** | ~168 |

**Why 90 tok/s is still climbing at c=50:** the *running batch* is gated by `mns=64`, NOT by KV — at the real 2304-token
workload the KV pool holds ~143–168 live seqs, so c=50 has slack. The wall it is approaching is **mns=64 + whether the
per-step compute feeds a wider batch** (the route-b UNFUSED per-expert Python loop). KV only becomes the wall at FULL
8192-context (40–47 seqs).

## (1) What caps the climb
- **Not KV (yet)** for short seqs — ~143 seqs fit at 2304 tok / util 0.90.
- **mns=64** caps the running batch — at c=50 with prefill gaps the steady batch is < 64; raising the cap lets it fill.
- **The route-b all-expert PREFILL loop** (`_fp8_route_b_allexpert`: E=16 per-expert dequant+2 GEMMs over the FULL token
  block) is the throughput tax at concurrency — every 2K prefill runs all 16 local experts in a Python loop. Decode is
  the cheap routed-K(=4) path. As batch widens, **serialized 2K prefill** (single prefill stream) is the next ceiling.
- At full 8192 ctx, KV pool (40 seqs/card @0.90) caps it hard.

## (2) mns 64→128?  YES — it is the #1 lever here (unlike int4)
On the **int4** serve mns was INERT (per-token int4→bf16 dequant serialized every step → batch slots couldn't help). The
**fp8** path removes that dequant (native E4M3 MME), so the expert stage no longer serializes → **batch capacity becomes a
real lever again.** Pair with `VLLM_DECODE_BS_BUCKET_MAX=128` so warmup covers the full batch. KV is NOT the wall for the
short bench workload, so 128 should fill further. **But** at util 0.90 / ml=8192 only ~40 *full-context* seqs fit per card
— so mns=128 at 8192 risks KV-OOM for long requests. Free the KV with knob (4).

## (3) The unfused per-expert loop vs native fp8 op
- **At concurrency the all-expert prefill loop IS a real bottleneck** (16× dequant+GEMM per prefill block, Python-driven),
  unlike decode where it is only 4 captured matmuls. This is the throughput tax that keeps prefill serialized.
- **Native `bias_fp8_fused_weights` (route a) is NOT worth chasing on `:next`.** It is `frontend_blocklist:[eager]` →
  CPU composite (slow), and it is the SAME fused-swiglu family that was BROKEN on the int4 side (perm=False→~0,
  perm=True→**card hang**, §IV.8a). The env already pins route b for cause. It is only worth a probe on a **LAZY** frontend
  image — single-card, true-idle, one config at a time — before trusting it. Do NOT flip route a on this serve.
- Real prefill win if needed: **batched fp8 bmm for the all-expert PREFILL path only** (decode stays the routed-K loop —
  batching decode regressed on int4). Untested for fp8; medium-effort follow-up.

## (4) The fp8 KV-vs-throughput tradeoff — the headline knob
fp8 weights leave only ~19–22 G/card for KV (vs int4's ~58 G). **max-model-len is the dial:** the scheduler reserves
block-table KV per admitted seq against max-model-len, so **halving it ~doubles the running batch the pool admits.**
- ml 8192→**4096**: full-ctx seqs/card 40→**80** (0.90) / 47→**95** (0.94) — mns=128 now fits even at full context.
- ml→**3072** if the workload tolerates it → ~107 seqs/card @0.94.
This is exactly Guide §IV.3's "fp8 trades context for concurrency." Set `max_model_len × max_num_seqs ≲ KV_pool_toks`
(≈0.33 M @0.90, 0.39 M @0.94 per card).

## RECOMMENDED CONFIG (maximize aggregate tok/s)
```
--max-num-seqs 128                 # fp8 makes the batch lever live again (int4's was inert)
--gpu-memory-utilization 0.94      # +KV pool 18.8→22.2 G/card (int4 run proved 0.94 safe, no OOM thru c=50/150)
--max-model-len 4096               # halves KV reservation -> mns=128 fits at full ctx; raise to 6144 only if needed
-e VLLM_DECODE_BS_BUCKET_MAX=128   # warm the full batch (else a 128-wide step recompiles mid-serve)
# keep: route b, fp8_inc KV, MSA-OFF, block-size 128, --async-scheduling, no prefix cache, TP8+EP
```
If a card OOMs at load+KV, drop mns 128→96 (fp8 weights are fixed; the slack is KV), or ml 4096→3072.

## #1 change, expected ceiling, what caps it
- **#1 config change:** `--max-num-seqs 128` + `--max-model-len 4096` + `--gpu-memory-utilization 0.94` (+ DECODE_BS_BUCKET_MAX=128).
- **Expected aggregate ceiling:** ~**130–180 tok/s** at c≈80–100. The c=50→still-climbing slope says there is real
  headroom above 90; the next wall is **serialized 2K prefill** (single prefill stream) + the all-expert prefill loop,
  which flattens aggregate well before KV-OOM. Do NOT expect linear scaling past c≈100 — prefill, not KV, will cap it.
- **What caps it (in order):** (1) mns=64 → raise it; (2) serialized 2K prefill + the route-b all-expert prefill loop
  (the real flattening cost); (3) host/sync wall on the whole decode step; (4) KV pool only at full 8192-ctx.

**Confidence:**
- KV math (60 KB/tok, ~40 seqs @8192/0.90, ~143 @2304) — **HIGH** (config-derived, matches Guide §IV.3 dense-GQA case).
- mns 128 lifts aggregate (fp8 un-serializes the expert stage) — **MEDIUM-HIGH** (mechanism solid; int4-inert proves the
  dequant was the blocker; must MEASURE on-card — the int4 lesson is "test, don't assume the batch lever").
- 130–180 ceiling — **MEDIUM** (extrapolation from a still-climbing curve; prefill flattening is the uncertainty).
- Route a not worth chasing on :next — **HIGH** (blocklisted + same family that hung the card).
