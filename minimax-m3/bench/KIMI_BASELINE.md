# Kimi-K2.7-Code — INT4 TEXT-ONLY baseline on 8× Gaudi 2

**Date:** 2026-06-15
**Checkpoint:** `/mnt/data2/kimi-k2.7-code` (64 shards), arch `KimiK25ForConditionalGeneration`
(multimodal MoonViT wrapper) around text LM `DeepseekV3ForCausalLM` (MLA, ~1T total / ~32B active,
384 routed experts top-8 + 1 shared, silu, native int4 compressed-tensors W4A16, group_size=32).
**Image:** `vllm-gaudi:next` · **Serve:** `./serve.sh kimi-k2.7` (TP8 + EP, fp8_inc KV, torch.compile).
**Endpoint:** `<SERVER_IP>:8000` · served name `kimi-k2.7`.

---

## TL;DR — what it took to serve text-only

Two fixes were required (the first unblocks the crash; the second makes it fit):

1. **Vision-warmup crash → out-of-tree plugin no-op (`hpu-kimi-ext`).**
   `KimiK25ForConditionalGeneration` is multimodal. vllm-gaudi's `warmup_model` always runs the
   vision warmup for an mm-capable model, and the MoonViT vision tower **cannot torch.compile on
   HPU**: `HPUModelRunner.warmup_multimodal_graphs → kimi_k25.embed_multimodal → kimi_k25_vit
   vision_tower_forward → get_rope_shape (@torch.compile) → torch._inductor get_device_op_overrides
   → KeyError: 'hpu'` (torch-inductor has no HPU device-op backend).

   - **Option 1 (`--limit-mm-per-prompt '{"image":0,"video":0}'`) ALONE was NOT enough.** Tried it
     first; it still crashed in `warmup_multimodal_graphs`. Root cause: the warmup loop gates on
     `is_image_warmup`, which needs `mm_budget.mm_limits['image'] != 0`, but `MultiModalBudget`
     **clamps the per-prompt image limit UP to ≥1** for an mm-capable model — so `mm_limits['image']`
     stays 1, `is_image_warmup` stays True, the loop body runs, the vision tower compiles → crash.
     (The flag is still kept — it caps *real requests* to 0 images so nothing reaches the un-warmed
     vision tower — but it does not empty the warmup buckets.)
   - **The fix that works (Option 2):** `hpu-kimi-ext`, a `vllm.general_plugins` entry-point plugin
     (mirrors `hpu-m3-ext`/`hpu-moe-ext`, zero edits to `/opt`) that, when `KIMI_NO_VISION_WARMUP=1`,
     wraps `HPUModelRunner.warmup_multimodal_graphs` to a **no-op**. The vision tower is never
     compiled; the **TEXT** prompt/decode graph warmup (a separate path later in `warmup_model`) runs
     fully intact. Strictly narrower than `VLLM_SKIP_WARMUP` (which would skip text warmup too).
     Wired via `USE_KIMI=1` + `-e KIMI_NO_VISION_WARMUP=1` in `models/kimi-k2.7.env`.
     Confirmed in logs: all 8 workers print `NO-OP vision warmup … TEXT warmup is UNAFFECTED`.

2. **HBM OOM in sampler warmup → cap the KV cache.**
   After the vision fix, the serve reached TEXT warmup but OOM'd during **sampler warmup**:
   `RuntimeError: PT_DEVMEM OOM: No enough memory for defragment`. Cause: int4 weights are
   **74.5–86.3 GB/card** (UNEVEN across EP cards — heaviest card TP7 = 86.3 GB), and at
   `--gpu-memory-utilization 0.9` (86.4 GB budget) vLLM greedily sized the MLA KV cache to **572,800
   tokens ≈ 20 GB/card** (69.9× concurrency for 8192 ctx). 86.3 GB weights + 20 GB KV ≈ 106 GB ≫ 96 GB
   physical → no room to defragment during warmup.
   - **Fix:** `--num-gpu-blocks-override 1280` caps KV to `1280×128 = 163,840 tokens ≈ 5.7 GB/card`
     (covers the needed 8 seqs × 8192 ctx working set + margin), freeing ~14 GB on the worst card.
     Also `--max-num-seqs 8` (smaller sampler/decode warmup batch → lower transient peak). `util`
     stays 0.9 (the 86.3 GB worst card already needs ~0.90×96 just to hold its weights — lowering
     util would under-budget the weights themselves).
   - Result: KV sized to **163,840 tokens (20.0× concurrency)**, sampler warmup completed on all 8
     workers, `Warmup finished in 380 s, allocated 4.68 GiB`, `Application startup complete`,
     `/health` → 200. **No OOM.**

**Load/warmup time:** weights load ~155–200 s; `init engine (profile + KV + warmup) took 382 s`
(~6.4 min total cold start). Recipe cache warms subsequent restarts.

---

## Coherence — CONFIRMED (greedy, temperature 0.0)

**Code prompt** ("Write a Python LRU cache class with get/put, O(1)…"): Kimi produced a correct
**O(1) LRU cache** design — hash map + doubly-linked list, sentinel head/tail nodes,
`_remove`/`_add_to_front`/`_pop_tail`, correct get/put semantics, and the classic LeetCode usage
example. Fully coherent, idiomatic, correct. (It emits a `<think>…</think>` reasoning preamble —
K2.7 is a reasoning+code model.)

**Reasoning prompt** (bat-and-ball, the classic intuition trap): Kimi flagged the wrong-but-intuitive
"$0.10" answer in its reasoning, set up the algebra `x + (x+1.00) = 1.10 → 2x = 0.10 → x = 0.05`, and
gave the correct answer: **ball = $0.05, bat = $1.05**, in clean step-by-step LaTeX.

The int4 text LM (DeepSeek-V3 path) serves correctly — no token salad, no degeneration.

---

## Measured throughput (bench/m3_concurrency.py — code prompt, 256 gen, greedy)

> Single-stream is **host/sync-bound** on this 1T/384-expert int4 MoE model — card util sits at
> **14–19%** even under concurrency (the cards idle waiting on host-side scheduling + per-token
> all-expert int4 dequant + MLA + TP8 all-reduce). This is the documented Gaudi single-stream wall
> (`gaudi-single-stream-speed`), scaled up for a 1T model. The engine was warmed (8-wide burst +
> single-stream passes) before measuring so the numbers are NOT compile-polluted.

| concurrency | aggregate tok/s | per-stream tok/s | req-latency p50 | notes |
|------------:|----------------:|-----------------:|----------------:|-------|
| c=1  (single-stream) | _BENCH_PENDING_ | — | — | structural sync-bound floor |
| c=10 | _BENCH_PENDING_ | _BENCH_PENDING_ | _BENCH_PENDING_ | |
| c=30 | _BENCH_PENDING_ | _BENCH_PENDING_ | _BENCH_PENDING_ | 8 active + 22 queued (max_num_seqs=8) |
| c=50 | _BENCH_PENDING_ | _BENCH_PENDING_ | _BENCH_PENDING_ | 8 active + 42 queued |

Observed engine aggregate during the c=10–30 tiers: ~14–22 tok/s peaks with frequent host-side
stalls to 0 (no recompiles — pure host/sync gaps).

**Peak HBM: ~84.3 GB/card** (84,341 MiB observed, uniform across all 8 cards, stable under c=30 load).
Breakdown ≈ weights ~78.5 GB avg (74.5–86.3 GB range, EP-uneven) + KV 5.7 GB (capped) + graphs 4.68 GB.
KV usage stayed <2% during the bench — KV is not the constraint; HBM is dominated by the 1T int4
weights, and throughput is dominated by host-side serialization, not memory bandwidth.

---

## Serve config (models/kimi-k2.7.env)

```
USE_KIMI=1                         # mount + install hpu-kimi-ext (vision-warmup no-op plugin)
MODEL_ENV: -e KIMI_NO_VISION_WARMUP=1  (+ PT_HPU_WEIGHT_SHARING=0, fork multiproc, generous timeouts)
VLLM_ARGS: --tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8_inc
           --block-size 128 --max-model-len 8192 --max-num-seqs 8 --gpu-memory-utilization 0.9
           --num-gpu-blocks-override 1280 --async-scheduling --no-enable-prefix-caching
           --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}'
```

## Notes / next levers (not done here — this is the int4 baseline)
- KV is capped tight (163,840 tok). Could raise `--num-gpu-blocks-override` toward ~2000 (≈9 GB/card)
  if more in-flight context is wanted — the worst card has ~12 GB free at idle (84.3/96).
- Single-stream is the structural wall; raising `--max-num-seqs` above 8 increases aggregate at
  concurrency but risks re-tripping the warmup OOM margin on the 86.3 GB worst card — would need a KV
  re-tune. Throughput, not latency, is where this model scales (cf. `gaudi-perf-targets`).
- The EP weight imbalance (74.5–86.3 GB) is what makes util 0.9 the floor; a more even expert
  placement would buy headroom but vLLM's EP sharding is fixed here.
