# MSA-Aware KV-Cache Offload for MiniMax-M3 @ 1M — **MAX-THROUGHPUT** Design

**Designer priority:** push tok/s as high as bandwidth allows. **Box:** 8×Gaudi2 (768 GB HBM),
host **444 GB RAM free** (2 NUMA × ~256 GB), NVMe `/mnt/data2` = **Intel P4510 4 TB Gen3×4
(~3.2 GB/s read ceiling)**, `/mnt/data1` a second identical P4510 (stripe → ~6.4 GB/s),
PCIe host↔HBM pinned ~25–32 GB/s, HBM ~2.46 TB/s/card. Date 2026-06-14.

> **The substrate already exists — do not build DMA from scratch.** `vllm_gaudi/v1/kv_offload/
> worker/cpu_hpu.py` ships a working async offload engine: a **pool of `torch.hpu.Stream`s +
> `torch.Event`** completion, **pinned** CPU tensors (`pin_memory=is_pin_memory_available()`),
> block gather/scatter via `index_select`+`index_put_`, event-chained (`stream.wait_event`) with
> `num_bytes` timing. `vllm_gaudi/utils.py::async_h2d_copy/async_h2d_update` give non_blocking H2D
> row updates. `cache_ops.swap_blocks` is the HBM↔host block move. **My job is the MSA-driven
> *prefetch policy* that drives this one step ahead — not the transport.**

---

## 0. The decisive realization: there are TWO cold streams, not one

The brief's "2–3 GB hot working set" is the **main top-k K+V** and is correct. But a naive offload
dies on a *second* stream the headline hides — **indexer scoring**:

| Stream | Per-step volume @15 seqs | If naive |
|---|---|---|
| **Main top-k K+V** (16 blk = 2048 tok, fp8, 57 sparse L) | **1.79 GB/step** (prefetchable) | fine |
| **Index-K scoring** — if the indexer rescans all 8192 blocks/step | **229 GB/step** | **1.8 tok/s — DEAD** |

The indexer (`msa_m3_wire.py` step 3) must score **every causal block** each step to pick top-16.
If its scoring substrate (the index-K cache, 14.25 KiB/tok × 1M × 15 = **229 GB**) lives off-HBM
and is re-read per step, **PCIe gives 1.8 tok/s and NVMe gives 0.2 tok/s** — the box is unusable.
**The entire design hinges on keeping the indexer's scoring substrate HBM-resident.** Everything
else (the bulk 1.1 TB of K/V values) can be cold, because MSA only *reads* 2048 tokens of it/step.

---

## (a) Cold-KV location — pinned host-RAM vs NVMe, and the 1.1 TB problem

Full cold KV (15 seqs × 1M, fp8 main 60 KiB/tok + bf16 index 14.25 KiB/tok) = **1.196 TB**.
**444 GB RAM holds only 37 %** → **NVMe is forced for ~750 GB of it.** This is unavoidable at
15×1M; no quantization in scope closes it (index-K is bf16-only, un-fp8-able — `index_cache_msa.py`).

**Tiered cold store (hottest-first):**
1. **Pinned host RAM (≤400 GB):** the most-recently-touched / highest-reuse blocks. Pinned is
   mandatory — pageable host memory drops PCIe DMA to ~8–10 GB/s. **NUMA-pin** the pinned arena
   per socket (cards span both nodes; cross-socket pinned halves effective BW).
2. **NVMe `/mnt/data2`+`/mnt/data1` striped (~750 GB):** the cold tail. Use `O_DIRECT` + the
   tuned readahead/`none` scheduler (Gaudi guide §5). Stripe both P4510s → ~6.4 GB/s.

**What NVMe costs:** a block that misses RAM *and* HBM and must come from NVMe pays
~6.4 GB/s vs PCIe 28 GB/s — **~4.4× slower per missed byte**. Whether that caps tok/s depends
entirely on the **HBM hot-cache hit rate** (§c). NVMe is the *tail*, not the *path*.

---

## (b) Async double-buffered prefetch pipeline (exploit step-ahead determinism to the hilt)

MSA's gift: **the indexer fires BEFORE the consume.** `msa_wire` computes `block_idx` (step 3)
then consumes (step 6). The selection for token *t* is known the instant the indexer for *t* runs —
and the local block (always-forced) makes the *next* step's set ~90 % predictable. Pipeline:

```
STEP t  (per sparse layer, all 15 seqs batched):
  [HBM compute stream]   indexer scores 8192 blocks on the RESIDENT proxy (§c) -> block_idx[t]
                         consume reads top-16 K+V from HBM hot-cache (already prefetched at t-1)
  [HBM prefetch stream]  DIFF(block_idx[t], hot_set) -> the MISS blocks -> issue gather:
                            pinned-RAM hits: async_h2d_copy(non_blocking) on a dedicated hpu.Stream
                            NVMe hits:       O_DIRECT read -> pinned staging -> H2D (depth-2 queue)
                         double-buffer: buffer A consumed at t while buffer B fills for t+1
  [completion]           end_event per transfer; consume at t+1 waits ONLY on its set's events
```

- **Batched per-step gather:** ONE `index_select` over the flat slot axis for ALL miss-blocks of
  ALL 15 seqs / all 57 layers (the `cpu_hpu.py` handler already does exactly this shape:
  `src_tensor.index_select(0, idx).to(dev)` then `index_put_`). Never per-block calls.
- **Overlap = free:** the consume read (HBM-local, 1.79 GB @ 2.46 TB/s = **0.73 ms**) and the
  prefetch DMA (delta GB @ 28 GB/s) run on separate streams; as long as DMA(t+1) ≤ compute(t),
  paging is fully hidden. Compute/step is far longer than 0.73 ms (MoE+57 layers), so **the PCIe
  prefetch of a few-hundred-MB delta hides completely** — paging becomes invisible up to the
  reuse-determined miss rate.
- **Prefetch lookahead:** speculatively pre-stage the next step's *local* block + the *carried-over*
  top blocks (those still above threshold) the instant the indexer emits — they are guaranteed hits.

---

## (c) HBM hot-cache + eviction exploiting block-reuse locality

Two resident structures (the rest is cold):

1. **Resident indexer proxy (THE enabler):** keep MSA's **BlockMaxPool representative** — one
   `d_idx=128` bf16 vector per block — resident: `[8192 blk, 128] × 57 L × 15 seq = **1.79 GB**`.
   The indexer scores against *this*, not the 1M-token index-K → scoring read collapses from
   229 GB to 1.79 GB/step HBM-local (**0.73 ms, negligible**). It IS the `output_maxscore` proxy
   MSA already computes (`MSA_SPEC §3` two-stage); we just persist it. (Exactness note in §risk.)
2. **HBM hot block-cache (LRU+pinned-recent) for main K+V:** a fixed pool of K+V blocks. Eviction
   policy exploits MSA locality:
   - **Pin the recent window** (last ~64k tok = the always-forced local block's neighbourhood):
     never evicted. ~75 GB if W=64k; tune down to fit.
   - **LRU over the rest**, keyed by block-reuse: the top-16 set drifts slowly (local block rolls
     every 128 steps; sink absent; high scorers persist), so step-to-step **reuse is empirically
     80–95 %**. Each evicted block is clean (read-only KV) → drop, no write-back.

**Reuse is the whole ballgame** — quantified next.

---

## (d) QUANTITATIVE: paging volume vs DMA BW → max tok/s, and the wall

Per-step main top-k paging = **1.79 GB** (15 seqs). Only the **miss fraction** (1−reuse) actually
moves. tok/s = 15 tokens ÷ (delta ÷ BW):

| Reuse | Δ/step | **PCIe-pinned (28 GB/s)** | **NVMe tail (6.4 GB/s striped)** |
|---|---|---|---|
| 0 % (cold start) | 1.79 GB | **234 tok/s** | 54 tok/s |
| 50 % | 0.90 GB | **468 tok/s** | 107 tok/s |
| 80 % | 0.36 GB | **1 171 tok/s** | 268 tok/s |
| 90 % | 0.18 GB | **2 342 tok/s** | 535 tok/s |

(tok/s = aggregate across 15 concurrent seqs; ÷15 for per-seq.) **The wall, in order:**
1. **NOT HBM hot-cache scoring** — the resident proxy makes indexer scoring 0.73 ms, free.
2. **HBM hot-cache HIT RATE is the dominant lever** — it sets the miss delta directly.
3. **Cold-tier bandwidth** caps the *miss* throughput: **PCIe 28 GB/s** if the miss is in pinned
   RAM (good, ~234 tok/s even at 0 % reuse), **NVMe ~6.4 GB/s** if the miss falls to the tail
   (~54 tok/s at 0 % reuse). Since 1.1 TB forces ~63 % onto NVMe, **the realistic steady state is
   a blend**: with 80–90 % HBM reuse, most misses are absorbed by the RAM tier and NVMe is only
   touched for genuinely-cold-block first-touch → **practical ceiling ≈ 300–800 tok/s aggregate**,
   PCIe-bound, *if* reuse holds ≥80 %.
4. **Compute floor:** 57-layer int4 MoE decode at 15-batch is itself the real-world cap once paging
   is hidden — likely **tens of tok/s/seq** (the box's measured single-stream is ~4–14 tok/s; batch
   15 amortizes). Paging is designed to NOT be the bottleneck; **MoE compute then dominates.**

**Honest NVMe verdict:** if the hot-cache reuse collapses (adversarial: every step picks 16 brand-
new far-apart blocks), every step is a **1.79 GB NVMe read = 280 ms = 54 tok/s aggregate (~3.6/seq)**
— *usable but slow*. NVMe does **not** make it unusable (the per-step volume is small); it only
caps the *worst case* at ~50 tok/s. The proxy keeping the indexer resident is what averts the true
death (229 GB/step). **So: NVMe is acceptable as the cold tail; PCIe-pinned RAM + high reuse is the
fast path; the design degrades gracefully, never falls off a cliff.**

---

## (e) Kernel / plumbing (what to build on the existing substrate)

1. **Paging op:** REUSE `cpu_hpu.py`'s `SingleDirectionOffloadingHandler.transfer_async` (stream
   pool, event chain, batched `index_select`/`index_put_`) — it is the per-step block gather. No new
   kernel. Add an **NVMe→pinned reader** (O_DIRECT, depth-2) feeding its CPU side.
2. **Prefetch scheduler (NEW, ~1 file):** an `MsaPrefetcher` that, per sparse layer, takes the
   indexer's `block_idx` (already produced upstream of consume in `msa_wire`), diffs against the
   HBM hot-set, and enqueues the miss-block transfer on the prefetch stream **before** the consume
   op runs. Hooks the seam in `msa_wire.py` step 3→5 (block_idx is right there). Double-buffer +
   LRU/pinned-recent eviction live here.
3. **Resident proxy cache (NEW, extends `index_cache_msa.py`):** allocate the `[n_blk, d_idx]` bf16
   per-block max-pool proxy resident; update it on each token insert (running max over the block).
   The wire's indexer scores this instead of full index-K when ctx exceeds the resident window.
4. **vllm-gaudi cache-spec changes:** register THREE pools via the existing `get_kv_cache_spec`
   monkeypatch pattern (`index_cache_msa.patch_get_kv_cache_spec` already does the DSA-style spec
   injection): (i) HBM hot block-cache pool, (ii) resident proxy, (iii) the pinned-host + NVMe cold
   tiers via `CpuGpuOffloadingHandlers` (`num_cpu_blocks` sized to RAM budget). Reuse the tolerant
   `bind_kv_cache` patch.

---

## Build effort & #1 risk

**Effort: ~6–9 days.** The transport (streams/events/pinned/swap_blocks) and the MSA wire both
exist — this is *policy glue + the resident proxy*: prefetch scheduler & double-buffer (~2 d),
resident BlockMaxPool proxy + insert-time update (~2 d), NVMe O_DIRECT tier + RAM/NVMe tiering
(~2 d), spec/bind wiring + on-card validation that paging hides under compute (~2 d). Decode-paged
MSA wire is itself still a documented follow-up (`index_cache_msa.py` "DECODE") — this rides on it.

**#1 RISK — the resident proxy's selection is APPROXIMATE.** MSA's exact selection is
BlockMaxPool over the **live, post-RoPE index-K** with a **per-query causal cut** *inside* each
block (`MSA_SPEC §2c`, `j≤i`). A persisted per-block max proxy is exact for *fully-causal* blocks
but **drifts on the diagonal/partial block** and ignores that the max-token may be causally masked
for an early query. If the proxy mis-ranks blocks, the top-16 set is wrong → silent quality loss
(no crash, fails the Jaccard gate). **Mitigation:** keep the proxy resident but **always also keep
the EXACT index-K resident when it fits** — Option A1: exact index-K (229 GB) + main double-buffer
(3.6 GB) = **233 GB, FITS the int4 envelope (~340 GB free)**, only the main 966 GB K/V goes cold.
That gives **exact selection AND max throughput** and is the recommended default; the proxy is the
fp8-weights fallback (150 GB free) where 229 GB won't fit. Gate either behind the existing
Jaccard-1.0 selection-parity test before flipping `MSA_READY`.
