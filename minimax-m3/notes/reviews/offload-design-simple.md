# MSA-aware KV offload for MiniMax-M3 @ 1M — the SIMPLICITY + CORRECTNESS design

**Designer 3 of 3. Priority: the minimal plumbing that correctly serves 15 seqs @ 1M at
fp8 KV quality.** Modest throughput accepted. "Simplest thing that could possibly work."
Date 2026-06-14. Target: 15 concurrent sequences, 1M context each.

---

## 0. The decisive arithmetic (drives every choice below)

Per-token KV (verified vs `build-kv-envelope.md` + config): main fp8 = 60.00 KiB (K+V, 60 L);
index-K bf16 = 14.25 KiB (1 head, 57 sparse L). **Split the KV by who reads it each step:**

| Tier | What | 15×1M size | Where |
|---|---|---|---|
| **Resident-A** | index-K, all 57 sparse L (the SELECT scan reads ALL of it) | **204 GiB** | **HBM** |
| **Resident-B** | dense-layer main K+V, 3 L (layers 0–2 attend FULL history) | **43 GiB** | **HBM** |
| **Cold-C** | sparse-layer main K+V, 57 L (MSA reads only top-16 blocks) | **815 GiB** | **NVMe** (host RAM is only 451 GiB → too small) |

**Resident floor = 247 GiB.** This FITS only under **int4 weights** (~340 GiB KV pool); the fp8-weights
path (~150 GiB pool) does **not** fit the floor → **int4 weights are a hard prerequisite.** Cold-C
(815 GiB) is the only thing offloaded, and only its MSA-selected blocks ever move.

The crucial simplification: **MSA never sparsifies the dense layers or the index scan**, so those two
*must* stay hot — but they're small enough (247 GiB) to just keep in HBM permanently. We offload **only
the part MSA actually makes sparse.** No clever paging of index/dense KV at all.

## (a) Where cold KV lives

**Cold-C on NVMe `/mnt/data2` as one flat file per (layer, kv-head), fp8, slot = `global_block_id ×
128 + offset`** — identical slot convention to `IndexerKCache` (`index_cache_msa.py:43`) and the main
paged cache, so the existing `slot_mapping`/`block_table` index it unchanged. 815 GiB of 3.3 TB. Use a
single `np.memmap` (or `O_DIRECT pread`) per shard; the OS page cache (the 451 GiB host RAM) becomes a
**free L2** — recently-touched cold blocks stay in RAM, so re-reads hit ~28 GB/s PCIe, not 5 GB/s NVMe.
**No bespoke host-RAM tier built — `mmap` + page cache IS the tier.**

## (b) How MSA-selected blocks page into HBM per step

A fixed **HBM hot pool** = a paged KV region sized for `15 seqs × 16 blocks × 57 sparse-L × (K+V fp8)`
= **1.67 GiB**, double-buffered → **~3.4 GiB**. Per decode step, per request:
1. The wire already computes `block_idx[Sq, 16]` one layer at a time (`msa_m3_wire.py` step 3,
   `force_local_sort_pad`). Union the 16 selected **global** block ids across the 57 sparse layers.
2. For each selected global block **not already in the hot pool**, copy its fp8 K+V from the NVMe
   memmap into a free hot-pool slot (`tensor.copy_` H2D, or `index_copy_`).
3. **Rewrite the request's `block_table` row** so the 16 logical blocks point at the hot-pool slots.
   The existing paged-attn consume (`sparse_attn` op, `K=2048`) then reads HBM exactly as today —
   **zero kernel change.** The block→token expansion and gather are already in the wire.

This is the entire mechanism: **a memmap, a copy, and a block_table indirection.** The MSA kernel,
the index cache, and the consume op are all untouched.

## (c) Latency hiding via the step-ahead index

The indexer runs **before** the consume (the whole point of the lever). Concretely: in `msa_wire`,
after `force_local_sort_pad` produces `block_idx` for a layer, **issue the NVMe→HBM copies for layer
L+1's blocks on a side HPU copy-stream while layer L's consume runs on the compute stream.** Because
all 57 layers' selections for the *current* token are computed in the same forward pass, we actually
get **57-deep prefetch within one step** — the first layer's copy is the only unhidden one; layers
2–57 overlap their copy with the prior layer's MoE+attn compute. Implementation = one extra
`torch.hpu.Stream` + a per-layer "blocks needed" list the wire already has. The MoE compute per layer
(22.6B active params, the dominant cost) is far longer than a 16-block copy, so **paging hides behind
compute whenever churn ≤ ~4 new blocks/layer/seq** (see (e)).

## (d) HBM hot-cache size + eviction

Hot pool = **3.4 GiB** (1.67 GiB working set × 2 for double-buffer). Eviction = **LRU by global block
id**, but MSA makes it nearly free: top-16 selection is dominated by the **forced-local block** (always
the growing tail, 1 truly-new block per 128 steps) plus a slowly-drifting set of high-score blocks.
Keep a per-seq `{global_block → hot_slot}` dict; on miss, evict the least-recently-selected slot. **No
ref-counting, no fancy policy** — a Python dict + a free-list ring. Worst case (selection fully churns)
just re-reads, bounded by (e).

## (e) Quantitative bandwidth → throughput ceiling

Per-step paged-in volume = `new_blocks × 128 × (K+V fp8) × 57 L × 15 seqs`. Each step advances all 15
seqs by 1 token (15 tok/step). **NVMe is the floor; page-cache hits run at PCIe.**

| new blk/seq/step | vol/step | NVMe 5 GB/s | PCIe 28 GB/s (cache hit) |
|---|---|---|---|
| 1 (local-tail only) | 107 MiB | 22 ms → **669 tok/s** | 4 ms → 3748 tok/s |
| 2 | 214 MiB | 45 ms → **335 tok/s** | 8 ms → 1874 tok/s |
| 4 (realistic drift) | 428 MiB | 90 ms → **167 tok/s** | 16 ms → 937 tok/s |
| 16 (full churn, worst) | 1710 MiB | 359 ms → **42 tok/s** | 64 ms → 234 tok/s |

**Honest estimate: 4 new blocks/seq/step (forced-local + modest top-16 drift), with the page cache
absorbing most re-reads → steady state ~150–250 tok/s aggregate (10–17 tok/s/seq).** This brackets the
brief's "warm-lazy 14 tok/s" GLM baseline — i.e. **offload does not slow us below where we already are.**
The ceiling is **NVMe-bound on cold misses, compute-bound (hidden) on cache hits.** Note: the dense-layer
and index-scan work stays in HBM, so it never touches this budget.

## (f) Exact vllm-gaudi plumbing + kernel

**Reuse, with three small monkeypatches (all card-free, out-of-tree, mirroring the DSA/MSA pattern):**
1. **`get_kv_cache_spec`** — already patched by `index_cache_msa.apply()` for index-K. Add: register the
   sparse-layer main-KV as the **small hot-pool** spec (16 blocks/seq, not 8192), not a full-context
   cache. The runner allocates HBM for the hot pool only.
2. **`bind_kv_cache`** — reuse the tolerant bind already written (`index_cache_msa._bind_tolerant`).
3. **The cold store + pager** — one new module `kv_offload_msa.py` (~250 lines): opens the memmap shards,
   exposes `page_in(global_block_ids) → hot_slots` and rewrites the `block_table`. Hook it at the top of
   `msa_wire` (read `attn_module._msa_hidden`-adjacent metadata) on a side stream.

**Kernel: NONE.** The MSA indexer + consume `.so` are unchanged; the consume already reads a paged HBM
region by `token_idx`. Offload is pure Python + memmap + an HPU copy-stream.

## What I do NOT build (the simplicity dividend)
- **No host-RAM cache tier** — `mmap` + OS page cache is it.
- **No new kernel, no consume change, no indexer change** — block_table indirection only.
- **No quantization of index-K or dense KV** — they stay HBM-resident (correctness: full fp8/bf16).
- **No prefix caching, no cross-request sharing** (MSA assumes none; matches `--no-enable-prefix-caching`).
- **No async multi-step paging engine** — one side stream, per-layer copies hidden behind MoE compute.
- **No defragmenter, no ref-count GC** — a per-seq dict + free-list ring.
- **No NVMe write path optimization** — prefill writes Cold-C once, sequentially (write-once, read-sparse).

## Build effort & risk

- **Effort: ~5–7 engineer-days.** (~2 d `kv_offload_msa.py` memmap+pager+block_table rewrite; ~1 d the
  side-stream prefetch in `msa_wire`; ~1 d spec/bind reuse + hot-pool sizing; ~2 d correctness gate:
  prove the offloaded path is bit-identical to an all-HBM small-context run + a 1M smoke.) Gated behind
  `MSA_READY` like every prior step; **prerequisite = int4 weights + the on-card MSA kernel** (both in
  flight, not yet green).
- **#1 RISK: NVMe random-read bandwidth at 16-block granularity.** The 5 GB/s figure assumes large
  sequential reads; 256 KiB block reads scattered across an 815 GiB file may deliver far less IOPS-bound
  throughput, collapsing the ceiling toward ~40 tok/s on a cold/high-churn workload. **Mitigation: store
  Cold-C layer-major + block-contiguous so a selected block is one 64 KiB×57 = 3.6 MiB sequential
  pread per seq, and lean on the page cache (host RAM holds ~55% of one seq's cold KV) — but measure
  real 4K/256K NVMe random throughput on `/mnt/data2` before trusting any number above 100 tok/s.**

**Confidence: HIGH** on the tier split + resident floor (config-verified) and that no kernel is needed
(block_table indirection is the existing seam); **MEDIUM** on the tok/s (NVMe random-read dependent).
