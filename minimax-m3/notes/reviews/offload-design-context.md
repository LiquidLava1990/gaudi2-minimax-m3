# Offload design — MAX CONTEXT × CONCURRENCY via TIERED MEMORY (MiniMax-M3 MSA)

**Designer priority:** push `context × concurrency` as far past **1M×15** as the storage hierarchy
allows, by holding KV in a 3-tier cache (HBM hot / host-RAM warm / NVMe cold) and exploiting MSA's
top-16-block selection to keep the per-step hot working set tiny.

**Date:** 2026-06-14. **Verdict up front:** §7. Numbers are config- + hardware-derived (see §0).
This is one of three competing designs; it makes the strongest case for the tiered-memory lever.

---

## 0. The numbers this rests on (all verified)

| Quantity | Value | Source |
|---|---|---|
| Main GQA K+V, **fp8** | **60.00 KiB/tok** (2·60·4·128·1B) | config.json L=60 Hkv=4 d=128 |
| Main GQA K+V, bf16 | 120.00 KiB/tok | — |
| Main K+V, **TurboQuant k8v4** (K fp8, V int4) | 45.00 KiB/tok | `kernel-kv/` (built, `KVQ_READY`-gated) |
| **Index K cache, bf16** (57 sparse layers, 1 head) | **14.25 KiB/tok** | `index_cache_msa.py:31` bf16, separate spec |
| TOTAL fp8 + index | **74.25 KiB/tok** (76.0 MB / Mtok) | build-kv-envelope.md §1 |
| HBM KV pool | **~337 GB** (691 usable − 285 wt − 69 graph) | build-kv-envelope.md §2 |
| Host RAM free | **444 GiB avail** (`free`: 444Gi) ≈ 477 GB | live box |
| NVMe `/mnt/data2` (Intel P4510 4TB, Gen3 x4) | **2.5 TB free, ~3.2 GB/s** | `nvme list`, `df` |
| NVMe `/mnt/data1` (2nd identical P4510, 4TB) | **3.4 TB free, ~3.2 GB/s** | available to stripe → 6.4 GB/s |
| PCIe HBM↔host | ~28 GB/s | brief |
| MSA selection | top-16 blocks = **2048 tok/query**, known **1 step ahead** | MSA_SPEC §2-3 |

**The one fact that reorganises the whole design (proven below):** the **index K cache cannot be
tiered.** Every decode step the indexer dots `q_idx` against **all** blocks across 57 sparse layers —
a *full sequential scan* of the per-seq index cache, every step. The main K/V, by contrast, reads
only the 16 selected blocks and selection is known a step ahead → it is the tier-able half. So the
right split is **index = pinned HBM; main K/V = tiered host→NVMe**, and tiering main K/V is precisely
what frees the HBM to pin a *larger* index — i.e. tiering is what buys the context×concurrency.

---

## 1. The 3-tier layout (what lives where)

```
                         per-token       access pattern              tier policy
  ┌─ HOT  (HBM, ~337 GB) ───────────────────────────────────────────────────────────────┐
  │  • INDEX K cache (bf16, 14.25 KiB/tok)  FULL-SCAN every step  → PINNED, never evicts  │
  │  • MAIN K/V working set (fp8): the union of recently-selected top-16 blocks per seq   │
  │      (~120 MiB/seq steady; churns a few blocks/step) → LRU-resident, prefetch-filled  │
  └──────────────────────────────────────────────────────────────────────────────────────┘
  ┌─ WARM (host RAM, ~444 GiB) ────────────────────────────────────────────────────────────┐
  │  • MAIN K/V blocks not currently hot but cheap to restage (PCIe 28 GB/s). The bulk of   │
  │    each long sequence's K/V lives here. Holds ~6.3 M tok of main K/V (fp8).             │
  └─────────────────────────────────────────────────────────────────────────────────────────┘
  ┌─ COLD (NVMe /mnt/data2 [+ data1 striped], 2.5–5.9 TB) ─────────────────────────────────┐
  │  • MAIN K/V for the far tail of very long sequences. Read only on a COLD BLOCK FAULT    │
  │    (the indexer selects a far block not in HBM/host). NVMe ~3.2 GB/s (6.4 striped).     │
  └─────────────────────────────────────────────────────────────────────────────────────────┘
```

**Index never leaves HBM.** Quantifying why (full-scan/step if offloaded):

| ctx×conc | index size | full-scan over PCIe | over NVMe(6.4) | usable? |
|---|---|---|---|---|
| 1M×15 | 204 GiB | 7.8 s/step | 34 s/step | **no** |
| 1M×30 | 408 GiB | 15.6 s/step | 68 s/step | **no** |

A 14 tok/s target is a ~70 ms step. Offloading the index turns every step into 8–68 **seconds**. The
index is non-negotiably HBM-resident; its footprint is therefore the **hard ceiling** on ctx×conc.

### Promotion / eviction policy (MSA-selection-driven)

1. **Prefetch on selection (the key move).** MSA computes the top-16 block ids for step *t+1* during
   step *t* (selection is one step ahead — MSA_SPEC §2d/§3). The allocator issues async copy-up of any
   selected block not already in HBM, from warm (PCIe) or cold (NVMe), to land before step *t+1*'s
   consume. This hides staging behind compute.
2. **LRU demotion HBM→host.** A main-K/V block not selected for *N* steps is evicted to host RAM
   (still instantly restage-able). The **forced local block** rolls forward 1 every 128 steps; the 15
   score-selected blocks are near-stationary for semantically-stable spans → low churn.
3. **host→NVMe spill** only when host RAM fills (sequences whose total main K/V exceeds ~6.3 M tok
   across the batch). Cold blocks are write-back on demotion, read on fault.
4. **Block-reuse locality is the cold-tier savior.** Consecutive decode steps advance 1 token and
   reselect *mostly the same* 16 blocks; queries within a step (and across nearby steps) overlap
   heavily. So NVMe is touched only on the *few* newly-selected far blocks per step, not the full set.

---

## 2. How far ctx×conc scales at each tier boundary

The ceiling is **index-HBM-pinned**: `ctx × conc ≤ (HBM index pool) / 14.25 KiB`. Tiering main K/V to
host+NVMe is what *frees* HBM for index, so the design's whole value is moving the main K/V out.

**Reserve ~30 GB HBM for the hot main-K/V working set; the remaining ~307 GB pins index.**

| Index precision | HBM index pool | **ctx×conc ceiling** | Examples |
|---|---|---|---|
| **bf16 (today)** | 307 GB | **21.0 M tok** | **1M×21**, 2M×11, 4M×5, 512K×42 |
| **fp8 index** (½ footprint; needs a tiny fp8 index path) | 307 GB | **42.1 M tok** | **1M×42**, 2M×21, 4M×10 |

Where the **main K/V** lives at each point (tiered; fp8 60 KiB/tok):

| Boundary | main-K/V total | resides in |
|---|---|---|
| ≤ 4.43 M tok | ≤ 337 GB | (could fit HBM, but we spend HBM on index) |
| ≤ 6.27 M tok | ≤ 477 GB | **host RAM warm tier** (PCIe restage) |
| ≤ 32.9 M tok | ≤ 2.5 TB | **+ NVMe data2 cold** |
| ≤ 77.6 M tok | ≤ 5.9 TB | **+ NVMe data1 striped cold** |

### Fit table (ctx × conc → total tokens → tier of the MAIN K/V; index assumed pinned & in budget)

| Config | total tok | main-K/V bytes | main-K/V tier | feasible? |
|---|---|---|---|---|
| **1M × 15** (baseline) | 15 M | 0.92 TB | host(0.48TB)+**NVMe cold** | ✅ index 214 GB fits |
| **1M × 21** | 21 M | 1.29 TB | host + NVMe cold | ✅ **bf16-index ceiling** |
| 1M × 30 | 30 M | 1.84 TB | host + NVMe cold | ❌ **index 428 GB > 337 HBM** (needs fp8 index) |
| **1M × 42** | 42 M | 2.57 TB | host + NVMe(2-drive) cold | ✅ **only with fp8 index** |
| 2M × 8 | 16 M | 0.98 TB | host + NVMe cold | ✅ index 233 GB fits |
| 2M × 11 | 22 M | 1.35 TB | host + NVMe cold | ✅ bf16-index ceiling |
| 2M × 21 | 42 M | 2.57 TB | host + NVMe(2-drive) | ✅ fp8 index |
| 4M × 5 | 20 M | 1.23 TB | host + NVMe cold | ✅ bf16-index ceiling |
| 1M × 100 | 100 M | 6.1 TB | overflows NVMe | ❌ index 1.4 TB ≫ HBM anyway |

**Headline:** past 1M×15 we reach **1M×21 / 2M×11 / 4M×5 today (bf16 index)**, and **1M×42 / 2M×21**
if the index cache is made fp8. NVMe is what holds the *main K/V* tail; **HBM index is the true cap.**

---

## 3. Throughput cost per tier (tok/s)

Decode step = `max(compute, uncovered_stage)`. Baseline batch decode step ≈ **140 ms** on this box
(int4 MoE, 22.6B active, sync/collective-bound — consistent with GLM traces). Prefetch (1 step ahead)
hides staging when stage < step.

| conc | HBM-hit | **HOST-warm (PCIe)** | NVMe2-cold *fault-only* (locality) | NVMe1-cold *full 16-blk* (worst) |
|---|---|---|---|---|
| **15** | 140 ms → **107 tok/s** | 67 ms hidden → **107 tok/s** | 74 ms hidden → **107 tok/s** | 590 ms → **25 tok/s** |
| **30** | 140 ms → **214 tok/s** | 135 ms hidden → **214 tok/s** | 147 ms → **~203 tok/s** | 1180 ms → **25 tok/s** |

- **HBM-hit and HOST-warm are FREE** (PCIe stage of the full 2048-tok×conc selection fits inside one
  140 ms step → fully prefetch-hidden). The warm tier is, for throughput purposes, as good as HBM.
- **NVMe-cold is usable ONLY because of block-reuse locality.** If only ~4 of 16 blocks fault to NVMe
  per step (the realistic steady state on a 2-drive stripe), staging stays hidden → ~107/203 tok/s.
- **NVMe-cold full-restage is the cliff:** if a step must pull *all* 16 selected blocks for every seq
  from a single NVMe drive (e.g. a context-switch / fresh long sequence with no warm history), the
  step balloons to **0.6–1.2 s → 25 tok/s**, a **4–8× slowdown**. **Be honest: cold-tail bandwidth is
  the design's soft floor.** It is acceptable for steady long-context decode (locality holds) and
  *unusable* for workloads that thrash the far tail (random long-range access, many cold seqs/step).

---

## 4. Quantitative bandwidth per tier + the index cache's place

| Tier | capacity | bandwidth | per-step hot-read budget (conc=30, top-16 = 3.6 GiB) | role |
|---|---|---|---|---|
| **HBM hot** | 337 GB | ~2 TB/s (on-die) | instant | **index (pinned)** + main-K/V working set |
| **Host warm** | 444 GiB | **28 GB/s PCIe** | 3.6 GiB → **135 ms** (hidden at ≥140 ms step) | bulk main K/V, restage-on-select |
| **NVMe cold** | 2.5–5.9 TB | **3.2 / 6.4 GB/s** | full 3.6 GiB → 590–1180 ms; **fault-only ~150 ms** | far-tail main K/V |

**Index's place: HBM, always, full-scan every step** — it is the reason ctx×conc is HBM-capped, not
NVMe-capped. The brief's "warm tier hit rate can be high" applies to the **main K/V** (block-reuse),
not the index. Making the index **fp8** (7.12 KiB/tok) is the single highest-leverage change: it
doubles the ceiling (1M×21 → 1M×42) for the price of a small fp8 index-K write/read path.

---

## 5. The plumbing

1. **Tiered paged allocator** (extends `index_cache_msa.py`'s spec/bind pattern). Today the main GQA
   KV is one flat HBM pool via `FullAttentionSpec`. Add a `TieredKVCache` that owns three backing
   stores per logical block table: HBM tensor, a pinned host-RAM tensor (`torch.empty(..., pin_memory)`
   for async H2D), and an NVMe-backed mmap/`O_DIRECT` file on `/mnt/data2` (optionally striped with
   `/mnt/data1`). A per-block **residency map** (block_id → {HBM,HOST,NVMe} + slot) drives gather.
2. **Cache-spec changes.** Keep `index_kcache` as `FullAttentionSpec(bf16, head=128)` **pinned in
   HBM** (it already is — `index_cache_msa.make_index_spec`); register the **main** GQA KV under a new
   `TieredAttentionSpec` whose `get_kv_cache_shape` allocates only the HBM *hot* slab and declares the
   host/NVMe slabs as overflow arenas the migration op fills. Mirror `cache_dtype_str` so the runner's
   merge stays uniform (the DSA precedent, `indexer_cache_hpu.py:99-112`).
3. **The migration op** (`kv_migrate`): given the next-step selection `block_idx[Sq,16]` (already
   produced by `msa_block_indexer_topk` one step ahead — `msa_m3_wire.py` step 3), diff against the
   residency map and issue async `copy_(non_blocking=True)` HBM←host (PCIe) / a batched NVMe read for
   cold blocks, into freed HBM hot slots; enqueue LRU demotions HBM→host and host→NVMe write-backs.
   Two small OpBackend ops (analogous to `kv_rotate_quant`'s gather/scatter): `gather_blocks_to_hbm`
   and `evict_blocks_from_hbm`. The consume op (`sparse_attn`) is **unchanged** — it reads the HBM hot
   slab after migration lands.
4. **Wire seam.** In `msa_wire` (step 4-5) the block→token expansion already names the selected
   blocks; insert a `kv_migrate(block_idx_next)` call right after `force_local_sort_pad` so prefetch
   overlaps the current step. Gate behind a `TIERED_KV_READY` flag (the `KVQ_READY` safety pattern —
   `platform_patch_kvq.py`): NO-OP until the host/NVMe path is on-card-validated, so the default serve
   (single-HBM-pool fp8) is untouched.
5. **NVMe prep:** `echo none > .../queue/scheduler`, raise `read_ahead_kb`, optional 2-drive RAID0 on
   data1+data2 for 6.4 GB/s (Gaudi2_Optimization_Guide §5).

---

## 6. Honest limits (where it breaks)

- **NVMe-cold full-restage = 25 tok/s** (4–8× slowdown). Tolerable for *steady* long-context decode
  (block-reuse keeps faults to a few blocks/step) but **unusable** for far-tail-thrashing workloads.
- **Prefill is worse than decode.** A 1M-token prefill writes the *entire* main K/V + index out;
  at fp8 that is ~76 GB/seq, host/NVMe write-bound (~2.4 s/seq to host, ~24 s to one NVMe). Tiering is
  a **decode**-residency win; prefill of very long contexts is bandwidth-bound regardless.
- **Index is the cap, and it's bf16.** Without an fp8 index path, **1M×30 does not fit** (index 428 GB
  > 337 HBM). The advertised >1M×15 reach is 1M×21 today; 1M×30+ *requires* the fp8-index change.
- **PCIe contention.** The 8 cards share host PCIe lanes; 28 GB/s is aggregate-optimistic under
  concurrent H2D from all ranks. Real warm-restage headroom should be measured, not assumed.

---

## 7. VERDICT

Tiered memory is the **right and necessary** lever to push past 1M×15 — but the win comes from an
inversion of the obvious: you tier the **main K/V** (host warm + NVMe cold, made cheap by MSA's
top-16-block locality and one-step-ahead prefetch) **in order to free HBM to pin a larger index
cache**, because the index — full-scanned every step — *cannot* be offloaded and is the true ceiling.

- **Reachable today (bf16 index): 1M×21, 2M×11, 4M×5** at full-rate **107 tok/s (conc 15) / 214 tok/s
  (conc 30)** when the hot+warm tiers serve (which they do, prefetch-hidden, under locality).
- **With an fp8 index path: 1M×42, 2M×21** — a clean 2× on the ceiling for a small kernel addition.
- **Cold NVMe tail** holds main K/V out to ~78 M tok of capacity but runs at **25 tok/s** on a
  full-restage step — usable as a safety reservoir for the long tail, not as a steady serving tier.

**Build effort:** ~**7–10 days**. Tiered allocator + residency map (3), the two migrate ops + wire
seam (4), `TieredAttentionSpec` (2): ~5–6 d. fp8-index path (the ceiling-doubler): ~2 d. Validation +
NVMe tuning + the `TIERED_KV_READY` gate: ~2 d. Reuses `index_cache_msa.py` and the `kernel-kv/`
gather/scatter precedent, so no new TPC ISA.

**#1 risk:** the **index cache is HBM-bound and un-tierable**, so tiering main K/V buys *capacity*
(more total tokens parked in host/NVMe) far more than it buys *the ctx×conc ceiling* — the ceiling
moves from 1M×15→1M×21 (bf16) only because we relocate main K/V off HBM, and reaching 1M×30+ **depends
on the fp8-index change, not on the tiering itself.** If the fp8 index path is harder than estimated
(quality regression on selection, or no fp8 index op), the design tops out at ~1M×21 / 2M×11.
