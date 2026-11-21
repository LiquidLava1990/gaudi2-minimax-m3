# Adversarial Review — KV + Long-Context Fit Envelope

**Lens:** KV memory vs the (concurrency × context) target. **Target:** MiniMax-M3 on 8×Gaudi2
(768 GB nominal), long context + high concurrency (30+), minimal quality loss.
**Date:** 2026-06-14. **Verdict up front:** see §6.

---

## 1. The per-token KV cost — the brief UNDERCOUNTS it

M3 is GQA (H_q=64 / **H_kv=4** / d=128), **60 layers**, all storing **full, uncompressed K/V**
(MSA_SPEC §1: selection cuts *compute*, never the stored K/V — `sparse_ref` gathers full per-block
K/V; no MLA/`kv_lora` fields in config — verified). So the main KV is ordinary GQA KV over all 60
layers. **MSA does not reduce KV-cache memory at all.**

But MSA adds a second cache the brief's headline number omits: the **index K cache** — one shared
index-key head (1×d_idx=128) per token, for the **57 sparse layers (3–59)**, stored **bf16** and
**NOT fp8** (`index_cache_msa.py:31` `INDEX_CACHE_DTYPE = torch.bfloat16`; it is a separate
`FullAttentionSpec`, not routed through `--kv-cache-dtype`). This is a hard, un-quantizable floor.

| Component | Formula | bf16 | fp8_inc |
|---|---|---|---|
| Main GQA K+V (60 L) | 2·L·H_kv·d·bytes | 120.00 KiB/tok | 60.00 KiB/tok |
| **Index K cache (57 L, bf16 always)** | 57·1·d·2 | **14.25 KiB/tok** | 14.25 KiB/tok (no fp8) |
| **TOTAL effective** | — | 134.25 KiB/tok | **74.25 KiB/tok** |

→ The real fp8_inc KV cost is **74.25 KiB/token, not 60.** The index cache is a fixed **+23.75 %**
surcharge on top of fp8 main KV and **does not shrink with fp8_inc** — it dominates more as you push
fp8. (Bytes/tok: main fp8 = 61 440; index bf16 = 14 592; total = 76 032 B.)

---

## 2. HBM budget left for KV

8×96 = 768 GB nominal · `--gpu-memory-utilization 0.9` → **691 GB usable**.
`VLLM_GRAPH_RESERVED_MEM` default **0.10 of usable** → ~69 GB graph reserve.
Weights ~**285 GB** (int4 experts + bf16 rest; bf16 checkpoint is 252 GB on disk, so 285 is a
conservative high estimate — see §5 sensitivity).

```
usable 691  − weights 285  − graph 69  =  KV pool ≈ 337 GB   (central estimate)
```
**Total token budget across all sequences (fp8+index): ≈ 4.43 M tokens.**

---

## 3. The envelope (fp8_inc main KV + bf16 index, KV pool = 337 GB)

| Context / seq | GB per seq | **Max concurrent seqs** |
|---|---|---|
| 32 K | 2.49 | **135** |
| 128 K | 9.97 | **34** |
| 256 K | 19.93 | **17** |
| 1 M | 79.73 | **4** |

Concurrency-first view (token budget ÷ conc):

| Concurrency | **Max ctx / seq (fp8+idx)** | (bf16+idx, for contrast) |
|---|---|---|
| **30** | **144 K** | 80 K |
| 32 | 135 K | — |
| 48 | 90 K | — |
| 64 | 68 K | — |
| 128 | 34 K | — |

---

## 4. Where fp8 KV becomes the BINDING constraint

The high-concurrency-long-context target is a **product**: `conc × ctx ≤ 4.43 M tokens` (fp8+index).
KV is the binding cap (weights and graph are fixed) the moment you cross that line.

- **conc 30 @ 32 K** → 0.96 M tok — **comfortable** (fp8 ample; could push conc to ~135).
- **conc 30 @ 128 K** → 3.94 M tok — **fits, but barely (≈34 ceiling)**; essentially no headroom for
  KV fragmentation, longer outputs, or a heavier weight estimate. This is the **knee**.
- **conc 30 @ 256 K** → 7.9 M tok — **EXCEEDS the budget by 1.8×.** fp8 cannot do it; real ceiling
  is ~17 seqs @256 K.
- **conc 30 @ 1 M** → 24 M tok — off the map (~4 seqs max).

**Binding point (the exact ctx×conc where fp8 runs out at the target conc=30):
ctx ≈ 144 K. At/above ~150 K context with 30 concurrent sequences, fp8_inc KV is exhausted.**

---

## 5. Sensitivity (does the verdict hold?)

| weights | graph% | KV pool | conc@128K | maxctx@conc30 | conc@256K |
|---|---|---|---|---|---|
| 265 | 8% | 371 | 37 | 159 K | 19 |
| 285 | 10% | 337 | **34** | **144 K** | **17** |
| 305 | 10% | 317 | 32 | 136 K | 16 |

Even on the **optimistic** corner (265 GB weights, 8% graph) the conc-30 ceiling is **~159 K** and
256 K@30 never fits on fp8. The knee at ~128–150 K and the 256 K wall are **robust to ±20 GB weights
and the graph-reserve knob.**

**If sub-fp8 main KV were unlocked** (TurboQuant-class int4 main KV, index still bf16):
44.2 KiB/tok total → **1.68× more context**. conc-30 ceiling rises **144 K → ~242 K**; 256 K@30
becomes reachable (~242 K is right at the edge). Note the index cache caps the win at 1.68× (not 2×)
because it stays bf16; making the **index** fp8 too would add only ~10 %.

---

## 6. VERDICT

**fp8_inc KV is SUFFICIENT for conc≥30 up to ~128 K context. KV-minimization (sub-fp8 / TurboQuant-
class) becomes REQUIRED at ctx ≥ ~256 K with conc ≥ 30** (and for any conc-30 use beyond ~150 K).
The crossover is sharp and lives at **conc 30 × ctx ≈ 150 K (≈4.4 M total tokens)**.

- 32 K / 30+ conc: fp8 trivially sufficient (room for ~4× more concurrency).
- 128 K / 30 conc: fp8 sufficient **but at the ceiling (~34 seqs)** — treat as the design limit, not
  a comfortable point.
- 256 K & 1 M at conc 30: **fp8 cannot reach the target**; need either fewer seqs (17 / 4) or sub-fp8
  KV. On this stack sub-fp8 is **blocked by the HPU platform gate** (`is_kv_cache_dtype_supported`
  returns True only for `fp8_inc`; Gaudi2_Optimization_Guide §"KV-cache compression status") — so it
  is **engine-upgrade scope, not a flag**.

**Confidence: HIGH** on the arithmetic and the index-cache surcharge (config + code verified);
**MEDIUM** on the exact KV-pool GB (depends on the realized weight footprint and the actual graph
reserve the engine prints — read the startup `usable/graph/KV` split to confirm 337 GB).

### #1 recommendation
**Set `--max-model-len` and the concurrency target to a per-request KV budget of ≈148 K tokens
(`max_model_len × max_num_seqs ≤ ~4.3 M`, leaving ~3 % slack), and stop treating 60 KiB/tok as the
fp8 cost — it is 74.25 KiB/tok because the bf16 index cache is unquantizable.** If the product needs
≥256 K context at conc≥30, fp8_inc is insufficient and a sub-fp8 main-KV path (TurboQuant-class)
must be unblocked at the platform gate — that, not the index cache, is the only lever that buys the
1.68× needed. Until then, position M3 as **"≤128 K context at 30+ concurrency"** and validate the
real KV split from the engine's startup memory report before committing SLAs.
