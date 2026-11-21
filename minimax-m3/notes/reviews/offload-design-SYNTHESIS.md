# MSA-aware KV Offload — Synthesized Design (3-agent deliberation verdict)

Synthesized from offload-design-{simple,throughput,context}.md. This is the design to build (phase-2 of the benchmark, after c0–c5 + on-card MSA decode wire validate).

## The decisive insight (all 3 agents converged)
The **MSA index-K cache (14.25 KB/tok, 57 sparse layers, ~230 GB @ 1M×15) is FULL-SCANNED every decode step** by the indexer to score all blocks. Offloading it = seconds/step = dead. So:
- **Index-K stays HBM-resident, EXACT** (no approximate proxy → Jaccard-1.0 selection, zero quality loss).
- **Only the main K/V** (which MSA reads sparsely — 2048 of 1M tokens/step) is tiered/offloaded.

## DECISIVE CONSTRAINT: 1M offload requires INT4 weights
Resident index (~230 GB) + double-buffer fits ONLY the int4-weights HBM envelope (~340 GB free), NOT fp8 weights (~150 GB free). **→ Benchmark c7/c8 (fp8 × offload @ 1M) will OOM. The viable 1M-offload config is c6 (int4-in-HBM × offload).** This proves int4 weights are mandatory for 1M-at-scale.

## The design (graft of all three)
1. **Index-K: HBM-resident, exact** (throughput design) — 230 GB, fits int4 envelope.
2. **Main K/V (sparse layers): tiered** — host-RAM warm (444 GB) + NVMe cold (`/mnt/data2`, the 1.1 TB tail). Dense layers (0–2) + everything non-sparse stays resident.
3. **Async step-ahead prefetch** (throughput) — MSA selection known one step early → DMA the next step's miss-blocks on a dedicated HPU copy-stream, double-buffered A/B, hidden behind the dominant MoE compute. **Reuse the existing `vllm_gaudi/v1/kv_offload/worker/cpu_hpu.py`** (stream-pool + pinned + event-chained + batched index_select/index_put already ship — build policy glue, not DMA).
4. **`block_table` indirection seam** (simple design) — on each step, copy missing selected blocks → HBM hot pool (~3.4 GB, LRU, recent/forced-local pinned), rewrite the request's block_table to hot slots. **Consume op reads HBM unchanged — NO new TPC kernel.**
5. **Layout: layer-major, block-contiguous** on NVMe → one ~3.6 MB sequential pread/seq (avoid IOPS-bound random reads).

## Throughput estimate
~100–300 tok/s aggregate @ 15 seqs @ 1M. Swing factor = NVMe **random**-read at block granularity (sequential 5 GB/s ≠ random); worst case ~40–54 tok/s, never a hard cliff (per-step volume is small). 80–95% cross-step block reuse (forced-local + slow-drift) keeps NVMe faults low. **Measure real NVMe random throughput before trusting >100 tok/s.**

## Fit reach (context design)
Today (bf16 index ceiling): 1M×15 baseline, up to ~1M×21 / 2M×11 / 4M×5. 1M×30+ would need an **fp8 index cache** (2× the ceiling) — a separate follow-up, not the tiering.

## Plumbing / build
- `kv_offload_msa.py` pager (~250–400 LOC) + the tiered allocator (HBM hot + pinned-host warm + NVMe-mmap cold) + residency map.
- Reuse `index_cache_msa.py` spec/bind patches; add `TieredAttentionSpec` for main KV; a `kv_migrate` gather/evict pair driven by the one-step-ahead selection; wire seam in `msa_m3_wire.py` after `force_local_sort_pad`.
- Gate behind `OFFLOAD_READY` (default off; the DSA_READY/MSA_READY/KVQ_READY no-op-until-validated pattern).
- **Build ~6–9 days, AFTER the on-card MSA decode-paged wire validates** (it rides on that).

## #1 risk
NVMe random-read IOPS at 256 KB block granularity could floor throughput toward ~40 tok/s. Mitigation: layer-major contiguous layout (sequential preads) + OS page-cache as a free L2 + measure before trusting the high end.
