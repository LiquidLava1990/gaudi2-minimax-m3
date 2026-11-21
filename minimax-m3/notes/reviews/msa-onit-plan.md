# MSA on-it: on-card validation plan + honest value + now-vs-defer (deliberation 3/4)

**Lens:** plan + value + risk for getting MSA (and the MSA-aware KV "overflow") working on-card.
**CARD-FREE.** Date 2026-06-15. Confidence: **high** on plan/risk (read the built .so + tests +
wire directly); **high** on value framing (host-sync wall is measured, cross-confirmed by 4 int4-wall
angles); **medium** on exact effort days (FCD bring-up is iteration-count-unknown, like the MoE op).

---

## VERDICT (one line)

**MSA is ~90% done and CORRECT-on-CPU, but it is NOT one flip from live: the compiled indexer .so
encodes the OLD per-group `[H_kv,Sq,kk]` contract, while the official math (re-pinned 2026-06-14) is
HEAD-COLLAPSE → ONE `[Sq,kk]` set — so the .so must be re-edited (insert one head-axis reduce_max
before its final topk) + rebuilt + on-card-revalidated, and the decode/paged path is wholly
unvalidated. The on-card value is CONTEXT-ENABLEMENT (M3 is 1M-native, `max_position_embeddings=
1048576`), NOT throughput (single-stream is host-sync-bound; dense-GQA fallback already serves to
M3's full ctx). DEFER MSA bring-up to AFTER the fp8/Kimi wins — fp8 is higher-ROI (concurrency +
revives MTP, no kernel build) and MSA's payoff only materializes at long ctx + the KV-offload, which
is itself a separate ~6-9 day build that rides on the validated decode-paged MSA wire.**

---

## (1) The on-card MSA validation plan — effort + risk

The CPU parity is DONE and tight (wire == oracle bit-for-bit; oracle == real `MiniMaxM3VLIndexer`
block_scores |Δ|≤4.8e-7, selection Jaccard 1.0 — `tests/test_msa_hf_parity.py`,
`test_msa_wire_parity.py`). Both `.so`s are built (`kernel/build/libmsa_*.so`). What remains is
purely on-card numeric validation + ONE real kernel edit. Steps, in order:

**P0 — FIX the indexer .so to the corrected head-collapse contract (the one real code change).**
`msa_block_indexer_backend_shared.h` MSA branch (NODE 6b→7) keeps the H axis through topk and returns
`[H_kv,Sq,kk]` (per-group sets — the *now-wrong* premise; header comments still say "each GQA group
selects independently"). The official does `block_scores.amax(-1).amax(dim=1)` → ONE set. **Insert one
`reduce_max_multi_dim_fwd` over the head/batch axis (dim 0 of `[H,Sq,n_blocks]`) BETWEEN node 6b
(blkmax) and the final topk**, so topk runs on `[Sq,n_blocks]` → output `[Sq,kk]`. The wire already
expects this (`msa_m3_wire.py:574-579` "_cpu_block_indexer_topk KERNEL IMPLICATION"); the CPU stand-in
already does it (`_cpu_block_indexer_topk` does `s.amax(dim=0)`). The on-card test
(`test_msa_block_indexer_op.py`) still asserts the per-group `[H,Sq,kk]` shape — **must be updated to
the `[Sq,kk]` set** at the same time. GUID exists (proven by DSA NODE 4 head-sum reduce); ~1 node +
meta-shape change. **Effort: ~0.5-1 day code + the build.**

**P1 — Indexer op on-card vs oracle (the only path the FIX touches).** Rebuild (CPU-only, in
`vllm-gaudi:next`), run `test_msa_block_indexer_op.py` on a free card: block-id SET == fp32 oracle
(Jaccard ≥0.97 / recall ≥0.98, the DSA gate) + HPU-graph capture/replay (CHECK 2). This is the first
real device exercise of the new head-collapse reduce + the reshape→reduce_max→topk block chain.

**P2 — Consume op on-card.** `libmsa_sparse_attn.so` is **byte-identical to the already-on-card-
validated DSA consume** (`msa_sparse_attn_backend_shared.h:6` "BYTE-IDENTICAL", only namespace/names
differ; block→token expansion is upstream in Python). Lowest risk leg. Feed `expand_blocks_to_tokens`
output + GQA-expanded K/V (`expand` view, no 16× copy) at `K=topk_blocks*block_size=2048`; validate
vs `msa_gqa_gather_attention` oracle (cosine >0.9999, NaN-pattern). Needs a `test_msa_sparse_attn_op.py`
(doesn't exist yet — the DSA one is the template).

**P3 — Decode / paged (THE genuinely unvalidated path).** Everything above is PREFILL (T==S, abs pos
== arange). Decode needs: `index_cache_msa.IndexerKCache.gather_blocks` to assemble paged index-K
history, abs_pos==ctx-1 scoring, the main GQA paged-KV gather, and the runner `get_kv_cache_spec` /
`bind_kv_cache` patches to land (authored, **unvalidated on-card** — the same gap DSA hit; prefix-
caching must be OFF). This is the largest unknown: it is parity-gated against the official
`past_key_values.update_index`, which has not been done.

**P4 — Integrate + flip `MSA_READY`.** Open the gate (`platform_patch_msa.maybe_flip_ready`, env
`HPU_MSA_READY=1`) only after: (i) indexer-fired counter >0 on ctx>block_size, (ii) ctx≤K sparse ==
dense-GQA baseline, (iii) ctx≫K sparse ≠ dense (not-silently-dense). Then lazy `.so` re-validate
under `PT_HPU_LAZY_MODE=1`.

**Effort total:** ~**4-7 engineer-days** if FCD geometry behaves; realistically more — the indexer
chain has the EXACT hazard class the MoE op needed multiple on-card iterations for: a 4-D
reshape→`reduce_max` and a `topk` over a non-trailing-becomes-trailing axis must emit PyTorch-row-major
shapes or the complex-GUID SIFs null-deref (`DSA_OPBACKEND_KERNEL_DESIGN.md §8 risk #1`;
`moe_..._backend.cpp:162-191`). The consume leg is low-risk (proven twin); the indexer head-collapse
+ block-pool + decode-paged are the iteration sinks.

**#1 risk:** the indexer's new head-axis `reduce_max` + the `[H,Sq,Sk]→[H,Sq,nblk,B]` reshape +
block-topk getting the Synapse FCD/SIF geometry right on-card (multi-iteration bring-up, cards are
busy serving). **#2 risk:** decode-paged is unproven end-to-end (cache spec/bind + index-K history).

---

## (2) The VALUE — honest: context-enablement, NOT throughput

**M3's trained context is 1M** (`config.json text_config.max_position_embeddings = 1048576`,
`rope_theta 5e6` — MSA_SPEC.md CFG:17). This is the crucial contrast with the DSA/GLM-5.1 lane, whose
model caps at ~200K — **M3 is genuinely 1M-native**, so MSA has a real long-context job here.

But the throughput honesty (cross-confirmed, `int4-wall-strategy.md`, `gaudi-single-stream-speed.md`,
PROJECT_FINDINGS Finding 3): **single-stream decode is host/collective-sync-bound (~6-8 tok/s for M3's
all-512-expert path; ~29-34 the absolute MoE-agnostic sync cap), with ~90% of the step in host overhead,
not attention compute.** MSA cuts attention FLOPs (~16 blocks×128 of 1M = ~512× block reduction) — but
that compute is NOT what binds the step. So:

- **Does MSA add single-stream throughput? Essentially NO.** Attention is not the single-stream wall;
  shrinking it is invisible at batch=1, exactly as DSA's own design concedes
  (`DSA_OPBACKEND_KERNEL_DESIGN.md:441` "DSA's win is making long context tractable at all… not raising
  the batch=1 decode ceiling").
- **Does MSA add concurrency throughput? Marginally / indirectly** — by shrinking per-seq KV *read*
  and attention work it eases the concurrency×context knee, but the concurrency win that actually
  matters on this box is **fp8 experts (~1.5-2× at c=30)**, which is orthogonal to attention and needs
  no kernel.
- **What MSA uniquely enables: LONGER context per seq + more concurrent long seqs**, and it is the
  *prerequisite* for the 1M KV-offload (the index→block selection is what makes tiering the main K/V
  tractable). The dense-GQA fallback is correct and serves to M3's full ctx **today** — but dense KV at
  1M is O(L²)-memory-impossible at real concurrency; MSA is how 1M-at-scale becomes feasible.

**Is MSA worth the on-card risk NOW vs the fp8/Kimi wins? NO — defer.** fp8 is a code-proven GO with
**no kernel build** (`fp8-moe-gate`: stock `VllmMixtureOfExpertsOpFP8PerChannel` → native MME), gives
the concurrency multiplier M3 actually targets, and **revives MTP** (the *only* proven 1.70×
single-stream lever, which int4 kills). MSA delivers nothing on single-stream, and its long-ctx payoff
is only realized once you ALSO build the offload. fp8/Kimi is strictly higher ROI per on-card day.

---

## (3) The "MSA overflow" (KV offload, 1M) — separate large effort, defer

Per `offload-design-SYNTHESIS.md`: the MSA **index-K cache must stay HBM-resident + EXACT** (it is
FULL-scanned every decode step to score all blocks → offloading it = seconds/step = dead). At 1M×15
that resident index is ~230 GB, which fits **only the int4-weights envelope (~340 GB free), NOT fp8
(~150 GB)** — so the viable 1M-offload config is **int4-in-HBM × offload** (c6), and `fp8×offload@1M
OOMs`. Only the *main* K/V (read sparsely, 2048 of 1M tok/step) is tiered to host-RAM→NVMe with
one-step-ahead prefetch; the consume op reads HBM unchanged (no new TPC kernel).

**Worth building now? NO — it is a distinct ~6-9 day build that RIDES ON the validated decode-paged
MSA wire** (synthesis line 30: "AFTER the on-card MSA decode-paged wire validates — it rides on that").
You cannot start it until P3 above is green, and its #1 risk (NVMe random-read IOPS flooring throughput
toward ~40 tok/s) needs a real measurement first. It is a phase-2 effort gated behind MSA itself.

---

## BOTTOM LINE / sequencing

1. **NOW: fp8 (Tier-1) + the Kimi/giant int4 serve path** — no kernel build, concurrency + MTP, the
   real ROI. (`int4-wall-strategy`, `giant-int4-strategy`.)
2. **NEXT (when a card is free + a 1M-ctx need is real): MSA on-card** — start at P0 (fix the indexer
   .so to head-collapse `[Sq,kk]` + update its test), then P1→P4. ~4-7 days if FCD behaves; consume leg
   is low-risk (DSA twin), indexer head-collapse + decode-paged are the iteration sinks. Keep
   `MSA_READY=False` / dense-GQA fallback live until all three flip-gates pass.
3. **THEN (phase 2, rides on P3): the MSA-aware KV offload** — int4-in-HBM × tiered main-K/V, ~6-9 days,
   measure NVMe random-read first. This is what actually delivers "1M overflow."

The user's "make sure MSA overflow is working" is really TWO sequential builds (on-card MSA, then the
offload), both correctly behind fp8 on the priority list — and the immediate honest status to report is:
**MSA is CPU-correct but the built indexer .so is one head-collapse edit + a full on-card pass (incl. the
unvalidated decode path) away from being flippable; it buys long-context, not speed.**
