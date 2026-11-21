# Adversarial review — MSA attention-kernel speed (MiniMax-M3 on 8×Gaudi 2)

Lens: is this the FASTEST attention path on Gaudi 2? Reviewer read the built kernel
(`kernel/msa_block_indexer_backend_shared.h`, `kernel/msa_sparse_attn_backend_shared.h`),
the wire (`vllm/msa_m3_wire.py`), the oracle (`ref/msa_reference.py`), and the DSA design
(`Gaudi2_Kernel/DSA_OPBACKEND_KERNEL_DESIGN.md` §3a/§6) + `Gaudi2_Optimization_Guide.md`.

Date: 2026-06-14. Status of the artifact: both `.so` built; on-card numeric validation NOT
yet run (`tests/test_msa_block_indexer_op.py` is gated; `MSA_READY` closed, dense fallback live).

---

## VERDICT: KEEP the kernel design, CHANGE the consume invocation + the indexer precision/fusion.

The two GC graphs are correct, bridge-confirmed, and SRAM-resident — the right substrate.
The speed that is left on the table is almost entirely in **how the wire drives the consume op**
and in **the indexer running dense bf16 with no fusion**, not in the graphs themselves. Two of the
three biggest levers are *already named in the DSA design §6* but are NOT wired into MSA.

### Evidence (where speed is lost)

1. **Consume is called 4× per layer in a Python `for r in range(Hkv)` loop** (`msa_m3_wire.py:514-520`),
   one `sparse_attn` op per GQA group, results stitched. That is 4 host dispatches + 4 separate
   GC recipes per sparse layer × 57 layers = ~228 op launches/token — the exact per-op eager-launch
   wall the OpBackend path exists to kill (`gaudi-single-stream-speed.md`; design §6.3). The op's
   `[T,H,1,Dh]` schema already carries a head axis; the 4 groups differ ONLY in `token_idx`. This
   should be ONE op call with the head dim = H_q (or H_kv batched), token_idx `[T,H,K]` not `[T,K]`.

2. **K/V is `repeat_interleave(G=16)`'d to H_q heads BEFORE the gather** (`:502-503`). Real GQA K/V
   is 4 heads; inflating to 64 heads 16× the bytes that the `gather_fwd` then moves (`[S+1, H*Dh]`,
   `msa_sparse_attn_backend_shared.h:228-238`). The gather should run on the **4 real KV heads**, the
   G-fan-out happening inside SDPA (heads are a batch dim; SDPA does not require K-head==Q-head if the
   gather indices are per-group). This is pure wasted HBM traffic on the dominant long-context tensor.

3. **Indexer scores dense `[H_kv=4, Sq, Sk]` in bf16 via plain `batch_gemm`** (`..._indexer_..._shared.h:281-288`).
   No fp8 (design §6.5: fp8 index logits ≈ 2× MME, "the single biggest prefill-TTFT lever"); no
   `masked_batch_gemm` to fold the causal triangle into the MME (design §3a/§6.7, Gaudi-2-only). At
   128K–1M the indexer Q·Kᵀ is the prefill-TTFT dominator (it is O(L²) per the block-max-pool input),
   so this is where TTFT is won or lost. The block-max-pool reshape→`reduce_max` (NODE 6a/6b) is cheap
   and correct; the cost is the dense bf16 GEMM feeding it.

### Things that are RIGHT (keep)

- gather-consume O(L·K=2048) product form, NOT O(L²) masked — correct (design §6.1).
- block→token expansion kept upstream so consume == DSA op byte-identical — good reuse.
- topk indices declared Int32 directly (no `cast_i32_i32` the lazy GC refuses) — the DSA fix carried.
- dense-GQA fallback as first serve while the kernel validates — **acceptable and correct**: layers
  0–2 are genuinely dense, `MSA_READY` gates loudly, never silently-sparse. Ship it.

### Confidence

High on #1/#2 (read directly in the wire; they are unambiguous extra dispatches + 16× bytes). High on
#3 being the TTFT lever at long context (consistent with the DSA design's own §6.5 and the 200K
TTFT≈40–70s compute-wall finding). Medium on the *magnitude* of each — none is measured on-card yet
(cards busy; validation gated). Caveat flagged in the wire itself: `index_q_norm`/`index_k_norm` are in
the checkpoint but NOT in the paper/oracle (`msa_m3_wire.py:41-51`) — re-pin before trusting selection
quality; this is a correctness risk that gates the speed work, not a speed item.

---

## #1 highest-impact kernel optimization

**Collapse the 4-call-per-layer consume loop into ONE op call over all H_q heads with a per-head
`token_idx[T,H,K]`, and gather on the 4 REAL KV heads (drop the `repeat_interleave(16)`).**

This removes ~3 of every 4 consume dispatches (~170 launches/token saved) AND cuts the gathered-KV
HBM traffic 16× on the single largest long-context tensor — one change, both wins, no new graph node
(the consume schema already has the head axis; widen `token_idx` to `[T,H,K]` and index per-group). It
is strictly higher leverage than fp8 logits for **decode throughput at concurrency 30+**, where
per-op launch overhead and KV bytes dominate. Do fp8 index logits (§6.5) + `masked_batch_gemm`
(§6.7) as the immediate follow-on for **prefill TTFT at 128K–1M** — that is the second lever, and the
two together are the path to the design's "ultra-fast" bar. Sequence: validate the bf16 graphs on-card
(flip nothing until the index-norm spec is re-pinned), THEN land the single-call consume, THEN fp8.
