# M3 single-stream — FINAL verdict (deliberation 2/4, card-free)

**Date:** 2026-06-15  **Question (asked 3×):** is there ANY single-stream lever left for M3 int4/fp8?
**Answer: single-stream is genuinely host-sync-MAXED. Speculation is the ONE real remaining lever, and
for M3 it is weak (n-gram only) or needs training (EAGLE) — NOT a free 1.5–2×. Stop chasing batch=1.**

## 1. Is it maxed? YES — structurally.
The wall is host/collective-sync, not MoE compute (PROJECT_FINDINGS §3/§11; util 18–31%, ~90–116 W of
600 W = host-starved device). Every MoE-compute lever is dead **on `:next`**, confirmed from THREE
directions and MEASURED: native captured swiglu op returns ~0 / hangs the card; batched bmm REGRESSES
(5.0 vs 8.3 tok/s); fp8 experts buy ≈0 single-stream (remove a 0.16 ms cost from a ~130–190 ms step).
The two card-free host-cost cuts the prior round diagnosed — **gate the lazy-only mark_steps** and the
**widened spec-decode gate** — are **ALREADY LANDED in `m3_swigluoai_patch.py`** (mark_step gating L93/246/406;
`K_active=min(E,T*top_k)` + gate `(T*top_k)<=E` L461–519). Mark_step gating is the only thing that might
still nudge int4 to ~9–10; everything else is harvested.

## 2. n-gram / draft — worth wiring?
**The n-gram CRASH FIX (gate-widening) is in the live patch** (the old T==1-only gate is gone). But the
serve env (`minimax-m3-int4.env`) does **NOT** enable n-gram AND sets `--async-scheduling`, which n-gram
forbids. So it is **wired-but-unmeasured**. Worth a 30-min A/B (drop async-scheduling, add
`--speculative-config '{"method":"ngram","num_speculative_tokens":3}'`, run the codegen probe) — it only
helps prompt-echo / structured output, **~1.1–1.4× on code/JSON, ~1× on prose**, and the T× heavier verify
loop can make it net-neutral on novel tokens. Real 1.5–2× needs a **novel-token drafter**: MTP is
IMPOSSIBLE (zero MTP tensors in either precision — verified header scan of all 59 bf16 shards); EAGLE works
arch-agnostically here but needs a **trained head** (few GPU-days); a tiny tokenizer-compatible draft model
is no-train but modest (~1.3–1.8×) and needs M3's custom tokenizer.

## 3. Realistic single-stream tok/s
**int4: ~8–10 warm** (8.3 measured; +mark_step-gate→~9–10). fp8: same single-stream (host-bound). With
n-gram: ~9–13 on structured only. Absolute MoE-agnostic sync cap ~29–34 is unreachable for M3's path.
The 3.6 "cold" is recipe-cache-cold — fixed by warming, not a real floor.

**DECISION:** declare single-stream MAXED. Do ONE cheap n-gram A/B (already coded). Do NOT build EAGLE
unless spec-decode is a funded goal. The box's real M3 win is **throughput@concurrency**, not batch=1.

**Confidence:** maxed-by-host-sync **HIGH**; fixes-already-landed **HIGH** (read in source); n-gram
~1.1–1.4× structured-only **HIGH**; EAGLE 1.7–2.5× *if trained* **MEDIUM**.
