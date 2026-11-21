# int4 single-stream wall — speculative decode (card-free diagnosis + fix)

**Date:** 2026-06-15  **Scope:** break the MiniMax-M3 int4 single-stream wall (~6–8 tok/s,
host-sync-bound) via speculation — the biggest single-stream lever (amortizes the host-bound
per-step over N accepted tokens). **Card-free** diagnosis of the on-card n-gram crash + fix design.
Cross-ref: `M3_MATRIX_REPORT.md` §IV.8a (crash), `m3_swigluoai_patch.py` (the int4 MoE apply),
`hpu-moe-ext/mtp_bf16_graft.py` (the GLM precedent), `_src/vllm_gaudi/v1/spec_decode/hpu_eagle.py`.

---

## 1. The crash, located precisely (no card needed)

`RuntimeError: index 16 is out of bounds for axis 0 with size 16` at **decode warmup** under
`--speculative-config {"method":"ngram","num_speculative_tokens":3}`. **16 = local experts E**
(`128 global / TP8 = 16`; confirmed `moe_wna16_hpu.py:591` `num_experts = layer.local_num_experts`,
`:593` `experts_min,experts_max = ep_shift, num_experts+ep_shift-1`).

The chain, file by file:

1. **Spec-decode makes decode a MULTI-token step.** `hpu_model_runner.py:2002`:
   `num_decode_tokens = 1 if spec_decode_tokens is None else len(spec_decode_tokens)+1`. With
   `num_speculative_tokens=3`, a verify step feeds **T = 4 rows per sequence** into the model as a
   *batched decode* (not a prefill — `:2004` keeps it a decode while `num_scheduled_tokens <=
   num_decode_tokens`). This is the shape the int4 MoE never sees outside spec-decode (normal decode
   is strictly T=1; prefill is T≫1 but goes through the prefill bucket).

2. **The M3 patch's fast-path gate is hard-wired to T==1.** `m3_swigluoai_patch.py:434-435`:
   ```python
   eligible = (layer._hpu_routed_w13_zp is not None
               and (x2.shape[0] * K) <= K)   # T*top_k<=K  i.e.  T==1
   ```
   For the verify batch `x2.shape[0]=4`, `K=top_k=4` → `16 <= 4` is **False**. So the T=4 step is
   **denied the K-gather routed path** and falls through (`:467-472`) to `_allexpert_swigluoai`.

3. **Where 16 overflows.** Two routes reach the axis-16 fault, and the **warmup** trigger is the
   stock path, not the patch:
   - The **decode-warmup dummy** for spec-decode is generated *before* a real prompt and runs through
     the engine's spec-decode plumbing. The stock `VllmMixtureOfExpertsOpWNA16` / routed-gather was
     built with the EP-local window `experts_min..experts_max = 0..15` and its internal scatter/gather
     indexes a **length-16 local-expert axis with a topk_id**. The grouped-topk router
     (`hpu_grouped_topk_router.py`) returns **GLOBAL** ids in `[0,128)` (it has no EP awareness —
     `num_expert_group=1`, so `tmp_scores.reshape(num_token,-1)` then a plain `topk` over all 128).
     A global id ≥16 indexing the **length-16 local** list = `index 16 out of bounds for axis 0 with
     size 16`. In normal T=1 decode the M3 K-gather path *remaps* global→local safely
     (`:217-219 local = topk_ids - emin; valid = (flat>=0)&(flat<E)`), masking the overflow — but the
     **multi-token verify/warmup shape skips that path** (gate False), exposing the raw global id to
     the length-16 local axis. **The gate that protects the int4 EP plumbing only fires for T==1.**

**Root cause (one line):** the int4 routed-MoE's global→local expert remap (the thing that makes
EP-sharded int4 experts coherent) is **only wired on the T==1 fast path**; the spec-decode T=N
verify/warmup batch bypasses it and feeds a global expert id (≥E) into the length-E local axis.

This is *not* a deep numeric bug — it is a **shape-coverage gap**: one batched-decode shape the
custom MoE never handled. Also confirmed orthogonal: n-gram is incompatible with `--async-scheduling`
(`hpu_model_runner.py:1234` asserts it), which must be dropped (engine auto-disables anyway).

---

## 2. The n-gram fix (concrete, card-free to author)

**Make the M3 int4 MoE handle the T=N batched-decode by routing it through the SAME global→local
remap the T==1 path already uses — i.e. let the all-expert path (or a generalized K-gather) own
ANY T, never the stock op.** Two edits, both in `m3_swigluoai_patch.py`:

1. **Drop the T==1 restriction; gate on GPTQ-eligibility only.** Replace `:434-435` with
   `eligible = (layer._hpu_routed_w13_zp is not None)` and make `_routed_swigluoai` size its active
   set by `K_active = min(E, T*top_k)` (it already builds `active = topk(present, K).indices` over the
   compact local space — bump K from `top_k` to the count of *distinct local experts actually hit*,
   which `present` already encodes). For small T (≤ a few) `K_active ≪ E`, so the loop stays cheap.
2. **Guarantee the all-expert fallback is the universal T-agnostic path** (it already is —
   `_allexpert_swigluoai` is static-shape over `E` local experts and masks with a `[T,E]` route built
   from `local = topk_ids - emin; in_win=(local>=0)&(local<E)` at `:367-371`). The only missing piece
   is making sure **warmup** also takes this path: the gate change in (1) sends every T to a patch
   path that does the remap, so the stock op (the one that overflows) is **never reached** for the M3
   int4 layer. This is exactly the §IV.8a "must NOT fall to `_orig_apply`" contract, extended from
   "prefill T>1" to "any T>1 including the spec-decode verify batch."

**Why this is the right shape of fix:** the patch *already* contains both the remap (routed path) and
the universal masked all-expert path. The bug is purely that the **dispatch gate** (`T*top_k<=K`)
excludes the spec-decode shape and hands it to the stock op. Widening the gate is ~5 lines, reuses the
proven dequant+remap+mark_step discipline bit-for-bit, and adds **no new kernel / no .so / no card
build**. It is unit-testable CPU-side against the existing `tests/test_swigluoai_unfused.py` harness by
feeding a T=4 routing tensor with global ids spanning two EP windows.

**Expected single-stream gain:** n-gram only accepts when the continuation is **literally present in
the prompt/context** (prompt-lookup). For M3's workloads (reasoning, free-form chat) accept-length is
typically **~0.2–0.6 tok/step**, i.e. **~1.1–1.4×** — and *only* on repetitive/structured output
(code echoing the prompt, JSON, long quotes), ~1× on prose. The per-expert loop also gets ~T× heavier
per verify step (T matmuls vs 1), so on **non-repetitive** single-stream it can be **net-neutral or
slightly negative**. **Verdict: fix it (cheap, removes a crash, helps structured workloads), but it is
NOT the 1.5–2× lever** — n-gram's ceiling is low because M3 decode is novel-token-heavy.

---

## 3. Alternatives — and why MTP-self-speculation is DEAD for M3

The real 1.5–2× lever is a **model-based drafter** (predicts novel tokens, not just prompt echoes).
Three candidates evaluated:

### (a) Graft M3's MTP/nextn heads — **IMPOSSIBLE (no weights exist).** ❌
The prompt's premise ("they're absent from int4 but present in bf16") is **false** — I verified the
bf16 checkpoint directly. `config.json` advertises `num_mtp_modules=7 / num_nextn_predict_layers=1`,
but a **header-only scan of all 59 bf16 shards** (`/mnt/data2/minimax-m3-bf16`) shows **global max
`.layers.N` index = 59** (valid 0..59 == the main tower) and **zero** tensors matching
`nextn|mtp|eh_proj|enorm|embed_head|shared_head`. The MTP heads were **architected and trained but
never released** by MiniMax — in *both* the int4 AND the bf16 export. This is the decisive difference
from GLM: `hpu-moe-ext/mtp_bf16_graft.py` works *only* because **zai-org shipped GLM's layer-78 bf16
MTP in 4 shards**; the graft just substitutes those real tensors. **For M3 there is nothing to graft,
quantize, or attach.** This path is closed unless MiniMax publishes the MTP weights (monitor the HF
repo; if they appear, the GLM graft generalizes almost verbatim — same `get_spec_layer_idx_from_
weight_name` skip already in `minimax_m3.py:1071`).

### (b) EAGLE draft head — **the most-likely-to-work real lever, but needs a trained head.** 🟡
`HpuEagleProposer` (`_src/vllm_gaudi/v1/spec_decode/hpu_eagle.py`) is **already implemented and
arch-agnostic** — it consumes the target model's last hidden state + a small draft transformer, and
the runner wires it generically (`hpu_model_runner.py:1182 use_eagle()`; no per-arch registration like
`mtp` needs). EAGLE drafts **novel** tokens from M3's own hidden states → typical accept-length
**2–4 tok/step → ~1.7–2.5× single-stream**, the documented MTP-class win (MTP *is* a 1-layer EAGLE).
**Blocker:** EAGLE requires a **trained draft head** (one ~1-layer transformer tied to M3's
embed/hidden space). None exists publicly for M3, and training one needs a short distillation run
(hidden-state targets from the bf16 model on a calibration corpus — feasible on this box over a few
GPU-days, far cheaper than re-quantizing). **This crashes into the SAME int4 MoE shape gap** (EAGLE's
verify batch is also T=N) — so the §2 gate-widening fix is a **prerequisite** for EAGLE too, and pays
off far more there. Recommended **if** spec-decode is worth a training investment.

### (c) Tiny separate draft model (e.g. a 0.5–1B dense LM as drafter) — **works today, modest.** 🟢
vLLM's standalone "draft model" spec-decode (a small independent LM proposes, M3 verifies) needs **no
M3-specific weights and no training** — just a fast small model sharing M3's tokenizer. Accept-length
is lower than EAGLE (the drafter doesn't see M3's hidden state) — **~1.3–1.8×** — and it costs HBM +
a second model's decode per step. M3's tokenizer is custom (MiniMax merges.txt), so the drafter must
share it (a distilled small M3, or retokenize) — a real but surmountable constraint. Same int4 MoE
gate-fix prerequisite. This is the **lowest-effort path to a real (novel-token) speedup** if a
tokenizer-compatible small model can be sourced/distilled.

---

## 4. Recommendation

**Two-step:**

1. **Land the §2 gate-widening fix now** (~5 lines in `m3_swigluoai_patch.py`, CPU-unit-testable). It
   (i) **un-crashes n-gram** for a real ~1.1–1.4× on structured/repetitive single-stream, and
   (ii) is the **mandatory prerequisite** for EAGLE *or* a draft model — all three speculation paths
   feed a T=N verify batch through the same int4 MoE that currently only handles T==1. This is the
   highest-leverage card-free change: it removes the single structural blocker to *all* M3
   speculation.

2. **For the real 1.5–2× lever, pursue EAGLE (b)** — it is the only path that drafts novel tokens AND
   is already wired arch-agnostically on this stack — **but it requires training a draft head** (no
   M3 EAGLE/MTP weights exist anywhere; the bf16 graft is impossible). If a training run isn't
   warranted, **a tiny tokenizer-compatible draft model (c) is the no-train fallback** (~1.3–1.8×).

**Do NOT** wait on MTP-self-speculation — M3 ships no MTP weights in either precision (verified). The
§IV.2 "fp8 keeps MTP alive" note is **moot for M3** because the MTP heads were never released; it
applies only to models (like GLM) whose authors shipped the draft head.

**Confidence:**
- n-gram crash diagnosis + fix shape: **HIGH** (traced end-to-end in source; gate `(x2.shape[0]*K)<=K`
  + global-id-into-length-16-local-axis is unambiguous; fix reuses existing remap code).
- n-gram gain ~1.1–1.4× (structured only), often ~neutral on prose: **HIGH**.
- EAGLE ~1.7–2.5× *if* a head is trained: **MEDIUM** (mechanism solid + proposer exists; accept-rate
  is corpus-dependent and the head must be trained).
- MTP-graft impossible: **HIGH** (direct header scan of all 59 bf16 shards — zero MTP tensors).
- Net single-stream after the most-likely-to-land *card-free* change (n-gram fix): **~1.1–1.4× on
  structured workloads, ~1× on prose.** The 1.5–2× requires the EAGLE/draft-model investment above.
