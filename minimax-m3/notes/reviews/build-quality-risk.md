# MiniMax-M3 / Gaudi2 MSA — Adversarial Build-Quality Review

**Lens:** silent quality-loss / correctness risk at LONG CONTEXT + HIGH CONCURRENCY.
**Date:** 2026-06-14. **Reviewer stance:** skeptic. Every claim below is checked against the live
checkpoint (`/mnt/data2/minimax-m3-bf16`), the official MSA repo (`/tmp/MSA_repo`), and the build
files. Where I could not resolve something from sources, I say so and rank it by *risk*, not by how
likely I think it is.

> Scope note: the build's SAFETY posture is genuinely good — `MSA_FULL_ATTENTION_FALLBACK=True`,
> the loud one-time dense-fallback log, the sentinel-row (never `clamp(min=0)`) consume, the LOUD
> `MsaDims.validate()`, and the "never serve dense in place of sparse" platform gate all mirror the
> DSA contract that already shipped. So the risk is NOT "it silently serves garbage tomorrow."
> The risk is **what flips to wrong the moment `MSA_READY` is set True at 128K–1M and c≥64** — i.e.
> the quality cliffs that the current gates (CPU set-match parity at small T, prefill-only) do NOT
> exercise. That is exactly where I focused.

---

## VERIFIED FACTS (so the ranking rests on ground truth, not the spec docs)

- **Index RMSNorms are REAL.** Probed the landed safetensors: every sparse layer (3,4,5,…) carries
  `self_attn.index_q_norm.weight [128]`, `index_k_norm.weight [128]`, `index_q_proj [512,6144]`,
  `index_k_proj [128,6144]`. Dense layers 0–2 carry none. So the checkpoint UNAMBIGUOUSLY has a
  per-index-head RMSNorm (over `d_idx=128`) on BOTH index branches. `ref/msa_reference.py` and
  MSA_SPEC.md §2b model a **bare** scaled dot product. **The oracle is wrong about the real model.**
- **No official M3 modeling code exists on the box or (per the wire's note) upstream.** Only
  `configuration_minimax_m3_vl.py` (config classes, no math) + `image_processor.py`. The paper has
  no index-norm. So the index-norm semantics (eps, fp32 discipline, whether K-norm is truly shared,
  whether there is *also* an index rope the paper omitted) **cannot be confirmed from any local
  source.** This is the crux landmine (Risk #1).
- **The wire is PREFILL-ONLY.** `msa_m3_wire.py:448,460` — `S = T`, "PREFILL contract: T == S (no
  paged history). Decode/paged is a follow-up (§C)." The goal (long ctx + high concurrency) is
  DECODE-dominated over a PAGED KV cache. The thing that must be correct for the goal is unbuilt.
- **fp8_inc is the ONLY KV dtype the platform accepts** (`platform.py:389`). Nothing in the build
  addresses whether the index/proxy K cache is also fp8 or what fp8 KV does to selection/consume at
  128K+. config: 1M ctx, theta 5e6, partial rope dim 64, topk 16×128, GQA 64/4.
- MSA repo `force_end_blocks` = "blocks at the end of the VALID sequence (nvp-N..nvp-1, closest to
  the current query)" (`api.py:1210-1217`) — confirms the local-block = block(i) semantics the
  oracle/wire encode.

---

## THE SINGLE BIGGEST QUALITY LANDMINE

**The `index_q_norm` / `index_k_norm` spec delta — an UNVERIFIED per-element transform on the exact
tensor that decides which KV blocks every token reads.** It is the biggest because: (a) it is
*confirmed present* in the weights yet *absent* from the only math source (the paper), so we are
guessing its exact form; (b) it sits on the SELECTION path — if the norm is applied wrong (wrong
eps, wrong axis, applied where the real model does not, or *missing* a paired index-rope the paper
also omitted), the top-16 blocks are silently the WRONG 16, and the consume then runs perfect
attention over the wrong evidence; (c) the existing parity gate is **structurally blind** to it — a
SET-match / Jaccard gate is robust to any positive monotone rescale, but RMSNorm is per-element and
genuinely re-orders which blocks win, AND the dense-fallback control masks it completely. So both
guardrails that "validated" DSA are blind here. The build's own header flags it (good), but flags
are not validation. **Do not flip `MSA_READY` until this is reconfirmed against official code.**

---

## RANKED MUST-VALIDATE LIST (top 5, by quality risk)

### 1. index_q_norm / index_k_norm spec delta (SELECTION-path, unverified) — CRITICAL
**Why:** above. Wrong selection = silent, unbounded quality loss that no SET-match gate catches.
**Validate:** (a) BLOCKING — obtain the official `modeling_minimax_m3*.py` (HF/vLLM/SGLang when it
lands) and byte-confirm: is the norm RMSNorm over `d_idx`? eps? fp32? is it on BOTH branches? **is
there ALSO a rope on the index branch the paper omitted** (the norm discovery proves the paper is
incomplete — re-open the rope question for the index path, do not trust §7.1's "no"). Until then,
treat sparse as un-shippable. (b) On-card A/B once code-confirmed: run the SAME long needle through
norm-ON vs norm-OFF (`MSA_APPLY_INDEX_NORM`) vs the FULL-DENSE control; measure **recall@16-blocks
vs the dense top-attention-mass blocks**, not set-match-to-oracle. The right metric is "do the
selected blocks contain the tokens dense attention actually weights," computed from real dense
attention scores on a needle prompt at 8K/32K/128K.

### 2. Block-selection RECALL at 128K–1M (top-16×128 = 2048 tokens of ≤1M) — CRITICAL
**Why:** 16 blocks of 128 = 2048 tokens retained out of up to 1,048,576 — a **512× reduction**. The
paper validates this for ITS model; nothing here validates that M3's *index head* still ranks the
right blocks when 8190 of 8192 blocks are dropped. Selection recall that is fine at T=4K can fall
off a cliff at 256K (more distractor blocks, BlockMaxPool saturating, fp8 KV noise). This is the
"needle in 1M" failure and it is **silent** — output stays fluent.
**Validate:** multi-needle (≥8 needles at varied depths) RULER/NIAH-style recall sweep at
4K/32K/128K/512K, sparse vs dense-full control on the SAME serve (DSA used exactly this — and note
its 3051-tok needle MISS matched the dense control, i.e. recall must be judged vs the dense control,
never vs ground truth, because the int4 checkpoint has its own floor). Report recall@depth; a
sparse-minus-dense gap that widens with context = selection breaking. Tie to Risk #1 (run after the
norm is pinned, else you measure the wrong selector).

### 3. PREFILL-ONLY wire → no validated paged/decode path — CRITICAL (for the stated goal)
**Why:** the goal is long-ctx + HIGH CONCURRENCY = many sequences, each generating tokens over a
PAGED KV cache. The wire only does `T==S` prefill (`msa_m3_wire.py:448`). Decode needs: per-step
single-query selection against the *paged* historical K/V in absolute order, an index/proxy K cache
kept in sync per step (BLOG: "K-only index buffer alongside the KV cache to avoid stale state"), and
correct abs-pos masking across page boundaries. None is built/validated. The DSA findings (§7) show
this is the HARD part — DSA is *op-validated* but **lazy/decode serve is still not done** ("the
production lazy serve runs the dense bypass"). High concurrency also means the per-GQA-group
Python loop in the consume (`for r in range(Hkv)`) and per-request orchestration will not capture
under lazy HPU-graph — same wall DSA hit.
**Validate:** before any quality claim, a decode parity test: prefill N tokens, then decode M tokens
through the paged path, compare to the prefill-everything oracle token-by-token (cosine > 0.9999,
NaN-pattern match). Then a high-concurrency soak (c≥64, mixed long prompts) watching for the
selection buffer / index-cache desync. Until decode is wired, "MSA quality at concurrency" is
untested by construction.

### 4. fp8_inc KV (and index/proxy-K) accuracy at 128K+ — HIGH
**Why:** fp8 is forced (`platform.py:389`). At 128K–1M the consume gathers REAL K/V — in fp8 — and
the dot products span a huge dynamic range; fp8 KV error compounds with sequence length and is worst
on the rare long-range key that a needle depends on. Separately, if the **index/proxy K cache** is
also fp8, selection scores are computed from quantized keys → noisier top-k exactly where recall is
already thin (Risk #2). The project's own fp8 finding: a hard ~6.4e-2 relNorm floor for fp8 compute;
KV-fp8 is different but unmeasured at this context length for this model.
**Validate:** fp8_inc vs bf16-KV (if any path allows it for a bench) A/B on the recall sweep at
128K; and explicitly determine + document the index/proxy-K cache dtype. If selection uses fp8 keys,
test bf16-index-cache vs fp8-index-cache recall — keep the index cache higher precision if recall
moves (it is tiny: 1 head × 128 per token).

### 5. Within-block causal masking + force-local correctness at scale — HIGH
**Why:** two coupled correctness traps, each a *silent* leak: (a) the **within-block future-token
leak** — the consume op masks ONLY pad/-1 slots; the wire must push `token_pos > abs_pos → -1`
(`causal_mask_token_idx`, `msa_m3_wire.py:497`). This is EXACTLY the bug DSA shipped and had to fix
(`PROJECT_FINDINGS §7` bug 3: "short-row top-k tail = future-token leakage," which the full-causal
on-card gate COULD NOT catch). The MSA wire has the guard, but it is unproven on-card and the
analogous gate is equally blind. (b) **force-local eviction order** (`force_local_sort_pad`) must
evict the lowest-SCORED op block, not the highest-id — a subtle reconstruction of +inf/topk
semantics from set+rank; if wrong, the forced local block displaces a high-value block or vice
versa, degrading the diagonal/recent context every token.
**Validate:** a dedicated causal-leak unit test at T just above and below a block boundary and at kk
(query rows with fewer causal keys than topk) — assert NO selected token has abs-pos > query pos,
and assert the forced local block is present AND the evicted block is the min-score one. Run on-card
(the CPU harness already covers the math; the leak only manifests with the real op's masking
contract). Cross-check force-local against MSA repo `test_sparse_topk_forced.py`.

---

## Lower but log-worthy (not in the top 5)

- **int4 MoE routing cliffs:** today's checkpoint is bf16 (no `quantization_config`) so int4 is
  inert; the production path is an int4 AutoRound export. When it lands, the sigmoid-router + routing
  bias (`e_score_correction_bias`, fp32 gate) quantization-sensitivity is unvalidated — but it is
  ORTHOGONAL to MSA and gated by hpu-moe-ext autodetect; validate at export time, not now.
- **MTP / spec-decode:** `num_mtp_modules=7` in config; M3 reuses the M2 spec-layer SKIP at load
  (correct). DSA's MTP experience (§8) was a DEAD int4 draft head requiring a bf16 graft — same risk
  class will recur for M3's int4 export, but MTP is OFF in phase 1 and does not affect MSA quality.
- **bf16 reference vs real checkpoint:** the oracle is pure-torch fp32; the real model is bf16 (→
  int4). Parity gates run vs the oracle, so they validate the WIRE, not the model's bf16/int4 quality
  floor. Judge final quality vs the DENSE control on the SAME dtype/serve (the only honest baseline),
  never vs the fp32 oracle or ground truth — DSA's needle-miss lesson.
- **use_gemma_norm=true** (confirmed in config): the RMSNorm `(1+w)` convention. If the index norm
  also follows gemma-norm (`_IndexRMSNorm` uses plain `x*w`, NOT `x*(1+w)`), selection is wrong.
  **This is a concrete sub-case of Risk #1** and is independently checkable: the index_*_norm weights
  are loaded — inspect whether they center near 0 (→ `1+w`) or near 1 (→ `x*w`). Flagged for the
  Risk-#1 code-confirm.

---

## CONFIDENCE

- **Findings 1, 3, 4, 5 (presence of unverified index norms; prefill-only; fp8-forced; causal-leak
  contract): HIGH** — read directly from the checkpoint tensors, the platform code, and the wire
  source.
- **Ranking / "biggest landmine = index norm": HIGH** — it is the only unverified transform on the
  selection path, and both existing guardrails (set-match parity, dense fallback) are provably blind
  to it.
- **Magnitude of the long-context recall cliff (Risk #2): MEDIUM** — the mechanism is certain
  (512× reduction, unvalidated for M3's selector at scale); the size of the quality hit is unmeasured
  and could be small if M3's index head is strong. Must be measured, not assumed.
- **Overall: the build's SAFETY (no silent dense) is HIGH-confidence sound; the build's READINESS to
  produce correct sparse output at 128K–1M and c≥64 is LOW — three of the top-5 risks are unbuilt or
  unverified for exactly that regime.**
