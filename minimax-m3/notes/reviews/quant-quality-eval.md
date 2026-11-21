# Adversarial Review — Quality-Measurement / Validation Plan for the MiniMax-M3 int4+fp8KV+MSA Build

**Lens:** How do we PROVE "minimal quality loss" before spending card time, especially at LONG CONTEXT + HIGH CONCURRENCY?
**Reviewer verdict up front:** the *current* plan does not measure quality at all — it ships a generic-calibrated int4 recipe (`pile-10k`) and relies on card-free oracle parity for the kernel. That is necessary but **not sufficient**. Below is the eval matrix, thresholds, the cheap cliff-catch subset, and the single eval to run first.

Sources read: `quant/quantize_m3_int4.py`, `notes/MSA_SPEC.md`, `Desktop/Gaudi2_Optimization_Guide.md` (§2/§4 lm-eval guidance, "never uncalibrated FP8 on MoE"), and the **proven precedent** — the GLM-5.1 DSA validation in `$HOME/AI/Gaudi2_Kernel/PROJECT_FINDINGS.md §7` and its e2e harness `$HOME/AI/dsa_e2e/e2e_long_needle.py` + `$HOME/AI/bench/accuracy_probe.py`.

---

## 0. The core principle: isolate THREE independent loss sources

The GLM-5.1 build's hardest-won lesson (PROJECT_FINDINGS §7, "Needle-recall honesty"): a long-context miss had **three confounded causes** and they were untangled only by same-serve controls. We design for that from day one. Total quality delta decomposes as:

```
Δquality(served)  =  Δ_quant      (int4 experts + fp8 KV)        ← weights/cache numerics
                  +  Δ_sparse     (MSA top-16 selection vs dense GQA) ← attention algorithm
                  +  Δ_baseline   (the bf16 M3 checkpoint's own floor) ← NOT our bug
```

**Every PASS threshold below is a Δ against the correct control**, never an absolute score, because an absolute MMLU number conflates all three and tells you nothing about *what to fix*. The controls are cheap because MSA collapses to dense GQA for any prompt ≤ `block_size × topk = 128×16 = 2048` tokens (MSA_SPEC §2d: <16 visible blocks ⇒ all selected ⇒ identical to dense). **So a ≤2048-token eval run on the MSA build IS the dense control** — you get it for free on the same serve, exactly as GLM did with its 1889-token dense-path control.

---

## 1. The A/B matrix (rungs, cheapest→most expensive)

Reference run = **A0**. Quantize ONE axis per rung so a cliff names its own cause (do NOT change two at once — the GLM K=8+MLA "two-bug confound" §6b is the cautionary tale).

| ID | Weights | KV | Attention | What it isolates | Cost |
|----|---------|-----|-----------|------------------|------|
| **A0** | bf16 | bf16 | dense GQA (≤2048 ctx) | gold reference (the checkpoint floor Δ_baseline) | ref |
| **A1** | **int4 experts** | bf16 | dense GQA | **Δ_quant(weights)** — the MoE int4 cliff, in isolation | cheap |
| **A2** | int4 experts | **fp8_inc** | dense GQA | **Δ_quant(KV)** added on top of A1 | cheap |
| **A3** | int4 experts | fp8_inc | **MSA sparse** (long ctx) | **Δ_sparse** at the configs that actually trigger it | medium |
| **A4** | int4 experts | fp8_inc | MSA sparse | **high-concurrency** quality (c=30+) — does batching/bucketing change answers? | medium |

A1 vs A0 = the int4 question. A2 vs A1 = the fp8-KV question. A3 vs A2 = the MSA question. A4 vs A3 = the concurrency question. Each arrow is a single variable. **Skip nothing**, but A1 is the gate that can kill the whole build cheapest (see §6).

**Out of scope / explicitly NOT tested** (justified):
- *3–4 bit KV (TurboQuant):* not on the table until `fp8_inc` is proven to bind at long ctx (per `minimax-m3-build` memory); add an **A2′ (3-4bit KV)** rung only then, same Δ-vs-A1 method.
- *int4-mixed shared-expert variants:* the recipe is int4-routed-experts-ONLY by contract; shared expert + attn + router stay bf16. No mixed-precision sweep needed unless A1 fails (then bump w2 or the most-sensitive layer to int8 and re-A1).
- *MSA-sparse-vs-dense-GQA as a throughput A/B:* that's the perf review's job, not quality's. Here dense GQA is only the **correctness control**.

---

## 2. Evals, by capability — with datasets and PASS thresholds

Run with `lm-eval` (Optimization Guide §2/§4 names it as the post-quant gate) against the OpenAI endpoint (`--model local-completions --model_args base_url=...`), so the *served* stack is what's measured, not an offline transformers forward. Greedy/`temperature=0`, fixed seed, identical prompts across rungs (the `accuracy_probe.py` discipline: same set ⇒ comparable deltas).

### 2a. General knowledge / reasoning — `mmlu`, `gsm8k`
- **mmlu** (57 subj, 14k Q) — broad knowledge regression. `lm_eval --tasks mmlu --num_fewshot 5`.
- **gsm8k** (8-shot CoT) — multi-step arithmetic; *first* thing int4 MoE damage shows up in (long reasoning chains compound small per-token errors).
- **PASS:** `Δ_quant ≤ 1.0 pp` absolute (A1→A0 and A2→A1 each). Rationale: Optimization Guide §1 cites "<1% accuracy loss" as the FP8 bar on 70B-class; int4-experts-only on a 428B model should be **easier** (experts are a fraction of params, attention/router/embeddings stay bf16), so 1.0 pp is a *generous* ceiling — exceeding it means the MoE int4 path has a real cliff (suspect group_size, sym zero-point — GLM hit the `_sym_zero` 9-vs-8 off-by-one, §III — or a too-generic calibration set, §4). **Hard fail at >2 pp.**

### 2b. LONG CONTEXT — `ruler` + needle (THE POINT OF MSA)
This is where the build lives or dies; MSA only engages > 2048 tokens.
- **RULER** at lengths **{4k, 8k, 16k, 32k, 64k, 128k}** (`niah_single`, `niah_multikey`, `vt`, `qa`). RULER is the right tool because it has structured retrieval/aggregation/tracing subtasks, not just a single needle — it catches "selects the wrong 16 blocks for aggregation" which a single needle misses.
- **Needle-in-haystack** — reuse `$HOME/AI/dsa_e2e/e2e_long_needle.py` (already proven against GLM sparse), swept depth × length, code → just retarget URL/model name + the `glm51ar`→`m3` served-model string.
- **The decisive A3-vs-A2 control** (do exactly what GLM did): run the SAME RULER/needle item at a length **below 2048** (dense path, same serve) and **above 2048** (sparse path). If the long one misses but the short one *also* misses, the miss is Δ_baseline/Δ_quant, **NOT** the MSA wiring — do not chase a sparse bug. If short passes and long fails, **that is a real MSA selection/wiring defect** (the case GLM's §7 gates (b)/(c) were built to separate).
- **PASS:** `Δ_sparse(A3→A2) ≤ 2 pp` on RULER at each length **up to the length where A2-dense itself degrades** (you cannot ask sparse to beat the dense checkpoint floor). Needle: sparse recall must **equal** dense recall at matched depth (±0, it's exact retrieval) for lengths where dense ≥ 90%. **Hard fail:** sparse RULER falls off a cliff (>10 pp drop vs dense) at a length where dense is still strong — that's lost blocks (top-k too small for the task, or the §7-bug-3 causal-tail leak / wrong gather order).
- **Critical confirm from MSA_SPEC §7 ambiguities BEFORE trusting 1M:** the spec flags **index-branch RoPE** and **index activation (ReLU?)** as unconfirmed-from-code. A wrong choice is INVISIBLE at 4–16k and only corrupts selection at very long ctx. So RULER **must** extend to ≥128k (ideally a 256k/512k spot-check) — a build that passes 32k can still be wrong at 1M. This is the highest-leverage long-context check and the one a generic MMLU/GSM8K plan would completely miss.

### 2c. Code + agentic (M3's stated strength — `minimax-m3-build`)
- **HumanEval** / **MBPP** (pass@1, greedy). Code is brittle to quant: one wrong token breaks execution, so it's a sharper int4 probe than prose.
- **A lightweight tool-call/agentic transcript set** — even 20–30 multi-turn function-calling prompts graded for valid-JSON-args + correct-tool (extend `accuracy_probe.py`'s exact-match style). M3 is sold as agentic; a routing/quant regression that prose hides will surface as malformed tool calls.
- **PASS:** `Δ_quant ≤ 2 pp` pass@1 (code tolerates a touch more than knowledge). Agentic: **0 regressions** in valid-call rate vs A0.

### 2d. MoE-routing-sensitive probe (the int4-cliff-specific instrument)
int4 quantizes **only the 128 routed experts** (top-4/token, `num_experts_per_tok=4`). The unique failure mode: quant perturbs expert *outputs* enough to **flip the router's top-4 selection**, compounding error non-locally. MMLU won't localize this. Two cheap instruments:
1. **Router-agreement metric (card-free-ish, decisive):** on a fixed 200-prompt calibration-like set, log the **top-4 expert IDs per token** for A0 (bf16) vs A1 (int4), every MoE layer. Report **top-4 set overlap (Jaccard) and top-1 agreement**. This is the *exact* analogue of the Jaccard/top-k-recall gate the DSA build used for selection parity (PROJECT_FINDINGS §7, dsa-build-quality-bar memory). **PASS: mean top-1 router agreement ≥ 0.98, top-4 Jaccard ≥ 0.95.** Below that, int4 is silently re-routing and *no downstream eval delta can be trusted as stable* — fix the recipe (calibration / int8-bump the offending experts) before anything else.
2. **Per-expert / per-layer activation error** during quant (AutoRound already computes this): flag any expert matrix whose post-quant relative error is a high outlier and selectively raise it to int8. (GLM's lesson: the *stored format is ground truth* and mislabeled bits silently drop weights, §III — assert the router/shared/attn really stayed bf16 in the output checkpoint, which `quantize_m3_int4.py`'s dry-run already checks; extend that assertion to the *loaded* serve.)

---

## 3. Calibration data — the recipe's biggest unforced error

`CALIB_DATASET = "NeelNanda/pile-10k"` is generic web text. **This is wrong for this workload and this architecture**, on two counts:
1. **Domain mismatch:** the serve target is code + agentic + long-context (`minimax-m3-build`). AutoRound's group scales are fit to the calibration activation ranges; calibrating on prose under-represents the code/tool-call token distributions and the long-range activation statistics. Optimization Guide §2/§4 are explicit: *"Calibrate activation ranges on a representative dataset"* / *"never uncalibrated FP8 on MoE."* The same logic binds int4 expert scales.
2. **Length mismatch:** `SEQLEN=2048` calibration cannot exercise the activation regime MSA operates in (>2048, the sparse path). Experts that only ever fire on long-context tokens are calibrated on data that never reaches that length.
3. **Routing coverage:** with 128 experts and top-4, a small generic corpus may **never activate some experts**, leaving their scales effectively un-calibrated (RTN-only). A domain-matched set that exercises the real routing distribution is far safer.

**Recommendation:** replace/augment with a **domain-and-length-matched mix** — code (`bigcode/the-stack` sample or `codeparrot`), agentic/tool-call transcripts, and **long documents at seqlen ≥ 8192** (raise `SEQLEN`), `nsamples ≥ 256` so all 128 experts are likely hit. Keep `iters=0` (RTN, matches GLM) for the first pass, but **the calibration corpus is the highest-ROI knob** and should be fixed before the run, not after a failed eval. *Make calibration choice itself an A/B if A1 is marginal:* generic-calib-int4 vs domain-calib-int4 vs A0 — if domain calibration closes a gap generic leaves open, that's proof the data mattered.

---

## 4. The CHEAP-CLIFF-CATCH subset (run this before committing real card time)

Full RULER@128k × HumanEval × MMLU × the matrix = many GPU-hours. To catch a *cliff* (the build is fundamentally broken) for **<1 card-hour**, run only:

1. **Router-agreement (§2d.1)** on 200 prompts — A0 vs A1. *Card-free-capable on CPU/1-card for a forward pass; no full eval.* If top-1 agreement < 0.98, **STOP** — int4 is mis-routing; no point evaluating further.
2. **GSM8K (200-Q subset), A1 vs A0.** Fastest knowledge/reasoning cliff detector; arithmetic chains amplify per-token int4 error. Δ > 2 pp ⇒ stop.
3. **Needle at 4k AND 16k (dense-vs-sparse control), A3 vs A2**, via `e2e_long_needle.py` — ~10 prompts. Catches MSA wiring/order/causal-tail bugs and the §7 confound in one shot. Sparse-16k miss while dense-4k passes ⇒ real MSA bug, stop.
4. **`accuracy_probe.py` (12-case, already exists), A2 vs A0** — 30-second coherence smoke; word-salad here = a gross dequant/dispatch break (the GLM "Paris, capital, capital" signature).

These four = the **kill-switch gate**. Only if all four pass do you spend hours on full MMLU/RULER-sweep/HumanEval/concurrency (A4). This mirrors the GLM `DSA_READY`-gating discipline: never run the expensive validation (or serve) on a build that fails the cheap parity gate.

---

## 5. High-concurrency-specific quality check (don't assume c=1 ⇒ c=30 is fine)

The goal is c≥30. Batching changes bucketing/padding and, with MoE expert-parallel, the per-step expert grouping. **Re-run the §4 cheap subset against the load-balanced endpoint under live concurrent load (c=30+)** and assert answers are **identical to the c=1 run** (greedy ⇒ they must be, modulo documented non-determinism). A divergence under load = a batching/bucketing/EP correctness bug (analogue of the GLM scheduler-shape A/B, §6c, which proved determinism). Cheap, and the only place concurrency-induced quality loss can hide.

---

## 6. Single most important eval to run FIRST

**A1 vs A0 on the GSM8K-200 subset *and* the router-agreement metric, together (§4.1+§4.2).**

Why first, above even the long-context RULER that is "the point of MSA": **MSA is dense-equivalent ≤2048 and the kernel already has card-free oracle parity** (the algorithm is validated; `tests/test_msa_*` pass). The genuinely *new, unproven, and irreversible-once-you-commit-card-hours* risk is **the int4 MoE quantization of a 428B model with a generic calibration set** — that is the one thing with no precedent here (GLM/Qwen used *pre-made* int4 checkpoints; we are self-quantizing for the first time). If int4 cliffs or silently re-routes, **nothing downstream matters** and every long-context number is built on sand. GSM8K-200 gives the reasoning-cliff signal; router-agreement gives the *mechanism* (is it mis-routing?) so you know whether to fix calibration, bump experts to int8, or proceed. They share one cheap forward sweep over the same 200 prompts. Run them the moment the int4 checkpoint exists, **before** wiring MSA serve or benchmarking.

---

## Confidence

**High** on the method (decompose into Δ_quant/Δ_sparse/Δ_baseline; A/B one axis vs a same-serve control; gate cheap before expensive) — it is the exact discipline that worked on GLM-5.1 DSA and is grounded in this box's own Optimization Guide. **High** on the calibration critique (generic `pile-10k` @ seqlen-2048 is a concrete, fixable defect for a code/agentic/long-ctx + 128-expert target). **Medium** on absolute thresholds (1 pp / 2 pp / 0.98-agreement are principled from the §1 "<1%" precedent and the DSA Jaccard gate, but the true int4-428B floor is unmeasured here — treat the first A1 run as **calibrating the thresholds**, not just passing them). **Medium-low** on tooling readiness: `lm-eval` is *not installed* in any container here (verified) — it must go in the **scratch env** alongside `auto-round`/`transformers-with-m3`, same as the quant deps; RULER needs its synthetic-data generator. The needle + accuracy-probe harnesses already exist and are reusable today.
