# MiniMax-M3 int4 — Benchmark Matrix Report

**Date:** 2026-06-15  **Box:** 8× Gaudi 2 (HL-225)  **Image:** `vllm-gaudi:next` (vLLM 0.22.1rc1)
**Model:** `/mnt/fastmodels/MiniMax-M3-int4-experts-AutoRound` (428B MoE, GQA, AutoRound int4 experts; coherent)
**Serve:** `USE_M3=1 M3_SWIGLUOAI=1 ./serve.sh minimax-m3-int4` — TP8 + expert-parallel, **fp8_inc KV** (self-scaled, no INC json),
dense-GQA (MSA/sparse OFF), `block-size 128`, `max-model-len 8192`, `--async-scheduling`, no prefix caching, `M3_SWIGLUOAI=1` (coherent routed-expert gate).
**Workload:** chat-completions (chat template), 2K-token prompt, 256-token generation, temperature 0, streaming.
**Coherence verified:** "capital of France + river" → clean `<mm:think>` reasoning + correct "Paris / Seine", `finish_reason: stop`.

---

## MoE-speed optimization pass (2026-06-15) — three levers MEASURED, all REJECTED; loop KEPT

Goal was to lift single-stream (~6–8 tok/s) + concurrency by replacing the int4 routed-MoE swigluoai compute.
**Result: none landed. The per-expert Python loop in `m3_swigluoai_patch.py` is the fastest known-good form;
the single-stream wall is structural (host/sync-bound decode), not the MoE compute.** Coherence re-verified
each run ("Paris", "80 mph"=240/3, iterative Fibonacci — all correct). One container, graceful stops, cards idle on exit.

| lever | what | single-stream decode | aggregate (c=30) | verdict |
|---|---|---|---|---|
| **baseline (per-expert loop)** | the shipped `_routed_swigluoai`/`_allexpert_swigluoai` K/E matmul loop | **3.6–8.3 tok/s** (util 18–31%; see variance note) | ~17.4 (documented) | **KEEP** |
| native captured op | `torch.ops.hpu.mixture_of_experts.bias_fused_weights(alpha=1.702, limit=7.0)` | — | — | **REJECTED — broken on :next** |
| batched `torch.bmm` | collapse the K/E loop → one bmm pair + elementwise swigluoai | 5.0 tok/s | **7.7** (TTFT 128 s!) | **REJECTED — regression** |
| n-gram speculative | `--speculative-config {"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":3}` | — (crashes) | — | **REJECTED — crashes at warmup** |

> **Single-stream variance note (important):** the same byte-identical loop config measured **8.3 tok/s** (first run,
> on the pre-existing warm recipe cache) and **3.6–5.0 tok/s** (after a recipe-cache wipe forced fresh recompiles for the
> batched/ngram experiments) — TPOT 240→280 ms, util unchanged 18–28%, **no CPU fallback / no recompile in either**. So
> batch=1 decode is dominated by **recipe-cache/warmup state + the structural host/sync wall**, not the MoE compute form;
> a wiped/rebuilt recipe can be ~2× slower than the original warm cache for the same code. (Keep the warm `recipe_cache`;
> the `false` validation flag means a rebuild is silent — don't wipe it casually.) The batched/ngram comparisons above
> were all taken under the SAME post-wipe state, so they are apples-to-apples regressions regardless of the absolute floor.

**1. Native captured `bias_fused_weights` swiglu op — broken on this image.** The op schema EXISTS and carries the
exact M3 gpt_swiglu defaults: `…(…, float alpha=1.702, float limit=7.)`. But the gpt_swiglu **device kernel is not
viable on `vllm-gaudi:next`** (on-card probe, single Gaudi, lazy): fed the int4 orientation (`permuted_weights=False`,
contiguous `[in=H,out=2I]`) it returns **≈0** (relNorm ~1.0); fed `permuted_weights=True` it **HANGS the card** (Synapse
"Compute or DMA timeout", recoverable per-process). The **plain `silu` overload of the SAME op works perfectly**
(relNorm **4e-3** vs a per-expert reference) — so routing/shapes/zero-bias are all correct; the swiglu activation leg
itself is the problem (matches the capability note's `frontend_blocklist:[eager]` caveat — the device swiglu CGUID is
absent/blocklisted here, only a CPU composite or a degenerate path remains). **Do not pursue this op on :next.** It is the
one that *could* have given GLM-class single-stream; it does not exist in working form on this build.

**2. Batched `torch.bmm` over stacked experts — bit-equal but SLOWER everywhere.** Replacing the per-expert loop with
`stack→bmm(x,W13)→swigluoai→bmm(h,W2)→weighted-sum` is numerically identical (relNorm 4–5e-3, coherent) but **regresses
hard**: single-stream **5.0 vs 8.3** tok/s, c=30 aggregate **7.7 vs 17.4**, TTFT **128 s vs 25 s**. Why: at decode `top_k=4`
the loop is only 4 *already-graph-captured* matmuls; the batched form adds `stack`/`expand`/`transpose`/`sum` host overhead,
and the **all-expert prefill bmm `[E=16, T, 6144]` builds huge intermediates + a heavier recipe that explodes TTFT.**
The decode step *does* graph-capture cleanly either way (no CPU fallback, no mid-serve recompile observed) — so the
**14–19% util is the host/sync wall on the whole decode step (attention + TP all-reduce + sampler), not the MoE loop.**
Reverted to the loop.

**3. n-gram speculative decode — wires in (unlike MTP!) but CRASHES at warmup.** Unlike `deepseek_mtp` (rejected outright +
no int4 MTP weights — see below), **`ngram` is ACCEPTED** by vLLM 0.22 for arch `minimax_m3_vl`:
`SpeculativeConfig(method='ngram', model=None, num_spec_tokens=3)`, "Loading drafter model weights took 0.0000 GB"
(ngram needs no draft weights — its whole appeal). **Gotcha #1:** it is **incompatible with `--async-scheduling`** —
with async on, engine init raises `Value error: async scheduling is only supported with EAGLE/MTP/Draft Model/NGram GPU
kind…`; **drop `--async-scheduling`** (vLLM then auto-disables it anyway). **Gotcha #2 (fatal):** with async dropped it
loads + prompt-warms fine, then **crashes in decode warmup**: `RuntimeError: Worker failed with error 'index 16 is out of
bounds for axis 0 with size 16'` (16 == local experts E). The draft-token decode shape (T = 1+`num_speculative_tokens`)
does not compose with the int4 MoE routing plumbing — the spec-decode multi-token decode step overflows an expert index.
**No n-gram speedup available on this int4 serve.** (Shell: pass the JSON brace/comma-escaped through serve.sh's `bash -lc`:
`--speculative-config=\{\"method\":\"ngram\"\,\"num_speculative_tokens\":3\,\"prompt_lookup_max\":3\}`.)

**Graph-capture health (the one positive confirmation):** warm lazy decode of the kept loop **graph-captures with no eager
fallback and no mid-serve recompile** — `VLLM_HPU_LOG_STEP_CPU_FALLBACKS` logged zero MoE/attention CPU fallbacks, and no
`recompil`/`graph break` lines appear after warmup. So the patch's MoE path is already captured; the speed ceiling is the
**structural single-stream host/sync wall**, the same one the box's `gaudi-single-stream-speed` note documents (~4–8 tok/s
batch=1). The real decode-speed lever remains **fp8 experts** (native E4M3 MME, no int4→bf16 dequant — guide §IV.2), a
separate effort.

---

## The matrix (working int4 config = c0/c2 family: int4-in-HBM, fp8_inc KV, dense-GQA)

Single config (the proven int4 serve) swept over `--max-num-seqs` {16, 64, 128} at `gpu-mem-util` {0.90, 0.94}.
Each row measured at single-stream (batch=1) AND aggregate at concurrency {1, 10, 30, 50}. `block-size 128` held.

| config | max-num-seqs | gpu-mem-util | single-stream tok/s (decode) | aggregate tok/s @ c=10 | @ c=30 | @ c=50 | TTFT p50 (c=30) | peak HBM/card | MTP |
|---|---|---|---|---|---|---|---|---|---|
| int4-in-HBM / fp8_inc KV (as-shipped) | 16  | 0.90 | **5.8** | 10.6 | — | — | — | 85.2 GB | off |
| int4-in-HBM / fp8_inc KV | 64  | 0.94 | **7.6** | 9.6 | **17.4** | **17.6** | 25.4 s | 96.7 GB | off |
| int4-in-HBM / fp8_inc KV | 128 | 0.94 | (≈7.6) | 8.4 | 13.7 | 18.6 | 24.6 s | 96.6 GB | off |

Full per-point detail (aggregate tok/s | per-req decode median | TTFT p50 | TTFT p99 | TPOT):

| mns | c=1 | c=10 | c=30 | c=50 |
|---|---|---|---|---|
| 16  | 3.9 agg / 5.8 dec / 1.43s / 5.0s / 213ms | 10.6 / 1.5 / 5.34s / 31.1s / 764ms | — | — |
| 64  | 3.3 agg / 7.6 dec / 1.43s / 16.8s / 225ms | 9.6 / 1.6 / 9.93s / 41.5s / 827ms | 17.4 / 0.8 / 25.4s / 173s / 1355ms | 17.6 / 0.4 / 33.8s / 269s / 2209ms |
| 128 | (warm; ≈7.6 dec) | 8.4 / 1.6 / 18.95s / 35.9s / 733ms | 13.7 / — / 24.6s / 127s / 1747ms | 18.6 / — / 28.3s / 279s / 2026ms |

> Notes on columns: **single-stream tok/s (decode)** = per-request decode-rate median at batch=1 (the latency floor;
> the *aggregate* c=1 number is lower because a c=1 run of N sequential prompts includes prefill gaps).
> **aggregate tok/s** = total output tokens / wall over all prompts at that concurrency (the saturation headline).
> TTFT climbs steeply with concurrency because a 2K-token prefill blocks decode (single prefill stream).

## Best config + headline numbers

- **Best AGGREGATE throughput:** `max-num-seqs 64, gpu-mem-util 0.94` → **~17.4–17.6 tok/s aggregate, saturating at the
  c≈30 knee.** Going from c=30→c=50 buys nothing at mns=64 (17.6 vs 17.4, within noise) but ~doubles TTFT p50
  (33.8 s vs 25.4 s). **mns=64 is the best config**: it wins decisively at the c=30 knee and ties the field at c=50.
- **`max-num-seqs` does NOT scale M3 the way it scaled GLM (64→128 = 486→793 tok/s on the box).** Across the whole sweep
  the aggregate ceiling barely moves with mns:

  | concurrency | mns=16 | mns=64 | mns=128 |
  |---|---|---|---|
  | c=10 | 10.6 | 9.6 | 8.4 |
  | c=30 | — | **17.4** | 13.7 |
  | c=50 | — | 17.6 | **18.6** |

  mns=128 is *worse* at c=10/c=30 (bigger decode-graph overhead + longer TTFT) and only edges ahead at c=50 — where
  TTFT is already punishing (28 s p50 / 279 s p99). M3's aggregate ceiling is set by the **per-decode-step MoE cost**
  (all-512-expert int4→bf16 dequant per token) + **serialized 2K prefill**, NOT by batch capacity. Raising mns past 64
  only deepens the queue. **The headline knob (mns) that 4×'d GLM is inert on M3.**
- **Single-stream decode:** **5.8–7.6 tok/s** (mns=16 → mns=64; warm bucketing helps). Consistent with the box's known
  batch=1 wall (~4–8 tok/s, host/sync-bound: device util 14–19%, ~90–116 W of 600 W cap even under load).
- **gpu-mem-util 0.90 → 0.94** raised peak HBM 85.2 → 96.7 GB/card (more KV headroom; all 150 c=50 requests OK, no OOM).
  0.94 is the safe ceiling at 8K ctx (96.7/98 GB leaves a thin graph reserve); did not push 0.96.

## MTP (speculative decode) — SKIPPED, with cause

**Attempted** `--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'` (depth-1), 4 launches to get
the JSON past the shell. **Will not wire — two independent, decisive blockers:**

1. **vLLM 0.22 rejects the method.** The engine parses the config fine (`speculative_config: {'method':'deepseek_mtp',
   'num_speculative_tokens':1}`), warns *"method `deepseek_mtp` is deprecated and replaced with mtp"*, then raises
   **`NotImplementedError: Unsupported speculative method: 'mtp'`** in `SpeculativeConfig.__post_init__`
   (`vllm/config/speculative.py:730`). The generic `mtp` path is not registered for arch `minimax_m3_vl`
   (only specific archs like deepseek_v3 resolve a draft model). No flag value fixes this in-image.
2. **The int4 checkpoint has NO MTP weights.** `num_mtp_modules: 7` / `num_nextn_predict_layers: 1` are advertised in
   `config.json`, but the AutoRound int4 export contains **zero** MTP/nextn tensors — layers 60–66 are entirely absent
   from `model.safetensors.index.json` (0 `eh_proj`/`enorm`/`nextn`/`mtp` tensors). There is no draft head to load even
   if vLLM accepted the method. The model class itself *skips* these modules at load (`minimax_m3.py:1071-1079`).

This **empirically confirms** the guide's §IV.2 prediction ("MTP survives fp8, dies under int4"): the AutoRound int4
path drops the MTP modules. To get spec-decode on M3 you need the **fp8/bf16** path (MTP modules kept bf16) AND a vLLM
build that registers the `mtp` method for `minimax_m3_vl`. **No MTP speedup is available on the int4 serve. Skipped.**

Shell note (for the next attempt): serve.sh's `bash -lc "... $VLLM_ARGS"` brace-expands `{"a":...,"b":...}` (splits on
the inner comma → "duplicate keys"). Pass the JSON with escaped braces+comma:
`--speculative-config=\{\"method\":\"mtp\"\,\"num_speculative_tokens\":1\}`. (A file path is NOT accepted — this vLLM
runs the value through `json.loads`.)

## Fabric state (hl-smi) — CLEAN; numbers are real, not collective-masked

- All 8 cards present and healthy across the whole run; **no link down/fault/toggle on any card** (including the
  historically-flaky card 5, `0000:43:00.0`), **zero uncorrectable events**, temps 30–47 °C, power 70–116 W.
- Under concurrent load the engine batches the full request set (Running: up to 30 at c=30, 8 at c=10), but device
  **util stays 14–19 %** and instantaneous generation throughput is choppy (0 tok/s windows when a 2K prefill is in
  flight). This is the host/sync + MoE-step wall, **not** a fabric storm. The tok/s here are trustworthy.

## Method / reproduce

- Driver: `minimax-m3/bench/m3_bench.py` (chat-completions, configurable `--prompt-tokens`; same JSON schema as
  `bench/bench_client.py`, which only has ~20-token prompts). Per-config: warmup at the batch buckets, then measure
  single-stream + each target concurrency. Raw JSON under `minimax-m3/bench/runs/{mns16_gmem90,mns64_gmem94,mns128_gmem94}/`.
- One container at a time (`vllm-active`), every reconfig via `VLLM_ARGS_OVERRIDE` through serve.sh's graceful
  `safe_stop`. No `rm -f`, no scheduler wedge. Cards returned to 768 MB / 0 % idle on exit. Host Kimi download
  (`/mnt/data2`) untouched throughout.
