# Adversarial Review — KV Minimization / TurboQuant lens

**Scope:** MiniMax-M3 (428B MoE, `MiniMaxM3SparseForCausalLM`, text tower) on 8× Gaudi 2, vllm-gaudi.
Goal: **long context (128K–1M) + high concurrency (30+)** at minimal quality loss. The KV axis.
**Date:** 2026-06-14. Lens: KV cache minimization (TurboQuant / Kitty / KIVI / CommVQ / QJL / PolarQuant).

> TL;DR **VERDICT: ship `fp8_inc` NOW. Do NOT build a 3-4bit KV OpBackend kernel as the next move.**
> A sub-fp8 KV kernel is a *real but second-tier* lever that should sequence **after** MSA + lazy +
> fp8-MoE are live and *only if* a measured long-context concurrency wall proves KV is the binding
> constraint. When/if built, the method is **TurboQuant `k8v4` (fp8 K + 4-bit V, Walsh-Hadamard,
> calibration-free)** — *not* Kitty (2-bit), *not* the 3-bit modes. Confidence: **HIGH** on "fp8 now",
> **HIGH** on "4-bit not 2/3-bit", **MEDIUM** on the exact concurrency-gain numbers (no 1M data exists).

---

## 0. The fact that reframes this whole review (read first)

**MiniMax-M3 KV is GQA real K/V — and MSA already makes the KV READS sparse.** (`MSA_SPEC.md`)
- KV substrate: `H_kv=4`, `head_dim=128`, **NOT MLA latent** (`M2_BASE_MAP.md §0`). Per-token KV =
  `2 (K,V) × 4 heads × 128 × 60 layers`. At bf16 that's ~**122 KB/token**; fp8 ~**61 KB/token**.
- **MSA attends only `topk_blocks=16` of up to 8192 blocks** at 1M (~512× *read* reduction,
  `MSA_SPEC.md §4`). The indexer proxy K-cache adds only **1 head × 128** per token (`MSA_SPEC.md §3`).

Two orthogonal axes are being conflated in the "minimize KV" framing:
1. **KV READ / compute** (which tokens the SDPA touches) → **MSA already solves this** (top-16 blocks).
   This is the dominant 1M-context-tractability + prefill-TTFT lever, and it's the MSA kernel's job
   (`MSA_SPEC.md`), **not** the KV-quant kernel's.
2. **KV STORAGE / HBM footprint** (bytes per cached token) → this is what KV-*quant* (fp8 → 4bit)
   touches. It sets **how many concurrent 1M sequences fit in HBM** = the concurrency ceiling.

So KV quant here is a **pure HBM-capacity / concurrency play**, layered on top of MSA. It does **not**
help decode tok/s (that's MoE-bandwidth/sync-bound, `gaudi-perf-targets.md`, `gaudi-single-stream-speed.md`)
and it does **not** make 1M tractable (MSA does). Frame every number below as "concurrent long
sequences per box", not "tokens/sec".

---

## 1. The hard gate + the hardware floor (why this isn't a flag)

- **vllm-gaudi platform gate:** `is_kv_cache_dtype_supported(...) -> return kv_cache_dtype == "fp8_inc"`
  (`platform.py`, confirmed in repo + `M2_BASE_MAP.md §2a`, `Gaudi2_Optimization_Guide.md` Part III
  "KV-cache compression status"). **fp8_inc is the ONLY accepted KV dtype on HPU.** Period.
- **vLLM CORE now ships sub-fp8 KV — but CUDA/ROCm ONLY.** Google's **TurboQuant** merged upstream
  **2026-04** (PR #38479, official grouped modes #39890). It is the production 3-4bit KV path
  *everywhere except HPU*. **Zero mention of HPU/Gaudi in any TurboQuant PR/backend.** No open
  vllm-gaudi/vllm-fork issue or PR for sub-fp8 KV exists. (vllm-fork is EOL after 1.23; vllm-gaudi is
  the path.) → **Any sub-fp8 KV on Gaudi is "build it yourself."**
- **Gaudi 2 MME has NO native int4/int8 matmul.** MME does FP32/TF32/BF16/FP16/**FP8(E4M3/E5M2)** →
  FP32 accumulate (Habana arch docs; SqueezeBits: 865 TFLOPS FP8 vs 432 BF16). Integers live on the
  TPC (elementwise), not the GEMM engine. **The native attention-compute floor on Gaudi 2 is FP8.**
  This is *exactly why* the gate stops at fp8_inc, and it is the load-bearing constraint:

  > **A 3-4bit KV on Gaudi can ONLY save HBM bytes + bandwidth. The K/V MUST dequant to bf16 (or
  > repack to fp8) before the SDPA.** This matches TurboQuant's own storage-only design
  > (`cache → unpack → dequant-to-BF16 → Q·Kᵀ`, confirmed in PR #38479), so the *algorithm* ports —
  > but the compute never gets cheaper, only the footprint. The dequant adds TPC work on every read.

---

## 2. The 2026 SOTA landscape (what to copy if we build)

All calibration/kernel/transform facts verified against the arXiv papers + vLLM PRs (sources §6).

| Method | Bits | Calib-free? | Transform | Quality @ long-ctx | Kernel | Notes for us |
|---|---|---|---|---|---|---|
| **fp8_inc (INC)** | 8 (E4M3) | needs MEASURE pass | none | ~1–2pt, NIAH ~0.99 | **native MME** | **the gate. per-tensor scales only (INC).** baseline. |
| **TurboQuant k8v4** | fp8 K / 4b V | **yes** | **Walsh-Hadamard (cheap, butterfly-fusible)** | **near-lossless** (NIAH 0.997, GSM8K −0.3..−4pt) | Triton/CUDA | **the one to port.** ~2.6× vs bf16 (~1.3× vs fp8). |
| TurboQuant 4bit_nc | 4b K+V | yes | WHT + 1-bit QJL resid | near-lossless 4× | Triton/CUDA | ~3.8× vs bf16; V-quant is the risk. |
| TurboQuant k3v4_nc | 3b K / 4b V | yes | WHT+QJL | **good if V≥4b** | Triton/CUDA | 4.3×; 3-bit *keys* OK, 3-bit *values* are not. |
| TurboQuant 3bit_nc | 3b K+V | yes | WHT+QJL | **CLIFF** | Triton/CUDA | community: **~0% GSM8K naive**, 71% w/ fixes; AVOID. |
| **Kitty** (2511.18643) | 2b +12.5–25% K@4b | yes (online mag) | **none** (axis mixed-prec) | 2-bit within ~1–2pt; **−3..4pt @32k reasoning** | **Triton, A100/H100** | true 2-bit, but residual long-ctx gap + bespoke page layout. |
| KIVI (2402.02750) | 2b | yes | none (per-chan K/per-tok V) | near-lossless 2-bit (older models) | CUDA | the asymmetric template Kitty builds on. |
| CommVQ (2506.18879) | 1–2b VQ | **NO (EM+train)** | RoPE-commuting codebook | 2b lossless, **1b holds** | Triton | only credible 1-bit; **costs codebook training**. 128K max tested. |
| QJL (2406.03482) | ~3b (1-bit sign) | yes | JL random proj | ~lossless @3b, no quant consts | CUDA | the residual-correction trick inside TurboQuant. |
| PolarQuant (2502.02617) | ~3.9b (K-only) | yes | polar + precond | SOTA @4.2× (K-only) | fused | K-only; pair w/ V scheme. |

**Winner if we build: TurboQuant `k8v4`.** It is (a) the upstream-blessed design (port, don't invent),
(b) calibration-free (no MEASURE-pass fragility, unlike fp8_inc), (c) Walsh-Hadamard = a **structured,
cheap, butterfly-fusible** rotation that maps to our OpBackend toolkit, (d) keeps **K in fp8 + V at
4-bit** — and the research is unanimous that **values dominate sub-4bit quality loss**, so k8v4 sits
exactly at the safe knee. **Reject Kitty/2-bit and all 3-bit-value modes:** the 1M reasoning gap and
the 3-bit cliff are real, and we have no 1M quality data to derisk them. **Reject CommVQ:** training a
codebook violates the calibration-free / no-slop bar and there's no 1M evidence.

---

## 3. The gain/cost math — does 4-bit KV buy enough at 1M × 30+?

**The capacity arithmetic (the only place KV quant helps):** at 1M tokens, per-sequence KV is the cost.
- bf16 KV ≈ 122 KB/tok → **122 GB / 1M-seq**. Won't fit even one across the box's KV budget → bf16 is a
  non-starter at 1M regardless. (This is moot — the gate forbids bf16 KV anyway.)
- **fp8_inc** ≈ 61 KB/tok → **~61 GB / 1M-seq.** On 8×96GB with the model + graph reserve, the KV pool
  is the few-hundred-GB range → **order ~3–8 concurrent 1M sequences**, scaling ~linearly to 30+ as
  context drops (e.g. 128K ≈ 7.6 GB/seq fp8 → tens of concurrent seqs).
- **TurboQuant k8v4** ≈ fp8 K + 4-bit V ≈ ~46 KB/tok → **~1.33× the fp8 sequence count.**
- **4bit_nc** ≈ ~31 KB/tok → **~2× the fp8 sequence count** (but 4-bit *keys* adds risk + more dequant).

**So the realistic ceiling on the prize:** a sub-fp8 KV kernel buys **~1.3× (k8v4) to ~2× (4bit_nc)
more concurrent long sequences / more context-per-seq** over fp8_inc — at **near-zero quality cost for
k8v4**, modest-but-real cost for 4bit. It buys **~0% extra decode tok/s** (compute-bound elsewhere) and
**0 extra 1M-tractability** (MSA owns that). The *only* metric it moves is **HBM-bound concurrency/context**.

**The cost to capture it (OpBackend kernel weeks):** real and non-trivial.
- New OpBackend op(s): `kv_quant_store` (WHT-rotate → Lloyd-Max scalar quant → pack 4-bit V into HBM)
  + `kv_dequant_load` (unpack → dequant-to-bf16/fp8 → feed SDPA). The chain primitives exist in the
  bridge (`convert_to_fp8`, `gemm`/Hadamard via `bmm`, `scatter`/`index_copy` pack, the `sdpa`/`fp8_sdpa`
  consume) per `DSA_OPBACKEND_KERNEL_DESIGN.md §3a` — **same toolkit as MoE/DSA/MSA**, so it's "compose
  existing nodes", not a TPC ISA write. But it must also **extend the platform gate**
  (`is_kv_cache_dtype_supported` → accept a new dtype, out-of-tree like the DSA `platform_patch`), and
  thread a **new KV layout** through the vllm-gaudi paged-attention / cache-spec path — the same
  "hardest plumbing" risk the indexer cache hit (`DSA_OPBACKEND_KERNEL_DESIGN.md §8 risk 3`).
- Lloyd-Max codebooks per head_dim are data-free (computed from the known Beta marginal) — good, no
  calibration. But the **WHT must fuse into the store/load** or the per-read rotation becomes a TPC tax
  that, on a box already **sync/host-bound at low batch** (`gaudi-single-stream-speed.md`), can *eat*
  the bandwidth win. This is the #1 measurement risk.
- Honest estimate: **1.5–3 kernel-weeks** (store/load ops + gate + cache-spec plumbing + on-card parity
  vs a CUDA TurboQuant oracle + the WHT-fusion measurement), i.e. comparable to an MVP DSA/MSA phase.

**Verdict on worth-it:** **NOT worth it now.** Paying ~2 kernel-weeks for a **1.3–2× concurrency**
bump is poor ROI **while MSA, lazy capture, and fp8-MoE are the items that actually unlock the workload
and move tok/s.** It becomes worth it **iff** a measured run shows the box is **HBM-KV-bound** at the
target (e.g. "we want 32 concurrent 128K seqs and fp8 KV OOMs at 22") — a falsifiable trigger, not a
foregone conclusion.

---

## 4. Sequencing (where this slots)

0. **NOW:** serve M3 with **`--kv-cache-dtype fp8_inc`** (already in the draft env, `M2_BASE_MAP.md §4`).
   It's the gate, native MME, ~2× footprint vs bf16, ~1–2pt loss. **This is the KV answer for bring-up
   and likely for production.** Run the INC MEASURE pass; validate NIAH + a long-ctx reasoning probe.
1. Land **MSA sparse** (the real long-ctx + TTFT lever) → then **lazy capture** → then **fp8 MoE**.
   These dominate both tractability and throughput; KV quant changes neither.
2. **MEASURE the binding constraint** at the real target (e.g. 30+ × 128K, and the 1M case): is the box
   KV-HBM-bound, or MoE/compute/concurrency-scheduler-bound? Capture per-card KV-pool occupancy at the
   concurrency wall. **This run decides step 3.** (Per `always-find-a-way`: measure the real constraint
   before declaring KV the knob.)
3. **IFF KV-HBM-bound:** build the **TurboQuant `k8v4` OpBackend KV kernel** (store/load ops + gate
   extension + cache-spec plumbing), gated `KV4_READY=False` + loud-fallback-to-fp8_inc exactly like
   `DSA_READY`/`MSA_READY` (`dsa-build-quality-bar`, `platform_patch.py` pattern). Parity-gate against a
   CUDA TurboQuant reference; ship only after on-card NIAH ≥ fp8 and a long-ctx reasoning probe within
   tolerance. **Skip 3-bit and 2-bit entirely** unless a *measured* capacity shortfall forces a documented,
   separately-gated experiment.

---

## 5. #1 recommendation

**Ship `fp8_inc` now; do not build a sub-fp8 KV kernel until a measured long-context run proves the box
is KV-HBM-bound at the concurrency target.** The dominant long-context wins are **MSA (read-sparsity →
1M-tractable + TTFT)** and the **MoE/lazy/fp8 path (throughput)** — KV *storage* quant is a pure ~1.3–2×
concurrency multiplier on top, worth ~2 kernel-weeks only after those land and only if HBM is the wall.
When that trigger fires, port **TurboQuant `k8v4` (fp8 K + 4-bit V, Walsh-Hadamard, calibration-free)** —
never the 2-bit (Kitty) or 3-bit-value modes (no 1M quality data; documented reasoning cliffs).

---

## 6. Sources (verified)

- **Gate / HW floor:** vllm-gaudi `platform.py` `is_kv_cache_dtype_supported`; Habana Gaudi Architecture
  (MME dtypes — no int matmul); SqueezeBits "Intel Gaudi #4: FP8 Quantization"
  (https://blog.squeezebits.com/intel-gaudi-4-fp8-quantization--40269); INC FP8 docs
  (https://intel.github.io/neural-compressor/latest/docs/source/3x/PT_FP8Quant.html — per-tensor scales only).
- **TurboQuant (the SOTA + upstream):** arXiv 2504.19874 (Google; random-rotation + per-coord scalar
  quant; data-oblivious; ≈2.7× of info-theoretic optimal; **3.5b lossless / 2.5b −0.6 LongBench**; theory
  uses a *dense* Haar rotation — the WHT is the vLLM impl's structured substitution). vLLM PR #38479
  (k8v4/4bit_nc/k3v4_nc/3bit_nc presets; WHT; calibration-free; **storage-only dequant-to-BF16**; CUDA
  SM86/SM90; **no HPU**), PR #39890; blog https://vllm.ai/blog/2026-05-11-turboquant
  (NIAH 0.997 @4× to 104K; 4-bit <0.3pt on MMLU/HumanEval/GSM8K). Community 3-bit cliff (mgoin "0% gsm8k"
  tq3; fixed only w/ fp8 values) — PR #38479 thread.
- **Kitty (2-bit):** arXiv 2511.18643 (MLSys'26; dynamic channel-wise INT4 boost over KIVI; **−3..4pt @
  32k AIME**; Triton, A100/H100; code github.com/Summer-Summer/Kitty). **KIVI** 2402.02750 (asymmetric
  per-chan-K/per-tok-V 2-bit; tuning-free; 2.6× mem / 2.35–3.47× tput; CUDA).
- **Others:** CommVQ 2506.18879 (1–2b VQ; **needs EM training**; 128K max). QJL 2406.03482 (1-bit JL
  sign; no quant consts; calib-free). PolarQuant 2502.02617 (K-only polar; ~3.9b; calib-free).
  QuaRot 2404.00456 (fused Hadamard, W4A4KV4, calib-free) / SpinQuant 2405.16406 (learned rotation,
  needs training) — the Hadamard-rotation precedent. SageAttention 2410.02367 (fp8 attention baseline).
- **This box:** `Gaudi2_Optimization_Guide.md` (Part III KV status, perf), `DSA_OPBACKEND_KERNEL_DESIGN.md`
  (OpBackend mechanism + fp8 chains + gate-patch pattern), `MSA_SPEC.md` / `M2_BASE_MAP.md` (M3 = GQA
  real-KV, 1M, MSA read-sparsity, fp8_inc env), memory: `gaudi-perf-targets`, `gaudi-single-stream-speed`,
  `dsa-build-quality-bar`, `always-find-a-way`.
