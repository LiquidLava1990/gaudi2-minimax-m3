# Giant Open-Model SIZE Reality on 8×Gaudi2 (768 GB HBM)

**Box:** 8 × Gaudi2 × 96 GB = **768 GB HBM total** (decimal GB to match vendor sizing).
**Date:** 2026-06-15. **Bottom line:** the ~1T cohort that just landed **forces int4** — fp8 (~1 TB) blows the box; int4 (~0.5 TB naive / ~0.6 TB native) is the only thing that fits, and even then KV/concurrency headroom is tight.

## Method / weight math

MoE keeps **all** expert weights resident, so footprint scales with **total** params, not active. Bytes/param:

| precision | bytes/param | rule |
|---|---|---|
| bf16 | 2.0 | total × 2 |
| fp8 (e4m3/MXFP8) | 1.0 | total × 1 |
| int4 (naive W4) | 0.5 | total × 0.5 |
| **int4 native (shipped)** | **~0.61** | naive × ~1.22 — scales (per-group), RMSNorms, router, embeddings & lm_head stay bf16/fp32. Calibrated to Kimi 1T → 610 GB native. |

The native-int4 column is the one that matters for real checkpoints (Moonshot ships W4A16 with bf16/f32/i32 metadata). Naive 0.5 B/param is a floor you do **not** actually hit on disk.

## Sizing table — weights only (GB)

| Model | Total | Active | Attn | bf16 | fp8 | int4 naive | int4 native | fp8 fits 768? | int4 fits 768? |
|---|---|---|---|---|---|---|---|---|---|
| **Kimi K2.7-Code** (2026-06-12) | 1.0 T | 32 B | MLA | 2000 | 1000 | 500 | **~610** | **NO** | **YES** |
| Kimi K2 / K2.x (1.04 T) | 1.04 T | 32 B | MLA | 2080 | 1040 | 520 | ~635 | **NO** | YES |
| **GLM-5.2** (2026-06-13) | 744 B | 40 B | MLA+DSA | 1488 | 744 | 372 | ~454 | YES (tight) | YES |
| GLM-5.1 | 744 B | 40–44 B | MLA+DSA | 1488 | 744 | 372 | ~454 | YES (tight) | YES |
| DeepSeek V3 / R1 | 671 B | 37 B | MLA | 1342 | 671 | 336 | ~410 | YES | YES |
| Qwen3-Coder-480B | 480 B | 35 B | GQA | 960 | 480 | 240 | ~293 | YES | YES |
| MiniMax-M3 | 427 B | 26 B | MSA (sparse) | 854 | 427 | 214 | ~261 | YES | YES |
| MiniMax-M3 (428 B figure) | 428 B | ~22–26 B | MSA | 856 | 428 | 214 | ~261 | YES | YES |
| Qwen3-235B-A22B | 235 B | 22 B | GQA | 470 | 235 | 118 | ~143 | YES | YES |

"fp8 fits 768" = weights + a *minimal* KV/runtime reserve (~40–80 GB) still ≤ 768.

## int4-MANDATORY list (fp8 will NOT fit 768 GB)

Any model whose **fp8 weights ≥ ~700 GB** is int4-only on this box (need ≥ ~50–70 GB for KV + activations + framework):

1. **Kimi K2.7-Code** — fp8 = 1000 GB ❌ → int4 native 610 GB ✅ **(int4 MANDATORY)**
2. **Kimi K2 / K2.5 / K2.6 (1.04 T family)** — fp8 = 1040 GB ❌ → int4 ~635 GB ✅ **(int4 MANDATORY)**
3. **GLM-5.2 / GLM-5.1 (744 B)** — **borderline.** fp8 = 744 GB leaves only ~24 GB → no usable KV headroom. **Treat as int4-required in practice** for any real context/concurrency; fp8 only "fits" as a static-weight stunt.

**Still comfortable in fp8** (weights ≤ ~671 GB): DeepSeek V3/R1, Qwen3-Coder-480B, MiniMax-M3, Qwen3-235B. These keep meaningful KV room in fp8.

→ **Thesis CONFIRMED.** For the trillion-param tier, **fp8 (~1 TB) does not fit 768 GB; int4 (~0.5–0.6 TB) does. int4 is mandatory, not optional.** And 744 B is the knee where fp8 stops being viable.

## KV / concurrency headroom for a ~1T int4 model

Take **Kimi K2.7-Code int4 native = 610 GB** on 768 GB:

- **Free HBM after weights ≈ 768 − 610 = ~158 GB.** Reserve ~20–30 GB for the framework/graphs/activations → **~128–138 GB usable for KV cache.**
- Kimi uses **MLA**: KV is a compressed latent, not full per-head K/V. Per-token KV is roughly **~70 KB/token bf16** (≈ kv_lora 512 + rope 64 per layer × 61 layers × 2 B). fp8 KV ≈ **~35 KB/token**.
- **Capacity (fp8 KV, ~130 GB):** ~130 GB ÷ 35 KB ≈ **~3.7 M token-slots.** That is e.g.
  - **~14 concurrent requests @ 256K ctx** (full window), or
  - **~28 @ 128K**, **~57 @ 64K**, **~115 @ 32K**.
- bf16 KV halves those (~7 @ 256K). MLA is what makes ~1T-on-768 GB workable at all — a GQA model this size would starve KV.

**Practical read:** a ~1T int4 model leaves only ~150 GB for everything-not-weights. You can serve long context at *low* concurrency or moderate concurrency at *short* context, but you cannot do both. Weights eat 79% of the box.

## GLM-5.2 (744 B) headroom note

int4 native ~454 GB → **~314 GB free** → ~284 GB KV. With MLA+DSA (sparse) the effective KV per token is even smaller than Kimi's, so GLM-5.2 in int4 is the **comfortable** giant: room for high concurrency *and* long context. fp8 GLM-5.2 (744 GB) is the trap — it "fits" with ~0 KV.

## Caveats / verify-before-build

- int4 *native* sizes (~0.61 B/param) are calibrated from Kimi's published 610 GB. Other vendors' int4 builds vary ±10% with group size and what they keep in bf16. The naive-0.5 column is a hard floor only.
- 768 GB is gross HBM; usable is lower after reserved/fragmentation. Budget ~5–8% off the top.
- Sparse-attn (MiniMax MSA, GLM DSA) and MLA shrink *KV*, not *weights* — weights still set the int4/fp8 cliff.
- Community sub-int4 (e.g. Unsloth dynamic 2-bit GGUF = **340 GB** for Kimi) exists but is **not** the native release and trades quality; not assumed here.

## Sources

- Kimi K2.7-Code release / 1T·32B / native INT4: https://huggingface.co/moonshotai/Kimi-K2.7-Code · https://cryptobriefing.com/kimi-k2-7-code-open-source-release/ · https://www.marktechpost.com/2026/06/12/moonshot-ai-releases-kimi-k2-7-code-a-coding-model-reporting-21-8-on-kimi-code-bench-v2-over-k2-6/
- Kimi K2.7 native INT4 ~610 GB (vs FP16 ~2 TB; 2-bit GGUF 340 GB): https://www.modemguides.com/blogs/ai-news/kimi-k2-7-code-open-source-release
- Kimi K2 1.04T / 32B active / MLA / 384 experts / 8+1: https://arxiv.org/abs/2507.20534 · https://huggingface.co/moonshotai/Kimi-K2-Instruct
- bf16 ≈ 2 TB / fp8 ≈ 1 TB / 4-bit ≈ 500 GB scaling + "MoE keeps all experts resident": https://apxml.com/models/kimi-k2-thinking · https://runaihome.com/blog/kimi-k2-local-inference-hardware-guide-2026/
- GLM-5.2 744B/40B, MLA+DSA, 1M ctx (2026-06-13): https://aitoolly.com/ai-news/article/2026-06-14-zhipu-ai-releases-glm-52-a-fully-open-source-frontier-model-featuring-a-1m-context-window · https://pandaily.com/zhipu-ai-glm-5-dot-2-open-source-mit-jun2026 · https://glm5.net/
- GLM-5.1 744B / 40–44B active / MLA+DSA+MTP: https://lambda.ai/inference-models/zai-org/glm-5.1 · https://www.qubrid.com/blog/glm-5-1-next-generation-agentic-engineering-model
- DeepSeek V3 671B/37B, MLA, 256 experts/8: https://arxiv.org/abs/2412.19437 · https://huggingface.co/deepseek-ai/DeepSeek-V3
- Qwen3-Coder-480B-A35B (GQA 96Q/8KV) & Qwen3-235B-A22B: https://docs.api.nvidia.com/nim/reference/qwen-qwen3-coder-480b-a35b-instruct · https://apxml.com/models/qwen3-235b-a22b-thinking
- MiniMax-M3 427B/26B active, MSA sparse attn: https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3 · https://developer.nvidia.com/blog/deploy-long-context-reasoning-and-agentic-workflows-with-minimax-m3-on-nvidia-accelerated-infrastructure
