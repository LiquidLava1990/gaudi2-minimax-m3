# MiniMax-M3 on Intel Gaudi 2 with vLLM (FP8 Mixture-of-Experts)

`gaudi2-minimax-m3` · [all Gaudi 2 recipes](https://github.com/LiquidLava1990)

Experimental recipes for serving MiniMax-M3 (a ~428B-parameter mixture-of-experts model) on one node of 8x Intel Gaudi 2 (HL-225, 96 GB each) with vLLM and vllm-gaudi. The repo holds serve configs, an out-of-tree vLLM plugin, runtime patches, quantization scripts, benchmark scripts, and the design notes written along the way. Both int4-expert and fp8-expert variants were built and measured.

## Status

Experimental reference material, not a production recipe. The champion config is `models/minimax-m3-fp8-static.env`: the files call it "the coherent champion" (`models/minimax-m3-fp8-chunkprefill.env`, `hpu-m3-ext/hpu_m3_ext/m3_moe_fuse_patch.py`). It serves a static on-disk fp8 checkpoint with the unfused SwiGLU-OAI route-b MoE path and native fp8 matmul (`M3_FP8_MME=1`). Its env file says it was validated through a C256 queued overload with 256/256 requests succeeding and zero restarts. Measured at about 405 tok/s aggregate with 128 concurrent requests (see Results). Model weights are not included; produce the fp8 checkpoint with `minimax-m3/quant/quantize_m3_fp8.py`.

### Champion config settings (from `models/minimax-m3-fp8-static.env`)

| Setting | Value |
|---|---|
| Checkpoint | `MiniMax-M3-fp8-experts-AutoRound` (experts float8_e4m3fn with per-output-channel scale; attention, router, shared expert, embeddings and MTP modules in bf16) |
| Image | `vllm-gaudi:next` |
| Parallelism | `--tensor-parallel-size 8 --enable-expert-parallel` |
| Quantization / KV | `--quantization compressed-tensors`, `--kv-cache-dtype fp8_inc` |
| Batch / memory | `--max-num-seqs 128`, `--gpu-memory-utilization 0.75`, `--block-size 128` |
| Context | `--max-model-len 4096` |
| Scheduling | `--async-scheduling --no-enable-prefix-caching` |
| Key env | `M3_FP8_STATIC=1 M3_SWIGLUOAI=1 M3_FP8_SWIGLU_ROUTE=b M3_FP8_MME=1 PT_HPU_LAZY_MODE=0 VLLM_SKIP_WARMUP=true VLLM_HPU_FORCE_CHANNEL_FP8=true` |

## What's here

| Path | Contents |
|---|---|
| `models/*.env` | 18 serve configs, one per variant (base, int4, int4 + KV quant, fp8 runtime, fp8 static eager/lazy/fused/grouped, EAGLE/EAGLE3, chunked prefill, max concurrency, TurboQuant). Each file's header comments explain its purpose. |
| `hpu-m3-ext/` | Python package `hpu-m3-ext`, a vLLM general plugin (entry point `05.hpu_m3_ext`). Registers the M3 model class and installs the M3 patches (SwiGLU-OAI MoE gate, static fp8, MoE fuse, MTP). |
| `minimax-m3/vllm/` | M3 model class (`minimax_m3.py`) and MSA (sparse attention) wiring and platform patch. |
| `minimax-m3/int4_to_fp8_load.py`, `minimax-m3/m3_expert_runtime_patch.py` | Load-time int4 to fp8 conversion, and the `M3_EXPERT_RUNTIME=int4|fp8` switch. |
| `minimax-m3/quant/` | `quantize_m3_int4.py`, `quantize_m3_fp8.py`, and fp8 INC calibration helpers. |
| `minimax-m3/kernel/`, `kernel-kv/` | C++ TPC backends and build scripts for MSA block-indexer, MSA sparse attention, and KV rotate/quantize kernels, with tests. |
| `minimax-m3/mtp/` | Draft-head (MTP/EAGLE3) training, calibration, and CPU-oracle scripts. |
| `minimax-m3/bench/` | Benchmark, concurrency, probe and profiling scripts, plus `M3_MATRIX_REPORT.md` and `KIMI_BASELINE.md`. |
| `minimax-m3/notes/` | Design notes and about 35 review documents (quantization, int4 speed ceiling, SwiGLU-OAI, offload, KV). |
| `minimax-m3/tests/`, `ref/` | Parity and unit tests and reference implementations. |

## How it runs

1. **Config.** A `models/*.env` file sets the checkpoint name, served name, image tag, extra mounts, and vLLM arguments. The champion uses tensor parallel 8 with expert parallel, `--kv-cache-dtype fp8_inc`, `--block-size 128`, and dense grouped-query attention (MSA sparse attention off). Launch is `USE_M3=1 ./serve.sh <config-name>`.
2. **Extension.** With `USE_M3=1` the launcher installs `hpu-m3-ext` inside the container. The plugin loads after the stock vllm-gaudi plugins and after `hpu-moe-ext`, so the M3 architecture registration wins and the patches wrap the already-installed MoE method. The pinned image is not modified.
3. **Patches.**
   - `M3_SWIGLUOAI=1` enables the SwiGLU-OAI routed-expert gate, which M3 needs for coherent output. It runs as a per-expert loop.
   - `M3_EXPERT_RUNTIME=int4` keeps int4 weights in HBM and dequantizes the routed experts per token. `M3_EXPERT_RUNTIME=fp8` converts the same int4 checkpoint to per-channel fp8 (E4M3) once at load and uses the native fp8 MoE op.
   - The `fp8-static*` configs serve a true on-disk fp8 checkpoint (`quantize_m3_fp8.py`) through `m3_fp8_static_patch.py`, with no int4 artifact.

Image: configs use the locally built tag `vllm-gaudi:next` (17 of 18 configs). One config uses `vllm-gaudi:1.24`. The benchmark report records vLLM 0.22.1rc1 for `:next`. The Habana base image used to build these tags is not recorded in this repo.

## Results

Measured on the champion (`minimax-m3-fp8-static`, 2026-06-27), 4,096-token context:

| Concurrency | Throughput |
|---|---|
| 1 (single stream) | about 10.7 tok/s |
| 32 | about 203 tok/s aggregate |
| 64 | about 306 tok/s aggregate (up to about 403, depending on the benchmark tool) |
| 128 | about 405 tok/s aggregate, about 617 tok/s decode; 128/128 requests succeeded, output coherent |

It also passed a 256-request queued overload with 256/256 succeeding and zero restarts (from the config file).

Related configs, for comparison (these are not champion results):

| Config | Measurement | Source |
|---|---|---|
| `minimax-m3-fp8` (int4 checkpoint converted to fp8 at load, `--max-num-seqs 64`) | Aggregate 39.9 / 63.4 / 90 tok/s at c=10 / 30 / 50, still rising at c=50; single-stream about 4.6 tok/s | `minimax-m3/notes/reviews/fp8-push-concurrency.md` |
| `minimax-m3-int4` (`--max-num-seqs 64`, util 0.94) | Single-stream 7.6 tok/s; aggregate 9.6 / 17.4 / 17.6 at c=10 / 30 / 50 | `minimax-m3/bench/M3_MATRIX_REPORT.md` |

Single-stream speed for an identical int4 config varied between 3.6 and 8.3 tok/s depending on whether the recipe cache was warm (same report). The notes conclude fp8 gives about no single-stream gain because decode is host and sync bound.

## Quick start

Prerequisites: an 8x Gaudi 2 node, Docker, a built `vllm-gaudi` image, and the launcher scripts (`serve.sh`, `common.env`) from the sibling repo. This repo does not include them.

```bash
# 1. Make the fp8 checkpoint (needs the original MiniMax-M3 weights, not provided)
python minimax-m3/quant/quantize_m3_fp8.py --help

# 2. Serve the champion config (USE_M3=1 installs hpu-m3-ext in the container)
USE_M3=1 ./serve.sh minimax-m3-fp8-static

# 3. Coherence check, then benchmark the running endpoint
minimax-m3/bench/m3_fp8_vs_int4.sh fp8
```

The env file asks you to confirm coherent output on-card before trusting any tok/s or deleting the int4 or bf16 sources. Paths such as `$HOME/AI/minimax-m3` and the model directory are placeholders; adjust them. The bench scripts read `BIND_IP`, `PORT` and `API_KEY` from `common.env`.

## Other experiments

| Config | What differs from the champion or purpose |
|---|---|
| `minimax-m3-fp8-chunkprefill` | Champion plus `--enable-chunked-prefill --max-num-batched-tokens 4096` and a step timer |
| `minimax-m3-fp8-static-lazy`, `-lazy-fused`, `-lazy-grouped128`, `-eager-mme`, `-eager-mme-fix` | Lazy vs eager frontend, grouped fp8 MoE, and fp8 matmul orientation variants |
| `minimax-m3-fp8` | Int4 checkpoint converted to fp8 at load (`M3_EXPERT_RUNTIME=fp8`) |
| `minimax-m3-int4` | Int4 experts kept in HBM, dense GQA; first coherent M3 serve |
| `minimax-m3-int4kv`, `minimax-m3-k8v8` | Static fp8 experts with KV-cache variants (see each header) |
| `minimax-m3-turboquant` | TurboQuant sub-fp8 KV (fp8 K, int4 V) |
| `minimax-m3-eagle`, `minimax-m3-base` | Int4 with a from-scratch EAGLE/MTP draft, and its no-draft baseline |
| `minimax-m3-fp8-eagle`, `minimax-m3-fp8-eagle3` | Static fp8 base plus a from-scratch MTP draft head |
| `minimax-m3-fp8-maxc` | Max-concurrency tuning of the load-time fp8 path (`max-model-len` 3072, util 0.94) |
| `minimax-m3` | Original bf16 plus INC fp8 bring-up |

Rejected on the int4 image (per `M3_MATRIX_REPORT.md`): the native fused SwiGLU op (returns about zero or hangs the card), a batched `bmm` MoE (5.0 tok/s single, 7.7 aggregate at c=30), and n-gram speculative decode (crashes at warmup).

## Known issues

- Single-stream decode is host and sync bound; the notes conclude it is limited by the whole decode step, not the MoE compute.
- Dense GQA only for serving. MSA sparse attention is gated off pending on-card validation.
- n-gram speculative decode fails at decode warmup (expert index out of bounds).
- Wiping the recipe cache can roughly halve single-stream speed until it is warm again.
- The custom TPC kernels build against version-exact bridge headers (tag `v1.24.0-1007` per the build scripts).
- Notes carry dated, point-in-time findings from June 2026; later notes may supersede earlier ones.

## Related repos

- [gaudi2-common](https://github.com/LiquidLava1990/gaudi2-common): shared kernels (including `hpu-moe-ext`), patches, and launchers used by this recipe.

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights are not included and keep their own licenses.
