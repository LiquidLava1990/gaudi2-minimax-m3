# MiniMax-M3 on vllm-gaudi — port plan from `minimax_m2.py` + MSA sparse-attn gate

Status: **PLAN ONLY — no code written.** All paths absolute. Line cites are against the
files as read 2026-06-14.

Goal: serve `MiniMaxM3SparseForConditionalGeneration` (config arch) **text-first** (skip the
vision tower) on 8× Gaudi 2, by porting the existing `minimax_m2` HPU model class and lifting
the platform's sparse-attention hard-raise behind an `MSA_READY` gate — mirroring exactly how
`hpu-dsa-ext` lifts the same raise for DSA, never silently falling back to dense.

---

## 0. The one architectural fact that drives everything

**MSA (MiniMax Sparse Attention) is GQA, NOT MLA.** From `/mnt/data2/minimax-m3-bf16/config.json`
`text_config`: `num_attention_heads=64`, `num_key_value_heads=4`, `head_dim=128`,
`rotary_dim=64` / `partial_rotary_factor=0.5`, `use_qk_norm=true`, `qk_norm_type="per_head"`.
There is **no** `kv_lora_rank` / `q_lora_rank` / `index_topk`. The sparse config lives under
`text_config.sparse_attention_config` (keys: `use_sparse_attention`, `sparse_index_dim=128`,
`sparse_num_index_heads=4`, `sparse_topk_blocks=16`, `sparse_block_size=128`,
`sparse_attention_freq` [60-len 0/0/0/1…], `sparse_disable_index_value`, `sparse_score_type="max"`,
`sparse_init_block=0`, `sparse_local_block=1`).

Consequences:
- M3 uses the **standard (non-MLA) HPU attention backend** `HPUAttentionBackendV1`
  (`$HOME/AI/_src/vllm_gaudi/v1/attention/backends/hpu_attn.py:21-22`, impl
  `HPUAttentionImpl` at `$HOME/AI/_src/vllm_gaudi/attention/backends/hpu_attn.py:362`),
  **not** `HPUMLAAttentionBackend`. So this is a *different gate branch* than DSA in
  `platform.py` (DSA was `use_sparse AND use_mla`; MSA is `use_sparse AND NOT use_mla`).
- The DSA wiring (`oot_mla.py`, `sparse_mla_consume`, indexer K-cache, MLA forward monkeypatch)
  is the **pattern to mirror, not the code to reuse** — M3's kernel attaches to the GQA
  `HPUAttentionImpl.forward`, not to `HPUMLAAttention.forward_impl`.
- M3 is **bf16** today (`config.json` has no `quantization_config`; `torch_dtype: bfloat16`).
  The AutoRound int4 autodetect path (`hpu-moe-ext`) is **inert** on this checkpoint and only
  matters once an int4 export exists. Plan keeps it wired (free, autodetected) but text-first
  bring-up runs bf16.

---

## 1. Structural map of `$HOME/AI/_src/vllm_gaudi/models/minimax_m2.py`

| Component | Class / fn | Lines | Notes |
|---|---|---|---|
| Imports | — | `26-50` | Pulls `Attention` (non-MLA) from `vllm.model_executor.layers.attention:33`; `FusedMoE:37`; `MiniMaxText01RMSNormTP` (the per-head q/k norm) from `vllm…mamba.linear_attn:38`; `get_rope:43`; loaders `46`; interfaces `SupportsPP, SupportsLoRA:48`. |
| **MoE block** | `HpuMiniMaxM2MoE` | `53-111` | `tp_size` guard `62-66`; **routing bias** `e_score_correction_bias` (fp32 param + custom `ebias_weight_loader`) `67-72,98-101`; `self.experts = FusedMoE(...)` `74-87` with `use_grouped_topk=True, num_expert_group=1, topk_group=1, scoring_func=config.scoring_func, renormalize=True`; `self.gate = ReplicatedLinear(..., params_dtype=fp32, quant_config=None)` `89-96`. `forward` `103-111`: gate in fp32 → `experts(hidden, router_logits)`. |
| **Attention (GQA)** | `HpuMiniMaxM2Attention` | `114-204` | TP head split `134-147`; `qkv_proj=QKVParallelLinear` `154-162`; `o_proj=RowParallelLinear` `164-170`; partial-rope wiring (`partial_rotary_factor = rotary_dim/head_dim`) `172-178`; **`self.attn = Attention(...)`** the standard non-MLA layer `179-188` (takes `per_layer_sliding_window`, `cache_config`, `quant_config`); per-head q/k RMSNorm `190-191`. **`forward` `193-204`**: qkv → split → `MiniMaxText01RMSNormTP.forward_qk(q_norm,k_norm,q,k)` `200` → rope `201` → `self.attn(q,k,v)` `202` → o_proj. **This is the single dispatch point into the Gaudi attention backend** — `self.attn` (vllm core `Attention`) is what calls `get_attn_backend(...)` and lands on `HPUAttentionImpl`. |
| **Decoder layer** | `HpuMiniMaxM2DecoderLayer` | `207-272` | `layer_idx` parsed from prefix `224`; builds `self_attn=HpuMiniMaxM2Attention` `227-240` and `self.block_sparse_moe=HpuMiniMaxM2MoE(prefix=f"{prefix}.mlp")` `242-246` (note: weight prefix is `.mlp`, attr is `block_sparse_moe`); pre/post RMSNorm `247-248`. `forward` `250-272`: residual-style norm, attn, post-norm, MoE. **Every layer is MoE here — M2 has no dense-first-k layers.** |
| **Model** | `HpuMiniMaxM2Model` (`@support_torch_compile`) | `275-437` | `embed_tokens` (PP-gated) `290-298`; **`make_layers(config.num_hidden_layers, lambda…DecoderLayer)`** `300-310`; final `norm` `312-315`. `forward` `322-343`. **`get_expert_mapping`** `345-352` → `FusedMoE.make_expert_params_mapping(ckpt_gate="w1", down="w2", up="w3", num_experts=num_local_experts)`. **`load_weights` `354-437`** — see §1a. |
| **ForCausalLM** | `HpuMiniMaxM2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP)` | `440-491` | `packed_modules_mapping` (qkv) `441-447`; builds `self.model` `457`, `lm_head` (PP-gated) `458-461`, `logits_processor` `462`. `load_weights` `486-488` delegates to `AutoWeightsLoader(self)`. `get_expert_mapping` `490-491`. |
| Spec-decode skip | `get_spec_layer_idx_from_weight_name` | `494-500` | Skips MTP/nextn layers (`num_mtp_modules`) during main-model load. M3 has `num_mtp_modules=7` / `num_nextn_predict_layers=1` — same mechanism applies. |

### 1a. Weight loading entry points (the load path to preserve)

- Top: `HpuMiniMaxM2ForCausalLM.load_weights` → `AutoWeightsLoader(self).load_weights` (`486-488`).
- Real work: `HpuMiniMaxM2Model.load_weights` (`354-437`):
  - `stacked_params_mapping` (qkv fuse: q/k/v_proj → qkv_proj) `355-360`.
  - `expert_params_mapping = get_expert_mapping()` (w1/w2/w3 → fused experts) `364`.
  - Per weight: skip `rotary_emb.inv_freq` `369`; skip spec-decode layers `372-374`; **the
    critical `mlp.experts.` guard** `386-387` (skip stacked-rewrite for expert tensors so the
    expert loader handles them); stacked attn-proj branch `376-399`; expert branch `401-419`
    (loader called with `(param, weight, name, shard_id, expert_id)`); else-branch handles
    kv-scale remap (`maybe_remap_kv_scale_name`) `425-435`.
- **Experts vs attention split:** attention/gate/norm tensors go through `default_weight_loader`
  or the qkv stacked loader; routed-expert tensors (`w1/w2/w3`, the bulk) go through the FusedMoE
  per-expert `weight_loader` with `expert_id`. This split is exactly what `hpu-moe-ext`'s
  autodetect later overrides for int4 (experts → `MoeWNA16HPUMethod`).

### 1b. How attention reaches the Gaudi backend (no model-side backend choice)

The model never names a backend. `self.attn = Attention(...)` (`minimax_m2.py:179`) is vllm core's
non-MLA attention layer; at construction it calls `get_attn_backend(...)` which calls
`HpuPlatform.get_attn_backend_cls(...)` (`$HOME/AI/_src/vllm_gaudi/platform.py:60-85`). With
`use_mla=False, use_sparse=False` it returns `HPUAttentionBackendV1` (`platform.py:83-85`). The MLA
twin (`MultiHeadLatentAttentionWrapper`, `$HOME/AI/_src/vllm_attn/mla.py:33`;
`mla_attention.py:386-391` passes `use_sparse`) is the path DSA uses — **M3 does not touch it.**

---

## 2. The platform sparse-raise + how DSA lifts it (the pattern to replicate)

### 2a. The raise

`$HOME/AI/_src/vllm_gaudi/platform.py:75-76`:
```python
if attn_selector_config.use_sparse:
    raise NotImplementedError("Sparse Attention is not supported on HPU.")
```
This fires **before** the `use_mla` branch (`78-81`) and the default non-MLA return (`83-85`).
So *any* sparse model — MLA or GQA — is blocked here. M3 (GQA, `use_sparse=True`, `use_mla=False`)
hits this and would otherwise raise.

Two more relevant gates:
- KV-cache dtype: `is_kv_cache_dtype_supported` (`platform.py:388-389`) returns **only**
  `kv_cache_dtype == "fp8_inc"`. → the serve env must use `fp8_inc` KV (or `auto`/bf16 which the
  runner maps; but the box convention is explicit `fp8_inc`).
- Block size forced to 128 (`platform.py:135-136`), prefix-caching auto-disabled under
  Contiguous-PA (`platform.py:210-212`).

### 2b. How `hpu-dsa-ext` lifts it WITHOUT editing `/opt` (out-of-tree plugin pattern)

`$HOME/AI/hpu-dsa-ext/hpu_dsa_ext/platform_patch.py`:
- Saves the original unbound fn: `_ORIG = HpuPlatform.get_attn_backend_cls.__func__` (`:15`).
- `DSA_READY = False` (`:19`) — the readiness gate; `FULL_ATTENTION_FALLBACK = True` (`:44`) — the
  "never silently dense → run *genuine* dense, loudly, until the kernel validates" safety flag.
- `_patched(cls, selected_backend, attn_selector_config, num_heads)` (`:47-59`): when
  `use_sparse` **and** `use_mla`, if `not DSA_READY` → **raise a clear NotImplementedError**
  (`:50-56`, "Refusing to serve DENSE in place of sparse"); else return the MLA backend. Sparse
  **without** MLA falls through to `_ORIG` (`:58-59`) — *this is exactly the M3 case today, so it
  still raises the stock message.*
- `apply()` rebinds: `HpuPlatform.get_attn_backend_cls = classmethod(_patched)` (`:62-63`).
- Installed via the `vllm.general_plugins` entry point `05.hpu_dsa_ext` →
  `hpu_dsa_ext/__init__.py:register()` calls `platform_patch.apply()` first (`__init__.py` register
  body), then registers the sparse stack, then flips `DSA_READY` **only** if `HPU_DSA_READY=1` AND
  every component imported cleanly (`glm_moe_dsa_wire.maybe_flip_ready`, `:230-241`).
- The mount/install is done by `serve.sh` for every model:
  `-v $HERE/hpu-moe-ext:/opt/hpu-moe-ext`, and DSA only when `USE_DSA=1`
  (`${USE_DSA:+-v "$HERE/hpu-dsa-ext:/opt/hpu-dsa-ext"}` + `pip install -e`). **No file under
  `/opt/vllm-gaudi` is ever edited** — the patch monkeypatches the imported class in-process.

**Replicate for MSA:** an `hpu-msa-ext` plugin whose `platform_patch.py` extends the same
`_patched` so the `use_sparse and NOT use_mla` branch returns `HPUAttentionBackendV1` when
`MSA_READY` is True (gated by `HPU_MSA_READY=1`), and otherwise raises the same "never dense"
message. Keep an `MSA_FULL_ATTENTION_FALLBACK` flag with identical semantics to DSA's: until the
MSA kernel validates, route MSA layers through **genuine dense GQA** (the stock
`HPUAttentionImpl.forward`) — loudly logged, never a silent empty-topk consume.

> **Co-existence note:** if both plugins are mounted, both rebind `get_attn_backend_cls`. The MSA
> patch must chain through DSA's `_patched` (capture whatever `get_attn_backend_cls` is at apply
> time as its `_ORIG`, not the stock one), so the MLA-sparse branch keeps working. Entry-point
> name `06.hpu_msa_ext` (sorts after `05.hpu_dsa_ext`) gives that ordering. For text-first
> bring-up, simplest is to mount **only** `hpu-msa-ext` (no `USE_DSA`).

---

## 3. M3 PORT PLAN — how `minimax_m3.py` differs from `minimax_m2.py`

Create `$HOME/AI/_src/vllm_gaudi/models/minimax_m3.py` by copying `minimax_m2.py` and changing:

1. **HF-config handling — read `text_config`.** M3's top config is `model_type=minimax_m3_vl` with
   the LM under `text_config` and the tower under `vision_config`. The HPU model must operate on
   the **text config**. Two options (pick at build time in `__init__`):
   `text_cfg = getattr(hf_config, "text_config", hf_config)` and pass `text_cfg` everywhere M2 used
   `config`. Confirm vllm surfaces `hf_config.text_config` (transformers nested config); if
   `get_text_config()` is available on `ModelConfig`, prefer it. **All of: `hidden_size=6144`,
   `num_hidden_layers=60`, `num_local_experts=128`, `num_experts_per_tok=4`, `scoring_func=sigmoid`,
   `use_routing_bias=true`, `rms_norm_eps=1e-6`, `head_dim=128`, `rotary_dim=64`,
   `partial_rotary_factor=0.5`, `rope_theta=5e6`, `max_position_embeddings=1048576` live in
   `text_config`.**

2. **Dense-first-k + per-layer MoE freq.** Unlike M2 (all-MoE), M3 has `moe_layer_freq`
   `[0,0,0,1,1,…]` → **layers 0,1,2 are DENSE** (`dense_intermediate_size=12288`), layers 3-59 are
   MoE. The decoder layer must branch: `if moe_layer_freq[layer_idx]==1:` build
   `HpuMiniMaxM3MoE` (the M2 MoE block), else build a dense MLP (`SwiGLU`,
   `dense_intermediate_size`). Also M3 has **shared experts** (`n_shared_experts=1`,
   `shared_intermediate_size=3072`) added alongside routed experts on MoE layers — M2 has none.
   Activation is `swigluoai` with `swiglu_alpha=1.702`, `swiglu_limit=7.0`; `use_gemma_norm=true`.
   `routed_scaling_factor=2.0`. (These are net-new vs M2; check whether the in-tree MiniMax-M2 /
   reference M3 HF modeling file under `$HOME/AI/minimax-m3/ref/` already encodes the
   dense/shared/activation details before re-deriving.)

3. **The MSA sparse-attention attention class.** Replace `HpuMiniMaxM2Attention.self.attn` usage so
   that for **sparse layers** (`sparse_attention_freq[layer_idx]==1`, i.e. the same 3-dense-then-
   sparse pattern) attention runs the MSA kernel. The MSA kernel is **block-sparse top-k over GQA**
   (an index head set of `sparse_num_index_heads=4`, `sparse_index_dim=128`, selecting
   `sparse_topk_blocks=16` blocks of `sparse_block_size=128`, with `sparse_init_block`+
   `sparse_local_block` always-kept) — it comes from **task `hpu-msa-ext`** (the MSA analogue of
   `hpu-dsa-ext`; the DSA repo's indexer/consume code is the template). Wire it the DSA way:
   - Build the indexer/selection as a sibling module; expose a `topk_blocks` buffer per sparse
     layer (analogue of DSA's `topk_indices_buffer`).
   - Monkeypatch the **GQA** impl `HPUAttentionImpl.forward`
     (`$HOME/AI/_src/vllm_gaudi/attention/backends/hpu_attn.py:362`), NOT the MLA
     `forward_impl`, restricting the paged-attention key set to the selected blocks. Mirror
     `sparse_mla_wire.py` structure: non-sparse layers + profiling run → original dense forward
     verbatim; sparse layer with `MSA_FULL_ATTENTION_FALLBACK` True → genuine dense; only with the
     gate flipped does the real block-sparse consume fire. **No silent dense; raise on unsupported
     cases** (`sparse_mla_wire.py:9` mandate).
   - Until `hpu-msa-ext` exists and validates, the model is fully functional as **dense GQA**
     (FULL_ATTENTION_FALLBACK) — coherent, just not yet sparse. This is the bring-up baseline.

4. **Register `MiniMaxM3SparseForConditionalGeneration`.** Add to
   `$HOME/AI/_src/vllm_gaudi/models/__init__.py:register_model()` (alongside the M2 line at
   `:32-33`):
   ```python
   from vllm_gaudi.models.minimax_m3 import HpuMiniMaxM3ForConditionalGeneration
   ModelRegistry.register_model(
       "MiniMaxM3SparseForConditionalGeneration",
       "vllm_gaudi.models.minimax_m3:HpuMiniMaxM3ForConditionalGeneration")
   ```
   Note the config's **top** arch is `MiniMaxM3SparseForConditionalGeneration` (multimodal) while
   `text_config.architectures=["MiniMaxM3SparseForCausalLM"]`. vLLM resolves on the **top-level**
   arch, so register the `…ForConditionalGeneration` name. For **text-first**, the registered class
   can be a *ForCausalLM-shaped* model that ignores image inputs (see #5) — it does not need to
   implement `SupportsMultiModal` yet, but it must accept the `…ForConditionalGeneration` arch name.

5. **TEXT-FIRST — skip the vision tower.** Config keys that gate vision:
   `vision_config` (CLIP, 32 layers), `image_token_index=200025`, `video_token_index=200026`,
   `image_seq_length=576`, `process_image_mode=dynamic_res`, `multimodal_projector_bias`,
   `vision_feature_layer`. **Plan:** build **only** `text_config` modules (embed/layers/norm/
   lm_head). Do **not** instantiate `vision_config` or the projector. Implement the class as a pure
   `ForCausalLM` (text tokens only) registered under the `…ForConditionalGeneration` name; do not
   declare `SupportsMultiModal`. Bring-up traffic is text-only (no `<image>` tokens). The vocab
   already includes the image/video token ids but they simply won't appear in text prompts. (A
   later vision pass adds the tower + `get_multimodal_embeddings` — out of scope here.)

6. **AutoRound int4 experts via `hpu-moe-ext` (autodetect, no flag).** Today's checkpoint is bf16
   (no `quantization_config`) so this is inert. The mechanism, already mounted for every model by
   `serve.sh` (`-v …hpu-moe-ext`, `pip install -e`):
   `hpu_moe_ext/__init__.py:_install_config_patches` (`:48-110`) patches GPTQ/AWQ configs;
   `auto_round_hpu.AutoRoundHPUConfig` (`:102-104`) registers `auto-round`/`auto_round` and
   **autodetects** from `config.json::quantization_config.quant_method`
   (`override_quantization_method`, `auto_round_hpu.py:153-161`) — **no `--quantization` flag**.
   Routed experts → `MoeWNA16HPUMethod` (`auto_round_hpu.py:305-313`); ground-truth quant detection
   reads safetensors shard headers for `.qweight` (`_load_quantized_prefixes`, `:182-230`) using
   `HPU_MOE_EXT_CKPT` (set by `serve.sh` to `/models/$MODEL_PATH`). Because M3's MoE block reuses
   `FusedMoE` exactly like M2 and `get_expert_mapping` uses the same `w1/w2/w3` mapping, an int4
   AutoRound M3 export will autodetect and route experts to the HPU kernel **with no model-side
   change** — provided the M3 model class keeps the M2 expert wiring (it does). The router `gate`
   stays bf16 via the `last=="gate"` skip (`__init__.py:_skip_linear:32-33`).

7. **Readiness gate (`MSA_READY`).** New `hpu-msa-ext/hpu_msa_ext/platform_patch.py` with
   `MSA_READY=False` + `MSA_FULL_ATTENTION_FALLBACK=True`, lifted only when `HPU_MSA_READY=1` AND
   the MSA indexer/consume registered cleanly (copy `hpu_dsa_ext.__init__._load_component` strict
   logic, `__init__.py:_load_component`). Until then: sparse layers run **genuine dense GQA**,
   loudly; never empty-topk garbage.

---

## 4. Draft serve env — `$HOME/AI/models/minimax-m3.env`

Aligned to the box convention (`glm51ar-dsa.env` / `glm51ar-lazy.env`; consumed by
`$HOME/AI/serve.sh`). Two blockers to resolve first (see §6): the model lives at
`/mnt/data2/minimax-m3-bf16` but `serve.sh` mounts `MODELS_DIR=/mnt/fastmodels:/models:ro` — either
move/symlink the model under `/mnt/fastmodels` or override `MODELS_DIR`; and `hpu-msa-ext` does not
yet exist (text-first dense bring-up needs no plugin, but sparse does).

```sh
# MiniMax-M3 (MiniMaxM3SparseForConditionalGeneration) — TEXT-FIRST bring-up on 8x Gaudi 2.
# Phase 1: bf16, DENSE GQA (MSA kernel not yet wired -> FULL_ATTENTION_FALLBACK). Coherence first.
# Sparse (MSA) is gated behind HPU_MSA_READY=1 + the hpu-msa-ext plugin; until then it is genuine
# dense GQA (never silent-empty). Recipe cache: its own dir 'minimax-m3' (never reuse another).
MODEL_PATH=minimax-m3-bf16             # must resolve under whatever is mounted at /models
SERVED_NAME=minimax-m3
IMAGE=vllm-gaudi:1.24                  # base image; use :next only if the MSA .so needs it (DSA did)
# NOTE: model currently at /mnt/data2/minimax-m3-bf16, but serve.sh mounts /mnt/fastmodels.
# Resolve via ONE of: (a) move model to /mnt/fastmodels/minimax-m3-bf16; (b) EXTRA_MOUNTS to add
# -v /mnt/data2/minimax-m3-bf16:/models/minimax-m3-bf16:ro ; (c) override MODELS_DIR for this run.
MODEL_ENV="-e PT_HPU_LAZY_MODE=0 -e VLLM_SKIP_WARMUP=true -e PT_HPU_WEIGHT_SHARING=0 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 -e VLLM_RPC_TIMEOUT=100000000 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 -e VLLM_HPU_LOG_STEP_CPU_FALLBACKS=true"
# Phase 1 = dense GQA, eager (correctness). For sparse later add: -e HPU_MSA_READY=1 and mount
# hpu-msa-ext (USE_MSA=1 once serve.sh learns that flag, mirroring USE_DSA).
VLLM_ARGS="--tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8_inc \
  --max-model-len 32768 --max-num-seqs 64 \
  --gpu-memory-utilization 0.9 --block-size 128 \
  --async-scheduling --no-enable-prefix-caching --trust-remote-code"
```
Rationale per flag:
- `--tensor-parallel-size 8` + `--enable-expert-parallel`: 128 experts across 8 cards (M2 TP guard
  `minimax_m2.py:62-66` needs `tp ≤ num_local_experts`; 8 ≤ 128 ✓).
- `--kv-cache-dtype fp8_inc`: the **only** KV dtype the HPU platform accepts
  (`platform.py:388-389`).
- `--block-size 128`: HPU kernel block size; platform forces it anyway (`platform.py:135-136`).
- `--no-enable-prefix-caching`: required once MSA is on (the block-selection / index buffers assume
  no cross-request prefix reuse — same rationale as DSA, `sparse_mla_wire.py:9`,
  `glm_moe_dsa_wire.LAUNCH_FLAGS:267`); harmless in dense phase. Contiguous-PA disables it anyway.
- `--max-num-seqs 64`: high concurrency (raise toward 128 for throughput once stable; cf.
  `qwen35-397b-tput128.env`).
- `--max-model-len 32768`: phase-1 target (full 1M `max_position_embeddings` needs the sparse
  kernel; mirror DSA's "32K phase-1, 200K needs the TPC kernel", `glm_moe_dsa_wire.py:269,291`).
- `PT_HPU_LAZY_MODE=0` (eager) for bring-up correctness, matching `glm51ar-dsa.env`; a lazy speed
  env (`PT_HPU_LAZY_MODE=1` + `PT_HPU_ENABLE_LAZY_COLLECTIVES=true`) is the follow-on, like
  `glm51ar-lazy.env`.
- `--trust-remote-code`: config has `auto_map`→`configuration_minimax_m3_vl` (remote config class).
- `--async-scheduling`: matches the box convention; platform auto-disables it with spec-decode
  (`platform.py:246-247`) — fine since MTP is off in phase 1.

`common.env` provides `BIND_IP/PORT/API_KEY/CONTAINER/RECIPE_CACHE`; `serve.sh` injects
`HPU_MOE_EXT_CKPT=/models/$MODEL_PATH` and mounts `hpu-moe-ext` automatically.

---

## 5. Concrete file-by-file CREATE / CHANGE list (text-first)

**Phase 1 — load + serve dense GQA (no sparse, no vision):**
1. **CREATE** `$HOME/AI/_src/vllm_gaudi/models/minimax_m3.py` — port of `minimax_m2.py` with:
   `text_config` plumbing (§3.1); dense-first-3 + per-layer MoE-freq decoder branch + shared
   experts + `swigluoai` activation (§3.2); class `HpuMiniMaxM3ForConditionalGeneration`
   (text-only, ForCausalLM-shaped, ignores `vision_config`, §3.5); reuse M2 MoE/FusedMoE wiring,
   `get_expert_mapping` (`w1/w2/w3`), and `load_weights` machinery verbatim (incl. the
   `mlp.experts.` guard `minimax_m2.py:386-387` and spec-layer skip `:494-500` for `num_mtp_modules=7`).
   Attention class initially uses the **stock** GQA `self.attn = Attention(...)` (dense) — identical
   to M2 — so phase 1 needs no platform patch at all.
2. **CHANGE** `$HOME/AI/_src/vllm_gaudi/models/__init__.py` — register
   `MiniMaxM3SparseForConditionalGeneration → minimax_m3:HpuMiniMaxM3ForConditionalGeneration`
   (after `:33`).
3. **CREATE** `$HOME/AI/models/minimax-m3.env` — §4 (dense phase).
4. **RESOLVE** model location vs mount (§4 note / §6 blocker).
   *At this point the model loads + serves coherent dense GQA bf16.* No sparse gate touched.

**Phase 2 — MSA sparse (gated, never silent-dense):**
5. **CREATE** task **`hpu-msa-ext`** (`$HOME/AI/hpu-msa-ext/hpu_msa_ext/`), mirroring
   `hpu-dsa-ext`:
   - `platform_patch.py`: extend `_patched` so `use_sparse and not use_mla` → `HPUAttentionBackendV1`
     when `MSA_READY`; else raise the "never dense" message. `MSA_READY=False`,
     `MSA_FULL_ATTENTION_FALLBACK=True`. Capture the *current* `get_attn_backend_cls` as `_ORIG`
     (chain after DSA if both mounted). (template: `hpu_dsa_ext/platform_patch.py`)
   - MSA indexer + block-select (template: `sparse_indexer_hpu.py`) producing a per-sparse-layer
     `topk_blocks` buffer from `sparse_index_dim/num_index_heads/topk_blocks/block_size`.
   - GQA consume wire: monkeypatch `HPUAttentionImpl.forward`
     (`$HOME/AI/_src/vllm_gaudi/attention/backends/hpu_attn.py:362`) restricting paged keys to
     selected blocks; non-sparse/profiling → verbatim dense; raise on unsupported (template:
     `sparse_mla_wire.py`).
   - `__init__.register()`: `_compat.verify()` → `platform_patch.apply()` → register MSA stack →
     flip `MSA_READY` only if `HPU_MSA_READY=1` and all components imported (template:
     `hpu_dsa_ext/__init__.py:register` + `_load_component` strict logic).
   - `vllm.general_plugins` entry point `06.hpu_msa_ext`.
6. **CHANGE** `$HOME/AI/serve.sh` — add a `USE_MSA` mount/install hook mirroring `USE_DSA`
   (`serve.sh` lines doing `${USE_DSA:+-v …hpu-dsa-ext}` + `pip install -e`), and the stale-recipe
   guard's plugin-source `find` (add `hpu-msa-ext/hpu_msa_ext`).
7. **CHANGE** the M3 model's attention to call the MSA buffer/consume on sparse layers (the wire is
   in the plugin; the model just exposes the per-layer sparse flag + `topk_blocks` buffer, exactly
   as DSA models expose `topk_indices_buffer`).
8. **CHANGE** `minimax-m3.env` — add `-e HPU_MSA_READY=1` + `USE_MSA=1`.

### What stays BYPASSED until the MSA kernel validates (DSA safety mirror)
- `MSA_READY=False` → platform keeps **raising** for `use_sparse` (never returns a backend that
  would run an empty/garbage sparse consume).
- `MSA_FULL_ATTENTION_FALLBACK=True` → if a sparse layer is reached before the kernel is trusted,
  it runs **genuine dense GQA** over all causal keys (coherent), loudly logged — the
  `platform_patch.FULL_ATTENTION_FALLBACK` / `sparse_mla_wire._sparse_forward_impl:231-238` pattern.
  **Never** the "consume an all-(-1) topk buffer → NaN-guard → self-only → token-salad" trap that
  PR #1301 fell into (`platform_patch.py:9`, `sparse_mla_wire.py:9,227-230`).
- Flip to real sparse only after a `sparse != dense` parity/coherence gate on-card (DSA did exactly
  this: `glm51ar-dsa.env` header — fired-log on >topk prompts, dense short-circuit on short ones,
  judged vs the teacher-force oracle).

---

## 6. Blockers / open items

- **B1 (hard, sparse only):** task **`hpu-msa-ext` does not exist yet** (`$HOME/AI/hpu-msa-ext`
  absent). Phase-1 dense bring-up does not need it; sparse does. It is a new task built from the
  `hpu-dsa-ext` template, but the consume target is the **GQA** `HPUAttentionImpl.forward`
  (`hpu_attn.py:362`), not the MLA `forward_impl` — the kernel math (block-sparse top-k over paged
  GQA KV) is genuinely different from DSA's lightning-indexer + latent gather and must be authored.
- **B2 (easy):** model is at `/mnt/data2/minimax-m3-bf16` but `serve.sh` mounts
  `MODELS_DIR=/mnt/fastmodels`. Per memory `gaudi-storage-layout`, models live on `/mnt/fastmodels`.
  Resolve before serving (move/symlink, or `EXTRA_MOUNTS`/`MODELS_DIR` override).
- **B3 (verify):** confirm vllm surfaces `text_config` as `hf_config.text_config` (or
  `ModelConfig.get_text_config()`) for `model_type=minimax_m3_vl`, so the model class can build on
  the LM sub-config. The reference HF modeling file (look under `$HOME/AI/minimax-m3/ref/`)
  should be consulted for the exact dense-MLP / shared-expert / `swigluoai` (`swiglu_alpha`,
  `swiglu_limit`) details + `use_gemma_norm` semantics before re-deriving them.
- **B4 (capacity):** M3 is **bf16** (`config.json` has no `quantization_config`). 60 layers ×
  (6144 hidden, 128 routed experts of 3072, dense-first-3 of 12288, +1 shared) at bf16 across 8×
  Gaudi 2 (~96 GB usable/card) needs a memory check; an **int4 AutoRound export** (which would
  autodetect through `hpu-moe-ext` with zero model changes, §3.6) is the likely production path —
  but is a separate quant task.
- **B5 (co-exist):** if `hpu-dsa-ext` and `hpu-msa-ext` are ever mounted together, the MSA patch
  must chain DSA's `_patched` (capture current `get_attn_backend_cls`, not stock). Entry-point
  ordering `06.` after `05.` handles it; document/assert.
