# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The MiniMax AI team.
# Copyright 2025 The ZhipuAI Team.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Inference-only MiniMax-M3 MTP (Multi-Token-Prediction / depth-1 speculative draft) head
for vllm-gaudi.

Mirrors the proven GLM-4.x MTP serving module (``_src/vllm_models/glm4_moe_mtp.py`` ->
``Glm4MoeMTP`` / ``Glm4MoeMultiTokenPredictor`` / ``Glm4MoeMultiTokenPredictorLayer`` /
``SharedHead``) and the DeepSeek-MTP layer-name contract, with M3's deltas:

  * GEMMA RMSNorm everywhere (``text_config.use_gemma_norm=True``): enorm / hnorm / the draft
    block's norms / the shared-head norm all scale by ``(1 + w)`` (vLLM ``GemmaRMSNorm``), NOT the
    plain ``x*w`` ``RMSNorm`` GLM/DeepSeek use. The CPU twin is ``m3_cpu_mtp.gemma_rms``.
  * swigluoai activation (alpha=1.702, limit=7.0) in the draft block's dense MLP, contiguous
    [gate|up] chunk layout (NOT the interleaved stock ``SwigluOAIAndMul``) — exactly as the main
    tower's ``HpuMiniMaxM3MLP`` does it.
  * mtp_start_layer_idx = num_hidden_layers = 60; num_nextn_predict_layers = 1 (depth-1).
  * partial RoPE (rotary_dim=64 of head_dim=128, theta=5e6) + per-head qk-norm + GQA(64/4) — the
    M3 attention substrate, reused from ``minimax_m3.HpuMiniMaxM3Attention``.

DESIGN DECISION — the draft block is LEAN, NOT a 128-expert MoE
--------------------------------------------------------------
A draft head runs on EVERY decode step, so it must be cheap at inference AND trainable in
reasonable wall-clock. M3's real decoder block is a 128-expert top-4 MoE — far too heavy on both
counts for a one-layer draft that we train from scratch. GLM shipped a MoE MTP block only because
zai exported one; EAGLE/MTP heads are canonically ONE small transformer layer. So this draft block
is::

    M3 attention (GQA + per-head qk-norm + partial-RoPE)  +  ONE dense swigluoai MLP

i.e. ``M3MTPDraftLayer`` below — the SAME attention as the tower, but a single dense ``HpuMiniMaxM3MLP``
in place of ``HpuMiniMaxM3MoE``. The dense MLP's intermediate width is the accept-rate-vs-cost knob
(``mtp_intermediate_size``; see ``_draft_intermediate_size``): wider -> higher acceptance, more FLOPs
per decode step. Default is a lean 8192 (between the routed-expert 3072 and the dense-prefix 12288):
big enough to learn next-token structure on top of the target's hidden state (which already carries
the heavy lifting via ``hnorm(prev_hidden)``), small enough that a depth-1 head still nets the
~1.7x single-stream win (GLM-proven) instead of paying for a second full MoE pass. Retrain with a
larger width if measured acceptance is below target; shrink it if the head is decode-bound.

The OUTER wiring (enorm/hnorm/eh_proj/shared_head + the layer-60 tensor names) is byte-identical to
what the serving MTP loader expects, so a trained draft checkpoint loads through the standard MTP
path regardless of the lean inner block.

WEIGHT SOURCE
-------------
The bf16/int4 M3 checkpoints ship NO MTP-module tensors (only the 0..59 tower + ``embed_tokens`` +
``lm_head`` + final ``norm``). The draft's layer-60 tensors (enorm / hnorm / eh_proj / the draft
block / shared_head.norm) come from the TRAINED draft checkpoint. ``embed_tokens`` and
``shared_head.head`` are SHARED with the target when the draft checkpoint omits them — handled
generically by hpu-moe-ext's ``mtp_share_head_patch`` (the EagleProposer.load_model tie), exactly as
for GLM-5.1 whose layer-78 shipped only ``shared_head.norm``.

CPU TWIN / SELF-TEST
--------------------
This file IMPORTS vLLM + HPU classes, so it does NOT import on the CPU calib venv. The card-free
gate is a pure-python AST/structure self-check at the bottom (``_cpu_struct_selftest``) plus a
name-mapping unit test of ``_rewrite_spec_layer_name`` — run with the m3calib venv. The numerical
core is gated by ``m3_cpu_mtp.py`` (``M3MTPFrontend`` / ``M3SharedHead`` / ``gemma_rms`` /
``swigluoai``), which this module's forward mirrors exactly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

# ----------------------------------------------------------------------------------------------------
# vLLM + mounted-minimax_m3 imports. These are REQUIRED for the serving module but are NOT importable
# on the CPU calib venv. The card-free structure self-test (`_cpu_struct_selftest`, run via __main__)
# parses THIS FILE with `ast` and needs NO imports, so we guard the heavy imports: if they fail (e.g.
# the m3calib venv has no vllm), the class bodies below will NameError if instantiated — but they are
# never instantiated on CPU; only the AST gate runs. On the serve image every import resolves and the
# guard is a no-op. (We deliberately do NOT swallow the import error on the serve path — see __main__:
# the gate re-imports vllm and FAILS LOUDLY there if it is genuinely missing on a card box.)
# ----------------------------------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from transformers import PretrainedConfig

    from vllm.config import CacheConfig, VllmConfig
    from vllm.distributed import get_pp_group  # noqa: F401  (parity with sibling MTP modules)
    from vllm.logger import init_logger
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.gemma import GemmaRMSNorm
    from vllm.model_executor.models.utils import maybe_prefix
    from vllm.sequence import IntermediateTensors

    # Reuse the validated M3 tower pieces verbatim (NEVER redefine the attention / dense MLP /
    # text-config resolver / rope builder / spec-layer helper — they are the parity-validated
    # main-model code). The plugin (hpu_m3_ext/__init__.py::_ensure_on_path) puts the mounted
    # minimax-m3/vllm dir on sys.path so `minimax_m3` imports as a top-level module, exactly as the
    # main-model registration relies on.
    from minimax_m3 import (  # type: ignore[import-not-found]
        HpuMiniMaxM3Attention,
        HpuMiniMaxM3MLP,
        _build_rope_parameters,
        get_spec_layer_idx_from_weight_name,
        get_text_config,
    )

    logger = init_logger(__name__)
    _VLLM_AVAILABLE = True
except ImportError:  # CPU calib venv (no vllm) — only the AST self-test runs; classes are never built.
    _VLLM_AVAILABLE = False
    import torch.nn as nn  # noqa: F401 — nn.Module base; available on the CPU torch too.

    class _MissingBase(nn.Module):  # minimal stand-in so `class X(nn.Module)` still parses+defines.
        pass

    GemmaRMSNorm = HpuMiniMaxM3MLP = HpuMiniMaxM3Attention = _MissingBase  # type: ignore[assignment,misc]
    PretrainedConfig = object  # type: ignore[assignment,misc]


# ==================================================================================================
# Draft-block intermediate width — the accept-rate-vs-cost knob (see module docstring DESIGN
# DECISION). Resolution order: explicit text_config.mtp_intermediate_size -> a lean default of 8192.
# We do NOT default to dense_intermediate_size (12288): that is the heavy dense-prefix width and a
# depth-1 draft runs EVERY decode step. The trainer must build its dense MLP at exactly this width
# (the loaded gate_up/down shapes are asserted by the linear layers at load time -> LOUD on mismatch).
# ==================================================================================================
_DEFAULT_DRAFT_INTERMEDIATE = 8192


def _draft_intermediate_size(text_config: PretrainedConfig) -> int:
    v = getattr(text_config, "mtp_intermediate_size", None)
    if v is None:
        v = getattr(text_config, "draft_intermediate_size", None)
    if v is None:
        return _DEFAULT_DRAFT_INTERMEDIATE
    return int(v)


# ==================================================================================================
# M3MTPDraftLayer — the LEAN draft block: M3 attention + ONE dense swigluoai MLP (NOT 128-expert MoE).
#
# Structurally a stripped HpuMiniMaxM3DecoderLayer: same Gemma input/post-attn layernorms, same
# attention substrate (GQA + per-head qk-norm + partial RoPE), but the FC sub-block is a single
# HpuMiniMaxM3MLP (dense swigluoai) instead of HpuMiniMaxM3MoE. Forward signature matches the tower
# decoder layer ((positions, hidden_states, residual) -> (hidden_states, residual)) so the MTP layer
# can drive it with residual=None exactly like Glm4MoeMultiTokenPredictorLayer drives its mtp_block.
#
# Attention is built NON-sparse (is_sparse=False): the draft is a single layer with no MSA kernel —
# it runs genuine dense GQA over the (short) draft context, never the block-sparse path. The weight
# prefix is `<prefix>.self_attn` / `<prefix>.mlp` so the trained-draft tensor names match the loader.
# ==================================================================================================
class M3MTPDraftLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = max(max_position_embeddings, config.max_model_len)

        # M3 partial-rope parameters from the flat text-config fields (rotary_dim=64, theta=5e6).
        rope_parameters = _build_rope_parameters(config)

        # Same attention substrate as the tower, but FORCED dense GQA (is_sparse=False): a one-layer
        # draft has no MSA index path. layer_idx is informational only (the seam never fires here).
        self.self_attn = HpuMiniMaxM3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=getattr(config, "rotary_dim", config.head_dim),
            rope_parameters=rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            is_sparse=False,  # M3 MTP: draft attention is genuine dense GQA (no MSA kernel)
            layer_idx=int(prefix.split(".")[-1]) if prefix.split(".")[-1].isdigit() else 0,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # LEAN FC: a single dense swigluoai MLP (contiguous gate|up chunk, alpha/limit from config) —
        # NOT HpuMiniMaxM3MoE. Width is the accept-rate-vs-cost knob (_draft_intermediate_size).
        self.mlp = HpuMiniMaxM3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=_draft_intermediate_size(config),
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # M3 Gemma-style layernorms (x*(1+w)).
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Identical residual discipline to HpuMiniMaxM3DecoderLayer (GemmaRMSNorm fuses the add).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ==================================================================================================
# SharedHead (M3) — final Gemma-norm + ParallelLMHead. Mirrors glm4_moe_mtp.SharedHead, but the norm
# is GemmaRMSNorm (M3 use_gemma_norm). forward() returns the NORMED hidden state (the LM head is
# applied by the logits_processor in compute_logits, exactly as GLM/DeepSeek do it). CPU twin:
# m3_cpu_mtp.M3SharedHead.
# ==================================================================================================
class M3SharedHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


# ==================================================================================================
# M3MultiTokenPredictorLayer — the per-spec-layer module. Mirrors Glm4MoeMultiTokenPredictorLayer:
#   enorm / hnorm (Gemma) + eh_proj Linear(2H -> H) + the draft block + shared_head.
# forward: mask position 0 -> enorm(embed) , hnorm(prev_hidden) -> eh_proj(cat[e, h]) -> draft block
#          -> + residual. The cat ORDER is [enorm(embed_{t+1}), hnorm(prev_hidden)] (matches
#          m3_cpu_mtp.M3MTPFrontend and the glm/deepseek contract); swapping it is a silent bug the
#          CPU gate catches.
# ==================================================================================================
class M3MultiTokenPredictorLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        # M3 Gemma-style enorm/hnorm (x*(1+w)).
        self.enorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = M3SharedHead(config=config, prefix=prefix, quant_config=quant_config)
        # LEAN draft block (attention + dense swigluoai MLP) — see M3MTPDraftLayer.
        self.mtp_block = M3MTPDraftLayer(
            config=config,
            prefix=prefix,
            cache_config=cache_config,
            quant_config=quant_config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # Mask position 0 — MTP does not predict a token there (no t+1 target at BOS). Matches
        # m3_cpu_mtp.M3MTPFrontend and Glm4MoeMultiTokenPredictorLayer.
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        # cat ORDER: [enorm(embed_{t+1}), hnorm(prev_hidden)] -> eh_proj(2H -> H).
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states
        return hidden_states


# ==================================================================================================
# M3MultiTokenPredictor — the inner predictor: embed_tokens + the per-spec-layer ModuleDict (keyed by
# the absolute layer index, e.g. "60") + logits_processor + compute_logits. Mirrors
# Glm4MoeMultiTokenPredictor. The layer dict spans [mtp_start_layer_idx, +num_nextn_predict_layers)
# == {"60"} for M3 (depth-1).
# ==================================================================================================
class M3MultiTokenPredictor(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = get_text_config(vllm_config.model_config.hf_config)
        if hasattr(vllm_config.model_config, "max_model_len") and not hasattr(config, "max_model_len"):
            config.max_model_len = vllm_config.model_config.max_model_len
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers  # 60
        self.num_mtp_layers = config.num_nextn_predict_layers  # 1
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): M3MultiTokenPredictorLayer(
                    config,
                    f"{prefix}.layers.{idx}",
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits


# ==================================================================================================
# MiniMaxM3MTP — the top spec-decode model class. Mirrors Glm4MoeMTP (minus the MoE bookkeeping, since
# our draft block is dense — there is no FusedMoE to extract). Registered under the MTP arch name and
# driven by vLLM's EagleProposer/spec-decode runner: forward(prev hidden) -> draft hidden;
# compute_logits(draft hidden) -> draft logits; load_weights maps the trained-draft tensor names onto
# the layer-60 module.
# ==================================================================================================
class MiniMaxM3MTP(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = get_text_config(vllm_config.model_config.hf_config)
        self.model = M3MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    # ----------------------------------------------------------------------------------------------
    # Weight loading. Mirrors Glm4MoeMTP.load_weights / DeepSeekMTP: rewrite the trained-draft tensor
    # names onto the layer-60 module, fuse qkv + gate_up shards, share embed/lm_head when the draft
    # checkpoint omits them (the EagleProposer.load_model tie in hpu-moe-ext does the actual module
    # sharing; here we just route the shards that ARE present). No expert mapping — the draft block is
    # dense (no FusedMoE), so we only need the stacked qkv/gate_up fusion + plain loads.
    # ----------------------------------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name == "lm_head.weight":
                spec_layer = self.model.mtp_start_layer_idx
                name = f"model.layers.{spec_layer}.shared_head.head.weight"
            elif name == "model.embed_tokens.weight":
                spec_layer = self.model.mtp_start_layer_idx
            else:
                spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
                if spec_layer is None:
                    continue
                name = self._rewrite_spec_layer_name(spec_layer, name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip routed-expert shards if any leak in (the dense draft has none); guard like the
                # tower loader so a gate_proj/up_proj inside an `mlp.experts.` name is never fused.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".weight_scale") and name not in params_dict:
                    continue
                # MTP modules share embedding/head with the target (DeepSeek-V3 report); for the
                # non-spec-layer shared weights, only load the first (mtp_start) layer's copy.
                if (
                    spec_layer != self.model.mtp_start_layer_idx
                    and ".layers" not in name
                ):
                    continue
                if name not in params_dict:
                    # embed_tokens / shared_head.head are SHARED from the target by the
                    # EagleProposer.load_model tie (hpu-moe-ext mtp_share_head_patch) when the draft
                    # checkpoint omits them; a name with no home param here is therefore expected for
                    # the shared tensors and must be skipped, not KeyError'd.
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """Rewrite a trained-draft weight name to the served layout. Adds ``.mtp_block`` for the
        transformer-block tensors of the spec layer (self_attn / mlp / input_layernorm /
        post_attention_layernorm) and lifts the shared embed to top level. The MTP-frontend tensors
        (enorm / hnorm / eh_proj / shared_head) stay at ``model.layers.{spec}.<name>``.

        Mirrors Glm4MoeMTP._rewrite_spec_layer_name exactly (same spec_layer_weight_names set + the
        shared embed_tokens lift), so a draft checkpoint that names its tensors like the served module
        loads with no extra mapping.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # transformer block tensors (self_attn / mlp / layernorms) live under .mtp_block.
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            # shared embedding -> top level.
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name


# ==================================================================================================
# ============================ CARD-FREE STRUCTURE SELF-TEST ========================================
# This module imports vLLM + HPU + the mounted minimax_m3, so it does NOT import on the CPU calib
# venv. The gate below parses THIS FILE'S source with `ast` (no import) and asserts the required
# classes, forward signatures, the cat-order, the Gemma-norm usage, and the lean-block decision are
# present; then it unit-tests _rewrite_spec_layer_name's name mapping by re-implementing the same AST-
# extracted logic on dummy names (so the test exercises the ACTUAL mapping table, not a hand copy).
#
# Run (card-free, ~0.1s):
#   $HOME/.venvs/m3calib/bin/python $HOME/AI/hpu-m3-ext/hpu_m3_ext/m3_mtp.py
# ==================================================================================================
def _strip_strings_and_comments(code: str) -> str:  # pragma: no cover - test helper
    """Return `code` with all comments and string/docstring literals blanked out, via tokenize. Used
    for the prose-sensitive ABSENCE/COUNT checks so that explanatory comments, docstrings, AND the
    self-test's own check-string literals (e.g. "HpuMiniMaxM3MoE") cannot create false positives — we
    assert only over REAL code tokens. Falls back to the raw text if tokenize chokes."""
    import io
    import tokenize

    try:
        out: list[str] = []
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
                continue
            out.append(tok.string)
        return " ".join(out)
    except Exception:  # noqa: BLE001
        return code


def _cpu_struct_selftest() -> bool:  # pragma: no cover - exercised only via __main__
    import ast
    import os

    checks: list[tuple[str, bool]] = []

    def ck(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    src = open(os.path.abspath(__file__)).read()
    code = _strip_strings_and_comments(src)  # real tokens only (no comments/strings) for absence checks
    tree = ast.parse(src)
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    # 1. all required classes present (mirrors glm4_moe_mtp's set, M3-named).
    for cname in (
        "M3MTPDraftLayer",
        "M3SharedHead",
        "M3MultiTokenPredictorLayer",
        "M3MultiTokenPredictor",
        "MiniMaxM3MTP",
    ):
        ck(f"class {cname} present", cname in classes)

    def method(cls_name: str, meth: str) -> Optional[ast.FunctionDef]:
        cls = classes.get(cls_name)
        if cls is None:
            return None
        for n in cls.body:
            if isinstance(n, ast.FunctionDef) and n.name == meth:
                return n
        return None

    def argnames(fn: Optional[ast.FunctionDef]) -> list[str]:
        return [a.arg for a in fn.args.args] if fn is not None else []

    # 2. forward signatures match the GLM/DeepSeek MTP contract the serving runner drives.
    ck(
        "MTP layer forward(input_ids, positions, previous_hidden_states, inputs_embeds, ...)",
        argnames(method("M3MultiTokenPredictorLayer", "forward"))[:5]
        == ["self", "input_ids", "positions", "previous_hidden_states", "inputs_embeds"],
    )
    ck(
        "predictor forward(input_ids, positions, previous_hidden_states, ...)",
        argnames(method("M3MultiTokenPredictor", "forward"))[:4]
        == ["self", "input_ids", "positions", "previous_hidden_states"],
    )
    ck(
        "top forward(input_ids, positions, hidden_states, ...)",
        argnames(method("MiniMaxM3MTP", "forward"))[:4]
        == ["self", "input_ids", "positions", "hidden_states"],
    )
    ck(
        "draft layer forward(positions, hidden_states, residual)",
        argnames(method("M3MTPDraftLayer", "forward"))
        == ["self", "positions", "hidden_states", "residual"],
    )
    ck("top compute_logits present", method("MiniMaxM3MTP", "compute_logits") is not None)
    ck("top load_weights present", method("MiniMaxM3MTP", "load_weights") is not None)
    ck("_rewrite_spec_layer_name present", method("MiniMaxM3MTP", "_rewrite_spec_layer_name") is not None)

    # 3. the MTP-layer forward must: mask position 0, enorm/hnorm, cat [inputs_embeds, prev] in THAT
    #    order, eh_proj, then +residual. Assert on the source text of the method.
    lyr_fwd_src = ast.get_source_segment(src, method("M3MultiTokenPredictorLayer", "forward")) or ""
    ck("layer masks position 0", "positions.unsqueeze(-1) == 0" in lyr_fwd_src)
    ck("layer applies enorm", "self.enorm(" in lyr_fwd_src)
    ck("layer applies hnorm", "self.hnorm(" in lyr_fwd_src)
    ck(
        "cat order [inputs_embeds, previous_hidden_states]",
        "torch.cat([inputs_embeds, previous_hidden_states]" in lyr_fwd_src,
    )
    ck("layer applies eh_proj", "self.eh_proj(" in lyr_fwd_src)
    ck("layer adds residual (residual + hidden_states)", "residual + hidden_states" in lyr_fwd_src)

    # 4. Gemma-norm everywhere (M3 use_gemma_norm). The file must import + use GemmaRMSNorm and must
    #    NOT build a plain RMSNorm for any MTP norm site (that would silently apply the wrong scale).
    #    Checked over `code` (comments/strings stripped) so the self-test's own check-strings + the
    #    import-guard fallback assignment cannot create false positives.
    ck("imports GemmaRMSNorm", "from vllm.model_executor.models.gemma import GemmaRMSNorm" in src)
    # tokenized form: `self . enorm = GemmaRMSNorm`. Exactly one real assignment of each, to GemmaRMSNorm.
    ck("enorm is GemmaRMSNorm",
       code.count("self . enorm =") == 1 and "self . enorm = GemmaRMSNorm" in code)
    ck("hnorm is GemmaRMSNorm",
       code.count("self . hnorm =") == 1 and "self . hnorm = GemmaRMSNorm" in code)
    ck("shared_head norm is GemmaRMSNorm",
       code.count("self . norm =") == 1 and "self . norm = GemmaRMSNorm" in code)
    # the only RMSNorm CONSTRUCTION token allowed is GemmaRMSNorm ; a bare `RMSNorm (` call is the bug.
    bare_rmsnorm = code.replace("GemmaRMSNorm", "")
    ck("no bare RMSNorm( construction", "RMSNorm (" not in bare_rmsnorm and "RMSNorm(" not in bare_rmsnorm)

    # 5. LEAN-block decision: the draft block builds HpuMiniMaxM3MLP (dense), NOT HpuMiniMaxM3MoE.
    #    Use the AST line span of M3MTPDraftLayer + strip strings/comments so the docstring/comment
    #    mentions of HpuMiniMaxM3MoE ("NOT a 128-expert MoE") do not trip the absence check.
    draft_cls = classes["M3MTPDraftLayer"]
    draft_lines = src.splitlines()[draft_cls.lineno - 1: draft_cls.end_lineno]
    draft_code = _strip_strings_and_comments("\n".join(draft_lines))
    ck("draft block uses dense HpuMiniMaxM3MLP", "HpuMiniMaxM3MLP" in draft_code)
    ck("draft block does NOT use HpuMiniMaxM3MoE", "HpuMiniMaxM3MoE" not in draft_code)
    ck("draft attention is non-sparse", "is_sparse = False" in draft_code)
    ck("draft MLP width via _draft_intermediate_size", "_draft_intermediate_size" in draft_code)

    # 6. mtp_start_layer_idx == num_hidden_layers (60) and depth-1 (num_nextn_predict_layers).
    pred_src = ast.get_source_segment(src, classes["M3MultiTokenPredictor"]) or ""
    ck("mtp_start_layer_idx = num_hidden_layers", "self.mtp_start_layer_idx = config.num_hidden_layers" in pred_src)
    ck("num_mtp_layers = num_nextn_predict_layers", "self.num_mtp_layers = config.num_nextn_predict_layers" in pred_src)

    # ---- name-mapping unit test --------------------------------------------------------------------
    # Re-implement _rewrite_spec_layer_name's mapping by EXTRACTING its constant lists from the AST
    # (so the test tracks the real table). Then assert dummy trained-draft names map to the layer-60
    # served names the loader's params_dict will contain.
    rewrite_fn = method("MiniMaxM3MTP", "_rewrite_spec_layer_name")
    rewrite_src = ast.get_source_segment(src, rewrite_fn) or ""
    # extract the two name lists literally from the function source.
    spec_names = _extract_str_list(rewrite_src, "spec_layer_weight_names")
    shared_names = _extract_str_list(rewrite_src, "shared_weight_names")
    ck(
        "rewrite spec_layer_weight_names == {embed_tokens,enorm,hnorm,eh_proj,shared_head}",
        set(spec_names) == {"embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head"},
    )
    ck("rewrite shared_weight_names == {embed_tokens}", set(shared_names) == {"embed_tokens"})

    def rewrite(spec_layer: int, name: str) -> str:
        spec_layer_weight = False
        shared_weight = False
        for wn in spec_names:
            if wn in name:
                spec_layer_weight = True
                if wn in shared_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            return name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        if shared_weight:
            return name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    SPEC = 60
    cases = {
        # transformer-block tensors -> .mtp_block.
        f"model.layers.{SPEC}.self_attn.q_proj.weight": f"model.layers.{SPEC}.mtp_block.self_attn.q_proj.weight",
        f"model.layers.{SPEC}.self_attn.o_proj.weight": f"model.layers.{SPEC}.mtp_block.self_attn.o_proj.weight",
        f"model.layers.{SPEC}.mlp.gate_proj.weight": f"model.layers.{SPEC}.mtp_block.mlp.gate_proj.weight",
        f"model.layers.{SPEC}.mlp.down_proj.weight": f"model.layers.{SPEC}.mtp_block.mlp.down_proj.weight",
        f"model.layers.{SPEC}.input_layernorm.weight": f"model.layers.{SPEC}.mtp_block.input_layernorm.weight",
        f"model.layers.{SPEC}.post_attention_layernorm.weight": f"model.layers.{SPEC}.mtp_block.post_attention_layernorm.weight",
        # MTP-frontend tensors -> unchanged (stay at the layer, owned by the predictor layer module).
        f"model.layers.{SPEC}.enorm.weight": f"model.layers.{SPEC}.enorm.weight",
        f"model.layers.{SPEC}.hnorm.weight": f"model.layers.{SPEC}.hnorm.weight",
        f"model.layers.{SPEC}.eh_proj.weight": f"model.layers.{SPEC}.eh_proj.weight",
        f"model.layers.{SPEC}.shared_head.norm.weight": f"model.layers.{SPEC}.shared_head.norm.weight",
        f"model.layers.{SPEC}.shared_head.head.weight": f"model.layers.{SPEC}.shared_head.head.weight",
        # shared embed -> top level.
        f"model.layers.{SPEC}.embed_tokens.weight": "model.embed_tokens.weight",
    }
    for src_name, want in cases.items():
        got = rewrite(SPEC, src_name)
        ck(f"map {src_name.split('.', 3)[-1]} -> {want.split('.', 1)[-1]}", got == want)

    # the qkv/gate_up FUSION the loader relies on must be expressible AFTER the rewrite: the rewritten
    # block names must contain the shard substrings the stacked_params_mapping fuses on.
    ck(
        "rewritten q_proj name fuses to qkv_proj",
        "q_proj" in cases[f"model.layers.{SPEC}.self_attn.q_proj.weight"],
    )
    ck(
        "rewritten gate_proj name fuses to gate_up_proj",
        "gate_proj" in cases[f"model.layers.{SPEC}.mlp.gate_proj.weight"],
    )

    ok = all(p for _, p in checks)
    for nm, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {nm}")
    print(f"\n{'=' * 60}\nM3 MTP SERVING-MODULE STRUCTURE GATE: "
          f"{'GREEN — all pass' if ok else 'RED — FIX ABOVE'}\n{'=' * 60}")
    return ok


def _extract_str_list(func_src: str, var: str) -> list[str]:  # pragma: no cover - test helper
    """Pull a string-list literal assigned to `var` out of a function's source via AST (robust to
    formatting). Returns [] if not found."""
    import ast

    mod = ast.parse(func_src.strip())
    for node in ast.walk(mod):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var and isinstance(node.value, ast.List):
                    return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    return []


if __name__ == "__main__":  # pragma: no cover
    import sys

    # The gate is PURE AST over this file's own source — it runs identically whether or not vllm is
    # importable. We print which mode we are in so a card-box run that UNEXPECTEDLY lost vllm is
    # visible (the structure gate still passes, but the serving import would fail — surfaced here).
    if not _VLLM_AVAILABLE:
        print("[m3_mtp self-test] vllm NOT importable in this interpreter (CPU calib venv expected) "
              "— running the card-free AST structure gate only.\n")
    sys.exit(0 if _cpu_struct_selftest() else 1)
