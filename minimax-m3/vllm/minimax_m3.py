# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The MiniMax AI team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniMax-M3 (text tower) model for vllm-gaudi.

Ported from ``vllm_gaudi/models/minimax_m2.py``. The differences from M2 are intentionally
isolated and each carries a ``# M3:`` comment so the diff stays auditable:

  1. CONFIG: M3 is a nested-config multimodal model (``model_type=minimax_m3_vl``). The LM lives
     under ``hf_config.text_config``; this class operates ENTIRELY on that text sub-config and
     never instantiates the vision tower (TEXT-FIRST — image/video inputs are out of scope).
  2. DENSE PREFIX + PER-LAYER MoE: M2 is all-MoE; M3 has a dense prefix (layers 0..2 per
     ``moe_layer_freq``) of plain SwiGLU-OAI MLPs, then MoE (layers 3..59) with 128 routed
     experts (top-4) + 1 shared expert.
  3. ACTIVATION: ``swigluoai`` (alpha=1.702, limit=7.0) everywhere (dense MLP, shared expert,
     and the routed FusedMoE), not silu.
  4. MSA SPARSE SEAM: layers 3..59 are MSA-sparse (``sparse_attention_freq``). The attention
     class exposes a CLEAR seam — an external ``msa_wire`` hook to be installed by the kernel/
     plugin (``hpu-msa-ext``) later. Until that is wired, sparse layers run GENUINE DENSE GQA
     and log ONCE that MSA is in dense-fallback (mirrors hpu-dsa-ext's FULL_ATTENTION_FALLBACK
     safety contract). NEVER a silent empty-topk consume.
  5. ROPE: partial rope (rotary_dim=64 / factor 0.5, theta=5e6) built from the flat
     ``rope_theta``/``rotary_dim`` text-config fields (M2 read a ``rope_parameters`` dict).

Everything else — the GQA attention substrate, per-head QK-norm (``MiniMaxText01RMSNormTP``),
the ``w1/w2/w3`` fused-expert weight mapping, the ``mlp.experts.`` load guard, the spec-decode
(MTP) layer skip, and the AutoRound-int4 autodetect path (handled by hpu-moe-ext with ZERO
model changes) — is preserved verbatim from M2.
"""

from collections.abc import Iterable
from typing import Any, Callable, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

import vllm.envs as envs  # noqa: F401  (kept for parity with sibling models)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SwigluOAIAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP
from vllm.model_executor.layers.layernorm import RMSNorm  # noqa: F401  (kept for parity / fallbacks)
# M3: the WHOLE tower uses Gemma-style RMSNorm `x*(1+w)` (config text_config.use_gemma_norm=True;
# class MiniMaxM3VLRMSNorm). The stored norm weights are Gemma-init (cluster near 0, include
# NEGATIVE values — impossible for a plain x*w norm). vLLM's GemmaRMSNorm implements exactly
# `x*(1+w)` with weight zero-init and the same (hidden, residual) fused signature as RMSNorm, so it
# is a drop-in for every norm site (qk-norm, input/post-attention layernorm, final norm). Using the
# plain RMSNorm here (as M2 does) silently applies the wrong per-coordinate scale on EVERY norm of
# all 60 layers -> incoherent output. See notes/reviews/index-norm-reconfirm.md.
from vllm.model_executor.models.gemma import GemmaRMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (default_weight_loader, maybe_remap_kv_scale_name)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (SupportsPP, SupportsLoRA)
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                                              make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)

logger = init_logger(__name__)


# ==================================================================================================
# DIAGNOSTIC: M3_LOGIT_PROBE=<N> dumps, for the first N compute_logits calls, the FINAL hidden-state
# stats + the lm_head weight stats + the top-predicted token(s) + logit stats. Localizes whether a
# (verified-sane) hidden state is mapped to garbage logits by the lm_head (lm_head/vocab bug) vs the
# hidden state itself being corrupt. @torch.compiler.disable + host sync; first-N only; off by default.
# ==================================================================================================
_m3_logit_probe_n = [0]

# ==================================================================================================
# DIAGNOSTIC: M3_LAYER_PROBE=1 captures the embed output + each dense-prefix layer's TRUE post-layer
# hidden state (residual stream) on the FIRST prefill, saving the raw tensors to
# M3_LAYER_PROBE_OUT (default /tmp/m3refdir/m3_served_layer.pt) for offline numerical parity vs the
# HF reference. Gated on a fixed probe length (M3_LAYER_PROBE_LEN, default 8 == the parity prompt's
# token count) so it fires on the parity prefill, not on warmup/dummy runs. eager + host sync; the
# whole-tensor .cpu() breaks lazy capture so this is a measurement-only switch (off by default).
# ==================================================================================================
_m3_layer_probe_state = {"fired": False, "fired_started": False, "caps": {}}


@torch.compiler.disable
def _m3_layer_probe_capture(tag, hidden_states, residual, positions, input_ids=None):
    import os, sys
    if os.environ.get("M3_LAYER_PROBE", "0") not in ("1", "true", "True"):
        return
    try:
        import torch
        want_len = int(os.environ.get("M3_LAYER_PROBE_LEN", "5") or "5")
        st = _m3_layer_probe_state
        # The parity prefill is a single seq of want_len real tokens at positions 0..want_len-1.
        # On HPU the batch is PADDED to a bucket (num_tokens >> want_len, tail positions are pad/0),
        # so gate on "positions start 0..want_len-1 are present and this is a prefill" rather than an
        # exact token count. We capture rows [0:want_len] (the real parity tokens) from every state.
        # IMPORTANT (HPU lazy mode): bring everything to CPU as Python lists via .tolist() — never mix
        # an HPU tensor with a CPU tensor in an op (that raises 'input on cpu'); .tolist() is a clean
        # host transfer that also acts as the graph sync.
        pos_list = positions.detach().reshape(-1).cpu().tolist()
        ntok = len(pos_list)
        if ntok < want_len:
            return  # smaller than the parity prompt -> not it
        head = [int(p) for p in pos_list[:want_len]]
        if head != list(range(want_len)):
            return  # not a fresh prefill starting at position 0
        if st["fired"]:
            return  # already captured + saved one full prefill; never re-enter (avoids deep-layer churn)
        if tag == "embed":
            st["fired_started"] = True
            if input_ids is not None:
                ii = input_ids.detach().reshape(-1)[:want_len].cpu()
                st["caps"]["input_ids"] = ii
                print(f"[m3-layer-probe] PREFILL ntok={ntok} first{want_len}_ids={ii.tolist()} "
                      f"first{want_len}_pos={head}", file=sys.stderr, flush=True)
        elif not st.get("fired_started"):
            return  # a layer tag arrived without our embed capture (different batch) -> skip
        # reconstruct the TRUE hidden state: for layer outputs it's (mlp_out + residual)
        if residual is not None:
            h = (hidden_states + residual).detach().float()
        else:
            h = hidden_states.detach().float()
        # slice the real parity rows [0:want_len] (HPU pads the prefill batch to a bucket). h is
        # either [num_tokens, hidden] (HPU flat) or [1, num_tokens, hidden]; normalize to [T, hidden],
        # then move to CPU in ONE shot.
        h2 = h.reshape(-1, h.shape[-1])[:want_len].cpu()
        st["caps"][tag] = h2
        h = h2
        msg = (f"[m3-layer-probe] {tag}: shape={tuple(h.shape)} norm={float(h.norm()):.4f} "
               f"mean={float(h.mean()):.6f} max={float(h.abs().max()):.4f} "
               f"nan={int(torch.isnan(h).sum())} | first6={[round(float(v),5) for v in h.reshape(-1)[:6].tolist()]}")
        print(msg, file=sys.stderr, flush=True)
        # dump to disk + mark fired once we've captured through M3_LAYER_PROBE_SAVE_AT (default L59_out,
        # i.e. the whole stack). Set e.g. M3_LAYER_PROBE_SAVE_AT=L4_out to dump early (MoE localization).
        save_at = os.environ.get("M3_LAYER_PROBE_SAVE_AT", "L59_out")
        if tag == save_at:
            out = os.environ.get("M3_LAYER_PROBE_OUT", "/tmp/m3refdir/m3_served_layer.pt")
            try:
                torch.save(st["caps"], out)
                print(f"[m3-layer-probe] SAVED {out} ({list(st['caps'].keys())})", file=sys.stderr, flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"[m3-layer-probe] save error: {ex!r}", file=sys.stderr, flush=True)
            st["fired"] = True
    except Exception as ex:  # noqa: BLE001
        import traceback
        print(f"[m3-layer-probe] error: {ex!r}\n{traceback.format_exc()}", file=sys.stderr, flush=True)


_m3_moe_probe_state = {"fired": False}


@torch.compiler.disable
def _m3_moe_probe_capture(x, router_logits, routed_out, shared_out):
    """Capture the FIRST MoE layer's (layer 3) internals on the parity prefill: routed output (the
    FusedMoE result, BEFORE +shared), the shared-expert output, and the router_logits — each sliced
    to the first M3_LAYER_PROBE_LEN rows. Isolates which MoE sub-path (routing / routed experts /
    shared / scaling) diverges from the HF reference. Gated by M3_MOE_PROBE=1, first MoE call only."""
    import os, sys
    if os.environ.get("M3_MOE_PROBE", "0") not in ("1", "true", "True"):
        return
    st = _m3_moe_probe_state
    if st["fired"]:
        return
    try:
        import torch
        want = int(os.environ.get("M3_LAYER_PROBE_LEN", "5") or "5")
        T = x.shape[0]
        if T < want or T > 256:
            return  # skip warmup (huge T) / undersized
        # require the first `want` rows to be real (non-zero) activations (warmup is all-zeros)
        head = x.reshape(-1, x.shape[-1])[:want].detach().float()
        if float(head.abs().max()) == 0.0:
            return
        st["fired"] = True
        out = os.environ.get("M3_MOE_PROBE_OUT", "/probeout/m3_served_moe.pt")
        # per-rank filename so we can confirm the TP reduce state (routed should be identical across
        # ranks == reduced; shared should DIFFER across ranks == unreduced partial == the bug).
        try:
            from vllm.distributed import get_tensor_model_parallel_rank as _tpr
            rk = int(_tpr())
        except Exception:  # noqa: BLE001
            rk = -1
        out = out.replace(".pt", f".rank{rk}.pt")
        caps = {
            "x": head.cpu(),
            "router_logits": router_logits.reshape(-1, router_logits.shape[-1])[:want].detach().float().cpu(),
            "routed_out": routed_out.reshape(-1, routed_out.shape[-1])[:want].detach().float().cpu(),
        }
        if shared_out is not None:
            caps["shared_out"] = shared_out.reshape(-1, shared_out.shape[-1])[:want].detach().float().cpu()
        for k, v in caps.items():
            print(f"[m3-moe-probe] {k}: shape={tuple(v.shape)} norm={float(v.norm()):.4f} "
                  f"max={float(v.abs().max()):.4f}", file=sys.stderr, flush=True)
        try:
            torch.save(caps, out)
            print(f"[m3-moe-probe] SAVED {out} ({list(caps.keys())})", file=sys.stderr, flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[m3-moe-probe] save error: {ex!r}", file=sys.stderr, flush=True)
    except Exception as ex:  # noqa: BLE001
        import traceback
        print(f"[m3-moe-probe] error: {ex!r}\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def _m3_logit_probe(hidden_states, logits, lm_head):
    import os, sys
    n = int(os.environ.get("M3_LOGIT_PROBE", "0") or "0")
    if n <= 0 or _m3_logit_probe_n[0] >= n:
        return
    _m3_logit_probe_n[0] += 1
    try:
        import torch
        h = hidden_states.detach().float()
        w = getattr(lm_head, "weight", None)
        wstat = (f"shape={tuple(w.shape)} norm={float(w.detach().float().norm()):.2f}"
                 if w is not None else "n/a")
        msg = (f"[m3-logit-probe #{_m3_logit_probe_n[0]}] hidden: shape={tuple(hidden_states.shape)} "
               f"norm={float(h.norm()):.4f} max={float(h.abs().max()):.4f} "
               f"nan={int(torch.isnan(h).sum())} | lm_head.weight: {wstat}")
        if logits is not None:
            lg = logits.detach().float()
            row = lg[-1] if lg.dim() > 1 else lg
            top = torch.topk(row, 5)
            msg += (f" | logits: shape={tuple(logits.shape)} min={float(lg.min()):.3f} "
                    f"max={float(lg.max()):.3f} nan={int(torch.isnan(lg).sum())} "
                    f"top5_ids={top.indices.cpu().tolist()} "
                    f"top5_logits={[round(float(v),3) for v in top.values.cpu().tolist()]}")
        else:
            msg += " | logits=None"
        print(msg, file=sys.stderr, flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[m3-logit-probe] error: {ex!r}", file=sys.stderr, flush=True)


_m3_mtp_dump_n = [0]


def _m3_mtp_dump_tee(hidden_states, logits):
    """M3_MTP_DUMP=1: tee (post-norm hidden, greedy-target) shards for MTP draft-head training.

    Worker-side by construction — this module is imported INSIDE each TP worker, the only place
    the per-position post-norm hidden stream exists under vLLM-V1 multiproc (the main LLM process
    has no model to reach; that is why an external _install_on_engine tap fails). Captures PREFILL
    calls only (hidden_states.shape[0] > 1 == a full prompt sequence in position order); drive the
    dump one prompt at a time with prompt_logprobs so each call is exactly one ordered sequence ->
    one shard. Only TP rank 0 writes (the post-norm hidden is replicated across ranks on the logits
    path). NO-OP unless M3_MTP_DUMP is set; never raises into the serve."""
    import os
    if os.environ.get("M3_MTP_DUMP", "0") not in ("1", "true", "True"):
        return
    try:
        import torch
        h = hidden_states
        if h.dim() != 2 or h.shape[0] <= 1:          # skip decode (single-token) calls
            return
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() != 0:
                return
        except Exception:  # noqa: BLE001
            pass
        S = h.shape[0]
        hid = h.detach().to(torch.bfloat16).cpu()
        if not torch.isfinite(hid.float()).all():
            return                                    # never write a garbage shard
        d = {"hidden": hid,
             "target_ids": logits.detach().argmax(dim=-1).to(torch.int64).cpu(),
             "positions": torch.arange(S, dtype=torch.int64)}
        ddir = os.environ.get("M3_MTP_DUMP_DIR", "/opt/m3-root/mtp/dumps")
        os.makedirs(ddir, exist_ok=True)
        n = _m3_mtp_dump_n[0]
        _m3_mtp_dump_n[0] += 1
        torch.save(d, os.path.join(ddir, f"shard_{n:06d}.pt"))
        if n < 3 or n % 50 == 0:
            import sys
            print(f"[m3-mtp-dump] wrote shard {n} ({S} rows)", file=sys.stderr, flush=True)
    except Exception as ex:  # noqa: BLE001 — never break the serve on a dump-tee error
        import sys
        print(f"[m3-mtp-dump] tee error (non-fatal): {ex!r}", file=sys.stderr, flush=True)


_m3_trace_prof = [None]
_m3_trace_step = [0]


def _m3_trace_hook(positions=None):
    """M3_TRACE=1: inject torch.profiler around N decode forwards (bypasses vLLM-gaudi:next's broken
    profiler — /start_profile 404s and VLLM_PROFILE_STEPS isn't read). Trigger: touch
    <M3_TRACE_DIR>/START_TRACE AFTER warmup; the rank-0 worker then captures M3_TRACE_N forwards,
    exports a Kineto chrome trace, removes the sentinel. NO-OP unless M3_TRACE set; never raises."""
    import os
    if os.environ.get("M3_TRACE", "0") not in ("1", "true", "True"):
        return
    if _m3_trace_prof[0] == "DONE":
        return
    try:
        import torch
        d = os.environ.get("M3_TRACE_DIR", "/profiler_out")
        sentinel = os.path.join(d, "START_TRACE")
        if _m3_trace_prof[0] is None:
            if not os.path.exists(sentinel):
                return
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
                if get_tensor_model_parallel_rank() != 0:
                    return
            except Exception:  # noqa: BLE001
                pass
            p = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                record_shapes=False, with_stack=False)
            p.start()
            _m3_trace_prof[0] = p
            _m3_trace_step[0] = 0
            print("[m3-trace] profiler STARTED (sentinel seen)", flush=True)
        _m3_trace_step[0] += 1
        if _m3_trace_step[0] >= int(os.environ.get("M3_TRACE_N", "12")):
            p = _m3_trace_prof[0]
            p.stop()
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, "m3_decode_trace.json")
            p.export_chrome_trace(out)
            print(f"[m3-trace] EXPORTED {out} after {_m3_trace_step[0]} forwards", flush=True)
            _m3_trace_prof[0] = "DONE"
            try:
                os.remove(sentinel)
            except Exception:  # noqa: BLE001
                pass
    except Exception as ex:  # noqa: BLE001 — never break the serve
        import sys
        print(f"[m3-trace] error (non-fatal): {ex!r}", file=sys.stderr, flush=True)


# ==================================================================================================
# M3: text-config resolver. The model_config.hf_config for model_type=minimax_m3_vl is the TOP
# (multimodal) config; the LM params live under .text_config. Resolve once, here, so every module
# below sees the LM sub-config exactly as M2 saw its flat config.
# ==================================================================================================
def get_text_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    # transformers PretrainedConfig.get_text_config() returns the inner text config for nested
    # multimodal configs (and returns self for flat configs). Fall back to the attribute / self.
    fn = getattr(hf_config, "get_text_config", None)
    if callable(fn):
        try:
            tc = fn()
            if tc is not None:
                return tc
        except Exception:  # noqa: BLE001 — be defensive; a flat config has no text_config
            pass
    return getattr(hf_config, "text_config", hf_config)


def _is_moe_layer(text_config: PretrainedConfig, layer_idx: int) -> bool:
    """M3: layer is MoE iff moe_layer_freq[idx]==1 (leading zeros == the dense prefix)."""
    freq = getattr(text_config, "moe_layer_freq", None)
    if freq is None:
        # No per-layer array -> fall back to a scalar dense-prefix if present, else all-MoE.
        k = int(getattr(text_config, "first_k_dense_replace", 0) or 0)
        return layer_idx >= k
    return bool(freq[layer_idx])


def _is_sparse_layer(text_config: PretrainedConfig, layer_idx: int) -> bool:
    """M3: layer uses MSA sparse attention iff sparse_attention_freq[idx]==1 (else dense GQA)."""
    sa = getattr(text_config, "sparse_attention_config", None) or {}
    if not isinstance(sa, dict):
        sa = getattr(sa, "__dict__", {}) or {}
    if not sa.get("use_sparse_attention", False):
        return False
    freq = sa.get("sparse_attention_freq")
    if freq is None:
        return False
    return bool(freq[layer_idx])


# ==================================================================================================
# MSA SPARSE SEAM (the clear hook the kernel/plugin wires later).
#
# Mirrors hpu-dsa-ext's FULL_ATTENTION_FALLBACK safety contract (platform_patch.py /
# sparse_mla_wire.py): until a validated MSA kernel installs `MSA_WIRE`, sparse layers run GENUINE
# DENSE GQA (coherent), logged ONCE, never a silent empty-topk consume.
#
# A kernel/plugin makes MSA live by calling `set_msa_wire(fn)` where `fn(attn_module, positions, q,
# k, v) -> attn_output` runs the real block-sparse top-k consume (q/k are POST rope+qk-norm; the
# attn_module exposes the index projections + per-layer sparse params built below). With the wire
# installed AND the model's `msa_ready` gate True, sparse layers route through it; otherwise they
# fall back to dense, loudly.
# ==================================================================================================
MSA_WIRE: Optional[Callable[..., torch.Tensor]] = None

# Mirror of platform_patch.FULL_ATTENTION_FALLBACK: when True (default), even sparse layers run
# genuine dense GQA. The plugin flips this False only after the MSA consume is parity-validated.
MSA_FULL_ATTENTION_FALLBACK = True

_MSA_FALLBACK_LOGGED = [False]


def set_msa_wire(fn: Optional[Callable[..., torch.Tensor]], *, full_attention_fallback: bool = False) -> None:
    """Install (or clear) the external MSA consume. Called by hpu-msa-ext once its kernel validates.

    Passing fn=None restores the SAFE dense-fallback baseline (fallback forced True regardless of
    the kwarg — no wire means there is nothing to route to). `full_attention_fallback=True` with a
    wire present lets the plugin keep dense even though the wire exists (the DSA pattern: wire
    authored but the readiness gate still closed)."""
    global MSA_WIRE, MSA_FULL_ATTENTION_FALLBACK
    MSA_WIRE = fn
    # Clearing the wire ALWAYS restores the safe dense baseline; a sparse layer must never see
    # "wire is None AND fallback is False" (that would be the empty-topk trap the contract forbids).
    MSA_FULL_ATTENTION_FALLBACK = True if fn is None else bool(full_attention_fallback)
    _MSA_FALLBACK_LOGGED[0] = False  # allow one fresh log after a (re)wire


def _log_msa_dense_fallback_once(reason: str) -> None:
    if not _MSA_FALLBACK_LOGGED[0]:
        _MSA_FALLBACK_LOGGED[0] = True
        logger.warning(
            "[minimax_m3] MSA sparse attention is in DENSE-FALLBACK (%s). Sparse layers (3..59) "
            "run GENUINE dense GQA over all causal keys — coherent, but NOT yet block-sparse. This "
            "is the safe bring-up baseline (mirrors hpu-dsa-ext FULL_ATTENTION_FALLBACK); it is "
            "NEVER a silent empty-topk consume. Install a validated MSA kernel via "
            "minimax_m3.set_msa_wire(...) + flip the gate to enable real sparsity.", reason)


_MSA_INDEXER_SKIP_LOGGED = [False]


def _log_msa_indexer_skip_once() -> None:
    if not _MSA_INDEXER_SKIP_LOGGED[0]:
        _MSA_INDEXER_SKIP_LOGGED[0] = True
        logger.warning(
            "[minimax_m3] DENSE-FALLBACK weight load: skipping MSA indexer projections "
            "(self_attn.index_{q,k}_proj / index_{q,k}_norm) on the sparse layers (3..59). They are "
            "the block-select index path and are consumed ONLY when a validated MSA wire is installed "
            "(MSA_READY). In dense GQA they have no target module — skipped, not silently mis-loaded.")


# ==================================================================================================
# MoE block (M3). Same FusedMoE wiring as M2 (routing bias + fp32 gate), PLUS:
#   - swigluoai activation + swiglu_limit + routed_scaling_factor (M3 config),
#   - an optional shared expert (n_shared_experts=1) summed with the routed output.
# The w1/w2/w3 fused-expert mapping and the gate-stays-bf16 contract are unchanged, so an int4
# AutoRound export autodetects through hpu-moe-ext with ZERO changes here.
# ==================================================================================================
class HpuMiniMaxM3MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_local_experts:
            raise ValueError(f"Tensor parallel size {self.tp_size} is greater than "
                             f"the number of experts {config.num_local_experts}.")
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(torch.empty(config.num_local_experts, dtype=torch.float32))
            self.e_score_correction_bias.weight_loader = (HpuMiniMaxM3MoE.ebias_weight_loader)
        else:
            self.e_score_correction_bias = None

        # M3: swigluoai activation for the routed experts (config.hidden_act), with the SwigluOAI
        # limit/scaling carried through. FusedMoE applies the activation internally.
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            scoring_func=config.scoring_func,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            e_score_correction_bias=self.e_score_correction_bias,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            activation=getattr(config, "hidden_act", "swigluoai"),  # M3
            swiglu_limit=getattr(config, "swiglu_limit", None),  # M3
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),  # M3
            prefix=f"{prefix}.experts",
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # M3: shared expert (always-on dense MLP summed with the routed output). n_shared_experts=1
        # with shared_intermediate_size; absent on M2.
        self.n_shared_experts = int(getattr(config, "n_shared_experts", 0) or 0)
        if self.n_shared_experts > 0:
            shared_inter = getattr(config, "shared_intermediate_size", None)
            if shared_inter is None:
                shared_inter = config.intermediate_size * self.n_shared_experts
            # M3 TP-CORRECTNESS FIX (numerically localized via layer-3 MoE parity vs HF):
            # The shared expert's down_proj is a RowParallelLinear: under TP each rank computes a
            # PARTIAL over its 1/tp slice of the intermediate dim. The routed FusedMoE output that we
            # add it to is ALREADY all-reduced (full) internally, and HpuMiniMaxM3MoE.forward performs
            # NO further all-reduce. With reduce_results=False the shared partial was therefore added
            # un-reduced: the residual stream kept only THIS rank's ~1/tp of the shared contribution
            # (and differed per rank) -> shared_out direction/magnitude wrong (measured cos 0.84,
            # norm 88 vs HF 334) -> the hidden-state DIRECTION drifts at every MoE layer (3..59),
            # compounding to byte-garbage even though magnitudes stay sane. reduce_results=True makes
            # the shared expert all-reduce its OWN partial to the full output, so (full routed) +
            # (full shared) is correct on every rank. (The alternative — passing shared_experts= into
            # FusedMoE, the laguna pattern — would need the hpu-moe-ext monolithic int4 path to fuse
            # the shared+reduce, which the swigluoai override does not; an independent reduce here is
            # the minimal, kernel-agnostic fix.)
            self.shared_experts = HpuMiniMaxM3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_inter,
                config=config,
                quant_config=quant_config,
                reduce_results=True,  # all-reduce the shared partial -> full; routed is already full
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bs, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_out = None
        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states)

        # router_logits: (bs * seq_len, n_experts)
        router_logits, _ = self.gate(hidden_states.to(torch.float32))
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

        # M3 MoE-internals parity probe (first MoE layer == layer 3): capture the routed output
        # (before +shared), the shared output, and the combined, sliced to the parity rows. Isolates
        # WHICH MoE sub-path diverges. eager + host-sync; gated by M3_MOE_PROBE=1; first MoE call only.
        _m3_moe_probe_capture(hidden_states, router_logits, final_hidden_states, shared_out)

        if shared_out is not None:
            final_hidden_states = final_hidden_states + shared_out

        return final_hidden_states.view(bs, seq_len, hidden_dim)


# ==================================================================================================
# Dense MLP (M3) — used by the dense prefix (layers 0..2) AND the shared expert. SwiGLU-OAI gated
# MLP. M2 had no dense MLP at all (all-MoE).
# ==================================================================================================
class HpuMiniMaxM3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        # M3: swigluoai (alpha, limit) — NOT silu. CRITICAL LAYOUT FIX: vLLM's stock SwigluOAIAndMul
        # CustomOp splits gate/up INTERLEAVED (gate=x[...,::2], up=x[...,1::2]) like gpt-oss. But M3's
        # weights are CONTIGUOUS: MergedColumnParallelLinear([I,I]) fuses gate_proj(w1) then up_proj(w3)
        # as the FIRST and SECOND halves, and the HF reference _apply_gate uses `gate, up =
        # gate_up.chunk(2)` ("same as GPT OSS, but the weights are NOT interleaved",
        # modeling_minimax_m3_vl.py:210). Using SwigluOAIAndMul here applied swigluoai to the WRONG
        # gate/up pairing on EVERY dense-prefix MLP (layers 0..2) AND the shared expert -> corrupted
        # residual stream -> token salad. The contiguous-chunk activation below matches the checkpoint.
        act = getattr(config, "hidden_act", "swigluoai")
        if act not in ("swigluoai", "swiglu_oai"):
            raise ValueError(f"[minimax_m3] dense MLP expects hidden_act='swigluoai', got {act!r}")
        self.swiglu_alpha = float(getattr(config, "swiglu_alpha", 1.702))
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 7.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        # CONTIGUOUS swigluoai (matches _apply_gate L209-215 + the w1|w3 fused layout): gate=first
        # half, up=second half. NOT the interleaved stock SwigluOAIAndMul.
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)            # upper bound only
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        x = (up + 1.0) * (gate * torch.sigmoid(gate * self.swiglu_alpha))
        x, _ = self.down_proj(x)
        return x


# ==================================================================================================
# Attention (GQA, NOT MLA). Identical substrate to M2 (QKV proj, partial rope, per-head QK-norm,
# stock vllm-core Attention -> Gaudi backend), PLUS the MSA sparse seam: sparse layers route through
# the external MSA wire when installed+ready, else genuine dense GQA (logged once).
# ==================================================================================================
class HpuMiniMaxM3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        is_sparse: bool = False,  # M3: this layer is an MSA-sparse layer (3..59)
        layer_idx: int = 0,  # M3: for the seam / logging
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.is_sparse = bool(is_sparse)  # M3
        self.layer_idx = int(layer_idx)  # M3

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if (rope_parameters is not None and "partial_rotary_factor" not in rope_parameters):
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        # M3: GQA, NOT MLA. Stock vllm-core Attention -> Gaudi HPUAttentionImpl (dense GQA). The MSA
        # sparse path attaches at this layer via the seam below; it does NOT use the MLA backend.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=attn_window_size,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # M3: GENUINE per-head QK-norm (config qk_norm_type='per_head'). The checkpoint stores ONE
        # `[head_dim]` weight for q_norm and one for k_norm (shape [128] == head_dim), applied
        # IDENTICALLY to every head — the standard per-head RMSNorm (cf. Qwen3). This DIFFERS from M2,
        # which used MiniMaxText01RMSNormTP: a FLAT, TP-sharded norm over the whole `head_dim*n_heads`
        # vector. That flat form only matches when num_kv_heads >= tp_size (K sharded evenly); M3 has
        # num_key_value_heads=4 < tp=8, so K is REPLICATED (per-rank width = head_dim) and the flat
        # TP-sharded weight (=head_dim*kv_heads/tp = 64) mismatches the per-rank K width (=head_dim=128)
        # -> the broadcast crash. A per-head RMSNorm over head_dim with a replicated [head_dim] weight
        # is correct for ANY TP/GQA ratio and matches the [128] checkpoint tensors exactly. The weight
        # is identical on every rank (not sharded), so default_weight_loader loads it verbatim.
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # M3: per-head RMSNorm over head_dim, applied independently to every (local) q/k head, then
        # flatten back to the (.., n_heads*head_dim) layout rotary_emb + Attention expect. q has
        # self.num_heads local heads, k has self.num_kv_heads (GQA), each of width self.head_dim.
        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim))).flatten(-2, -1)
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim))).flatten(-2, -1)
        q, k = self.rotary_emb(positions, q, k)

        # ---- MSA SPARSE SEAM ------------------------------------------------------------------
        # Dense layers (0..2) and the dense-fallback baseline both run the stock GQA attention.
        # Only a sparse layer WITH a validated MSA wire installed AND the gate open runs the real
        # block-sparse consume. Anything else => genuine dense GQA (logged once). The post-rope/
        # post-qk-norm q/k/v handed to the wire are exactly the kernel boundary the MSA spec
        # defines (MSA_SPEC.md §6: "RoPE on main Q/K and QK-norm are assumed already applied").
        if self.is_sparse and (MSA_WIRE is not None) and (not MSA_FULL_ATTENTION_FALLBACK):
            attn_output = MSA_WIRE(self, positions, q, k, v)
        else:
            if self.is_sparse:
                _log_msa_dense_fallback_once(
                    "no validated MSA wire installed" if MSA_WIRE is None else "MSA gate closed "
                    "(MSA_FULL_ATTENTION_FALLBACK=True)")
            attn_output = self.attn(q, k, v)
        # ---------------------------------------------------------------------------------------

        output, _ = self.o_proj(attn_output)
        return output


# ==================================================================================================
# Decoder layer (M3) — branches dense-vs-MoE per moe_layer_freq, and marks sparse-vs-dense
# attention per sparse_attention_freq. M2 had neither branch (all-MoE, all-dense-attention).
# ==================================================================================================
class HpuMiniMaxM3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = max(config.max_position_embeddings, config.max_model_len)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        # M3: per-layer sparse-attention flag (sparse_attention_freq). Layers 0..2 are dense GQA;
        # 3..59 are MSA-sparse (run dense-fallback until the kernel is wired).
        is_sparse = _is_sparse_layer(config, layer_idx)

        # M3: build the partial-rope parameters from the flat text-config fields. M2 read a prebuilt
        # rope_parameters dict; M3's text_config exposes rope_theta + rotary_dim + factor directly.
        rope_parameters = _build_rope_parameters(config)

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
            is_sparse=is_sparse,  # M3
            layer_idx=layer_idx,  # M3
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # M3: dense prefix vs MoE. moe_layer_freq[idx]==0 -> dense SwiGLU-OAI MLP; ==1 -> MoE.
        # Note the weight prefix is `.mlp` for BOTH (matches the checkpoint), while the attribute is
        # `block_sparse_moe` (as in M2) so the dense/MoE branch is invisible to forward().
        self.is_moe = _is_moe_layer(config, layer_idx)
        if self.is_moe:
            self.block_sparse_moe = HpuMiniMaxM3MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.block_sparse_moe = HpuMiniMaxM3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=getattr(config, "dense_intermediate_size", config.intermediate_size),
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected (dense MLP or MoE; forward signature is identical for both)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.block_sparse_moe(hidden_states)

        return hidden_states, residual


# ==================================================================================================
# M3: rope parameters builder. text_config carries rope_theta + rotary_dim + partial_rotary_factor
# as flat scalars (no rope_parameters dict like M2). Build the dict get_rope expects.
# ==================================================================================================
def _build_rope_parameters(config: PretrainedConfig) -> dict[str, Any]:
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and rp:
        return dict(rp)
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    head_dim = getattr(config, "head_dim", None)
    rotary_dim = getattr(config, "rotary_dim", None)
    factor = getattr(config, "partial_rotary_factor", None)
    params: dict[str, Any] = {"rope_type": "default", "rope_theta": rope_theta}
    if factor is not None:
        params["partial_rotary_factor"] = float(factor)
    elif rotary_dim is not None and head_dim:
        params["partial_rotary_factor"] = float(rotary_dim) / float(head_dim)
    return params


@support_torch_compile
class HpuMiniMaxM3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # M3: operate on the TEXT sub-config (model_type=minimax_m3_vl is nested). Everything below
        # uses `config` == the LM config, exactly as M2 used its flat config.
        hf_config = vllm_config.model_config.hf_config
        config = get_text_config(hf_config)
        # Carry over the runtime max_model_len if the outer wrapper stamped it on the top config.
        if hasattr(hf_config, "max_model_len") and not hasattr(config, "max_model_len"):
            config.max_model_len = hf_config.max_model_len
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HpuMiniMaxM3DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(["hidden_states", "residual"],
                                                                                       config.hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        _m3_layer_probe_capture("embed", hidden_states, None, positions, input_ids=input_ids)

        # EAGLE-3 IN-GRAPH AUX CAPTURE: when the eagle3 spec path requests aux hidden states
        # (set_aux_hidden_state_layers -> self.aux_hidden_state_layers), tee the post-layer hidden
        # at the 3 low/mid/high layers DIRECTLY in the forward graph -> the bridge captures them so
        # the EAGLE-3 head serves LAZY (no Python forward-hooks, which force enforce-eager). Captures
        # `hidden_states` (the layer's first return) to EXACTLY match the training dump (M3_MTP_DUMP_AUX
        # register_forward_hook out[0]). Empty tuple by default -> byte-identical no-op.
        _aux_layers = getattr(self, "aux_hidden_state_layers", ())
        _aux_out = []
        for li, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            hidden_states, residual = layer(positions, hidden_states, residual)
            # M3 layer-parity probe: the TRUE post-layer hidden state is the running residual
            # stream reconstructed as (mlp_out + residual) (GemmaRMSNorm fuses the add on the NEXT
            # norm). Capture it for the dense-prefix layers only (0..2), eager + host-sync.
            _m3_layer_probe_capture(f"L{self.start_layer + li}_out", hidden_states, residual, positions)
            if _aux_layers and (self.start_layer + li) in _aux_layers:
                _aux_out.append(hidden_states)   # == dump's forward_hook out[0]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        if _aux_layers:
            return hidden_states, _aux_out
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # M3: same fused w1/w2/w3 expert mapping as M2 — the contract hpu-moe-ext autodetects for
        # int4 AutoRound experts with ZERO model changes.
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # M3: dense MLP + shared-expert gate/up fuse (gate_proj/up_proj -> gate_up_proj). These
            # only match the dense-prefix MLPs and the shared experts; routed experts (w1/w2/w3) are
            # handled by the expert mapping below and are guarded out by the `mlp.experts.` check.
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # M3: the DENSE-prefix layers (0..2) ship their MLP under `...layers.N.mlp.<proj>` in the
            # checkpoint, but every MLP/MoE block is built under the `block_sparse_moe` ATTRIBUTE
            # (the dense branch reuses that name, see HpuMiniMaxM3DecoderLayer). The MoE layers
            # already use `block_sparse_moe.` in the checkpoint, so only the dense `mlp.<proj>` form
            # (gate_proj / up_proj / down_proj — NOT `mlp.experts.`) needs rewriting onto the module
            # attribute. Do it BEFORE the stacked/expert mappings so gate_up fusion + down_proj land.
            if ".mlp." in name and ".mlp.experts." not in name:
                name = name.replace(".mlp.", ".block_sparse_moe.", 1)

            # M3: MSA sparse-attention INDEXER projections (`self_attn.index_q_proj/index_k_proj/
            # index_q_norm/index_k_norm`) exist on the MSA-sparse layers (3..59). In the dense-GQA
            # fallback phase the attention module does NOT build them (no sparse consume), so they
            # have no home param. Skip them LOUDLY-ONCE rather than KeyError — they are consumed only
            # when a validated MSA wire is installed (a later task). This is the weight-load twin of
            # the model's MSA dense-fallback seam.
            if "self_attn.index_" in name:
                _log_msa_indexer_skip_once()
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ==================================================================================================
# Top class. Registered under the config's TOP arch name
# `MiniMaxM3SparseForConditionalGeneration` (multimodal), but implemented TEXT-ONLY: a pure
# ForCausalLM that ignores vision_config and image/video inputs. (A later vision pass would add the
# tower + SupportsMultiModal; out of scope here.) Named `...ForConditionalGeneration` so it accepts
# the top arch; behaviourally it is a causal-LM (cf. M2's HpuMiniMaxM2ForCausalLM).
# ==================================================================================================
class HpuMiniMaxM3ForConditionalGeneration(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        # M3: dense-prefix MLP + shared-expert gate/up fuse.
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        # M3: keep the TOP config as self.config (the spec-layer skip + lm_head read vocab_size /
        # num_hidden_layers / num_mtp_modules, all of which live on the TEXT config) — resolve to
        # text for those reads.
        config = get_text_config(hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        self.model = HpuMiniMaxM3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=None)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        # TEXT-ONLY: image/video kwargs are ignored in this phase.
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        _m3_trace_hook(positions)  # NO-OP unless M3_TRACE=1 — direct torch.profiler capture
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        _m3_logit_probe(hidden_states, logits, self.lm_head)
        _m3_mtp_dump_tee(hidden_states, logits)  # NO-OP unless M3_MTP_DUMP=1 — tees training shards
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # M3: the checkpoint is the MULTIMODAL export (model_type=minimax_m3_vl): the text tower's
        # tensors are prefixed `language_model.` (e.g. `language_model.model.layers.0...`,
        # `language_model.lm_head.weight`) and the vision stack ships under `vision_tower.` /
        # `multi_modal_projector.` / `patch_merge_mlp.`. This class is TEXT-ONLY (it builds `model.*`
        # + `lm_head` and never instantiates the vision tower), so we (1) DROP the vision tensors and
        # (2) STRIP the `language_model.` prefix so AutoWeightsLoader maps onto this module's
        # `model.*` / `lm_head.*` params. Everything downstream (the dense/MoE split, the w1/w2/w3
        # int4 expert mapping, the MTP-layer skip) is unchanged — it operates on the stripped names.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(_strip_text_tower_prefix(weights))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> int | None:
    # M3: num_mtp_modules=7 (MTP / nextn predict layers) — skipped during main-model load, same
    # mechanism as M2.
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None


# M3: the multimodal export wraps the text tower under `language_model.` and ships a vision stack we
# do not build (TEXT-ONLY serve). Strip the text prefix + drop vision tensors so the loader sees the
# same `model.*` / `lm_head.*` names every downstream mapping (dense/MoE, int4 experts, MTP skip)
# already expects. Skipped vision prefixes are logged ONCE for auditability (never a silent drop of a
# tensor that SHOULD have loaded — only the known vision stack is dropped).
_TEXT_TOWER_PREFIX = "language_model."
_VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
_SKIPPED_VISION_LOGGED = [False]


def _strip_text_tower_prefix(
        weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
    for name, w in weights:
        if name.startswith(_TEXT_TOWER_PREFIX):
            yield name[len(_TEXT_TOWER_PREFIX):], w
        elif name.startswith(_VISION_PREFIXES):
            if not _SKIPPED_VISION_LOGGED[0]:
                _SKIPPED_VISION_LOGGED[0] = True
                logger.info(
                    "[minimax_m3] TEXT-ONLY serve: skipping vision-stack weights (%s) — the vision "
                    "tower is not built in this phase.", ", ".join(_VISION_PREFIXES))
            continue
        else:
            # Not under language_model. and not a known vision prefix: pass through unchanged so a
            # genuinely unexpected name still reaches the loader (and raises loudly there) rather than
            # being silently swallowed here.
            yield name, w
