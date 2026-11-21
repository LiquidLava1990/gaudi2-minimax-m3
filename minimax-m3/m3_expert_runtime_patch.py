#!/usr/bin/env python3
# ==================================================================================================
# m3_expert_runtime_patch.py  —  RUNTIME SWITCH: serve the SAME int4 M3 checkpoint as either
#   M3_EXPERT_RUNTIME=int4  (DEFAULT)  -> existing hpu-moe-ext fused int4->bf16 MoE kernel  (MAX KV)
#   M3_EXPERT_RUNTIME=fp8              -> int4->fp8 load-time transform + native fp8 op      (MAX SPEED)
#
# ONE on-disk artifact, two HBM formats. The disk checkpoint is ALWAYS the small int4 AutoRound
# export (auto_round:auto_gptq). The switch only changes what is materialized in HBM at load:
#   * int4 : qweight int4 + group scales stay in HBM; the op dequants K routed experts->bf16 per
#            token (smallest HBM footprint -> most room for KV cache / longest context).
#   * fp8  : at load, each expert is dequantized int4->bf16 ONCE and requantized bf16->fp8 per
#            output channel (int4_to_fp8_load.stage_layer_int4_to_fp8); HBM then holds fp8 weights +
#            per-channel scales, and decode runs the stock NATIVE fp8 MoE op (no per-token dequant ->
#            fastest expert compute; larger HBM than int4, smaller than bf16).
#
# OUT-OF-TREE — DOES NOT EDIT ANY INSTALLED PLUGIN. This is a sibling vllm general_plugin that, at
# register(), WRAPS (monkeypatches) two methods of hpu-moe-ext's already-installed MoeWNA16HPUMethod:
#   - process_weights_after_loading : run the int4 path verbatim, THEN (fp8 mode) convert+swap the op
#   - apply_monolithic              : route to the fp8 op when this layer was converted
# It mirrors hpu_moe_ext/__init__.py:register()'s own monkeypatch style. hpu-moe-ext must be
# registered FIRST (it owns the int4 dispatch); this patch refines the result. Nothing here touches
# vllm-active, the cards, or the installed packages — it lives in $HOME/AI/minimax-m3/.
#
# HOW TO LOAD IT  (two equivalent out-of-tree ways; pick ONE — do NOT modify installed plugins):
#
#  (A) Entry point (preferred, same channel hpu-moe-ext uses). In a SCRATCH editable install of THIS
#      dir, add to its pyproject/setup:
#          [project.entry-points."vllm.general_plugins"]
#          m3_expert_runtime = "m3_expert_runtime_patch:register"
#      vLLM auto-runs every vllm.general_plugins entry point at engine init (after hpu-moe-ext's,
#      since plugins load in name order and refinement is order-independent here — we wrap whatever
#      MoeWNA16HPUMethod is current).
#
#  (B) Zero-install, env-injected import (no packaging). Point vLLM at a tiny sitecustomize that
#      imports+registers this module, e.g. add this dir to PYTHONPATH and set in the serve env:
#          -e M3_EXPERT_RUNTIME=fp8
#          -e PYTHONPATH=$HOME/AI/minimax-m3:$PYTHONPATH
#      then in AI/models/<env> add a one-line `-e VLLM_..`? No: simplest is a sitecustomize.py on the
#      PYTHONPATH that does `import m3_expert_runtime_patch as m; m.register()`. See SERVE NOTE below
#      for the exact env block. (register() is idempotent and safe to call from anywhere post-import
#      of hpu_moe_ext.)
#
# SERVE NOTE (env block to add to the existing int4 serve env — NOT a new serve.sh):
#   int4 (default, unchanged):   M3_EXPERT_RUNTIME unset  (or =int4)  -> nothing changes
#   fp8 (this path):             -e M3_EXPERT_RUNTIME=fp8
#                                -e PYTHONPATH=$HOME/AI/minimax-m3
#                                -e VLLM_HPU_FORCE_CHANNEL_FP8=true   (already the default; keeps the
#                                                                       NATIVE per-channel fp8 op)
#   Everything else (TP=8, --enable-expert-parallel, lazy, the int4 model path) is IDENTICAL — the
#   model on disk is the same int4 checkpoint in both modes.
#
# CARD-FREE: importing this module needs no card and no torch beyond what vllm already imports. The
# actual conversion (stage_layer_int4_to_fp8) only runs at weight-load on a card. `--selftest` here
# imports the module, resolves M3_EXPERT_RUNTIME, and confirms the wrap is well-formed without a card.
# ==================================================================================================
from __future__ import annotations

import os
import sys

_REGISTERED = False


# --------------------------------------------------------------------------------------------------
def expert_runtime() -> str:
    """Resolve the switch. 'int4' (default) | 'fp8'. Anything else is a loud error."""
    v = (os.environ.get("M3_EXPERT_RUNTIME") or "int4").strip().lower()
    if v not in ("int4", "fp8"):
        raise ValueError(f"M3_EXPERT_RUNTIME must be 'int4' or 'fp8', got {v!r}")
    return v


def _log(msg: str) -> None:
    print(f"m3-expert-runtime: {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------------
#  The wrap. Refine hpu-moe-ext's MoeWNA16HPUMethod so fp8 mode converts experts at load + reroutes.
# --------------------------------------------------------------------------------------------------
def _install():
    runtime = expert_runtime()
    if runtime == "int4":
        _log("M3_EXPERT_RUNTIME=int4 -> using the existing hpu-moe-ext int4->bf16 MoE (no change). "
             "fp8 path NOT installed.")
        return  # nothing to do; the stock int4 path is exactly what we want.

    # fp8 mode — wrap the int4 method to convert + swap the op after loading.
    try:
        from hpu_moe_ext.moe_wna16_hpu import MoeWNA16HPUMethod
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "M3_EXPERT_RUNTIME=fp8 requires hpu-moe-ext (MoeWNA16HPUMethod) to be importable and "
            f"registered FIRST — it owns the int4 dispatch this patch refines. Import failed: {e!r}")

    # Bring in the native fp8 op + our transform. The op lives in the running image's vllm_gaudi
    # extension; the on-disk mirror is $HOME/AI/_src/vllm_gaudi/extension/ops.py.
    from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpFP8PerChannel

    # int4_to_fp8_load is colocated with this file (this dir on PYTHONPATH). Import robustly whether
    # we are run as a module or imported by sitecustomize.
    try:
        import int4_to_fp8_load as i2f
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import int4_to_fp8_load as i2f

    if getattr(MoeWNA16HPUMethod, "_m3_fp8_wrapped", False):
        _log("already wrapped; skipping.")
        return

    _orig_process = MoeWNA16HPUMethod.process_weights_after_loading
    _orig_apply = MoeWNA16HPUMethod.apply_monolithic

    def process_weights_after_loading(self, layer):
        # 1) Run the int4 path VERBATIM. This produces the op-ready int32 weights, the transposed
        #    group scales, the shared symmetric zero (layer._hpu_routed_w13_zp / _w2_zp), and builds
        #    layer.moe_op = VllmMixtureOfExpertsOpWNA16(...). We need all of that as the SOURCE.
        _orig_process(self, layer)

        # AWQ per-expert zeros are out of scope for the symmetric transform; leave such a layer on
        # the int4 op (do not convert). The M3 AutoRound export is GPTQ-symmetric, so this is just a
        # safety valve — a converted-vs-not flag drives apply_monolithic.
        if getattr(self.quant_config, "has_zp", False):
            _log(f"layer {getattr(layer, 'layer_name', '?')}: AWQ per-expert zeros -> kept int4 "
                 "(fp8 transform is GPTQ-symmetric only).")
            layer._m3_fp8_converted = False
            return

        # 2) Build the native per-channel fp8 op with the SAME local-window geometry the int4 op uses
        #    (LOCAL window [0, num_experts-1]; apply_monolithic already remaps global->local ids, so
        #    the fp8 op — like the int4 op — runs on local ids and must use experts_min=0).
        num_experts = layer.moe_op.num_experts
        global_num_experts = getattr(layer.moe_op, "global_num_experts", num_experts)
        fp8_op = VllmMixtureOfExpertsOpFP8PerChannel(
            global_num_experts, num_experts, experts_min=0, experts_max=num_experts - 1,
            dispatch_fn=getattr(layer.moe_op, "dispatch_func", None))

        # 3) int4 -> fp8 per channel for every local expert, written DIRECTLY into pre-allocated
        #    stacked fp8 tensors on fp8_op (fp8_op._w13_fp8_stack [E,H,2I] + _s13_stack [E,2I],
        #    _w2_fp8_stack [E,I,H] + _s2_stack [E,H]), with each per-expert MoeFP8Matmul handle
        #    pointed at a VIEW into the stack. This builds the SINGLE fp8 copy the route-b decode/
        #    prefill paths gather via index_select(0, active) (no per-element host read -> no host
        #    sync / torch.compile break), and frees the layer's int4 source + reclaims it to the pool
        #    BEFORE the next layer's repack — the load-time peak-HBM fix (see int4_to_fp8_load
        #    PEAK-HBM CONTRACT). HBM holds ONE fp8 copy; no transient int4+2x-fp8 per layer.
        i2f.stage_layer_int4_to_fp8(layer, fp8_op)

        # 4) Swap the op. apply_monolithic now drives the fp8 path instead of the int4 dequant.
        layer.moe_op = fp8_op
        layer._m3_fp8_converted = True
        # mark so a double process_weights call is a no-op (the int4 method has its own guard, but the
        # conversion must not re-run either).
        layer._hpu_moe_ext_done = True

        # 5) Harden the op's rebuild hooks. The stacks + per-expert VIEWS are already built+published
        #    by stage_layer_int4_to_fp8 (step 3); we must NOT re-stack here (that would build a THIRD
        #    full fp8 copy and defeat the peak-HBM fix). All this does is make a later model.to('hpu')
        #    -> _apply -> _cache_weight_lists a safe no-op (route b reads _w*_fp8_stack directly, and
        #    the stacks are already on HPU, so there is nothing to rebuild/re-move).
        _harden_fp8_op_hooks(fp8_op)

        # 6) ROUTE (a) staging (only if selected): the native bias_fp8_fused_weights op expects w13 in
        #    INTERLEAVED gate/up order; M3 is CONTIGUOUS. Reorder w13 (and its per-channel scale) to
        #    interleaved ONCE here, plus build the non-empty zero-bias lists the op's check_inputs
        #    demands (M3 experts bias=False; the +1 is in the gpt_swiglu formula, so zero bias is exact).
        #    Skipped for route b to avoid the extra interleaved-w13 HBM copy.
        if _fp8_route() == "a":
            _stage_interleave_for_route_a(fp8_op)

    def apply_monolithic(self, layer, x, router_logits, **kwargs):
        if getattr(layer, "_m3_fp8_converted", False):
            # fp8-converted layer -> the swigluoai fp8 apply (route a native gpt_swiglu, or route b
            # unfused-fp8). NEVER the silu-only native op (M3 needs swigluoai, not silu).
            return _apply_fp8_swigluoai(self, layer, x, router_logits)
        # not converted (e.g. AWQ layer) -> the original int4 apply.
        return _orig_apply(self, layer, x, router_logits, **kwargs)

    MoeWNA16HPUMethod.process_weights_after_loading = process_weights_after_loading
    MoeWNA16HPUMethod.apply_monolithic = apply_monolithic
    MoeWNA16HPUMethod._m3_fp8_wrapped = True

    # Hand the swigluoai patch a reference to our fp8 apply, so when it (the OUTERMOST wrap) sees an
    # fp8-converted layer it delegates here instead of running int4 plumbing on freed weights. The
    # swigluoai patch registers AFTER us (hpu_m3_ext.register() order: expert-runtime THEN swigluoai),
    # so by the time it wraps apply_monolithic, MoeWNA16HPUMethod._m3_fp8_apply is already set.
    MoeWNA16HPUMethod._m3_fp8_apply = staticmethod(_apply_fp8_swigluoai)
    _log("M3_EXPERT_RUNTIME=fp8 -> wrapped MoeWNA16HPUMethod: experts int4->fp8 at load, served with "
         f"SWIGLUOAI on the fp8 path (route={_fp8_route()}; fp8 weights, no per-token int4 dequant).")


# ==================================================================================================
#  SWIGLUOAI ON THE FP8 PATH
#  The native per-channel fp8 op (VllmMixtureOfExpertsOpFP8PerChannel) feeds fp8 weights straight to
#  torch.ops.hpu.mixture_of_experts(... activation=<str>) — but that string map is silu/gelu/relu
#  ONLY (mixture_of_experts_gen.cpp activationModeMap). M3 needs swigluoai. Two routes:
#    (a) bias_fp8_fused_weights  — the NATIVE fp8 gpt_swiglu kernel (alpha=1.702, limit=7.0). FAST.
#        Needs INTERLEAVED gate/up (op expects ::2/1::2) — M3 weights are CONTIGUOUS (gate=first half)
#        so w13 + its scale are reordered to interleaved AT LOAD. Needs non-empty zero-bias lists.
#        CATCH: hpu_op.yaml gives this schema `frontend_blocklist: [eager]`; on the eager-frontend
#        :next image (M3 runs under torch.compile, not lazy) it falls to the CPU/composite path =
#        SLOW. So route (a) is attempted only when explicitly forced or when probed live to be on the
#        device kernel; otherwise route (b).
#    (b) unfused fp8 (DEFAULT, robust) — dequant the routed-K experts fp8->bf16 via the op's CHANNEL
#        get_dequant_weight() (a single `weight.to(bf16)*scale` multiply — NOT int4's per-token
#        convert_from_uint4 nibble-unpack, which is the measured ~98% decode tax), apply the EXACT
#        swigluoai in torch, GEMM. Mirrors the proven _routed_moe routing + mark_step discipline.
# ==================================================================================================

# bit-exact swigluoai numerics (swigluoai-numerics.md). Prefer the installed hpu_m3_ext tail so there
# is ONE source of truth; fall back to an inline copy (identical math) if that import is unavailable.
try:
    from hpu_m3_ext.m3_swigluoai_patch import swigluoai_tail as _swigluoai_tail  # noqa: F401
    _ALPHA_DEFAULT, _LIMIT_DEFAULT = 1.702, 7.0
except Exception:  # noqa: BLE001
    _ALPHA_DEFAULT, _LIMIT_DEFAULT = 1.702, 7.0

    def _swigluoai_tail(gate_up, alpha, limit):
        import torch
        gate, up = gate_up.chunk(2, dim=-1)          # gate=FIRST, up=SECOND (contiguous halves)
        gate = gate.clamp(max=limit)                 # UPPER bound only
        up = up.clamp(min=-limit, max=limit)         # symmetric
        glu = gate * torch.sigmoid(gate * alpha)     # alpha INSIDE sigmoid
        return (up + 1.0) * glu                      # the +1, then (up+1)*glu


def _fp8_route() -> str:
    """auto (default) | b | a. 'b' = unfused-fp8 (robust, FAST on this image); 'a' = native
    bias_fp8_fused_weights gpt_swiglu (only worthwhile on a lazy frontend where it is NOT
    eager-blocklisted). 'auto' picks b (the safe default on the eager-frontend :next image)."""
    v = (os.environ.get("M3_FP8_SWIGLU_ROUTE") or "auto").strip().lower()
    if v not in ("auto", "a", "b"):
        raise ValueError(f"M3_FP8_SWIGLU_ROUTE must be auto|a|b, got {v!r}")
    return v


def _fp8_mme() -> bool:
    """M3_FP8_MME A/B switch (default OFF). The THIRD, independent route-b compute path: a TRUE
    fp8 x fp8 -> bf16 GEMM on the MME (torch.ops.hpu.fp8_gemm_v2), instead of dequant-to-bf16 +
    bf16 torch.matmul.

    WHY: the default/NOTRANSPOSE forms dequant the fp8 weight to bf16 (one multiply) then run a
    *bf16* GEMM — the MME never sees fp8 (matmul ~= 32% of decode device time, profiled). Here we
    keep the weight fp8, quantize the activation to fp8 (dynamic per-token, EXACTLY dynamic_quant /
    apply_fp8_linear_hpu), and call the native fp8 GEMM (~2x MME throughput, no bf16 dequant scratch).
    The swigluoai tail stays bf16/f32 (only the two MATMULS go fp8).

    PRECEDENCE: M3_FP8_MME is independent of M3_FP8_NOTRANSPOSE. If BOTH are set, MME WINS (the MME
    path is its own end-to-end branch; NOTRANSPOSE's bf16-no-transpose form is not used). Documented
    in _fp8_route_b_routed / _fp8_route_b_allexpert.

    ORIENTATION (researched from the native callers — see _fp8_gemm): fp8_gemm_v2 with trans_B=False
    consumes B as [out, in] (the SAME layout vLLM Linear stores; hpu_fp8.py:130, hpu_modelopt.py:163,
    hpu_compressed_tensors.py:304 all pass layer.weight=[out,in] with trans_B=False). M3's CONTIGUOUS
    [out,in] originals are op._w13_oi_stack/_w2_oi_stack (published by the static-fp8 producer); the
    int4->fp8 RUNTIME stacks are [in,out] (op._w13_fp8_stack) and use trans_B=True. The MME path picks
    the orientation per available handle, so it works in BOTH static-fp8 and int4->fp8 modes."""
    return (os.environ.get("M3_FP8_MME", "0").strip().lower()
            in ("1", "true", "yes", "on"))


def _fp8_notranspose() -> bool:
    """M3_FP8_NOTRANSPOSE A/B switch (default OFF == exact current transposed-view behavior).

    WHY: vLLM stores the experts CONTIGUOUS as [out, in] (w13_weight [E,2I,H], w2_weight [E,H,I]).
    The default route-b form dequants a TRANSPOSED VIEW ([in,out] = [H,2I]/[I,H]) and does
    `gate_up = x[T,H] @ w13_deq[H,2I]`. Because w13_deq came from a strided/transposed view, the
    Gaudi matrix engine must physically transpose the FULL ~52 GiB of expert weights every token
    (a DmaTranspose ~= 110 ms/forward, ~32% of single-stream decode device time).

    When ON, route-b consumes the CONTIGUOUS [out,in] storage directly via the identity
        x[T,H] @ Wdeq[H,2I]  ==  (W13_oi[2I,H] @ x.t()[H,T]).t()
    so ONLY the tiny T==1 activation x is transposed, never the big weights. Numerically bit-faithful:
    W13_oi[o,h] == w13_deq[h,o] (same fp8*scale dequant value, scale is per-OUT-channel either way),
    and both matmuls contract over the same `in` dim in the same index order.

    ENGAGES ONLY when the op also exposes the contiguous [out,in] handles op._w13_oi_stack /
    op._w2_oi_stack (set by the static-fp8 producer m3_fp8_static_patch._build_fp8_stacks, where the
    original [out,in] weights stay alive — zero extra HBM). The int4->fp8 RUNTIME path builds its
    op._w13_fp8_stack ALREADY-contiguous [in,out] (no [out,in] original survives the int4 free), so it
    has no transpose tax and is intentionally left on the current form even when this flag is set."""
    return (os.environ.get("M3_FP8_NOTRANSPOSE", "0").strip().lower()
            in ("1", "true", "yes", "on"))


def _gate_markstep() -> bool:
    """LEVER #5 (2026-06-24): default ON -> skip the LAZY-only mark_steps in the non-lazy
    (torch.compile / HPUGraph) serve, where the dataflow graph already orders ops and each mark_step is
    a graph-break / host fence. The routed decode path fired ~2-6 mark_steps PER MoE LAYER x 57 layers
    = ~114-342 host fences/decode step -> the single-stream host-overhead wall. Mirrors the ALREADY-
    gated all-expert path (m3_swigluoai_patch._gate_markstep). M3_GATE_LAZY_MARKSTEP=0 to A/B legacy
    (always-fire) for the baseline measurement."""
    return (os.environ.get("M3_GATE_LAZY_MARKSTEP") or "1").strip() not in ("0", "false", "False", "no", "off")


def _ms() -> None:
    """Gated mark_step: fire iff lazy OR the gate is disabled (legacy A/B). No-op in the non-lazy serve
    by default -> lets the decode step capture as ONE HPU graph instead of ~hundreds of fenced segments."""
    import habana_frameworks.torch.core as htorch
    if htorch.is_lazy() or not _gate_markstep():
        htorch.mark_step()


def _alpha_limit(layer):
    limit = getattr(layer, "swiglu_limit", None)
    if limit is None:
        limit = getattr(getattr(layer, "act_fn", None), "limit", None)
    alpha = getattr(getattr(layer, "act_fn", None), "alpha", None)
    return (float(alpha) if alpha is not None else _ALPHA_DEFAULT,
            float(limit) if limit is not None else _LIMIT_DEFAULT)


# one-shot announce / route-decision guards (per worker process)
_announced_b = [False]
_announced_a = [False]
_route_a_live = [None]   # None=undecided, True=device kernel confirmed, False=fell back to b
_shape_ok = [False]
_dbg_count = [0]         # M3_DEBUG_ROUTE: log (T,K,routed_eligible,path) for the first N apply calls
_dbg_seen = {}           # M3_DEBUG_ROUTE: T -> count, to log first occurrences of each DISTINCT T


def _route_topk(layer, x, router_logits):
    """Routing IDENTICAL to the int4 swigluoai/stock path: grouped-topk via the layer's router (so
    grouped-topk + correction bias + renorm all match the int4 serve bit-for-bit)."""
    import torch
    import torch.nn.functional as F
    if getattr(layer, "use_grouped_topk", False) or getattr(layer, "custom_routing_function", None) is not None:
        topk_weights, topk_ids = layer.router.select_experts(hidden_states=x, router_logits=router_logits)
    else:
        tw = F.softmax(router_logits, dim=1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(tw, layer.top_k, dim=-1)
        if getattr(layer, "renormalize", True):
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int64).view(-1, topk_ids.shape[-1])
    topk_weights = topk_weights.to(x.dtype).view(-1, topk_weights.shape[-1])
    return topk_ids, topk_weights


_announced_mono = [False]
_mono_warned = [False]
_mono_tcount = [0]
_mono_total = [0]


def _m3_fused_mono(method, layer, x2, topk_ids, topk_weights):
    """vllm-banz: device-resident dynamic all-E swigluoai-fp8 MoE via the validated
    fused_moe_monolithic_fp8 op (relNorm 0.04 on-card vs route-b, fits 0.91GB/card).
    Returns [T,H] or None (caller falls back to the unfused route-b loop)."""
    try:
        from hpu_moe_ext.fused_monolithic_moe_fp8 import fused_moe_monolithic_fp8
    except Exception as e:  # noqa: BLE001
        if not _mono_warned[0]:
            _mono_warned[0] = True
            _log(f"MMEL_M3_FUSED=1 but fused_monolithic_moe_fp8 import failed ({e}) -> route-b.")
        return None
    op = getattr(layer, "moe_op", None)
    if op is None or not hasattr(op, "_w13_fp8_stack"):
        return None
    # NOTE: contiguous-copy of the stacks was TRIED (cache op._w13_mono_contig) — it did NOT
    # speed the op in-serve (the bottleneck is recipe COMPILE-cost/caching, not stride) AND it
    # OOMed warmup (+0.9GB/card). Reverted: pass the route-b transposed-view stacks directly.
    emin = getattr(layer, "_hpu_experts_min", 0)
    alpha, limit = _alpha_limit(layer)
    out = fused_moe_monolithic_fp8(
        x2, op._w13_fp8_stack, op._s13_stack, op._w2_fp8_stack, op._s2_stack,
        topk_ids, topk_weights, emin, alpha=alpha, limit=limit)
    if out is not None and not _announced_mono[0]:
        _announced_mono[0] = True
        _log("LIVE: M3 routed fp8 experts via the FUSED MONOLITHIC swigluoai-fp8 op "
             "(device-resident all-E + dense [E,T,1] router, fp8-native MME) [vllm-banz].")
    return out


def _apply_fp8_swigluoai(method, layer, x, router_logits, **kwargs):
    """Apply the fp8 routed experts with the EXACT M3 swigluoai gate. Single-token decode uses the
    routed-K fast path; prefill/multi-token uses the all-expert path. Route (a) native gpt_swiglu is
    attempted only when selected & live; else route (b) unfused-fp8.

    Accepts **kwargs and IGNORES them: the swigluoai patch (the OUTERMOST apply_monolithic wrap that
    delegates here) forwards whatever kwargs the FusedMoE dispatch passed it — `fp8_apply(self, layer,
    x, router_logits, **kwargs)`. The stock hpu-moe-ext apply_monolithic also takes **kwargs, and the
    routing here is fully recomputed from (layer, router_logits) via _route_topk, so any extra dispatch
    kwargs are not needed. Swallowing them keeps a future vLLM that adds a monolithic kwarg from
    crashing the fp8 serve with a TypeError (DSA build-quality bar: no latent on-card-only failure)."""
    import torch
    input_shape = x.shape
    x2 = x.view(-1, x.shape[-1])
    topk_ids, topk_weights = _route_topk(layer, x2, router_logits)

    # vllm-banz: the VALIDATED device-resident dynamic fused monolithic swigluoai-fp8 op
    # (on-card relNorm 0.04 vs route-b, fits 0.91GB/card) replaces the unfused route-b loop
    # when MMEL_M3_FUSED=1 -- one fast recipe the owned-forward engine composes on.
    if os.environ.get("MMEL_M3_FUSED") == "1":
        _mono = _m3_fused_mono(method, layer, x2, topk_ids, topk_weights)
        if _mono is not None:
            return _mono.view(*input_shape)

    route = _fp8_route()
    K = layer.top_k
    T = x2.shape[0]
    # routed-K fast path eligibility: single-token decode (T*top_k <= K <=> T==1), same gate the int4
    # swigluoai/_routed_moe path uses. Multi-token (prefill) -> all-expert path.
    routed_eligible = (T * K) <= K

    if os.environ.get("M3_DEBUG_ROUTE"):
        _dbg_seen[T] = _dbg_seen.get(T, 0) + 1
        if _dbg_seen[T] <= 3:   # first 3 of EACH distinct T -> prefill T=2048 won't drown out decode T
            _log(f"DEBUG_ROUTE: T(x2.shape[0])={T} K(top_k)={K} route={route} "
                 f"routed_eligible={routed_eligible} -> "
                 f"{'ROUTED-K' if (route!='a' and routed_eligible) else 'ALL-EXPERT'} (T-seen#{_dbg_seen[T]})")

    if route == "a":
        out = _fp8_route_a_native(method, layer, x2, topk_ids, topk_weights)
        if out is not None:
            if not _announced_a[0]:
                _announced_a[0] = True
                _log("LIVE: routed fp8 experts via NATIVE bias_fp8_fused_weights (gpt_swiglu "
                     "alpha=1.702 limit=7.0, interleaved gate/up, zero-bias) [route a].")
            return out.view(*input_shape)
        # route a unavailable (op missing) -> fall through to b
        if _route_a_live[0] is None:
            _route_a_live[0] = False
            _log("route a (bias_fp8_fused_weights) unavailable -> falling back to unfused-fp8 [route b].")

    # route (b) unfused-fp8 (default + fallback)
    if routed_eligible:
        out = _fp8_route_b_routed(method, layer, x2, topk_ids, topk_weights, K)
    else:
        out = _fp8_route_b_allexpert(method, layer, x2, topk_ids, topk_weights)
    if not _announced_b[0]:
        _announced_b[0] = True
        _log("LIVE: routed fp8 experts via UNFUSED swigluoai on fp8 weights "
             "(get_dequant_weight = weight.to(bf16)*scale; gate.clamp(max=7)*sigmoid(1.702*gate)*"
             f"(up.clamp(-7,7)+1)) [route b, {'routed-K decode' if routed_eligible else 'all-expert'}].")
    return out.view(*input_shape)


def _assert_shapes(layer, x2, w13_deq0, w2_deq0, K):
    """One-shot LOUD orientation check (DSA build-quality bar). The fp8 get_dequant_weight returns
    the SAME [in, out] orientation the int4 convert_from_uint4 does: W13_deq [H, 2I], W2_deq [I, H]."""
    if _shape_ok[0]:
        return
    _shape_ok[0] = True
    H = x2.shape[-1]
    twoI = w13_deq0.shape[-1]
    if w13_deq0.shape[0] != H or (twoI % 2) != 0:
        raise RuntimeError(
            f"m3-fp8-swigluoai: W13_deq shape {tuple(w13_deq0.shape)} != [H={H}, 2I]; fp8 dequant "
            "orientation ([in=H, out=2I]) is wrong -> garbage. Refusing to serve.")
    I = twoI // 2
    if w2_deq0.shape[0] != I or w2_deq0.shape[1] != H:
        raise RuntimeError(
            f"m3-fp8-swigluoai: W2_deq shape {tuple(w2_deq0.shape)} != [I={I}, H={H}]; down_proj "
            "orientation ([in=I, out=H]) is wrong. Refusing to serve.")
    _log(f"fp8 shape OK: x[T,{H}] W13_deq[{H},{twoI}] W2_deq[{I},{H}] (gate=:I up=I:, K={K}).")


_shape_ok_oi = [False]


def _assert_shapes_oi(layer, x2, w13_oi0, w2_oi0, K):
    """One-shot LOUD orientation check for the M3_FP8_NOTRANSPOSE [out,in] path (DSA build-quality bar).
    Here the dequantized weight stays in the ORIGINAL contiguous [out, in] layout: W13_oi [2I, H],
    W2_oi [H, I]. gate_up = (W13_oi @ x.t()).t(); y = (W2_oi @ h.t()).t()."""
    if _shape_ok_oi[0]:
        return
    _shape_ok_oi[0] = True
    H = x2.shape[-1]
    twoI = w13_oi0.shape[0]
    if w13_oi0.shape[1] != H or (twoI % 2) != 0:
        raise RuntimeError(
            f"m3-fp8-swigluoai[notranspose]: W13_oi shape {tuple(w13_oi0.shape)} != [2I, H={H}]; the "
            "no-transpose dequant orientation ([out=2I, in=H]) is wrong -> garbage. Refusing to serve.")
    I = twoI // 2
    if w2_oi0.shape[0] != H or w2_oi0.shape[1] != I:
        raise RuntimeError(
            f"m3-fp8-swigluoai[notranspose]: W2_oi shape {tuple(w2_oi0.shape)} != [H={H}, I={I}]; the "
            "down_proj no-transpose orientation ([out=H, in=I]) is wrong. Refusing to serve.")
    _log(f"fp8 shape OK [notranspose]: x[T,{H}] W13_oi[{twoI},{H}] W2_oi[{H},{I}] "
         f"(gate=:I up=I:, K={K}); only the T-row activation is transposed, weights stay contiguous.")


def _dequant_fp8_slice(w_fp8, s, hi=None):
    """fp8 [.,.] * per-OUTPUT-channel scale [1,out] -> bf16. EXACTLY MoeFP8Matmul.get_dequant_weight's
    CHANNEL branch (ops.py:1133-1135): (weight.to(scale.dtype) * scale_inv_fp8).to(high_precision). A
    single elementwise multiply — NOT int4's convert_from_uint4 nibble-unpack (the ~98% decode tax).

    Orientation here is [in, out]: w_fp8 is [in, out], s is [1, out] -> scale broadcasts over `in`."""
    import torch
    return (w_fp8.to(s.dtype) * s).to(hi or torch.bfloat16)


def _dequant_fp8_slice_oi(w_fp8_oi, s, hi=None):
    """[out,in] sibling of _dequant_fp8_slice for the M3_FP8_NOTRANSPOSE path. w_fp8_oi is the
    CONTIGUOUS [out, in] expert slice; s is the per-OUTPUT-channel scale [out]. In [out,in] the scale
    indexes dim 0 (out), so it broadcasts over dim 1 (in) -> reshape to [out, 1].

    BIT-FAITHFUL to the transposed form: for every element, value == _dequant_fp8_slice(w_fp8_oi.t(),
    s.unsqueeze(0)).t() because (a) the fp8->scale.dtype cast, (b) the multiply in scale.dtype (f32),
    and (c) the .to(bf16) are all elementwise and transpose-invariant, and the SAME out-channel scale
    hits the SAME weight element in both layouts."""
    import torch
    return (w_fp8_oi.to(s.dtype) * s.view(-1, 1)).to(hi or torch.bfloat16)


# ==================================================================================================
#  M3_FP8_MME — TRUE fp8 x fp8 -> bf16 GEMM on the MME (the native fp8_gemm_v2 primitive)
#
#  Mirrors the proven vllm_gaudi.extension.ops fp8 path EXACTLY:
#    * activation quant:  dynamic_quant(x)  (ops.py:932-939) = per-token (dim=-1) amax scale,
#      data_fp8 = cast_to_fp8_v2(data, 1.0/scale, False, False, torch.float8_e4m3fn); returns
#      (data_fp8, scale.float()) where scale is the FORWARD (recovery) scale [T,1].
#    * GEMM:  torch.ops.hpu.fp8_gemm_v2 (ops.py:919-928, apply_fp8_linear_hpu):
#          fp8_gemm_v2(A=x_fp8, trans_A=False, B=w_fp8, trans_B=<orient>, D=None,
#                      out_dtype=bf16, A_scale_inv=x_scale, B_scale_inv=w_scale,
#                      bias=None, accumulate=False)
#      A_scale_inv / B_scale_inv are the FORWARD recovery scales (op multiplies by them after the
#      int8-domain accumulate); for B these are the per-OUTPUT-channel weight scales we already store
#      (op._s13_stack [.,2I], op._s2_stack [.,H]) — the SAME tensors the native silu op passes as
#      d_scale_w12/d_scale_w3 (ops.py:1292-1293). NO new scale is computed for the weights.
#
#  fnuz/fn: the dtype handed to the cast/GEMM is float8_e4m3fn (bias 7) — IDENTICAL to dynamic_quant
#  and to int4_to_fp8_load._cast_to_fp8 / FP8_DTYPE. The HW maps fn->fnuz internally, and the
#  cast(1/s) <-> op(d_scale=s) pair is the self-consistent round-trip that cancels the fn<->fnuz
#  factor-of-2 (int4_to_fp8_load comment, ops.py:30). The M3 weight stacks were ALREADY built with
#  this exact convention (FP8_DTYPE=fn, FP8_MAX=fnuz-240). So we pass them straight through; we do NOT
#  re-quantize or re-scale the weights, and we use the SAME float8_e4m3fn for the activation cast — a
#  consistent fn-domain GEMM, exactly like the native silu op.
#
#  ORIENTATION (fp8_gemm_v2, confirmed from every native Linear caller):
#    A=[T,in], trans_A=False ; output=[T,out].
#    trans_B=True   <=>  B is [out, in]  (vLLM's stored Linear layout; hpu_fp8.py:130 etc.)  -> y=A@B.T
#    trans_B=False  <=>  B is [in, out]  (our int4->fp8 runtime stack layout)                -> y=A@B
#    B_scale_inv is per-OUTPUT-channel either way (out is B-dim-0 if [out,in], B-dim-1 if [in,out]),
#    matching the 1-D [out] scale we store. Prefer the CONTIGUOUS [out,in] _oi_stack (trans_B=True);
#    fall back to the [in,out] _fp8_stack (trans_B=False) for the int4->fp8 mode.
# ==================================================================================================
_fp8_gemm_v2 = [None]   # cached torch.ops.hpu.fp8_gemm_v2 handle (resolved once, per worker)
_mme_shape_ok = [False]
_mme_announced = [False]


def _resolve_fp8_gemm():
    """Resolve torch.ops.hpu.fp8_gemm_v2 once. LOUD if absent (DSA build-quality bar: M3_FP8_MME was
    explicitly requested; silently falling back to a bf16 GEMM would hide that the fast path is dead)."""
    if _fp8_gemm_v2[0] is not None:
        return _fp8_gemm_v2[0]
    import torch
    fn = getattr(getattr(torch.ops, "hpu", None), "fp8_gemm_v2", None)
    if fn is None:
        raise RuntimeError(
            "M3_FP8_MME=1 but torch.ops.hpu.fp8_gemm_v2 is unavailable on this image. This is the "
            "native fp8 GEMM the per-channel fp8 MoE op (VllmMixtureOfExpertsOpFP8PerChannel) uses; "
            "its absence means the MME fp8 path cannot run. Refusing to silently serve a slower bf16 "
            "GEMM under an explicit speed flag — unset M3_FP8_MME or fix the image.")
    _fp8_gemm_v2[0] = fn
    return fn


def _quant_act_fp8(x):
    """Quantize an activation tensor to fp8 EXACTLY like vllm_gaudi.extension.ops.dynamic_quant
    (per-token / dim=-1 amax FORWARD scale; cast_to_fp8_v2 with 1/scale into float8_e4m3fn). Returns
    (x_fp8 [T, in] float8_e4m3fn, x_scale [T, 1] float32). On a card uses the HPU cast intrinsic; off
    a card (no torch.ops.hpu) falls back to the same math via the native fp8 dtype cast so the helper
    stays importable/testable, but the real GEMM still requires the card."""
    import torch
    FP8_MAX = float(torch.finfo(torch.float8_e4m3fnuz).max)   # 240.0 — the fnuz max, per ops.py:30
    scale = ((torch.abs(x)).max(dim=-1).values + 1e-8) / FP8_MAX
    scale = scale.unsqueeze(-1)                                # [T, 1] forward scale
    cast = getattr(getattr(torch.ops, "hpu", None), "cast_to_fp8_v2", None)
    if cast is not None:
        x_fp8 = cast(x, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
    else:
        x_fp8 = (x.float() / scale).to(torch.float8_e4m3fn)   # off-card emulation (no real GEMM)
    return x_fp8, scale.float()


def _fp8_gemm(x_fp8, x_scale, w_fp8, w_scale, trans_B, out_dtype=None):
    """One expert GEMM via the native torch.ops.hpu.fp8_gemm_v2 (mirrors apply_fp8_linear_hpu).
      x_fp8 [T, in] fp8 ; x_scale [T,1] f32 (FORWARD) ; w_fp8 fp8 ; w_scale [out] f32 (FORWARD, per
      output channel) ; trans_B selects the weight orientation (True=[out,in], False=[in,out]).
    Returns [T, out] in out_dtype (default bf16)."""
    import torch
    fn = _resolve_fp8_gemm()
    return fn(A=x_fp8, trans_A=False, B=w_fp8, trans_B=trans_B, D=None,
              out_dtype=(out_dtype or torch.bfloat16),
              A_scale_inv=x_scale, B_scale_inv=w_scale, bias=None, accumulate=False)


def _mme_handles(op):
    """Pick the weight stacks + trans_B for the MME path. Prefer the CONTIGUOUS [out,in] _oi_stack
    (trans_B=True — the GEMM transposes B); else the [in,out] _fp8_stack
    (trans_B=False — the int4->fp8 runtime stack). Returns
    (w13_stack, w2_stack, trans_B, layout_str)."""
    if getattr(op, "_w13_oi_stack", None) is not None and getattr(op, "_w2_oi_stack", None) is not None:
        return op._w13_oi_stack, op._w2_oi_stack, True, "[out,in] (trans_B=True)"
    return op._w13_fp8_stack, op._w2_fp8_stack, False, "[in,out] (trans_B=False)"


def _assert_shapes_mme(layer, x2, w13_0, w2_0, w13_scale_0, w2_scale_0, trans_B, K, layout_str):
    """One-shot LOUD orientation/scale gate for the M3_FP8_MME path (DSA build-quality bar). With
    trans_B=True the weight is [out,in] (w13 [2I,H], w2 [H,I]); with trans_B=False it is [in,out]
    (w13 [H,2I], w2 [I,H]). The per-out-channel scale length must equal the OUTPUT dim (2I for w13, H
    for w2) in either layout, or the GEMM silently corrupts -> we fail loud instead."""
    if _mme_shape_ok[0]:
        return
    _mme_shape_ok[0] = True
    H = x2.shape[-1]
    if trans_B:                                   # [out,in]: out==dim0, in==dim1
        if w13_0.shape[1] != H or (w13_0.shape[0] % 2) != 0:
            raise RuntimeError(
                f"m3-fp8-mme: W13 fp8 [out,in] shape {tuple(w13_0.shape)} != [2I, H={H}]; trans_B=True "
                "orientation is wrong -> garbage. Refusing to serve.")
        twoI = w13_0.shape[0]
        I = twoI // 2
        if w2_0.shape[0] != H or w2_0.shape[1] != I:
            raise RuntimeError(
                f"m3-fp8-mme: W2 fp8 [out,in] shape {tuple(w2_0.shape)} != [H={H}, I={I}]; trans_B=True "
                "down_proj orientation is wrong. Refusing to serve.")
    else:                                         # [in,out]: in==dim0, out==dim1
        if w13_0.shape[0] != H or (w13_0.shape[1] % 2) != 0:
            raise RuntimeError(
                f"m3-fp8-mme: W13 fp8 [in,out] shape {tuple(w13_0.shape)} != [H={H}, 2I]; trans_B=False "
                "orientation is wrong -> garbage. Refusing to serve.")
        twoI = w13_0.shape[1]
        I = twoI // 2
        if w2_0.shape[0] != I or w2_0.shape[1] != H:
            raise RuntimeError(
                f"m3-fp8-mme: W2 fp8 [in,out] shape {tuple(w2_0.shape)} != [I={I}, H={H}]; trans_B=False "
                "down_proj orientation is wrong. Refusing to serve.")
    # per-output-channel scale length must match the output dim (2I for w13, H for w2).
    s13_len = int(w13_scale_0.reshape(-1).shape[0])
    s2_len = int(w2_scale_0.reshape(-1).shape[0])
    if s13_len != twoI or s2_len != H:
        raise RuntimeError(
            f"m3-fp8-mme: per-out-channel scale length s13={s13_len} (want 2I={twoI}) s2={s2_len} "
            f"(want H={H}); weight-scale axis mismatch -> wrong dequant. Refusing to serve.")
    _log(f"fp8 MME shape OK: x[T,{H}] fp8 W13{tuple(w13_0.shape)} W2{tuple(w2_0.shape)} "
         f"scale(s13={s13_len},s2={s2_len}) layout={layout_str} (gate=:I up=I:, K={K}); "
         "native fp8_gemm_v2 (e4m3fn), activation dynamic-quant per-token, swigluoai tail bf16.")


def _fp8_route_b_routed(method, layer, x, topk_ids, topk_weights, K):
    """UNFUSED fp8 routed-K path. Gather ONLY the <=K routed experts from the stacked fp8 weights
    (index_select — no host sync), dequant fp8->bf16 (one multiply each), swigluoai in torch, GEMM,
    weighted-combine. Routing + mark_step discipline mirrors _routed_moe. Static shape (T==1 decode)."""
    import torch
    import habana_frameworks.torch.core as htorch
    op = layer.moe_op
    E = op.num_experts
    emin = getattr(layer, "_hpu_experts_min", 0)
    dev = x.device

    # --- routing -> K active local experts (VERBATIM from _routed_moe) ----------------------------
    local = topk_ids.to(torch.int64) - emin
    flat = local.reshape(-1)
    valid = (flat >= 0) & (flat < E)
    scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
    present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
        0, scatter_idx, valid.to(torch.float32))[:E]
    active = torch.topk(present, K).indices.to(torch.int64)        # [K] local ids
    _ms()

    # M3_FP8_MME (PRECEDENCE: wins over M3_FP8_NOTRANSPOSE): TRUE fp8 GEMM on the MME. Its own branch
    # below — keeps weights fp8, quantizes x to fp8, calls fp8_gemm_v2. When set, _notr is forced off
    # so the bf16 no-transpose branch is not also taken.
    _mme = _fp8_mme()
    # M3_FP8_NOTRANSPOSE: consume the CONTIGUOUS [out,in] originals so only x (T==1) is transposed,
    # never the big weights. Engages only when the producer published op._w13_oi_stack / _w2_oi_stack
    # (the static-fp8 path; the int4->fp8 stacks are already-contiguous [in,out] and stay on the form
    # below). _w13_oi_stack is the SAME storage as the transposed view -> no extra HBM.
    _notr = (not _mme) and _fp8_notranspose() and getattr(op, "_w13_oi_stack", None) is not None
    _need_clone = htorch.is_lazy()
    def _co(t):
        return t.clone() if _need_clone else t
    s13 = op._s13_stack.index_select(0, active)                    # [K, 2I]
    s2 = op._s2_stack.index_select(0, active)                      # [K, H]

    alpha, limit = _alpha_limit(layer)

    # --- per-(token,expert) routing weight matrix [T, K] (VERBATIM from the int4 swigluoai path) ----
    # Computed before the compute branch so the MME path (which returns early) and the bf16 paths share
    # the SAME combine. (Moved up from below; identical math.)
    remap = torch.zeros(E, dtype=torch.int64, device=dev).scatter(
        0, active, torch.arange(K, dtype=torch.int64, device=dev))
    flat_safe = torch.where(valid, flat, torch.zeros_like(flat))
    compact = remap[flat_safe].view_as(topk_ids)
    valid2d = valid.view_as(topk_ids)
    safe_w = torch.where(valid2d, topk_weights.to(x.dtype), torch.zeros_like(topk_weights).to(x.dtype))
    T = x.shape[0]
    gate_mat = torch.zeros(T, K, dtype=x.dtype, device=dev).scatter_add_(1, compact, safe_w)  # [T,K]

    if _mme:
        # ---- TRUE fp8 x fp8 -> bf16 GEMM (native fp8_gemm_v2). gather fp8 weights + per-out scales,
        #      quantize x to fp8 (dynamic per-token), two fp8 GEMMs per expert, swigluoai tail bf16. ---
        w13s, w2s, trans_B, layout = _mme_handles(op)
        w13p = w13s.index_select(0, active)                       # [K, *, *] fp8
        w2p = w2s.index_select(0, active)                         # [K, *, *] fp8
        _ms()
        _assert_shapes_mme(layer, x, w13p[0], w2p[0], s13[0], s2[0], trans_B, K, layout)
        x_fp8, x_scale = _quant_act_fp8(x)                        # [T,H] fp8, [T,1] f32 (once for all K)
        out = torch.zeros(T, x.shape[-1], dtype=x.dtype, device=dev)
        for k in range(K):
            # gate_up = x_fp8 @ W13_fp8 -> [T, 2I] bf16   (B scale = per-out-channel s13[k] [2I])
            gate_up = _fp8_gemm(_co(x_fp8), _co(x_scale), _co(w13p[k]), _co(s13[k]), trans_B, x.dtype)
            h = _swigluoai_tail(gate_up, alpha, limit)           # [T, I] bf16/f32 (tail stays high-prec)
            h_fp8, h_scale = _quant_act_fp8(h.to(x.dtype))       # quantize the swigluoai output to fp8
            y_k = _fp8_gemm(h_fp8, h_scale, _co(w2p[k]), _co(s2[k]), trans_B, x.dtype)  # [T, H] bf16
            out = out + gate_mat[:, k:k + 1] * y_k
            _ms()
        if not _mme_announced[0]:
            _mme_announced[0] = True
            _log("LIVE: routed fp8 experts via NATIVE fp8_gemm_v2 MME GEMM "
                 "(x dynamic-quant per-token e4m3fn; W per-out-channel fp8; swigluoai tail bf16) "
                 f"[route b, MME, {layout}, routed-K decode].")
        return out

    if _notr:
        # --- gather + dequant in ORIGINAL [out,in] layout (NO weight transpose) --------------------
        w13p = op._w13_oi_stack.index_select(0, active)           # [K, 2I, H] fp8 (contiguous orient)
        w2p = op._w2_oi_stack.index_select(0, active)             # [K, H,  I] fp8
        w13_deq = [_dequant_fp8_slice_oi(_co(w13p[k]), _co(s13[k])) for k in range(K)]   # [2I,H] bf16
        w2_deq = [_dequant_fp8_slice_oi(_co(w2p[k]), _co(s2[k])) for k in range(K)]       # [H,I]  bf16
        _ms()
        _assert_shapes_oi(layer, x, w13_deq[0], w2_deq[0], K)
    else:
        # --- gather + dequant ONLY the K routed experts (index_select on the stacks; NO host read) --
        w13p = op._w13_fp8_stack.index_select(0, active)           # [K, H, 2I] fp8
        w2p = op._w2_fp8_stack.index_select(0, active)             # [K, I, H]  fp8
        # fp8 -> bf16: a single multiply each (MoeFP8Matmul CHANNEL get_dequant_weight) — NOT int4's
        # convert_from_uint4 nibble-unpack (the measured ~98% decode tax this whole fp8 path removes).
        w13_deq = [_dequant_fp8_slice(_co(w13p[k]), _co(s13[k]).unsqueeze(0)) for k in range(K)]  # [H,2I]
        w2_deq = [_dequant_fp8_slice(_co(w2p[k]), _co(s2[k]).unsqueeze(0)) for k in range(K)]      # [I,H]
        _ms()
        _assert_shapes(layer, x, w13_deq[0], w2_deq[0], K)

    out = torch.zeros(T, x.shape[-1], dtype=x.dtype, device=dev)
    xt = x.t() if _notr else None                                 # [H,T]; transpose the TINY activation
    for k in range(K):
        if _notr:
            gate_up = torch.matmul(w13_deq[k], xt).t()            # ([2I,H]@[H,T]).t() -> [T,2I]
            h = _swigluoai_tail(gate_up, alpha, limit)            # [T, I]
            y_k = torch.matmul(w2_deq[k], h.t()).t()              # ([H,I]@[I,T]).t() -> [T,H]
        else:
            gate_up = torch.matmul(x, w13_deq[k])                 # [T,H]@[H,2I] -> [T,2I]
            h = _swigluoai_tail(gate_up, alpha, limit)            # [T, I]
            y_k = torch.matmul(h, w2_deq[k])                      # [T,I]@[I,H] -> [T,H]
        out = out + gate_mat[:, k:k + 1] * y_k
    return out


def _fp8_route_b_allexpert(method, layer, x, topk_ids, topk_weights):
    """UNFUSED fp8 ALL-EXPERT path (prefill T>1 / universal). Per-LOCAL-expert dequant fp8->bf16 +
    swigluoai + GEMM, masked by a [T, E] routing matrix. Static shape (E fixed)."""
    import torch
    import habana_frameworks.torch.core as htorch
    op = layer.moe_op
    E = op.num_experts
    emin = getattr(layer, "_hpu_experts_min", 0)
    dev = x.device
    T = x.shape[0]
    alpha, limit = _alpha_limit(layer)

    local = topk_ids.to(torch.int64) - emin
    in_win = (local >= 0) & (local < E)
    safe_local = torch.where(in_win, local, torch.zeros_like(local))
    safe_w = torch.where(in_win, topk_weights.to(x.dtype), torch.zeros_like(topk_weights).to(x.dtype))
    route = torch.zeros(T, E, dtype=x.dtype, device=dev).scatter_add_(1, safe_local, safe_w)  # [T,E]

    _need_clone = htorch.is_lazy()
    def _co(t):
        return t.clone() if _need_clone else t

    # M3_FP8_MME (PRECEDENCE: wins over M3_FP8_NOTRANSPOSE): TRUE fp8 GEMM on the MME. Keeps weights
    # fp8, quantizes x to fp8, calls fp8_gemm_v2 per local expert. swigluoai tail stays bf16/f32.
    if _fp8_mme():
        w13s, w2s, trans_B, layout = _mme_handles(op)
        _assert_shapes_mme(layer, x, w13s[0], w2s[0], op._s13_stack[0], op._s2_stack[0], trans_B, E, layout)
        x_fp8, x_scale = _quant_act_fp8(x)                       # [T,H] fp8, [T,1] f32 (once for all E)
        _ms()
        out = torch.zeros(T, x.shape[-1], dtype=x.dtype, device=dev)
        for e in range(E):
            gate_up = _fp8_gemm(_co(x_fp8), _co(x_scale), _co(w13s[e]), _co(op._s13_stack[e]), trans_B, x.dtype)
            h = _swigluoai_tail(gate_up, alpha, limit)          # [T, I] bf16/f32
            h_fp8, h_scale = _quant_act_fp8(h.to(x.dtype))      # quantize swigluoai output to fp8
            y_e = _fp8_gemm(h_fp8, h_scale, _co(w2s[e]), _co(op._s2_stack[e]), trans_B, x.dtype)  # [T,H]
            out = out + route[:, e:e + 1] * y_e
            _ms()
        if not _mme_announced[0]:
            _mme_announced[0] = True
            _log("LIVE: fp8 experts via NATIVE fp8_gemm_v2 MME GEMM "
                 "(x dynamic-quant per-token e4m3fn; W per-out-channel fp8; swigluoai tail bf16) "
                 f"[route b, MME, {layout}, all-expert].")
        return out

    # M3_FP8_NOTRANSPOSE: dequant from the CONTIGUOUS [out,in] originals (no big-weight transpose);
    # engages only when the static-fp8 producer published op._w13_oi_stack / _w2_oi_stack.
    _notr = _fp8_notranspose() and getattr(op, "_w13_oi_stack", None) is not None
    if _notr:
        w13_deq = [_dequant_fp8_slice_oi(op._w13_oi_stack[e], op._s13_stack[e]) for e in range(E)]  # [2I,H]
        w2_deq = [_dequant_fp8_slice_oi(op._w2_oi_stack[e], op._s2_stack[e]) for e in range(E)]      # [H,I]
        _ms()
        _assert_shapes_oi(layer, x, w13_deq[0], w2_deq[0], E)
    else:
        w13_deq = [_dequant_fp8_slice(op._w13_fp8_stack[e], op._s13_stack[e].unsqueeze(0)) for e in range(E)]
        w2_deq = [_dequant_fp8_slice(op._w2_fp8_stack[e], op._s2_stack[e].unsqueeze(0)) for e in range(E)]
        _ms()
        _assert_shapes(layer, x, w13_deq[0], w2_deq[0], E)
    out = torch.zeros(T, x.shape[-1], dtype=x.dtype, device=dev)
    xt = x.t() if _notr else None                                # [H,T]; transpose the TINY activation
    for e in range(E):
        if _notr:
            gate_up = torch.matmul(w13_deq[e], xt).t()           # ([2I,H]@[H,T]).t() -> [T,2I]
            h = _swigluoai_tail(gate_up, alpha, limit)           # [T, I]
            y_e = torch.matmul(w2_deq[e], h.t()).t()             # ([H,I]@[I,T]).t() -> [T,H]
        else:
            gate_up = torch.matmul(x, w13_deq[e])                # [T,H]@[H,2I] -> [T,2I]
            h = _swigluoai_tail(gate_up, alpha, limit)           # [T, I]
            y_e = torch.matmul(h, w2_deq[e])                     # [T,I]@[I,H] -> [T,H]
        out = out + route[:, e:e + 1] * y_e
    return out


def _fp8_route_a_native(method, layer, x, topk_ids, topk_weights):
    """ROUTE (a): the NATIVE fp8 gpt_swiglu kernel torch.ops.hpu.mixture_of_experts.bias_fp8_fused_weights
    (alpha=1.702, limit=7.0). Returns the output, or None if the op is unavailable (caller falls to b).

    Requires: (1) w13 in INTERLEAVED gate/up layout (op expects ::2/1::2) — staged at load into
    op._w13_il / scale op._w13_il_scale; (2) per-channel d_scale lists; (3) non-empty zero-bias lists;
    (4) x cast to fp8 + its scale. The op fuses dequant+gpt_swiglu+down internally.

    NOTE: on the eager-frontend image this schema is frontend_blocklist:[eager] -> CPU composite (slow).
    Only select route a (M3_FP8_SWIGLU_ROUTE=a) on a lazy frontend where the device kernel is live."""
    import torch
    moe = getattr(torch.ops.hpu, "mixture_of_experts", None)
    bias_fp8 = getattr(moe, "bias_fp8_fused_weights", None) if moe is not None else None
    if bias_fp8 is None:
        return None
    op = layer.moe_op
    il_w13 = getattr(op, "_w13_il", None)
    il_s13 = getattr(op, "_w13_il_scale", None)
    if il_w13 is None or il_s13 is None:
        # interleaved staging absent -> route a not prepared; fall back.
        return None
    emin = getattr(layer, "_hpu_experts_min", 0)
    E = op.num_experts
    alpha, limit = _alpha_limit(layer)

    local = (topk_ids.to(torch.int64) - emin)
    in_win = (local >= 0) & (local < E)
    local = torch.where(in_win, local, torch.zeros_like(local))
    rw = torch.where(in_win, topk_weights.to(x.dtype), torch.zeros_like(topk_weights).to(x.dtype))

    # fp8 activation
    from vllm_gaudi.extension.ops import dynamic_quant
    x_fp8, x_scale = dynamic_quant(x)

    w13_list = il_w13                # interleaved fp8 w13 (tuple per expert)
    w2_list = op._w2_views_a         # down-proj fp8 views from the stack
    s13_list = il_s13                # interleaved per-channel scale (tuple per expert)
    s2_list = op._s2_views_a
    z13 = op._zero_bias_w13          # non-empty zero bias (M3 experts bias=False; +1 is in the formula)
    z2 = op._zero_bias_w2
    out = bias_fp8(
        hidden_states=x_fp8, expert_routing_table=local, router_weights=rw,
        w12=w13_list, w3=w2_list, w12_bias=z13, w3_bias=z2,
        d_scale_hidden_states=x_scale,
        d_scale_intermediate_hidden_states=[],
        d_scale_w12=s13_list, d_scale_w3=s2_list,
        permuted_weights=True, experts_min=0, experts_max=E - 1,
        alpha=alpha, limit=limit)
    return out


def _harden_fp8_op_hooks(fp8_op) -> None:
    """Make the fp8 op's rebuild hooks safe no-ops. The per-channel fp8 stacks
    (fp8_op._w13_fp8_stack [E,H,2I] + _s13_stack [E,2I]; _w2_fp8_stack [E,I,H] + _s2_stack [E,H]) and
    the per-expert .weight/.scale_inv_fp8 VIEWS into them are ALREADY built+published by
    int4_to_fp8_load.stage_layer_int4_to_fp8 (the single physical fp8 copy — the peak-HBM fix). This
    helper deliberately does NOT (re)stack or (re)point anything: re-stacking would build a THIRD full
    fp8 copy of the layer and undo the fix.

    What it DOES: a later model.to('hpu') calls op._apply -> _cache_weight_lists, which would churn (or
    crash) on the staged weights. Make both no-ops — route b reads _w*_fp8_stack directly (not the
    _cached_* views), and the stacks + per-expert views are already on HPU (the conversion ran on
    device), so there is nothing to rebuild or re-move. _apply returns self (final placed state)."""
    import types as _types

    def _safe_cache(self):
        return  # no-op: route b uses _w*_fp8_stack, not _cached_* views

    def _safe_apply(self, fn):
        return self  # already on HPU; do NOT rebuild caches or re-move staged tensors

    fp8_op._cache_weight_lists = _types.MethodType(_safe_cache, fp8_op)
    fp8_op._apply = _types.MethodType(_safe_apply, fp8_op)


def _stage_interleave_for_route_a(fp8_op) -> None:
    """Reorder w13 + per-channel scale from M3's CONTIGUOUS [gate(I) | up(I)] to the op's INTERLEAVED
    [gate0,up0,gate1,up1,...] along the output dim, and build zero-bias lists. Reads the fp8 stacks
    built by stage_layer_int4_to_fp8 (op._w13_fp8_stack etc.). Stores op._w13_il / op._w13_il_scale
    (tuples per expert) + op._zero_bias_w13 / op._zero_bias_w2. Route a only."""
    import torch
    import habana_frameworks.torch.core as htorch
    E = fp8_op.num_experts
    twoI = fp8_op._w13_fp8_stack.shape[-1]
    I = twoI // 2
    dev = fp8_op._w13_fp8_stack.device
    # contiguous -> interleaved permutation on the OUTPUT (last) dim:
    #   gate cols = [0..I), up cols = [I..2I); interleaved wants col 2j=gate[j], 2j+1=up[j].
    perm = torch.empty(twoI, dtype=torch.int64, device=dev)
    idx = torch.arange(I, dtype=torch.int64, device=dev)
    perm[0::2] = idx           # even slots <- gate[0..I)
    perm[1::2] = idx + I       # odd  slots <- up[0..I)
    il_w13, il_s13 = [], []
    for e in range(E):
        il_w13.append(fp8_op._w13_fp8_stack[e].index_select(-1, perm).contiguous())   # [H, 2I]
        il_s13.append(fp8_op._s13_stack[e].index_select(0, perm).contiguous())        # [2I]
        _ms()
    fp8_op._w13_il = tuple(il_w13)
    fp8_op._w13_il_scale = tuple(il_s13)
    H = fp8_op._w2_fp8_stack.shape[-1]
    Idown = fp8_op._w2_fp8_stack.shape[1]
    # zero-bias lists (check_inputs demands non-empty w12_bias/w3_bias). bias=False on M3 experts; the
    # +1 on up is hardcoded inside gpt_swiglu, so a literal zero bias is numerically exact.
    fp8_op._zero_bias_w13 = tuple(torch.zeros(twoI, dtype=torch.bfloat16, device=dev) for _ in range(E))
    fp8_op._zero_bias_w2 = tuple(torch.zeros(H, dtype=torch.bfloat16, device=dev) for _ in range(E))
    # route a also needs the (down) w2 + its scale as per-expert views from the stacks
    fp8_op._w2_views_a = tuple(fp8_op._w2_fp8_stack[e] for e in range(E))
    fp8_op._s2_views_a = tuple(fp8_op._s2_stack[e] for e in range(E))
    _ms()


# --------------------------------------------------------------------------------------------------
def register():
    """vllm.general_plugins entry point. Idempotent. Safe to call after hpu_moe_ext.register()."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        _install()
        _REGISTERED = True
    except Exception as e:  # noqa: BLE001
        # LOUD failure (DSA build-quality bar) — never silently fall back to a wrong precision.
        _log(f"FAILED to install fp8 runtime switch: {e!r}")
        raise


# --------------------------------------------------------------------------------------------------
def _selftest() -> int:
    """Card-free wiring check: resolve the switch + confirm the wrap is structurally valid without a
    card and without a real checkpoint. Does NOT import vllm/hpu (which need the image); it asserts
    the env resolution, the colocated transform import, and that the documented routing constants
    exist. The numeric proof is `int4_to_fp8_load.py --validate`."""
    print("=" * 88)
    print("m3_expert_runtime_patch — card-free selftest")
    print("=" * 88)
    rt = expert_runtime()
    print(f"M3_EXPERT_RUNTIME resolves to : {rt}")
    print(f"  int4 -> hpu-moe-ext int4->bf16 fused MoE (max KV room)")
    print(f"  fp8  -> int4->fp8 load-time transform + native VllmMixtureOfExpertsOpFP8PerChannel (max speed)")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    try:
        import int4_to_fp8_load as i2f
        ok = all(hasattr(i2f, n) for n in
                 ("quant_weight_fp8_per_channel", "dequant_int4_to_bf16", "stage_layer_int4_to_fp8",
                  "validate_synthetic"))
        print(f"colocated int4_to_fp8_load importable + API present : {ok}")
        print(f"  FP8_DTYPE={i2f.FP8_DTYPE}  FP8_MAX={i2f.FP8_MAX}  UINT4_SYM_ZERO={i2f.UINT4_SYM_ZERO}")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to import colocated transform: {e!r}")
        return 1
    print("registration target : hpu_moe_ext.moe_wna16_hpu.MoeWNA16HPUMethod "
          "(process_weights_after_loading + apply_monolithic) — wrapped out-of-tree, installed plugin untouched.")
    print("VERDICT: wiring OK (card-free). Numeric proof: `python3 int4_to_fp8_load.py --validate`.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
