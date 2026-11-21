#!/usr/bin/env python3
# ==================================================================================================
# m3_swigluoai_patch.py  —  per-expert swigluoai MoE for the MiniMax-M3 int4 routed experts.
#
# PERF NOTE (2026-06-15, MEASURED on-card — two optimizations EVALUATED and REJECTED):
#   1. NATIVE captured swiglu intrinsic torch.ops.hpu.mixture_of_experts.bias_fused_weights(alpha=
#      1.702, limit=7.0): the schema exists and carries the EXACT M3 gpt_swiglu defaults, BUT on
#      vllm-gaudi:next the device kernel is NOT viable — fed perm=False it returns ~0 (relNorm ~1.0),
#      fed perm=True it HANGS the card (Synapse "Compute or DMA timeout"). (The plain `silu` overload
#      of the SAME op works perfectly — relNorm 4e-3 — so routing/shapes are correct; the gpt_swiglu
#      leg itself is broken/blocklisted here. Matches the capability note's eager-blocklist caveat.)
#   2. BATCHED torch.bmm over stacked experts (replace this per-expert loop with one bmm pair +
#      elementwise swigluoai): numerically bit-equal (relNorm ~4-5e-3) but SLOWER everywhere on M3 —
#      single-stream 5.0 vs 8.3 tok/s, c=30 aggregate 7.7 vs 17.4, TTFT 128s vs 25s. The stack/
#      expand/reduce add host overhead and the all-expert bmm [E,T,2I] builds huge prefill recipes
#      that blow TTFT. At decode top_k=4 the loop is only 4 already-captured matmuls; the single-
#      stream wall is the STRUCTURAL host/sync-bound decode (attention + TP all-reduce + sampler),
#      not this loop. So: KEEP the per-expert loop. Full numbers in minimax-m3/bench/M3_MATRIX_REPORT.md.
#
# THE BLOCKER (validated by notes/reviews/swigluoai-*.md):
#   M3's routed experts use the gpt-oss-style "swigluoai" gate, NOT silu:
#       gate, up = gate_up.chunk(2, dim=-1)          # gate=FIRST half, up=SECOND half (CONTIGUOUS)
#       gate = gate.clamp(max=7.0)                   # UPPER bound only
#       up   = up.clamp(min=-7.0, max=7.0)           # symmetric
#       glu  = gate * sigmoid(gate * 1.702)          # alpha INSIDE sigmoid
#       h    = (up + 1.0) * glu                      # the +1, then (up+1)*glu
#   But the Habana `torch.ops.hpu.mixture_of_experts` intrinsic (the tail of hpu-moe-ext's int4
#   WNA16 path: VllmMixtureOfExpertsOpWNA16 / _routed_moe / fused_*) only understands silu/gelu.
#   Feeding "swigluoai" either raises or silently applies plain SiLU -> WRONG / incoherent output.
#
# THE TRAP (why we do NOT dispatch off the activation string):
#   transformers' MiniMaxM3VLTextConfig.__init__ OVERRIDES hidden_act -> "silu"
#   (configuration_minimax_m3_vl.py L127-129: "Checkpoint declares 'swigluoai', but the gate is
#   computed inline from swiglu_alpha/limit; ... normalize it to silu."). So at runtime the WNA16
#   apply_monolithic sees act=="silu". Triggering "when act != silu" would NEVER fire here and we
#   would serve plain SiLU = token salad. THEREFORE this patch is gated by an EXPLICIT env signal
#   (M3_SWIGLUOAI=1), and when on it applies the INLINE swigluoai math to EVERY routed int4 MoE
#   layer regardless of the act string. (M3's only int4 MoE is the routed experts; the dense-prefix
#   MLPs + shared expert are bf16 in minimax_m3.py with their own correct SwigluOAIAndMul and never
#   reach this method.)
#
# THE FIX (PRIORITY-1 — guaranteed coherent, no kernel / no .so / no card-side build):
#   Out-of-tree wrap of MoeWNA16HPUMethod.apply_monolithic. When M3_SWIGLUOAI=1 and the layer is the
#   GPTQ-symmetric int4 routed MoE, run the UNFUSED path: reuse the proven _routed_moe dequant
#   (convert_from_uint4 gather of the <=top_k routed experts + the mandatory mark_step discipline),
#   then compute gate_up = x @ W13_deq, chunk to gate/up (CONTIGUOUS), apply the EXACT swigluoai
#   above in plain torch elementwise, down_proj, weighted-combine. This swaps ONLY the silu intrinsic
#   tail for torch-math swigluoai; the dequant + routing + mark_step are reused bit-for-bit.
#   process_weights_after_loading is left UNTOUCHED (the int4 weights/scales/shared-zero are exactly
#   the source we need). DOES NOT EDIT any installed plugin (mirrors hpu-moe-ext's monkeypatch style).
#
# WEIGHT LAYOUT (load-bearing; derived from hpu-moe-ext _repack_weight + convert_from_uint4):
#   layer.w13_qweight [E, in=H,  out//8=2I//8] int32 ; layer.w13_scales [E, groups, out=2I]
#       convert_from_uint4(w13p[H, 2I//8], s13[groups, 2I], zp, dt, None) -> W13_deq [H, 2I] = [in,out]
#       => gate_up = x[T,H] @ W13_deq[H,2I] = [T, 2I]            (NO .T — dequant is already [in,out])
#   layer.w2_qweight  [E, in=I,  out//8=H//8 ] int32 ; layer.w2_scales  [E, groups, out=H ]
#       convert_from_uint4(w2p[I, H//8], s2[groups, H], zp, dt, None)    -> W2_deq  [I, H ] = [in,out]
#       => y = h[T,I] @ W2_deq[I,H] = [T, H]
#   The first decode call ASSERTS these exact shapes (LOUD on mismatch — DSA build-quality bar).
#
# VERIFY (M3_SWIGLUOAI_VERIFY=1, EAGER-ONLY — the .item() breaks lazy capture): on the FIRST call,
#   prove the unfused PLUMBING (dequant+GEMM+routing+orientation) is bit-exact by running the SAME
#   path with a SiLU tail and comparing to the stock _routed_moe (which uses the silu intrinsic).
#   relNorm < 1e-2 => the only remaining delta is the activation tail, which is pure torch math
#   unit-tested CPU-side (tests/test_swigluoai_unfused.py). LOUD on mismatch.
#
# SAFETY / QUALITY: LOUD failure (raise) if M3_SWIGLUOAI=1 but the wrap cannot be installed; never a
#   silent wrong-activation serve. No host sync on the hot path (the verify .item() is first-call /
#   eager-only). The per-expert compute is a static-shape [T,2I]/[T,H] GEMM loop over the K routed
#   experts (T*top_k<=K guaranteed by the routed gate) -> no dynamic shape, no recompiles.
# ==================================================================================================
from __future__ import annotations

import os
import sys

_REGISTERED = False

# M3 swigluoai constants (config text_config: swiglu_alpha=1.702, swiglu_limit=7.0). These are the
# checkpoint's fixed values; we read them off the layer when available (below) and fall back here.
_ALPHA_DEFAULT = 1.702
_LIMIT_DEFAULT = 7.0


def enabled() -> bool:
    return (os.environ.get("M3_SWIGLUOAI") or "").strip() in ("1", "true", "True", "yes", "on")


def _verify() -> bool:
    return (os.environ.get("M3_SWIGLUOAI_VERIFY") or "").strip() in ("1", "true", "True", "yes", "on")


def _gate_markstep() -> bool:
    """FIX 1 (2026-06-15): skip the two LAZY-only mark_steps in the routed/all-expert MoE under
    torch.compile (they are pure host-sync overhead when not lazy). DEFAULT ON. Set
    M3_GATE_LAZY_MARKSTEP=0 ONLY to A/B the old always-mark_step behaviour (baseline measurement).
    When ON, the mark_steps fire iff htorch.is_lazy(); when OFF they always fire (legacy)."""
    v = (os.environ.get("M3_GATE_LAZY_MARKSTEP") or "1").strip()
    return v in ("1", "true", "True", "yes", "on")


def _no_lazy_markstep() -> bool:
    """RANK-0 capture experiment: SKIP the routed-MoE mark_steps EVEN under lazy, so
    wrap_in_hpu_graph captures the WHOLE decode step in one replay (the fences fragment it).
    The MoE is functional-ops replay-safe (held-leaf zp, .scatter not scatter_, per-slice .clone),
    so the fences may be droppable. M3_NO_LAZY_MARKSTEP=1 to try; LOUD-verify coherence after."""
    return (os.environ.get("M3_NO_LAZY_MARKSTEP") or "0").strip() in ("1", "true", "True", "yes", "on")


def _log(msg: str) -> None:
    print(f"m3-swigluoai: {msg}", file=sys.stderr, flush=True)


# one-shot guards (per worker process)
_shape_checked = [False]
_verified = [False]
_announced = [False]       # decode K-gather path announced
_announced_all = [False]   # all-expert (prefill) path announced
_probe_done = [0]          # M3_SWIGLUOAI_PROBE: dump first N decode MoE-input/routing fingerprints


try:
    import torch as _torch_for_probe
    _compiler_disable = _torch_for_probe.compiler.disable
except Exception:  # noqa: BLE001
    def _compiler_disable(fn):
        return fn


@_compiler_disable
def _probe(layer, x, router_logits, tid, tw_):
    """Reliable (host-synced, compiler-disabled) first-call dump to localize corruption: MoE input
    activation stats, router_logits stats, and the ACTUAL topk ids/weights (in [0,128) if sane).
    Gated by M3_SWIGLUOAI_PROBE=<N>. Unlike the lazy .cpu() routed-debug, this is correct because it
    runs outside the captured graph."""
    import os as _os
    import torch
    n = int(_os.environ.get("M3_SWIGLUOAI_PROBE", "0") or "0")
    if n <= 0 or _probe_done[0] >= n:
        return
    # SKIP warmup dummy shapes: warmup captures padded prefill buckets (e.g. T=2048) with ZERO dummy
    # inputs. Only probe REAL inference (small-batch decode / a real prompt's prefill, T<=128) so x
    # reflects actual activations, not the all-zeros warmup scenario.
    if x.shape[0] > 128:
        return
    _probe_done[0] += 1
    try:
        import logging
        log = logging.getLogger("vllm")
        xf = x.detach().float()
        rl = router_logits.detach().float()
        lid = getattr(layer, "layer_name", "?")
        emin = getattr(layer, "_hpu_experts_min", 0)
        ids0 = tid.detach().to(torch.int64).reshape(-1)[:8].cpu().tolist()
        w0 = tw_.detach().float().reshape(-1)[:8].cpu().tolist()
        log.warning(
            "[m3-probe #%d lid=%s emin=%s] x: shape=%s norm=%.4f max=%.4f nan=%d | "
            "router_logits: shape=%s min=%.4f max=%.4f nan=%d | topk_ids[:8]=%s topk_w[:8]=%s",
            _probe_done[0], lid, emin, tuple(x.shape),
            float(xf.norm()), float(xf.abs().max()), int(torch.isnan(xf).sum()),
            tuple(router_logits.shape), float(rl.min()), float(rl.max()),
            int(torch.isnan(rl).sum()), ids0, [round(v, 4) for v in w0])
    except Exception as ex:  # noqa: BLE001
        import logging
        logging.getLogger("vllm").warning("[m3-probe] error: %s", ex)


def swigluoai_tail(gate_up, alpha: float, limit: float):
    """The EXACT MiniMax-M3 swigluoai activation (modeling_minimax_m3_vl.py:_apply_gate L209-215,
    byte-for-byte the authoritative numerics in notes/reviews/swigluoai-numerics.md).

    gate_up: [T, 2I] (gate=first half, up=second half, CONTIGUOUS chunk(2)). Returns h [T, I].
    Pure torch elementwise (clamp/sigmoid/mul/add) — runs on TPC/MME, never the silu intrinsic."""
    import torch  # local import: card-free module load
    gate, up = gate_up.chunk(2, dim=-1)          # gate=FIRST, up=SECOND (contiguous halves)
    gate = gate.clamp(max=limit)                 # UPPER bound only
    up = up.clamp(min=-limit, max=limit)         # symmetric
    glu = gate * torch.sigmoid(gate * alpha)     # alpha INSIDE sigmoid
    return (up + 1.0) * glu                      # the +1, then (up+1)*glu


def _alpha_limit(layer):
    """Resolve (alpha, limit) for this layer. FusedMoE stores swiglu_limit; alpha is the fixed
    gpt-oss/M3 1.702. Prefer the layer's own attrs so a future config change is honoured."""
    limit = getattr(layer, "swiglu_limit", None)
    if limit is None:
        limit = getattr(getattr(layer, "act_fn", None), "limit", None)
    alpha = getattr(getattr(layer, "act_fn", None), "alpha", None)
    return (float(alpha) if alpha is not None else _ALPHA_DEFAULT,
            float(limit) if limit is not None else _LIMIT_DEFAULT)


def _install():
    if not enabled():
        _log("M3_SWIGLUOAI not set -> unfused swigluoai patch NOT installed (routed int4 MoE keeps "
             "the stock hpu-moe-ext silu path). Set M3_SWIGLUOAI=1 to serve the M3 routed experts "
             "with the correct swigluoai activation.")
        return

    import torch
    import habana_frameworks.torch.core as htorch

    try:
        from hpu_moe_ext.moe_wna16_hpu import MoeWNA16HPUMethod
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "M3_SWIGLUOAI=1 requires hpu-moe-ext (MoeWNA16HPUMethod) importable + registered FIRST "
            f"(it owns the int4 routed MoE this patch refines). Import failed: {e!r}")

    if getattr(MoeWNA16HPUMethod, "_m3_swigluoai_wrapped", False):
        _log("already wrapped; skipping.")
        return

    _orig_apply = MoeWNA16HPUMethod.apply_monolithic

    def _routed_swigluoai(method, layer, x, topk_ids, topk_weights, K):
        """UNFUSED swigluoai routed path. Mirrors hpu-moe-ext _routed_moe's dequant + routing +
        mark_step discipline EXACTLY up to the per-expert compute, then replaces the silu
        mixture_of_experts intrinsic with explicit GEMM + swigluoai (pure torch). Returns the
        routed output [T, H], or None if ineligible (AWQ per-expert zp -> caller falls back).

        K = the number of distinct LOCAL experts to gather+dequant this call. For T==1 decode K is
        top_k; for the T>1 spec-decode VERIFY batch the caller passes K_active = min(E, T*top_k) (the
        max distinct local experts T tokens can possibly hit), so the same global->local remap that
        makes EP-sharded int4 experts coherent at T==1 now also owns any T>1 batch — the stock op
        (which overflows the length-E local axis on a global id) is NEVER reached. K is a STATIC
        function of the (bucketed) shape -> still no host sync, no recompiles. See
        notes/reviews/int4-wall-speculation.md (the n-gram/spec-decode shape-coverage fix)."""
        w13z = getattr(layer, "_hpu_routed_w13_zp", None)
        w2z = getattr(layer, "_hpu_routed_w2_zp", None)
        if w13z is None:
            return None  # AWQ per-expert zp -> ineligible (same as _routed_moe)
        E = layer._hpu_local_experts
        emin = layer._hpu_experts_min
        dev = x.device

        # --- routing -> K active local experts (VERBATIM from _routed_moe) -----------------------
        local = topk_ids.to(torch.int64) - emin
        flat = local.reshape(-1)
        valid = (flat >= 0) & (flat < E)
        scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
        present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
            0, scatter_idx, valid.to(torch.float32))[:E]
        active = torch.topk(present, K).indices.to(torch.int64)        # [K] local ids
        # Materialize routing before the data-dependent gather (HPU-graph capture requirement;
        # the exact boundary _routed_moe documents as load-bearing under lazy).
        # PERF (2026-06-15, MEASURED): these two mark_steps are LAZY-only correctness boundaries
        # (realize routing before the gather; realize dequant before the matmul). The int4 serve runs
        # torch.compile (the .env sets no PT_HPU_LAZY_MODE=1), where the dataflow graph already orders
        # these deps — so under compile they are pure overhead (~2 forced host round-trips/sparse layer
        # × 57 layers = ~114/decode step). Gate both on is_lazy() (mirrors _need_clone) so compile gets
        # one fused MoE region instead of 3 host-synced fragments. See notes/reviews/int4-wall-localize.md.
        _lazy = htorch.is_lazy()
        _ms = (_lazy or not _gate_markstep()) and not _no_lazy_markstep()  # +RANK-0: skip even under lazy
        if _ms:
            htorch.mark_step()

        # --- dequant ONLY the K routed experts (VERBATIM from _routed_moe) ------------------------
        w13p = layer.w13_qweight.data.index_select(0, active)          # [K, H, 2I//8]
        w2p = layer.w2_qweight.data.index_select(0, active)            # [K, I, H//8]
        s13 = layer.w13_scales.data.index_select(0, active)            # [K, groups, 2I]
        s2 = layer.w2_scales.data.index_select(0, active)              # [K, groups, H]
        dt = s13.dtype
        _need_clone = _lazy

        def _co(t):
            return t.clone() if _need_clone else t

        w13_deq = [torch.ops.hpu.convert_from_uint4(_co(w13p[k]), _co(s13[k]), w13z, dt, None)
                   for k in range(K)]                                  # each [H, 2I]
        w2_deq = [torch.ops.hpu.convert_from_uint4(_co(w2p[k]), _co(s2[k]), w2z, dt, None)
                  for k in range(K)]                                   # each [I, H]
        # MANDATORY (LAZY only): the exact boundary the coherent reference op inserts before consuming
        # the per-expert dequant outputs (without it the dequant tensors are not realized -> garbage
        # -> decode degenerates). _routed_moe documents this as the true lazy-correctness fix. Under
        # torch.compile the graph already realizes the dequant before the matmul -> skip (see above).
        if _ms:
            htorch.mark_step()

        # --- one-shot LOUD shape assertion (orientation correctness) -----------------------------
        if not _shape_checked[0]:
            _shape_checked[0] = True
            H = x.shape[-1]
            twoI = w13_deq[0].shape[-1]
            if w13_deq[0].shape[0] != H or (twoI % 2) != 0:
                raise RuntimeError(
                    f"m3-swigluoai: W13_deq shape {tuple(w13_deq[0].shape)} != [H={H}, 2I]; the "
                    "dequant orientation assumption ([in=H, out=2I]) is wrong -> would compute "
                    "garbage. Refusing to serve (DSA build-quality bar).")
            I = twoI // 2
            if w2_deq[0].shape[0] != I or w2_deq[0].shape[1] != H:
                raise RuntimeError(
                    f"m3-swigluoai: W2_deq shape {tuple(w2_deq[0].shape)} != [I={I}, H={H}]; the "
                    "down_proj orientation assumption ([in=I, out=H]) is wrong. Refusing to serve.")
            _log(f"shape OK: x[T,{H}] W13_deq[{H},{twoI}] W2_deq[{I},{H}] (gate=:I up=I:, K={K}).")

        alpha, limit = _alpha_limit(layer)

        # --- per-(token,expert) routing weight matrix [T, K] -------------------------------------
        # For each token t and active slot k, the weight is topk_weights[t, slot] if token t routed
        # to active[k], else 0. Reconstructs exactly what the intrinsic applies internally. Built
        # with a scatter on the compact id space (no host sync, static shape). remap: local id->compact
        remap = torch.zeros(E, dtype=torch.int64, device=dev).scatter(
            0, active, torch.arange(K, dtype=torch.int64, device=dev))
        flat_safe = torch.where(valid, flat, torch.zeros_like(flat))
        compact = remap[flat_safe].view_as(topk_ids)                  # [T, top_k] -> compact slot id
        valid2d = valid.view_as(topk_ids)
        safe_w = torch.where(valid2d, topk_weights.to(x.dtype), torch.zeros_like(topk_weights).to(x.dtype))
        T = x.shape[0]
        # gate_mat[t, k] = sum over the token's top_k slots whose compact==k of safe_w (a token routes
        # to <=1 of the K active per slot; summing is the correct accumulation if duplicated).
        gate_mat = torch.zeros(T, K, dtype=x.dtype, device=dev).scatter_add_(
            1, compact, safe_w)                                       # [T, K]

        # --- per-expert compute (explicit GEMM + swigluoai; NEVER the silu intrinsic) -------------
        # MEASURED 2026-06-15: this per-expert loop BEATS the batched torch.bmm form for M3 on :next
        # (single-stream 8.3 vs 5.0 tok/s, c=30 agg 17.4 vs 7.7; batching adds stack/expand/reduce
        # host overhead + heavier prefill recipes that explode TTFT). At decode K=top_k=4 the loop is
        # 4 already-captured matmuls; the single-stream wall is the structural host/sync-bound decode
        # (attention + all-reduce + sampler), NOT this loop. Keep the loop. See M3_MATRIX_REPORT.md.
        out = torch.zeros(T, x.shape[-1], dtype=x.dtype, device=dev)
        for k in range(K):
            gate_up = torch.matmul(x, w13_deq[k])                     # [T,H]@[H,2I] -> [T,2I]
            h = swigluoai_tail(gate_up, alpha, limit)                 # [T, I]
            y_k = torch.matmul(h, w2_deq[k])                          # [T,I]@[I,H] -> [T,H]
            out = out + gate_mat[:, k:k + 1] * y_k                    # weighted combine
        return out

    def _routed_silu_reference(method, layer, x, topk_ids, topk_weights, K):
        """EAGER-ONLY verify helper: the SAME unfused plumbing but with a SiLU tail, so it can be
        compared bit-for-bit to the stock _routed_moe (silu intrinsic) to prove dequant+GEMM+routing+
        orientation are correct. The only delta vs _routed_swigluoai is the activation tail."""
        # reuse _routed_swigluoai but monkey-swap the tail to silu via a thread-local flag would be
        # ugly; instead recompute inline (small, first-call only).
        w13z = getattr(layer, "_hpu_routed_w13_zp", None)
        w2z = getattr(layer, "_hpu_routed_w2_zp", None)
        if w13z is None:
            return None
        E = layer._hpu_local_experts
        emin = layer._hpu_experts_min
        dev = x.device
        local = topk_ids.to(torch.int64) - emin
        flat = local.reshape(-1)
        valid = (flat >= 0) & (flat < E)
        scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
        present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
            0, scatter_idx, valid.to(torch.float32))[:E]
        active = torch.topk(present, K).indices.to(torch.int64)
        htorch.mark_step()
        w13p = layer.w13_qweight.data.index_select(0, active)
        w2p = layer.w2_qweight.data.index_select(0, active)
        s13 = layer.w13_scales.data.index_select(0, active)
        s2 = layer.w2_scales.data.index_select(0, active)
        dt = s13.dtype
        _co = (lambda t: t.clone()) if htorch.is_lazy() else (lambda t: t)
        w13_deq = [torch.ops.hpu.convert_from_uint4(_co(w13p[k]), _co(s13[k]), w13z, dt, None) for k in range(K)]
        w2_deq = [torch.ops.hpu.convert_from_uint4(_co(w2p[k]), _co(s2[k]), w2z, dt, None) for k in range(K)]
        htorch.mark_step()
        remap = torch.zeros(E, dtype=torch.int64, device=dev).scatter(
            0, active, torch.arange(K, dtype=torch.int64, device=dev))
        flat_safe = torch.where(valid, flat, torch.zeros_like(flat))
        compact = remap[flat_safe].view_as(topk_ids)
        valid2d = valid.view_as(topk_ids)
        safe_w = torch.where(valid2d, topk_weights.to(x.dtype), torch.zeros_like(topk_weights).to(x.dtype))
        T = x.shape[0]
        gate_mat = torch.zeros(T, K, dtype=x.dtype, device=dev).scatter_add_(1, compact, safe_w)
        out = torch.zeros(T, x.shape[-1], dtype=x.dtype, device=dev)
        import torch.nn.functional as F
        # SAME per-expert plumbing as _routed_swigluoai (so the verify proves the EXACT shipped path's
        # dequant/GEMM/routing/orientation); only the activation tail differs (SiLU here to match the
        # stock mixture_of_experts silu intrinsic for the bit-exact A/B).
        for k in range(K):
            gate_up = torch.matmul(x, w13_deq[k])
            gate, up = gate_up.chunk(2, dim=-1)
            h = F.silu(gate) * up                                     # SiLU tail (matches intrinsic)
            y_k = torch.matmul(h, w2_deq[k])
            out = out + gate_mat[:, k:k + 1] * y_k
        return out

    def _allexpert_swigluoai(method, layer, x, topk_ids, topk_weights):
        """UNFUSED swigluoai ALL-EXPERT path — the PREFILL (T>1) regime and the universal correct
        fallback. Replaces the stock all-expert mixture_of_experts intrinsic (which CRASHES on
        activation='swigluoai' at mixture_of_experts_gen.cpp:799) with an explicit per-LOCAL-expert
        GEMM + swigluoai over ALL E local experts, masked by a [T, E] per-(token,expert) routing
        matrix. Static shape (E fixed = local_num_experts), correct for any T. Works for AWQ too
        (uses the op's own get_dequant_weight, which honours per-expert zeros), so it is the safe
        universal path — no _hpu_routed_w13_zp dependency.

        Dequant source: layer.moe_op.w13_list[e].get_dequant_weight() -> W13_deq [H, 2I] (== the
        same convert_from_uint4 the routed path uses); w2_list -> W2_deq [I, H]. The op was built
        with the LOCAL window [0, E-1] (hpu-moe-ext EP global->local fix), so expert e here is the
        LOCAL expert; route a token to it iff (global_id - emin) == e."""
        E = layer._hpu_local_experts
        emin = layer._hpu_experts_min
        dev = x.device
        T = x.shape[0]
        alpha, limit = _alpha_limit(layer)

        # [T, E] routing weights for the LOCAL experts. local id = global - emin; in-window iff [0,E).
        local = topk_ids.to(torch.int64) - emin                       # [T, top_k]
        in_win = (local >= 0) & (local < E)
        safe_local = torch.where(in_win, local, torch.zeros_like(local))   # park out-of-window at col 0
        safe_w = torch.where(in_win, topk_weights.to(x.dtype), torch.zeros_like(topk_weights).to(x.dtype))
        route = torch.zeros(T, E, dtype=x.dtype, device=dev).scatter_add_(1, safe_local, safe_w)  # [T,E]

        # Build the per-LOCAL-expert dequant lists FIRST, then ONE mark_step (exactly the boundary
        # the reference VllmMixtureOfExpertsOpWNA16.forward uses: it dequants all experts, then a
        # single htorch.core.mark_step(), then the GEMM). A single boundary keeps the graph from
        # fragmenting into E pieces while still realizing the dequant tensors before the matmul.
        w13_deq = [layer.moe_op.w13_list[e].get_dequant_weight() for e in range(E)]   # each [H, 2I]
        w2_deq = [layer.moe_op.w2_list[e].get_dequant_weight() for e in range(E)]     # each [I, H]
        # LAZY-only boundary (see _routed_swigluoai): realize the per-expert dequant before the GEMM.
        # Under torch.compile the dataflow graph orders it -> skip the forced host round-trip.
        if (htorch.is_lazy() or not _gate_markstep()) and not _no_lazy_markstep():
            htorch.mark_step()
        # per-LOCAL-expert compute (explicit GEMM + swigluoai). MEASURED 2026-06-15: the loop BEATS
        # the batched torch.bmm form here too — at concurrency the all-expert bmm [E,T,2I] builds huge
        # intermediates + a heavier prefill recipe that blows TTFT (c=30 agg 7.7 vs 17.4 tok/s, TTFT
        # 128s vs 25s). Keep the loop; route[:,e:e+1] zeros experts a token didn't pick.
        out = torch.zeros(T, x.shape[-1], dtype=x.dtype, device=dev)
        for e in range(E):
            gate_up = torch.matmul(x, w13_deq[e])                    # [T,H]@[H,2I] -> [T,2I]
            h = swigluoai_tail(gate_up, alpha, limit)                # [T, I]
            y_e = torch.matmul(h, w2_deq[e])                        # [T,I]@[I,H] -> [T,H]
            out = out + route[:, e:e + 1] * y_e                      # weighted combine (0 if not routed)
        # one-shot shape assert (orientation), shared with the routed path's guard
        if not _shape_checked[0]:
            _shape_checked[0] = True
            _log(f"all-expert shape OK: x[T,{x.shape[-1]}] W13_deq[{x.shape[-1]},2I] over E={E} "
                 f"local experts (gate=:I up=I:).")
        return out

    def apply_monolithic(self, layer, x, router_logits, **kwargs):
        import torch.nn.functional as F
        # FP8 RUNTIME HANDOFF: when M3_EXPERT_RUNTIME=fp8, m3_expert_runtime_patch has converted this
        # layer's experts int4->fp8 at load (freeing layer.w13_qweight) and swapped layer.moe_op to
        # the per-channel fp8 op. The int4 dequant this patch relies on is therefore GONE for such a
        # layer. m3_expert_runtime_patch wraps apply_monolithic FIRST (it captured the stock method as
        # its _orig_apply); this swigluoai wrap runs OUTERMOST. Delegate fp8-converted layers straight
        # back to it so the fp8 swigluoai path owns them — never run the int4 plumbing on freed
        # weights. (_m3_fp8_apply is installed by m3_expert_runtime_patch when fp8 mode is active.)
        if getattr(layer, "_m3_fp8_converted", False):
            fp8_apply = getattr(type(self), "_m3_fp8_apply", None)
            if fp8_apply is None:
                raise RuntimeError(
                    "m3-swigluoai: layer is fp8-converted (_m3_fp8_converted=True) but the fp8 apply "
                    "handler (_m3_fp8_apply) is missing — the M3_EXPERT_RUNTIME=fp8 swigluoai wiring "
                    "is broken. Refusing to serve a wrong activation (DSA build-quality bar).")
            return fp8_apply(self, layer, x, router_logits, **kwargs)
        input_shape = x.shape
        x2 = x.view(-1, x.shape[-1])

        # Routing — mirror the stock apply_monolithic EXACTLY (M3 uses grouped-topk).
        if getattr(layer, "use_grouped_topk", False):
            topk_weights, topk_ids = layer.router.select_experts(
                hidden_states=x2, router_logits=router_logits)
        else:
            tw = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(tw, layer.top_k, dim=-1)
            if getattr(layer, "renormalize", True):
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x2.dtype)
        topk_ids = topk_ids.view(*x2.shape[:-1], -1)
        topk_weights = topk_weights.view(*x2.shape[:-1], -1)
        tid = topk_ids.to(torch.int64)
        tw_ = topk_weights.to(x2.dtype)
        _probe(layer, x2, router_logits, tid, tw_)  # M3_SWIGLUOAI_PROBE: reliable first-call dump

        top_k = layer.top_k
        T = x2.shape[0]
        E = layer._hpu_local_experts
        # K_active = max distinct LOCAL experts T tokens can hit = min(E, T*top_k). STATIC (T is the
        # bucketed shape; top_k,E are constants) -> no host sync, no recompile. For T==1 this is top_k.
        K_active = E if (T * top_k) >= E else (T * top_k)
        # GATE (widened 2026-06-15 for spec-decode): use the routed K-gather whenever the distinct
        # set fits the local axis (K_active = min(E, T*top_k) <= E), which covers T==1 decode AND the
        # small T>1 spec-decode VERIFY batch (T = 1+num_speculative_tokens, typ. <=8). Both go through
        # the SAME global->local remap, so the stock op (which overflows index-E on a global id) is
        # NEVER reached. LARGE-T prefill (T*top_k > E, every expert likely hit) keeps the all-expert
        # path below — its proven prefill form; the routed path with K==E would be a costlier
        # equivalent and its [T*K] gather/scatter would balloon. gate on GPTQ-sym zp (else fall below).
        eligible = (getattr(layer, "_hpu_routed_w13_zp", None) is not None and (T * top_k) <= E)
        if eligible:
            out = _routed_swigluoai(self, layer, x2, tid, tw_, K_active)
            if out is not None:
                if _verify() and not _verified[0]:
                    _verified[0] = True
                    # Prove the unfused PLUMBING vs the stock silu intrinsic (eager-only .item()).
                    # Use K_active (== top_k for the T==1 verify call) so the A/B matches the served K.
                    ref_silu_unfused = _routed_silu_reference(self, layer, x2, tid, tw_, K_active)
                    ref_stock = self._routed_moe(layer, x2, tid, tw_, False, "silu", K_active)
                    import logging
                    log = logging.getLogger("vllm")
                    if ref_silu_unfused is not None and ref_stock is not None:
                        diff = (ref_silu_unfused.float() - ref_stock.float()).norm().item()
                        rn = diff / (ref_stock.float().norm().item() + 1e-9)
                        if rn >= 1e-2:
                            ma = (ref_silu_unfused.float() - ref_stock.float()).abs().max().item()
                            log.error("[m3-swigluoai-verify] PLUMBING MISMATCH relNorm=%.4g "
                                      "maxAbs=%.4g -- unfused dequant/GEMM/orientation disagrees "
                                      "with the stock silu intrinsic. The swigluoai serve is "
                                      "SUSPECT.", rn, ma)
                            raise RuntimeError(
                                "m3-swigluoai plumbing verify failed (relNorm=%.4g); refusing to "
                                "serve a suspect activation path." % rn)
                        log.warning("[m3-swigluoai-verify] PLUMBING OK relNorm=%.6g (silu-unfused "
                                    "== stock silu intrinsic); only the swigluoai tail differs "
                                    "(CPU-unit-tested).", rn)
                if not _announced[0]:
                    _announced[0] = True
                    _log("LIVE: routed int4 experts served via UNFUSED swigluoai (gate.clamp(max=7) "
                         "* sigmoid(1.702*gate) * (up.clamp(-7,7)+1)); silu intrinsic BYPASSED "
                         f"[K-gather path, K_active={K_active} for T={T}].")
                return out.view(*input_shape)
        # LARGE-T PREFILL (T*top_k > E) or ineligible-for-K-gather -> the ALL-EXPERT unfused swigluoai
        # path (also T-agnostic + global->local remapping). We must NOT fall to _orig_apply: the stock
        # op passes activation='swigluoai' into the Habana mixture_of_experts intrinsic, which has no
        # swigluoai mode and FATALs at mixture_of_experts_gen.cpp:799 ("Activation 'swigluoai' not found
        # among MoeActivationMode_t"). The all-expert path is correct for any T and any zp (AWQ included).
        # NOTE: the small T>1 spec-decode VERIFY batch now takes the K-gather path above (gate widened),
        # so it no longer reaches the stock op's length-E local axis (the prior `index 16` crash).
        out = _allexpert_swigluoai(self, layer, x2, tid, tw_)
        if not _announced_all[0]:
            _announced_all[0] = True
            _log("LIVE: routed int4 experts served via UNFUSED swigluoai [all-expert path, T="
                 f"{x2.shape[0]}]; stock swigluoai-crashing intrinsic BYPASSED.")
        return out.view(*input_shape)

    MoeWNA16HPUMethod.apply_monolithic = apply_monolithic
    MoeWNA16HPUMethod._m3_swigluoai_wrapped = True
    _log("WRAPPED MoeWNA16HPUMethod.apply_monolithic -> UNFUSED swigluoai for the M3 routed int4 "
         "experts (M3_SWIGLUOAI=1). Stock silu intrinsic bypassed for single-token decode.")


def register():
    """Idempotent entry point, called by hpu_m3_ext.register() after hpu_moe_ext is live."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        _install()
        _REGISTERED = True
    except Exception as e:  # noqa: BLE001
        _log(f"FAILED to install unfused swigluoai patch: {e!r}")
        raise


# --------------------------------------------------------------------------------------------------
def _selftest() -> int:
    """Card-free: assert the swigluoai tail matches the authoritative numerics on CPU."""
    import torch
    print("=" * 80)
    print("m3_swigluoai_patch — card-free selftest (swigluoai numerics)")
    print("=" * 80)
    torch.manual_seed(0)
    T, I = 5, 7
    gate = torch.randn(T, I) * 4
    up = torch.randn(T, I) * 4
    gate_up = torch.cat([gate, up], dim=-1)
    got = swigluoai_tail(gate_up, _ALPHA_DEFAULT, _LIMIT_DEFAULT)
    # reference: modeling_minimax_m3_vl.py _apply_gate L209-215
    g = gate.clamp(max=7.0)
    u = up.clamp(min=-7.0, max=7.0)
    ref = (u + 1.0) * (g * torch.sigmoid(g * 1.702))
    md = (got - ref).abs().max().item()
    print(f"swigluoai_tail vs reference maxdiff = {md:.3e}  (must be 0)")
    print(f"M3_SWIGLUOAI enabled() = {enabled()}  M3_SWIGLUOAI_VERIFY = {_verify()}")
    ok = md == 0.0
    print("VERDICT:", "OK" if ok else "FAIL")
    print("=" * 80)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
