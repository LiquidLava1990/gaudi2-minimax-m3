# M3 int4 MoE — UNFUSED swigluoai path (fastest route to a WORKING M3 tonight)

Angle: the FASTEST path to a coherent (even if slow) M3 tonight — an unfused int4 MoE that
applies `swigluoai` correctly, bypassing every silu-hardcoded fused op. Card-free design.

## The blocker, confirmed at THREE levels
`silu` is hard-coded everywhere the int4 MoE could run:
1. **OpBackend fused ops** (`moe_wna16_hpu.py` L995-996, L1062-1063): `if act != "silu": return None`
   (the GC-captured `fused_expert` / `fused_monolithic` graphs bake `silu(gate)*up`).
2. **C++ TPC kernel** (`dsa_e2e/.../fused_moe_full.cpp` L11-12): `h = silu(gate)*up`
   (`sigmoid_fwd_bf16 + mult_fwd_bf16`).
3. **STOCK op too** — `VllmMixtureOfExpertsOpWNA16.forward` (ops.py L1475-1522) and the
   `_routed_moe` gather (L900-933) both pass `activation=<str>` straight into the
   `torch.ops.hpu.mixture_of_experts` **intrinsic**, which only understands `silu`/`gelu`. So
   **dequant-to-bf16-then-stock-op is ALSO blocked** — feeding `activation="swigluoai"` would
   either raise or silently apply the wrong gate. **There is no non-fused fallback in the plugin
   today** that applies an arbitrary activation; every path terminates in the same intrinsic.

Conclusion: a WORKING M3 tonight needs an activation computed in **plain torch/HPU ops**, not the
intrinsic — i.e. an unfused MoE.

## The exact numerics (from swigluoai-numerics.md, authoritative)
```python
gate = gate.clamp(max=7.0)            # upper only
up   = up.clamp(min=-7.0, max=7.0)    # symmetric
glu  = gate * torch.sigmoid(1.702 * gate)
h    = (up + 1.0) * glu               # NOTE the +1 and the (up+1)*glu order
```
`gate = first half, up = second half` of `gate_up` (concatenated, NOT interleaved). This matches
the dequant weight layout exactly: `convert_from_uint4(w13_packed, scale, zp, dtype, g_idx)` →
`W13_bf16 [2I, H]` with `gate = W13[0:I]`, `up = W13[I:2I]`; `W2_bf16 [H, I]` (cpp L27-35).

## Unfused path design (per-layer apply override, reuses ALL the int4 plumbing)
Mirror `m3_expert_runtime_patch.py` verbatim: an out-of-tree `vllm.general_plugins` entry point
that, after `hpu_moe_ext.register()`, **wraps `MoeWNA16HPUMethod.apply_monolithic`** for swigluoai
layers. `process_weights_after_loading` is left UNTOUCHED — the int4 weights, transposed scales,
shared symmetric zero (`layer._hpu_routed_w13_zp/_w2_zp`) and `layer.moe_op` are exactly the source
we need. Then in apply:

1. **Route**: reuse `layer.router.select_experts(...)` → `topk_ids, topk_weights` (grouped-topk +
   sigmoid scoring + correction bias all behave identically to the int4 serve). Global→local remap
   identical to the existing routed path.
2. **Dequant the K routed experts only** (NOT all 128): reuse the proven `_routed_moe` gather —
   `index_select(0, active)` the packed weights + scales, `convert_from_uint4` per expert, with the
   **mandatory `htorch.mark_step()`** boundaries already documented at L875/L917 (these are
   correctness-load-bearing under lazy). This gives bf16 `W13[k] [2I,H]`, `W2[k] [H,I]`.
3. **Per-expert unfused compute** (plain torch, runs on TPC/MME, never the silu intrinsic):
   ```
   gu   = x @ W13[k].T              # [T,2I]
   gate, up = gu[:, :I], gu[:, I:]
   h    = (up.clamp(-7,7) + 1.0) * (gate.clamp(max=7) * torch.sigmoid(1.702*gate))
   y_k  = (h @ W2[k].T)            # [T,H]
   out += topk_weight[k] * y_k     # weighted combine
   ```
   For K=4 routed experts this is 4 dequants + 8 small GEMMs + the elementwise swigluoai — a tiny,
   static-shape loop (tokens*top_k ≤ K guaranteed by the gather gate, so no host sync / no dynamic
   shape / no recompiles).
4. **Shared expert + dense-prefix MLPs** already use `SwigluOAIAndMul` in `minimax_m3.py`
   (HpuMiniMaxM3MLP, L330-333) — they are bf16 and ALREADY CORRECT; only the routed int4 experts
   need this override.

This is the minimal correct change: it swaps ONLY the activation tail (silu→swigluoai) and the
final GEMM; the dequant + routing + mark_step discipline is reused bit-for-bit from the validated
`_routed_moe`. LOUD-fail if `act` is unexpected (DSA build-quality bar); never silently silu.

## Speed (428B / ~26B active / top-4 +1 shared, TP=8, batch=1)
The unfused path **does not add dequant cost** — it dequants the same K experts the routed path
already does; it only replaces the fused `silu` tail with torch elementwise + an explicit bf16 GEMM.
At **batch=1 the box is host/sync-bound at ~29-34 tok/s regardless of the MoE kernel**
(int4-speed-ceiling.md §1) — the MoE compute is a tiny slice of the step. The routed-K gather is the
proven single-stream lever (the 3.84→21.7 tok/s, 5.6× arc on the Qwen-397B proxy). M3 is *smaller*
active (~26B vs Qwen's ~32B) and the unfused tail adds only a handful of extra elementwise/GEMM
kernels per routed expert.

**Estimate: ~10-20 tok/s single-stream** (a bit under the fused routed-K ~21 because the explicit
per-expert Python GEMM loop is less efficient than the single `mixture_of_experts` call, but the
same order of magnitude — it is NOT a 10× regression). **Comfortably usable for coherence
verification and as a working fallback** while the proper swigluoai-fused TPC kernel is built. At
concurrency it would trend worse than the fused op (the explicit loop doesn't batch experts as
tightly), so it is explicitly **correctness-first, throughput-later** — exactly the brief.

## Feasibility / does it yield a WORKING M3 TONIGHT?
**YES.** Everything required already exists and is validated:
- the exact numerics (swigluoai-numerics.md),
- the dequant primitive + K-gather + mark_step discipline (`_routed_moe`, proven coherent),
- the out-of-tree monkeypatch template (`m3_expert_runtime_patch.py`, card-free, installed-plugin-
  untouched),
- the bf16 weight layout matches `gate=first/up=second` by construction.
No new kernel, no `.so`, no card-side build. It is a ~1-file `apply_monolithic` override that
reuses the existing int4 dequant and substitutes the swigluoai math in torch. The only real risk is
the usual lazy-HPUGraph mark_step placement — and that is already solved and documented in
`_routed_moe`, which this path copies. This is the fastest route to a coherent M3 tonight.
