# swigluoai-op-capability — can the Gaudi MoE op do swigluoai directly / skip activation?

Angle: op SURFACE. Can a Gaudi MoE op do swigluoai natively, or be made to skip the
activation so we apply it outside? Fastest correct path. (card-free static read)

## TL;DR
- **YES — a native swigluoai path already exists in the bridge:**
  `torch.ops.hpu.mixture_of_experts.bias_fused_weights(... float alpha=1.702, float
  limit=7.0)`. It is the EXACT gpt-oss/SwigluOAIAndMul formula (interleaved gate/up,
  clamp(up,±limit), `(up+1)*gate*sigmoid(alpha*gate)`). M3's `SwigluOAIAndMul` is the
  same formula (model file says so: "CustomOp expects interleaved gate/up … same as
  gpt_oss"). Device flag `MOE_FLAGS_GPT_SWIGLU` + alpha/limit.
- **The stock int4 op never calls it.** `VllmMixtureOfExpertsOpWNA16.forward`
  (vllm_gaudi/extension/ops.py:1456) only calls plain `mixture_of_experts(activation=
  activation)`; activationModeMap = {gelu,relu,silu} only → `"swigluoai"` fails the
  `find()` at mixture_of_experts_gen.cpp:798-799. That IS the reported crash.
- **The custom int4 FUSED kernels** (fused_moe_monolithic / batched / routed) do NOT use
  the stock activation param at all — they emit SiLU explicitly as graph nodes
  (sigmoid_fwd+mult_fwd) in our own emit headers. So a "skip-activation / swap-to-
  swigluoai" mode there is a *local edit we own*, not an op-API request.

## (1) Does any Gaudi MoE op accept swigluoai / a skip mode?
- **swigluoai: yes**, via `.bias_fused_weights` (and `.measurement_bias_fused_weights`,
  `.bias_fp8_fused_weights[_scalars]`). All route to `FillMixtureOfExpertsBiasParams`
  → `gpt_swiglu=true, alpha, limit` (mixture_of_experts_gen.cpp:842-862). gelu/relu/silu
  remain the only *string*-selectable activations; swigluoai is reached by the bias
  overload, NOT the `activation=` string.
- **skip/identity activation: NO** on the stock op (no "none" enum; activationModeMap has
  no identity, and there's no "return pre-activation gate&up" output mode — the op always
  fuses act+down internally and returns only the final [T,H]).
- **Two caveats on `.bias_fused_weights`:**
  1. **Requires non-empty `w12_bias`/`w3_bias`** (check_inputs, mixture_of_experts.cpp:
     550-565). M3 experts are `bias=False` → must pass **zero-bias** tensor lists. The
     `+1` on up is hardcoded in the formula (not supplied via bias), so zero bias is
     numerically exact.
  2. **`frontend_blocklist: [eager]`** (hpu_op.yaml:8939). The vllm-gaudi:next image is
     the eager/upstream build → the *device* `moe`-guid swigluoai kernel is blocked there;
     it falls to the **CPU/composite** `mixture_of_experts_common_with_bias` (torch.bmm +
     explicit clamp/sigmoid). Correct, but that's the un-fused dequant→bmm reference, not
     the fused device kernel. The *lazy* container is NOT eager-blocklisted → device
     swigluoai kernel available there.
- Per-call activation selection: only gelu/relu/silu via the string; swigluoai is a
  separate overload (compile-time-distinct schema), not a runtime knob on the silu op.

## (2) Can we get un-activated gate&up out to apply swigluoai externally?
- **Stock op: no** clean pre-activation output. But the routed reference path ALREADY
  does this de-facto: dequant int4 → bf16 weight list → explicit `gate_up = x@W13`, split,
  apply activation in PYTHON, `@W2` (fused_routed_moe.py). Swapping `F.silu(gate)*up` →
  swigluoai there is trivial and is the simplest correctness fix (but it's the slow
  unfused leg).
- **Custom fused kernels: YES, cleanly** — they build the activation as explicit nodes
  (slice gate=gu[:,:,:I], up=gu[:,:,I:]; sigmoid_fwd; mult_fwd ×2). To do swigluoai
  in-kernel: replace those ~3 nodes with clamp(up,±7) + (gate*sigmoid(1.702*gate)) +
  (up+1)*glu. NOTE the split: our kernels use **contiguous** [:I]/[I:]; gpt-oss
  swigluoai is **interleaved** [0::2]/[1::2]. Since we own the emitter we can either
  emit interleaved slices or (cheaper) require the weight-loader to de-interleave w13 so
  contiguous-split == correct (the loader already interleaves for the CustomOp; we'd
  invert it). Either is a self-contained kernel edit.

## (3) FASTEST correct path (given the op surface)
1. **Correctness-first (hours), unfused:** in the int4 apply path, when activation is
   swigluoai, run the dequant→python-GEMM leg (fused_routed_moe / the WNA16 dequant list)
   but apply `SwigluOAIAndMul(1.702,7.0)` instead of `F.silu(gate)*up`. No op/kernel
   rebuild. Unblocks M3 int4 serving immediately; slow (the bf16→HBM leg) but correct.
2. **Native op (small), lazy container:** call `mixture_of_experts.bias_fused_weights`
   with the dequanted bf16 expert weight lists + **zero-bias** lists + alpha/limit. On
   the lazy build this hits the device swigluoai kernel; on eager it hits the bmm
   composite. Drop-in for the stock WNA16 op — just a new branch in
   `VllmMixtureOfExpertsOpWNA16.forward` / the moe_wna16_hpu dispatch. Still dequants
   int4→bf16 first (bias overload takes dense weights, no scales arg).
3. **Fused-kernel swigluoai (the perf path):** add a swigluoai variant to the fused
   emit headers (replace the 3 silu nodes; handle the interleave). Reuses all the
   int4-dequant/SRAM-resident fusion already proven; this is where the speed lives.
   Bigger change (kernel build + the gate/up interleave decision + revalidation).

**Recommendation:** ship #1 to unblock today; do #3 for production speed. #2 is a nice
middle rung only if running the lazy container (device kernel) — on eager it's no faster
than #1.

## Confidence
HIGH that `.bias_fused_weights` is a real, exact-formula native swigluoai path and that
the stock int4 op's silu-only string is the crash. HIGH that the fused kernels can take a
swigluoai mode (we own them). MEDIUM on the eager-blocklist meaning the device swigluoai
kernel is unavailable on :next (schema says so; not card-verified). The interleave-vs-
contiguous gate/up split is the one correctness gotcha to nail in any path.

## Key files
- vllm_gaudi/extension/ops.py:1456 (VllmMixtureOfExpertsOpWNA16 — silu-only int4 op),
  :619-748 (VllmMixtureOfExpertsOp — has the `.bias_fused_weights` call), :45 _as_activation_str
- bridge mixture_of_experts_gen.cpp:770-773 (activationModeMap={gelu,relu,silu}),
  :791-840 FillMixtureOfExpertsParams (gpt_swiglu/alpha/limit), :842-863 BiasParams
- bridge habana_eager/ops/mixture_of_experts.cpp:503-617 (swigluoai CPU formula +
  check_inputs bias requirement)
- bridge scripts/hpu_op.yaml:8939-8943 (bias_fused_weights schema; frontend_blocklist:[eager])
- hpu_moe_ext/fused_monolithic_moe.py / fused_routed_moe.py (silu hard-code, python leg)
- tpc_kernel/opbackend/moe_fused_*_emit.h (explicit silu nodes — the skip/swap surface)
- minimax-m3/vllm/minimax_m3.py:324-333 (M3 SwigluOAIAndMul, interleaved, alpha/limit), :246 bias=False
