# swigluoai in the fused-MoE OpBackend kernel — design + verdict

**My angle (assigned):** extend OUR fused-expert OpBackend kernel (`moe_fused_expert_backend.cpp`)
to support a config-driven `silu|swigluoai` activation mode — the dynamic-kernel fix that also
future-proofs gpt-oss-style models. Card-free, code-only. Evidence is file:line into the on-disk
source the running image builds from.

---

## VERDICT: the kernel extension is BUILDABLE and the node-chain is trivial — but it is **NOT the right fix for M3 serving**, for two independently-fatal reasons. Recommend NO-GO as the M3 path; use the stock op's native gpt_swiglu flag instead.

> **Reason 1 — this kernel is NOT the serving path, and it LOSES to the path that is.** int4
> AutoRound MoE serves through the stock Habana `torch.ops.hpu.mixture_of_experts` intrinsic
> (`moe_wna16_hpu.py:1279` all-expert / `:930` routed-gather). The fused-expert OpBackend op is
> opt-in only (`HPU_MOE_FUSED_OP=1`, set in 3 GLM/Qwen envs, **never** M3) and the project already
> measured it **slower than stock** (~23 vs ~30 single-stream; ~68 vs ~126 @c16 —
> `int4-speed-ceiling.md:115`, "Do NOT enable HPU_MOE_FUSED_OP for speed"). Extending it only helps
> if we ALSO re-route M3 serving onto it — paying that perf regression on every token — AND relax the
> hard `if act != "silu": return None` bail at `moe_wna16_hpu.py:996,1063`.
>
> **Reason 2 — the stock op ALREADY does swigluoai natively. The kernel work is redundant.** The
> stock `mixture_of_experts` intrinsic has a built-in **gpt_swiglu** mode with the exact M3 numerics:
> `mixture_of_experts_gen.cpp:786-788` (`bool gpt_swiglu; float alpha=1.702; float limit=7.0`),
> wired via `MOE_FLAGS_GPT_SWIGLU` (`:832`) into `ns_MoeKernel::ParamsV5{alpha, limit}` (`:836-837`).
> Two registered torch schemas already carry it: `hpu::mixture_of_experts.bias_fused_weights(...,
> float alpha=1.702, float limit=7.0)` and `.bias_fp8_fused_weights(...)` (`hpu_op.yaml:8939, 9005`),
> both hard-setting `gpt_swiglu=true` (`mixture_of_experts_gen.cpp:858, 903`). **The TPC kernel
> already computes M3's activation on-device today** — no .cpp, no rebuild, no bring-up.

Confidence: **high** on both reasons (read verbatim from the on-disk bridge + plugin the image builds
from). Confidence **high** that the node-chain below is correct (numerics cross-checked against HF
source via the sibling `swigluoai-numerics.md`).

---

## The kernel design (what I was asked to design — it works, it's just unnecessary)

The activation is the ONLY thing M3 changes; the int4 dequant, both MMEs, the slice, and the router
mult are byte-identical to the proven bf16 op. Today the recipe emits SiLU as three nodes
(`moe_fused_expert_backend_shared.h:255-269`):

```
sgte = sigmoid_fwd(gate); silu = mult_fwd(gate, sgte); h = mult_fwd(silu, up)
```

**swigluoai replaces ONLY that block** (numerics from `swigluoai-numerics.md`, HF-confirmed:
`gate=chunk0, up=chunk1` CONTIGUOUS halves — NOT interleaved; `alpha=1.702, limit=7.0`):

```
# config-driven: if mode==SWIGLUOAI emit this in place of the silu block (im_shape=[M,I] throughout)
gate_c = clamp_pt_fwd(gate, max=limit)              # UPPER bound only (min=-inf)
up_c   = clamp_pt_fwd(up,  min=-limit, max=limit)   # BOTH bounds, symmetric +/-7.0
ax     = mult_fwd(gate_c, const(alpha))             # alpha INSIDE sigmoid
sg     = sigmoid_fwd(ax)
glu    = mult_fwd(gate_c, sg)                       # gate*sigmoid(alpha*gate)
up1    = add_fwd(up_c, const(1.0))                  # the +1 bias (gpt-oss form)
h      = mult_fwd(up1, glu)                         # (up+1)*glu   -> feeds W2 gemm UNCHANGED
```

All GUIDs are real in the version-exact bridge (`v1.24.0-1007`), confirmed present:
`clamp_pt_fwd` (`clamp_gen.cpp:204,221`), `sigmoid_fwd` (`sigmoid_gen.cpp`), `add_fwd`/`mult_fwd`
(`binary_gen.cpp`). The clamp scalar bound is a param on `clamp_pt_fwd`; the `alpha`/`1.0` constants
are `OpBackend::BuildConstantTensor(this, graph, val, ...)` (`op_backend.h:389`) or `ConstantHelper`
(`op_backend.h:287`), both used by sibling ops (`addmm_gen.cpp:149`). **No TPC ISA, no missing GUID.**

**Config-driven mode:** add an `int activation_mode` (0=silu, 1=swigluoai) + optional `float alpha,
limit` to the op schema (`hpu_moe_ext::fused_expert`), branch in the shared `AddNode` header. Mirrors
the existing `BUILD_FP8_NODES` toggle discipline. Future-proofs gpt-oss (same math; only difference is
gpt-oss INTERLEAVES gate/up `::2`/`1::2` — handle as a separate split mode if a gpt-oss checkpoint
ever lands here; M3 is contiguous).

**Build effort (IF pursued):** the node-chain is ~7 BuildNode calls, ~1 hour to write. But the
realistic cost is the on-card geometry/SIF bring-up loop the bf16 op needed (clamp/const broadcast
shapes, the [M,I] FCD discipline) + a relNorm gate vs a CPU swigluoai oracle + the lazy frontend .so +
re-routing the plugin off its silu bail + an A/B proving it's coherent: **~2-4 focused days, almost
all on-card (box owner).** And it ends at a path that is *slower than stock* by Reason 1.

---

## #1 RISK (if the kernel path is pursued anyway)

**You pay a measured perf regression to rebuild something the stock op already does.** Re-routing M3
to the fused op (required, since it's off the serving path) inherits the Finding-2 debt (fused op
~23 vs stock ~30 single-stream), AND single-stream decode is host/collective-bound so the kernel's
SRAM-residency win buys ~0 tok/s anyway — net a slower, riskier path to identical numerics. The
correct move is **stock `mixture_of_experts` gpt_swiglu**: route M3 through the
`bias_fused_weights`/`bias_fp8` schema (alpha=1.702, limit=7.0), or have the WNA16 plugin pass
`gpt_swiglu` instead of bailing. Secondary risk on the stock route: the WNA16 int4 wrapper
(`VllmMixtureOfExpertsOpWNA16`) currently only forwards an `activation` *string* into the non-bias
schema (no alpha/limit, no gpt_swiglu arg) — so the small real task is plumbing M3 to the
**bias-variant** schema (or adding a gpt_swiglu str), NOT writing a kernel.
