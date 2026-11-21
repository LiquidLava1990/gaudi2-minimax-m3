# W4A8-FUSED kernel review — int4-in-HBM, dequant→fp8 IN-RECIPE, native fp8 MME

**Question:** For giant models that ONLY fit in int4 (~500 GB int4 = a 1T model), can the fused-expert
OpBackend kernel keep int4 weights in HBM but dequant int4→**fp8** (not bf16) INSIDE the SRAM-resident
recipe → feed the native Gaudi-2 fp8 MME (2× bf16)? = int4 footprint + fp8 compute speed.

**Method:** CARD-FREE, code-only. Did NOT touch cards / vllm-active. Evidence is file:line into the
on-disk source the running image builds from. Read: `DSA_OPBACKEND_KERNEL_DESIGN.md`,
`PROJECT_FINDINGS.md` (Findings 2/3/5/6, §11a dequant trace, §13c/§13d precision),
`hpu-moe-ext/tpc_kernel/opbackend/moe_fused_expert_fp8_backend.cpp` + `FUSED_GENERALIZE_FP8_DESIGN.md`,
`minimax-m3/int4_to_fp8_load.py`.

---

## VERDICT: **PARTIAL — GO as a 1T-class MEMORY-FIT enabler, NO-GO as a speed win on its own.**

> The W4A8-fused path is **real, already precisely scaffolded** (`moe_fused_expert_fp8_backend.cpp`),
> and is the **correct tool for the specific regime the question names**: a model so big that fp8-in-HBM
> does NOT fit, so the only way to touch the fast fp8 MME at all is to keep int4 in HBM and unpack to fp8
> *per-forward* inside the recipe. In that regime its competitor is **not** "stock fp8 MoE" (that needs
> fp8 resident — doesn't fit) but **"int4→bf16 fused / stock int4 dequant"** — and against THAT baseline
> it plausibly wins (½ the GEMM-input bytes + 2× MME). **BUT** two hard project findings bound it:
> (Finding 2) the *fused-expert OpBackend op itself* loses to the stock `mixture_of_experts` intrinsic,
> so any fused variant inherits that debt; and (Finding 6 / §11a) at **batch=1** decode is
> host/collective-bound (~30 tok/s sync cap, MoE ≈ 0.16 ms of a ~127 ms step) so MME 2× buys ~0 single-stream.
> The win is **throughput@concurrency for a model that otherwise cannot run fast at all**, plus the memory fit.

Confidence: **high** that the scaffold == W4A8-fused and the node-chain is right (read verbatim);
**high** on the ~6.4e-2 quality floor (measured on-device, CPU-matched); **medium** on the speedup
*magnitude* vs int4→bf16 (mechanism is sound — ½ GEMM bytes + 2× MME — but unmeasured, and the
fused-op overhead from Finding 2 partly offsets it; needs an on-card A/B).

---

## (1) Can ONE SRAM-resident recipe do int4-unpack → fp8 → fp8-GEMM? **YES — the exact node-chain exists.**

Per GEMM, the dequant stays bf16 (the ONLY safe `convert_from_int4` out_dtype — see fn-vs-fnuz below),
then a self-consistent bf16→fp8 cast feeds `fp8_gemm`. From `moe_fused_expert_fp8_backend.cpp:97-116`
(W13 path; W2 identical):

```
x[M,H] bf16                                                      (input, stays int4-free)
1a  W13_bf16 = dequantize_4_bit_bf16(reint_u4(w13_packed), w13_scale, reint_u4(w13_zp)) → [H,2I] bf16
1b  s_w13    = reduce_max(|W13_bf16|, dim=0)/240                 → [1,2I] f32  PER-CHANNEL fwd scale
1c  W13_fp8  = convert_to_fp8(W13_bf16, 1/s_w13)                 → [H,2I] float8_e4m3fn   (NON-PERSISTENT)
Xa  s_x      = reduce_max(|x|, dim=-1)/240                       → [M,1]  f32  PER-TOKEN act scale
Xb  x_fp8    = convert_to_fp8(x, 1/s_x)                          → [M,H]  float8_e4m3fn
2   gu       = fp8_gemm(x_fp8, W13_fp8, scaleA=s_x, scaleB=s_w13, out=bf16)  → [M,2I] bf16  (was `gemm`)
3-4 gate/up slice → silu = gate·σ(gate) → h = silu·up           → [M,I]  bf16  (UNCHANGED)
5a-6 W2_bf16 = dequantize_4_bit_bf16(...) ; W2_fp8 = convert_to_fp8(W2_bf16,1/s_w2) ;
     h_fp8 = convert_to_fp8(h,1/s_h) ; y = fp8_gemm(h_fp8, W2_fp8, s_h, s_w2, out=bf16) → [M,H] bf16
7   out = y · router_weight                                     → [M,H]  bf16  (FINAL)
```

Bridge GUIDs (all real, version-exact `v1.24.0-1007`, reused verbatim from this tag's live fp8 linear):
`dequantize_4_bit` (complex-guid, bf16 out), `abs_fwd` → `reduce_max_multi_dim_fwd`
(`ns_Reduction::ParamsV2{reductionDimensionMask, keepDim}`), `convert_to_fp8`
(`ns_ConvertToFp8::ParamsV2{round_mode=CAST_ROUND_HALF_NE}`, `fp8_ops.cpp:153`), `fp8_gemm`
(`ns_Fp8Gemm::ParamsV2{transpose_a=false, transpose_b=false}`, `fp8_gemm_gen.cpp:206-223`). The
`W*_fp8`/`x_fp8`/`h_fp8` are graph-internal **non-persistent** tensors → the GC still bundles
dequant→cast→MME with SRAM residency (the proven MoE lever), and the fp8 weight is HALF the bf16 bytes,
so the residency win is at least as large. **No TPC ISA, no missing GUID, one captured recipe.** YES.

> Naming note: it is "int4-unpack→fp8" via an intermediate bf16, NOT a direct `convert_from_uint4→fp8`.
> Direct int4→fp8 is the **fn-vs-fnuz factor-of-2 trap** (`convert_from_int4` writes fnuz-interpreted
> bits, the next read reinterprets at 2× with nothing to cancel → the historical "fp8_norm ≈ 2.06×ref").
> The dequant MUST emit bf16; the ONLY fp8 cast is a self-consistent round-trip (`convert_to_fp8` packs
> with 1/s, `fp8_gemm` recovers with s; the 2× cancels). `FP8_MAX = 240` (fnuz), even though the
> cast/gemm DTYPE is `float8_e4m3fn` (`moe_fused_expert_fp8_backend.cpp:57-83`, 186-189).

## (2) Is this the existing scaffold, or new build? **It IS the scaffold — and it is the SAME thing as W4A8-fused.**

`moe_fused_expert_fp8_backend.cpp` (registers `torch.ops.hpu_moe_ext_fp8.fused_expert_fp8`, separate
library so the proven bf16 `.so` is byte-untouched) is **precisely** the W4A8-fused op the question
describes: keeps int4 in HBM, dequants int4→bf16→fp8 in-graph, runs the two expert GEMMs in fp8 on the
MME. The header calls it "the int4→fp8 **DOUBLE-quant**" — **that is W4A8-fused, not a different thing.**
"Double-quant" is just the honest accuracy framing of the same path (the weight passes int4 *then* fp8
grids); it is NOT a separate "two-stage extra quant." So the answer to the question's own sub-question
("is the fp8 scaffold the same as W4A8-fused or different?") is: **the same.**

**Status = SCAFFOLD, not a built `.so`.** It compiles and, with the default `BUILD_FP8_NODES=0`, falls
back to the proven bf16 recipe so the `.so` always loads and the schema/registration/harness validate
CPU-side (`FUSED_GENERALIZE_FP8_DESIGN.md:74-82`). The fp8 node bodies (the `make_fp8` amax/cast helper +
the `fp8_gemm` helper) are **written out in full** as the spec, gated behind `BUILD_FP8_NODES=1`. What is
NOT done: the same multi-iteration **on-device geometry/SIF bring-up** the bf16 op needed — the 8 new fp8
nodes' param structs + the in-graph amax/scale wiring (`BuildConstantTensor(240)`, reciprocal, mult) need
the box owner's validation loop (`moe_fused_expert_fp8_backend.cpp:138-148`). With `BUILD_FP8_NODES==1`
but the helpers stubbed it aborts LOUDLY at build (no silent wrong-scale path).

> **Distinct from `minimax-m3/int4_to_fp8_load.py` — the LOAD-TIME transform (the contrast the question
> draws, and it is exactly right).** `int4_to_fp8_load.py` does the int4→fp8 conversion ONCE at model
> load and **expands the weights to fp8 in HBM** (M3: disk stays int4 ~285 GiB, **HBM holds fp8 ~471
> GiB**, `int4_to_fp8_load.py:17-19`), then runs the *stock native* `VllmMixtureOfExpertsOpFP8PerChannel`
> — NO per-forward dequant. That is **strictly better when fp8 fits** (no per-token cast/amax overhead;
> the stock native fp8 op beats the fused op per Finding 2). But for a **1T model fp8-in-HBM does NOT
> fit** — that is the entire reason W4A8-fused (int4 resident, unpack-to-fp8 per-forward) exists. The two
> are complementary and the right one is dictated by whether fp8 fits HBM: **fp8 fits → load-time
> transform + stock op; fp8 does NOT fit → W4A8-fused.**

## (3) Speed vs int4→bf16 (the current path): **gets fp8-MME 2× + smaller GEMM input — but caveated 3 ways.**

The *mechanism* is favorable vs the int4→bf16 baseline (the relevant baseline for a 1T model, since stock
fp8 needs resident fp8 that doesn't fit):
- **2× MME:** Gaudi-2 MME is FLOAT-ONLY (bf16+fp8, no integer matmul); fp8 e4m3 runs ~2× bf16 (PF §13c:
  bf16 432 TF → fp8 865 TF). int4→bf16 feeds the bf16 MME; W4A8-fused feeds the fp8 MME.
- **½ the GEMM-input bytes** for the matmul operands (fp8 vs bf16), and the fp8 intermediates stay
  non-persistent/SRAM-resident, so HBM traffic on the GEMM inputs halves.
- **Same int4 HBM footprint** as int4→bf16 (weights resident as int4; the dequant is per-forward either way).

The three caveats that make this PARTIAL not GO-for-speed:
1. **Finding 2 (the central hard-won result):** the *fused-expert OpBackend op itself* is SLOWER than the
   stock `mixture_of_experts` intrinsic (~23 vs ~30 single-stream; ~68 vs ~126 throughput@c16) because
   the op's overhead (chunked sub-chains for SRAM-residency, OpBackend dispatch, a much larger graph)
   outweighs the bandwidth saving. W4A8-fused is a *variant of that same fused op* + **adds 8 nodes**
   (4× abs/reduce/cast amax subgraphs + the two `gemm`→`fp8_gemm` swaps) → it inherits the debt AND adds
   per-forward cast/amax work. The 2× MME must overcome that overhead before it nets positive.
2. **Finding 3 / §11a — batch=1 is host/collective-bound, MME 2× buys ~0 single-stream.** MoE compute is
   ~0.16 ms of a ~127 ms decode step (~90% collective+host); ~30 tok/s is the sync cap. Halving fp8 GEMM
   time cannot move a step dominated by all-reduce + host launch. So **W4A8-fused is a
   THROUGHPUT@concurrency lever, not a single-stream one** — same as documented for plain fp8-compute
   (PF Finding 6, §13d).
3. **The real comparison the model forces is "W4A8-fused vs nothing":** for a 1T int4-only model the
   alternative to W4A8-fused is int4→bf16 (the dequant-tax path that §11a measured as ~98% of decode in
   the all-experts regime). W4A8-fused at least puts the work on the 2× MME with half the GEMM bytes — so
   **vs the only feasible baseline it should help throughput; vs the unattainable "stock fp8 resident" it
   loses.** Net: a memory-fit enabler that recovers *some* of fp8's throughput edge for a model that
   could not host fp8 at all. **Magnitude unmeasured — needs an on-card A/B (W4A8-fused vs int4→bf16 at
   max-num-seqs 64/128).**

## (4) Quality — int4→fp8 double-quant floor: **~6.4e-2 relNorm. Coherent, but a HARD floor; acceptable for giant-model experts only with eyes open.**

- **Measured on-device 6.43e-2**, matched by faithful CPU emulation (6.2–6.6e-2 across seeds, GLM shape)
  — `moe_fused_expert_fp8_backend.cpp:34-55`, PF Finding 6 / §4a. The on-box gate uses a LOOSER tol of
  **8e-2** (floor + margin). int4 checkpoint + bf16→fp8 e4m3 (3 mantissa bits) on TWO weights + TWO
  activations across gate‖up→silu→down stacks the error.
- **Scale granularity is NOT the fix (investigated + measured).** The op already uses production
  PER-CHANNEL weight + PER-TOKEN activation scales (the `apply_fp8_linear_hpu` /
  `VllmMixtureOfExpertsOpFP8PerChannel` convention) — it improves the floor only ~2–3% (6.44→6.28e-2).
  Per-GROUP(128) does NOT break 5e-2 either: in the GPTQ int4 layout the weight's dynamic range lives
  along the contraction/group axis (per-output-column range ~1.2×), so per-output-channel scaling is
  largely orthogonal; the binding constraint is **fp8's 3 mantissa bits + the intermediate-h requant**,
  not the scale (`moe_fused_expert_fp8_backend.cpp:45-56`).
- **Acceptable for giant-model experts?** Conditionally yes: it is **coherent** (the floor is a few-%
  relNorm, not garbage), and a 1T model's redundancy tolerates per-expert noise better than a small
  model. But it is a real accuracy give-up — gate it on a **task-level eval** (lm-eval), not just relNorm,
  and only accept it where the model **cannot** be served any other way (fp8 doesn't fit). This is
  strictly the W4A8 double-quant floor; the **load-time int4→fp8 per-channel** path measures *far* lower
  (a few e-3 added over int4, `int4_to_fp8_load.py:309-392`) because the stock native op consumes the
  per-channel fp8 directly with no per-forward intermediate-h requant — another reason to prefer load-time
  whenever fp8 fits.

## (5) Build effort: **~3–6 focused days, ALL on-device (the CPU spec is done).**

The hard part is done in code: the node-chain, every GUID/param, the fn-vs-fnuz discipline, the
LOUD-failure guards, the separate-library registration, the `build_fp8.sh` (CPU-only, `-DFUSED_FP8=1`),
and the looser-tol accuracy test (`test_fused_expert_fp8_op.py`) all exist. Remaining work, in order:
1. **Flip `BUILD_FP8_NODES=1`**, supply the two param structs from the bridge perf_lib headers, build the
   `.so` (`docker run --rm ... bash build_fp8.sh`) — hours.
2. **On-device geometry/SIF bring-up** of the 8 new fp8 nodes (the amax `reduce_max_multi_dim_fwd`
   reductionDimensionMask reversed-bit, the `[1,out]`/`[M,1]` scale broadcasts into `fp8_gemm`, the
   `convert_to_fp8` shapes) — the same multi-iteration loop the bf16 op needed, ~1–3 days, box owner.
3. **Accuracy gate** (`test_fused_expert_fp8_op.py`, looser 8e-2 tol; flag `|got|/|ref|≈2×` = the
   fn-vs-fnuz trap) — ~½ day.
4. **lazy frontend `.so`** (a `*_lazy.cpp` sharing the backend header, like the bf16 op) so it captures
   into the decode HPU-graph for serving — ~1 day. Then the K-expert routed/monolithic wrapper and the
   serve-env wiring (mirrors the existing int4 fused path) — those are plugin work, model-agnostic.

**#1 RISK:** **the 2× MME does not net a speedup** because (a) the fused-expert OpBackend op already loses
to stock (Finding 2) and W4A8-fused *adds* 8 per-forward cast/amax nodes on top, and (b) single-stream is
collective/host-bound so MME 2× is invisible there (Finding 3 / §11a). I.e. the speed case rests entirely
on **throughput@concurrency for a model too big to host fp8**, and that exact number is **unmeasured** —
the build could validate-correct (~6.4e-2, coherent) yet still not beat int4→bf16 on tok/s once the
per-forward fp8-cast overhead is paid. Mitigation: treat it as a **memory-fit enabler** (its real,
defensible value: lets a 1T int4-only model touch the fp8 MME at all), gate "enable for speed" behind an
explicit on-card W4A8-fused-vs-int4→bf16 throughput A/B, and prefer the **load-time transform + stock
native fp8 op whenever fp8 fits HBM** (lower error, no per-forward overhead, beats the fused op).
Secondary risk: the fn-vs-fnuz factor-of-2 — but the scaffold already enforces the bf16-only dequant +
self-consistent round-trip + `FP8_MAX=240` discipline that defuses it.
