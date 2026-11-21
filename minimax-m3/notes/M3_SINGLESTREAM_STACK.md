# M3 master perf game plan — MAX single-stream + MAX concurrency (2026-06-15)

**Decision (user):** invest in M3 ITSELF (not a fallback model), workload MIXED code+reasoning.
Targets: single-stream **~25–30 tok/s** (today not-usable), concurrency **maxed**.
Built from the 10-workstream game plan (wf wbktm8frf) + source verification.

## CORRECTED MECHANICS (supersede the earlier "lazy / mark_step fragmentation" framing)
- **Live serve runs torch.compile / EAGER backend — `PT_HPU_LAZY_MODE=0` on all 8 workers** (docker logs
  vllm-active: `bridge_mode: eager`, `regional_compilation: True`, `fullgraph_compilation: False`,
  `Wrapping in HPUGraph took 0.0000 GB`). NOT lazy. So:
  - The ~114 mark_steps + ~456 clones are **already gated OFF** (`m3_swigluoai_patch.py` FIX-1, default-on);
    the decode MoE is already ONE compiled recipe. The ~9–10 tok/s from that may be **already banked** (the
    5.8–7.6 numbers predate FIX-1 / were cold-cache). **CONFIRM via a warm-cache A/B** — do not re-claim it.
  - **MoE op-fusion on :next = NO-GO** (0 tok/s — nothing left to fuse). **Full-step HPUGraph capture = NO-GO**
    (no lazy fragmentation to collapse; HPUGraph only activates under lazy, where routed-MoE degenerates).
- **The real single-stream wall is the host op-dispatch of the WHOLE ~57-layer compiled step + structural
  (attention all-reduce + sampler)** — common to all models, ~29–34 MoE-agnostic cap. The M3-vs-GLM gap is
  most likely **graph-breaks / eager islands** in the swigluoai K-loop (dynamic expert indexing) vs GLM's
  one fused intrinsic. **UNMEASURED — the profile (Phase 0) is decisive.** Precision swaps = 0 single-stream.

## SINGLE-STREAM plan (the usable-speed goal)
1. **MTP native-shaped draft head — CRITICAL PATH, ~1.70× (CONDITIONAL-GO).** Train depth-1 from scratch
   (nothing to graft: 0 MTP tensors in all 59 bf16 shards). Build a native-MTP-shaped head (NOT EAGLE —
   EAGLE-HPU asserts Eagle3LlamaForCausalLM, Llama-only): `m3_mtp.py` modeled on `glm4_moe_mtp.py`
   (enorm/hnorm + eh_proj 2H→H + REUSE `HpuMiniMaxM3DecoderLayer` → swigluoai MoE for free + shared_head),
   patch vLLM speculative resolution (minimax_m3_vl→mtp; M3 lacks GLM's mapping), build the hidden-state dump
   harness + CPU-oracle accept-gate. **GATE: CPU-oracle accept >~60% BEFORE any card time.** T=N verify-batch
   crash already FIXED (`m3_swigluoai_patch.py:474`). Depth-1 only (GLM depth-2 net-flat). Multiplies the
   NOVEL-token (reasoning) half — the half prompt-lookup can't touch.
   - **Card-free now:** author m3_mtp.py, the speculative-method patch, the dump harness, the CPU oracle.
   - **Card-gated:** dump hidden+target (needs M3 running), train, validate, serve.
2. **Fused-op revive on the :1.24 LAZY image — the ONE structural single-stream lever, ~2–3× → GLM-class
   ~14–20 (CONDITIONAL-GO).** `bias_fused_weights(alpha=1.702,limit=7.0)` is the exact M3 swigluoai formula
   but `frontend_blocklist:[eager]` on :next forces a CPU composite; **the :1.24 lazy frontend is NOT
   eager-blocklisted → the device CGUID may be live.** One fused op/layer replaces the per-expert loop.
   **DECISIVE UNKNOWN:** does the CGUID execute on :1.24 lazy or ~0/hang like :next eager? Single-card probe.
   (This is the "master dynamic kernel = speed" path, via the stock gpt_swiglu mode on a different image.)
3. **Confirm the banked mark_step/clone gating** (warm-cache A/B) → ~9–10 base. Depends on Phase 0.

Stack: ~9–10 (banked) → fused-op-revive ~14–20 OR base×MTP 1.70 → **~24–30**. Both landing → high end.

## CONCURRENCY plan (already mostly WON — finish it)
1. **fp8 route-b is ALREADY ~3.6–5× (39.9/63.4/90 @ c=10/30/50, climbing) — measured.** The win is the native
   fp8 MME removing the §11a per-token all-expert dequant that serialized int4 (why mns was inert past 64).
2. **Quality gate — switch the fp8 SOURCE to INC-from-bf16.** The 90 number is on int4→fp8-at-load = DOUBLE
   quant (~6.4e-2 floor). INC-from-bf16 = single-quant <1% AND keeps non-expert modules bf16.
3. **Config sweep → ~130–180 @ c~80–100:** mns 64→128, max-model-len 4096→3072, util 0.94,
   `VLLM_DECODE_BS_BUCKET_MAX=128`. The #1 remaining concurrency lever. Next wall = serialized 2K prefill.
4. **fp8_gemm_v2 swap** (route-b matmul → native fp8 GEMM): INCREMENTAL ~2× on the PREFILL matmul leg only;
   second-order — do after the config lever. (~0 single-stream — host-bound.)
5. **MSA block-sparse attention** (~90% done; fix the head-collapse indexer .so [old per-group contract] →
   re-validate Jaccard≥0.97 + real-weight needle → flip HPU_MSA_READY): concurrency + long-context, ~0 single.

## SEQUENCE
- **Phase 0 (do first, card-light):** measure-infra — warm-recipe-cache durability fix + the ONE warm batch=1
  decode profile (sizes the single-stream split: graph-breaks vs attention vs all-reduce vs sampler). Decides
  whether the fused-op-revive is worth it and confirms the banked mark_step gain. ALL other A/Bs depend on it.
- **Phase 1 (single-stream):** MTP head (mostly card-free authoring NOW) ∥ fused-op-revive :1.24 probe.
- **Phase 2 (concurrency):** INC quality gate → config sweep → fp8_gemm_v2 increment → MSA.
- **HARD CONSTRAINT:** live Kimi bench owns vllm-active — every card-gated step BLOCKED until it drains + idle
  (affinity-deadlock rule). Author all card-free pieces meanwhile; MTP is the long pole → start it first.

## Honest ceiling: single-stream ~25–30 (MTP + base) or ~14–20 (fused-op-revive alone); concurrency ~130–180
agg. ~29–34 is the MoE-agnostic single-stream node cap; 80–150 single-stream is NOT reachable for a 428B model
on one node — that bar = a smaller model or a 2nd node (PP-fp8, both plumbed).
