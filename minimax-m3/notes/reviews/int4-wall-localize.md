# Localizing the M3 int4 single-stream host wall — WHERE the ~280 ms TPOT goes

**Angle:** at ~280 ms TPOT with cards 18–31 % busy, ~70–80 % of each decode step is host-side
stall. Find the SINGLE biggest *reducible* host cost. CARD-FREE op-count analysis from the code that
actually runs (`hpu-m3-ext/hpu_m3_ext/m3_swigluoai_patch.py`, `models/minimax-m3-int4.env`,
`config.json`). MEASURED context: M3_MATRIX_REPORT (8.3→3.6 tok/s, util 18–31 %, no CPU fallback, no
recompile); Opt-Guide §IV.8/IV.8a; PROJECT_FINDINGS §11/§13.

## The arithmetic (single-stream, T=1)

Architecture: **60 layers, 0–2 dense, 3–59 = 57 sparse MoE layers**, `top_k=4`, `num_local_experts=16`
(128/8, TP8+EP). The routed gate (`x2.shape[0]*K ≤ K`) fires only at **T==1** → every decode step
runs `_routed_swigluoai` with **K = top_k = 4** on all 57 sparse layers. Per MoE layer:

| segment | host ops (≈) | mark_step |
|---|---|---|
| `select_experts` grouped-topk routing | ~8 | |
| routing→K-active (to/reshape/where×2/scatter/topk) | ~8 | **#1** |
| dequant: index_select×4 + clone×8(lazy) + **convert_from_uint4×8** | ~20 | **#2** |
| gate-matrix scatter_add + remap | ~6 | |
| **K=4 expert loop**: 4 × (matmul + chunk/clamp²/sigmoid/mul²/add + matmul + mul + add) | **~44** | |
| **per MoE layer** | **~86** | **2** |

**Per decode step:** 57 × ~86 ≈ **~4,900 host op enqueues** + 57 × 2 = **~114 mark_steps** from the
MoE *alone*, before attention (60 layers FusedSDPA + KV-cache writes), router, sampler, and the TP=8
all-reduces. At ~280 ms that is **~50–55 µs of host dispatch per op** — pure Python/bridge enqueue
latency, the cards idle between segments. This is the host wall, quantified.

## The #1 reducible host cost: the 114 mark_steps (2 per sparse layer)

Each `mark_step()` is a **hard graph/stream boundary** — it flushes the lazy queue and forces a
host↔device round-trip the bridge cannot pipeline across. At 57 layers × 2 = ~114/step they are the
largest *structural, reducible* serialization, distinct from the irreducible op count. They exist for
**lazy** correctness (realize routing before the data-dependent gather; realize dequant before the
matmul). **But the int4 serve runs torch.compile, NOT lazy** (the .env sets neither `PT_HPU_LAZY_MODE=1`
nor INC; `_co` clones are even gated on `htorch.is_lazy()`). Under compile the two `mark_step`s and the
clones are **not required for correctness** — they were copied verbatim from the lazy `_routed_moe`
contract.

## The cut

Gate both `mark_step`s on `htorch.is_lazy()` (mirror the existing `_need_clone`): under the compile
serve, drop them so the whole MoE layer is one fused region instead of 3 host-synced fragments.
Removes **~114 forced host round-trips/step**. Low-risk (compile already realizes deps via the
dataflow graph), gated, reversible, LOUD-safe. Pair with the secondary cut — collapse the K=4 loop's
per-iter elementwise (chunk/clamp²/sigmoid/mul²/add, ~28 of the ~44 loop ops) since that is the
densest reducible op cluster after mark_steps; but mark_steps are the #1 *serializing* cost.

## Expected gain & confidence

If host stall is ~75 % of 280 ms (~210 ms) and ~114 mark_steps each cost a ~0.3–0.5 ms flush, they
are **~35–55 ms/step** → removing them is **~15–25 % TPOT** → ~7.6 → **~9–10 tok/s**, util ~18 →
~25–30 %. **Confidence: MEDIUM.** The op-count localization is HIGH-confidence (direct from code); the
*per-mark_step ms* is the soft variable (could be smaller if the bridge already coalesces). It is the
cheapest high-leverage probe: one `is_lazy()` guard, A/B in one warm serve. It will NOT clear 30 tok/s
(that needs the all-512-expert dequant gone → fp8 experts, §IV.2) — it shaves the host wall, the
correct first move for *this* regime.
