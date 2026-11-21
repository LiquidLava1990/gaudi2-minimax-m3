"""probe_fused_swigluoai_124.py — GO/NO-GO probe for the fused-op-revive single-stream lever.

WHAT THIS DECIDES
-----------------
The FAST, no-training path to GLM-class single-stream is the native captured swiglu op
`torch.ops.hpu.mixture_of_experts.bias_fused_weights(..., alpha=1.702, limit=7.0)` — the
device gpt_swiglu kernel that fuses dequant(not here; bf16)+swigluoai+down into ONE recipe
node, so per-token decode is ~free. M3's routed experts ARE exactly this formula.

On `vllm-gaudi:next` (the EAGER-frontend build) this op is REJECTED (M3_MATRIX_REPORT §1):
  - `permuted_weights=False` (contiguous in=H,out=2I)  -> returns ~0   (relNorm ~1.0)
  - `permuted_weights=True`                            -> HANGS the card (Synapse "Compute
                                                          or DMA timeout", recoverable/proc).
Cause: `frontend_blocklist: [eager]` on the schema (hpu_op.yaml:8939) — the *device* swiglu
CGUID is absent/blocklisted on the eager image; only a degenerate/CPU-composite path remains.

The bet THIS probe settles: the **:1.24 LAZY** image is NOT eager-blocklisted, so the device
swigluoai kernel may be LIVE there → GLM-class single-stream WITHOUT training an MTP head.

  GO  iff   relNorm < 1e-2  for a permuted_weights variant  AND no card hang.
  NO-GO if  the op returns ~0 (relNorm ~1) for both variants, OR it hangs (DMA timeout),
            OR the schema/op is absent on the image.

This script RUNS ON A CARD (single Gaudi, lazy, :1.24). It is NOT run by the authoring agent.
The CPU self-test below (`_cpu_selftest`, the default when run off-card) only validates the
swigluoai-MoE CPU REFERENCE path — that it returns finite, correctly-shaped output on the
synthetic inputs — which is the oracle the on-card relNorm is measured against.

LOUD on the hang risk: `permuted_weights=True` is the orientation that HUNG the card on :next.
Each op call is fenced with SIGALRM + a try/except for the Synapse DMA-timeout, and the
permuted_weights=False (safe-ish, returns-0-not-hang on :next) variant is run FIRST so a clean
NO-GO can be reported even if the True variant later wedges this process.

REFERENCE NUMERICS — the one gotcha (swigluoai-op-capability.md §2):
  The device op consumes gate/up **INTERLEAVED** on the w12 output dim ([g0,u0,g1,u1,...]).
  m3_cpu_mtp.swigluoai (the trusted oracle) consumes **CONTIGUOUS** halves ([gate(I)|up(I)]).
  So we build each expert's W13 CONTIGUOUS for the CPU reference, and feed the op an
  INTERLEAVED copy of that same W13 — then both see identical gate/up pairings and a faithful
  device kernel must match the reference. (permuted_weights is the orthogonal x@W vs x@W^T knob;
  we feed the matching w12 orientation for each variant.)

RUN ON THE BOX (human only — the authoring agent must NOT run this):
  see HUMAN_RUN_CMD at the bottom, or:
  PT_HPU_LAZY_MODE=1 python /opt/m3-root/mtp/probe_fused_swigluoai_124.py
  under the :1.24 image on ONE idle Gaudi.

CPU SELF-TEST (authoring agent, card-free):
  $HOME/.venvs/m3calib/bin/python \
      $HOME/AI/minimax-m3/mtp/probe_fused_swigluoai_124.py --cpu-selftest
"""
from __future__ import annotations

import os
import sys

# --- LAZY plugin selection MUST happen before habana_frameworks import. The :1.24 lazy
#     frontend is the whole point (eager is blocklisted for this op). Force + warn if unset;
#     refuse loudly if habana is already imported (the choice is locked at import time). ---
if "habana_frameworks" in sys.modules:
    print("FATAL: habana_frameworks imported BEFORE this probe set PT_HPU_LAZY_MODE=1; the "
          "lazy-vs-eager plugin is locked at import. Run as:\n"
          "  PT_HPU_LAZY_MODE=1 python probe_fused_swigluoai_124.py", file=sys.stderr)
    sys.exit(2)
_CARDLESS_FLAGS = ("--cpu-selftest", "--print-cmd")
if not any(f in sys.argv for f in _CARDLESS_FLAGS) and os.environ.get("PT_HPU_LAZY_MODE") != "1":
    print("NOTE: PT_HPU_LAZY_MODE was not '1'; forcing it to 1 (the device swigluoai kernel "
          "is blocklisted on the EAGER frontend — this probe must run LAZY).", file=sys.stderr)
    os.environ["PT_HPU_LAZY_MODE"] = "1"

# import the trusted CPU oracle (swigluoai) + constants from the validated foundation.
# Make the foundation importable whether we run from the repo, /opt/m3-root/mtp, or elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, "/opt/m3-root/mtp"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from m3_cpu_mtp import swigluoai, H, ALPHA, LIMIT  # the SINGLE source of truth
except Exception as ex:  # noqa: BLE001
    print(f"FATAL: cannot import m3_cpu_mtp (swigluoai/H/ALPHA/LIMIT): {ex}\n"
          f"       expected next to this file ({_HERE}) or at /opt/m3-root/mtp.",
          file=sys.stderr)
    sys.exit(2)

# --- synthetic problem shape (small, T=1 decode; a FEW experts, real M3 hidden/intermediate) ---
INTERMEDIATE = 3072          # M3 text_config intermediate_size
N_EXPERTS = 4                # a few local experts (cheap; enough to exercise routing)
TOP_K = 4                    # M3 routes top-4
RELNORM_GO = 1e-2            # GATE threshold
HANG_TIMEOUT_S = 90          # SIGALRM fence per op call (Synapse DMA-timeout is ~minutes; cut early)


def _relnorm(got, ref):
    import torch  # local: torch may be CPU-only in selftest, hpu on card
    g = got.detach().float().reshape(-1)
    r = ref.detach().float().reshape(-1)
    diff = (g - r).norm().item()
    return diff / (r.norm().item() + 1e-9), (g - r).abs().max().item()


def _build_inputs(device, dtype, seed=0):
    """Small synthetic routed-MoE problem on `device`.

    Returns x [T,H], routing (ids[T,K], weights[T,K]), per-expert CONTIGUOUS W13[H,2I] / W2[I,H]
    (bf16), and zero-bias lists. Contiguous = [gate(I) | up(I)] on the output dim — matches the
    m3_cpu_mtp.swigluoai oracle. (The op gets an interleaved view built in _run_op.)"""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = 1  # decode shape
    twoI = 2 * INTERMEDIATE
    x = (torch.randn(T, H, generator=g) * 0.1).to(dtype).to(device)
    # routing: top-K distinct experts, softmax-ish positive weights summing to ~1
    ids = torch.stack([torch.randperm(N_EXPERTS, generator=g)[:TOP_K] for _ in range(T)]).to(device)
    raw = torch.rand(T, TOP_K, generator=g)
    weights = (raw / raw.sum(-1, keepdim=True)).to(dtype).to(device)
    # small weights so x@W13 stays in a sane range (not all clamp-saturated → a real comparison)
    W13, W2, zb13, zb2 = [], [], [], []
    for e in range(N_EXPERTS):
        ge = torch.Generator(device="cpu").manual_seed(seed + 1 + e)
        W13.append((torch.randn(H, twoI, generator=ge) * (1.0 / H ** 0.5)).to(dtype).to(device))
        W2.append((torch.randn(INTERMEDIATE, H, generator=ge) * (1.0 / INTERMEDIATE ** 0.5)).to(dtype).to(device))
        # M3 experts are bias=False; the op's check_inputs DEMANDS non-empty bias -> zero bias
        # (the +1 on up is hardcoded in the kernel, so literal zero is numerically exact).
        zb13.append(torch.zeros(twoI, dtype=dtype, device=device))
        zb2.append(torch.zeros(H, dtype=dtype, device=device))
    return x, ids, weights, W13, W2, zb13, zb2


def _cpu_reference(x, ids, weights, W13, W2):
    """The trusted oracle: out = sum_k w_k * ( swigluoai(x @ W13_k) @ W2_k ).

    Uses m3_cpu_mtp.swigluoai (CONTIGUOUS gate/up). Pure matmuls + the foundation activation —
    deliberately NOT the device op, so it independently pins what 'correct' is."""
    import torch
    T = x.shape[0]
    out = torch.zeros(T, x.shape[-1], dtype=x.dtype, device=x.device)
    for t in range(T):
        for k in range(ids.shape[1]):
            e = int(ids[t, k].item())
            gate_up = x[t:t + 1] @ W13[e]          # [1, 2I]
            h = swigluoai(gate_up, ALPHA, LIMIT)    # [1, I]  (contiguous-halves swigluoai)
            y = h @ W2[e]                           # [1, H]
            out[t:t + 1] = out[t:t + 1] + weights[t, k] * y
    return out


def _interleave_w13(W13_contig):
    """Reorder a CONTIGUOUS [gate(I)|up(I)] W13 (cols 0..2I) to INTERLEAVED [g0,u0,g1,u1,...].

    The device op consumes gate/up interleaved; the oracle is contiguous. Feeding this view to
    the op makes its swigluoai see the SAME gate/up pairing the oracle used. (Mirrors
    m3_expert_runtime_patch._stage_interleave_for_route_a, on the output/last dim.)"""
    import torch
    twoI = W13_contig.shape[-1]
    I = twoI // 2
    perm = torch.empty(twoI, dtype=torch.int64, device=W13_contig.device)
    idx = torch.arange(I, dtype=torch.int64, device=W13_contig.device)
    perm[0::2] = idx       # even cols  <- gate[0..I)
    perm[1::2] = idx + I   # odd  cols  <- up[0..I)
    return W13_contig.index_select(-1, perm).contiguous()


class _Timeout(Exception):
    pass


def _run_op(moe_bias, x, ids, weights, W13, W2, zb13, zb2, permuted_weights):
    """One fenced call to bias_fused_weights. Returns (status, detail) where status is one of
    'OK' (with out tensor in detail), 'ZERO', 'HANG', 'ERR'.

    permuted_weights selects the matmul orientation the kernel assumes:
      False -> kernel does x @ W12   (W12 stored [in=H, out=2I])  — the contiguous-in orientation
      True  -> kernel does x @ W12^T (W12 stored [out=2I, in=H])  — the orientation that HUNG :next
    For each, we feed the matching w12 layout, both built from the INTERLEAVED-gate/up W13 so the
    swigluoai pairing matches the oracle."""
    import signal
    import torch
    import habana_frameworks.torch.core as htcore

    E = len(W13)
    w12_il = [_interleave_w13(w) for w in W13]                 # [H, 2I] interleaved
    if permuted_weights:
        # x @ W12^T  =>  store W12 as [out=2I, in=H]; W3 as [out=H, in=I]
        w12 = [w.t().contiguous() for w in w12_il]             # [2I, H]
        w3 = [w.t().contiguous() for w in W2]                  # [H, I]
    else:
        w12 = w12_il                                           # [H, 2I]
        w3 = W2                                                # [I, H]
    htcore.mark_step()

    def _alarm(signum, frame):  # noqa: ARG001
        raise _Timeout()

    prev = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(HANG_TIMEOUT_S)
    try:
        out = moe_bias(
            hidden_states=x,
            expert_routing_table=ids.to(torch.int64),
            router_weights=weights,
            w12=w12, w3=w3, w12_bias=zb13, w3_bias=zb2,
            permuted_weights=permuted_weights,
            experts_min=0, experts_max=E - 1,
            alpha=ALPHA, limit=LIMIT,
        )
        htcore.mark_step()
        out_c = out.to("cpu")            # forces device sync -> a hang trips the alarm here
        htcore.mark_step()
        signal.alarm(0)
        if not bool(torch.isfinite(out_c).all()):
            return "ERR", "non-finite output"
        if out_c.float().norm().item() < 1e-6:
            return "ZERO", out_c
        return "OK", out_c
    except _Timeout:
        return "HANG", (f"no return in {HANG_TIMEOUT_S}s — Synapse Compute/DMA timeout "
                        f"(the :next failure mode for permuted_weights=True)")
    except Exception as ex:  # noqa: BLE001
        m = str(ex)
        low = m.lower()
        if "timeout" in low or "dma" in low or "synapse" in low:
            return "HANG", f"Synapse/DMA timeout: {m}"
        return "ERR", m
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _on_card_probe() -> int:
    """The GO/NO-GO probe. RUNS ON A CARD (single Gaudi, lazy)."""
    try:
        import habana_frameworks.torch  # noqa: F401  (loads the LAZY plugin)
        import habana_frameworks.torch.core as htcore  # noqa: F401
        from habana_frameworks.torch.utils.internal import is_lazy
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.",
              file=sys.stderr)
        return 2

    if not is_lazy():
        print("FATAL: is_lazy() is False after import. This probe targets the LAZY frontend "
              "(the device swigluoai kernel is frontend_blocklist:[eager]). "
              f"PT_HPU_LAZY_MODE={os.environ.get('PT_HPU_LAZY_MODE')!r}.", file=sys.stderr)
        return 2

    moe = getattr(torch.ops.hpu, "mixture_of_experts", None)
    moe_bias = getattr(moe, "bias_fused_weights", None) if moe is not None else None
    if moe_bias is None:
        print("\nNO-GO (~0 / hang / blocklisted): "
              "torch.ops.hpu.mixture_of_experts.bias_fused_weights is ABSENT on this image.",
              file=sys.stderr)
        print("FUSED-OP :1.24 LAZY = NO-GO (op not registered)")
        return 1

    dev = "hpu"
    dtype = torch.bfloat16
    x, ids, weights, W13, W2, zb13, zb2 = _build_inputs(dev, dtype, seed=0)
    htcore.mark_step()

    # trusted CPU/device reference (NOT the op): build on device in bf16 then bring to CPU
    ref = _cpu_reference(x, ids, weights, W13, W2)
    htcore.mark_step()
    ref_c = ref.to("cpu")
    print(f"=== probe_fused_swigluoai_124 ===  lazy=True  T=1 H={H} I={INTERMEDIATE} "
          f"E={N_EXPERTS} top_k={TOP_K}  |ref|={ref_c.float().norm().item():.4f}")

    # --- SANITY: the plain silu overload of the SAME op works on :next (relNorm 4e-3). Run it
    #     first as a routing/shape/zero-bias control: if THIS fails too, the problem is our call
    #     plumbing, not the swiglu leg. (Best-effort: skip cleanly if the overload is absent.) ---
    moe_silu = getattr(moe, "fused_weights", None)
    if moe_silu is not None:
        import torch.nn.functional as F

        def _silu_ref():
            T = x.shape[0]
            out = torch.zeros(T, H, dtype=dtype, device=dev)
            for t in range(T):
                for k in range(ids.shape[1]):
                    e = int(ids[t, k].item())
                    gu = x[t:t + 1] @ W13[e]
                    gate = gu[:, :INTERMEDIATE]; up = gu[:, INTERMEDIATE:]
                    out[t:t + 1] = out[t:t + 1] + weights[t, k] * ((F.silu(gate) * up) @ W2[e])
            return out.to("cpu")
        try:
            import signal
            sref = _silu_ref(); htcore.mark_step()
            prev = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
            signal.alarm(HANG_TIMEOUT_S)
            so = moe_silu(
                hidden_states=x, expert_routing_table=ids.to(torch.int64), router_weights=weights,
                w12=W13, w3=W2, permuted_weights=False, activation="silu",
                experts_min=0, experts_max=N_EXPERTS - 1)
            htcore.mark_step(); so_c = so.to("cpu"); signal.alarm(0)
            rn_s, _ = _relnorm(so_c, sref)
            print(f"  [control] silu overload (permuted_weights=False): relNorm={rn_s:.3e} "
                  f"({'routing/shapes OK' if rn_s < RELNORM_GO else 'FAILS — call plumbing suspect'})")
        except _Timeout:
            print("  [control] silu overload HUNG — unexpected (silu worked on :next); "
                  "call plumbing or image suspect.", file=sys.stderr)
        except Exception as ex:  # noqa: BLE001
            print(f"  [control] silu overload skipped/failed: {ex}")
        finally:
            try:
                import signal
                signal.alarm(0)
            except Exception:  # noqa: BLE001
                pass

    # --- the real test: swigluoai via bias_fused_weights. permuted_weights=False FIRST (it
    #     RETURNED ~0, did not hang, on :next) so a clean verdict survives even if True wedges. ---
    best_rn = None
    verdict_go = False
    hang_seen = False
    for permuted in (False, True):
        if permuted:
            print("  [WARN] about to call permuted_weights=True — this HUNG the card on :next "
                  "(Synapse DMA timeout). Fenced with a "
                  f"{HANG_TIMEOUT_S}s alarm; a hang => NO-GO, not a wedge of the whole box.",
                  file=sys.stderr)
        status, detail = _run_op(moe_bias, x, ids, weights, W13, W2, zb13, zb2, permuted)
        if status == "OK":
            rn, ma = _relnorm(detail, ref_c)
            best_rn = rn if best_rn is None else min(best_rn, rn)
            ok = rn < RELNORM_GO
            verdict_go = verdict_go or ok
            print(f"  bias_fused_weights permuted_weights={permuted}: relNorm={rn:.4e} "
                  f"maxAbs={ma:.4e}  -> {'MATCH (device swiglu LIVE)' if ok else 'mismatch'}")
        elif status == "ZERO":
            print(f"  bias_fused_weights permuted_weights={permuted}: returned ~0 "
                  "(degenerate/blocked swiglu leg — the :next NO-GO signature).")
        elif status == "HANG":
            hang_seen = True
            print(f"  bias_fused_weights permuted_weights={permuted}: HANG — {detail}",
                  file=sys.stderr)
            # do NOT attempt the other orientation after a hang; the device may be wedged.
            break
        else:  # ERR
            print(f"  bias_fused_weights permuted_weights={permuted}: ERROR — {detail}",
                  file=sys.stderr)

    print(f"\n{'='*64}")
    if verdict_go:
        print(f"FUSED-OP :1.24 LAZY = GO (relNorm {best_rn:.2e})")
        print("  => the native swigluoai device kernel is LIVE on :1.24 lazy. This is the FAST,"
              " no-training path to GLM-class single-stream. Wire route 'a' "
              "(M3_FP8_SWIGLU_ROUTE=a / a bf16 bias_fused_weights branch) on a :1.24 lazy serve.")
        rc = 0
    else:
        why = "hang" if hang_seen else ("~0" if best_rn is None else f"relNorm {best_rn:.2e}")
        print(f"FUSED-OP :1.24 LAZY = NO-GO (~0 / hang / blocklisted) [{why}]")
        print("  => device swigluoai not viable on :1.24 lazy either. Stay on the per-expert "
              "loop; pursue the MTP draft-head path for single-stream speed.")
        rc = 1
    print(f"{'='*64}")
    return rc


# ============================ CPU self-test (card-free) ============================
def _cpu_selftest() -> int:
    """Validate the REFERENCE path only (the oracle the on-card relNorm is measured against):
    the swigluoai-MoE CPU reference returns finite, correctly-shaped output on synthetic data,
    and exercises the interleave helper + the routed weighted-sum. No HPU, no op call."""
    import torch
    torch.set_grad_enabled(False)
    dev = "cpu"
    dtype = torch.float32  # CPU: fp32 for a clean reference (bf16 matmul on CPU is lossy/odd)
    x, ids, weights, W13, W2, zb13, zb2 = _build_inputs(dev, dtype, seed=0)

    checks = []
    # 1. reference runs + shape [T, H]
    ref = _cpu_reference(x, ids, weights, W13, W2)
    checks.append(("reference shape [T,H]", ref.shape == (1, H)))
    # 2. finite
    checks.append(("reference finite", bool(torch.isfinite(ref).all())))
    # 3. non-trivial (small but nonzero — weights/W scaled to avoid all-clamp-saturation)
    checks.append(("reference nonzero", ref.float().norm().item() > 1e-4))
    # 4. routing well-formed: top-K distinct experts in range, weights ~sum to 1
    ok_ids = bool(((ids >= 0) & (ids < N_EXPERTS)).all()) and ids.shape == (1, TOP_K)
    distinct = ids.unique().numel() == TOP_K
    checks.append(("routing ids valid+distinct", ok_ids and distinct))
    checks.append(("router weights ~sum 1", abs(weights.sum().item() - 1.0) < 1e-4))
    # 5. interleave helper is a permutation (round-trips: de-interleave recovers contiguous)
    twoI = 2 * INTERMEDIATE
    il = _interleave_w13(W13[0])
    inv = torch.empty(twoI, dtype=torch.int64)
    idx = torch.arange(INTERMEDIATE)
    inv[idx] = 2 * idx              # gate j  <- interleaved col 2j
    inv[idx + INTERMEDIATE] = 2 * idx + 1  # up   j  <- interleaved col 2j+1
    recov = il.index_select(-1, inv)
    checks.append(("interleave is a permutation (round-trip)", torch.equal(recov, W13[0])))
    checks.append(("interleave preserves shape", il.shape == W13[0].shape))
    # 6. zero-bias lists non-empty + right shape (the op's check_inputs demands this)
    checks.append(("zero-bias w13 list", len(zb13) == N_EXPERTS and zb13[0].shape == (twoI,)
                   and float(zb13[0].abs().sum()) == 0.0))
    checks.append(("zero-bias w2 list", len(zb2) == N_EXPERTS and zb2[0].shape == (H,)))
    # 7. the reference truly uses the foundation swigluoai (contiguous): a hand re-derivation
    #    of ONE routed expert term must match the reference's contribution exactly.
    e0 = int(ids[0, 0].item())
    gu = x[0:1] @ W13[e0]
    gate, up = gu[:, :INTERMEDIATE], gu[:, INTERMEDIATE:]
    glu = gate.clamp(max=LIMIT) * torch.sigmoid(ALPHA * gate.clamp(max=LIMIT))
    h_manual = (up.clamp(-LIMIT, LIMIT) + 1.0) * glu
    term0 = weights[0, 0] * (h_manual @ W2[e0])
    # rebuild ref WITHOUT expert e0's k=0 term, add the manual term, compare to full ref
    ref_minus = ref.clone() - term0
    # the remaining terms (k=1..) recomputed via the module path:
    rem = torch.zeros(1, H, dtype=dtype)
    for k in range(1, TOP_K):
        e = int(ids[0, k].item())
        rem = rem + weights[0, k] * (swigluoai(x[0:1] @ W13[e], ALPHA, LIMIT) @ W2[e])
    checks.append(("swigluoai contiguous-halves matches hand formula",
                   torch.allclose(ref_minus, rem, atol=1e-4, rtol=1e-4)))

    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\n{'='*64}\nCPU REFERENCE SELF-TEST: "
          f"{'GREEN — reference path sound (on-card probe is card-gated)' if ok else 'RED — FIX ABOVE'}"
          f"\n{'='*64}")
    return 0 if ok else 1


# ============================ the exact human on-card command ============================
HUMAN_RUN_CMD = r"""
# === HUMAN ONLY — runs on ONE idle Gaudi under the :1.24 LAZY image. Mirror M3_MATRIX_REPORT
#     lever-1 on-card probe (single Gaudi, lazy). The authoring agent must NOT run this. ===
# (1) confirm cards idle + box not wedged:  hl-smi   (all cards ~0%, no link down)
# (2) run the probe in a throwaway :1.24 container, ONE Gaudi, lazy frontend:
docker run --rm \
  --device=/dev/accel/accel0 --device=/dev/infiniband/uverbs0 \
  --cap-add=sys_nice --cap-add=SYS_PTRACE --ipc=host --net=host --ulimit memlock=-1:-1 \
  -e HABANA_VISIBLE_DEVICES=0 \
  -e PT_HPU_LAZY_MODE=1 \
  -e PT_HPU_RECIPE_CACHE_CONFIG="" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v $HOME/AI/minimax-m3:/opt/m3-root \
  vllm-gaudi:1.24 \
  python /opt/m3-root/mtp/probe_fused_swigluoai_124.py
#
# Reads the verdict on the last line:
#   "FUSED-OP :1.24 LAZY = GO (relNorm X)"            -> wire route 'a' on a :1.24 lazy serve
#   "FUSED-OP :1.24 LAZY = NO-GO (~0 / hang / blocklisted)"  -> stay on the loop; do MTP head
#
# HANG NOTE: permuted_weights=True HUNG the card on :next. It is fenced (90s SIGALRM + DMA-timeout
# catch) and run AFTER permuted_weights=False, so a hang yields a clean NO-GO line, not a box wedge.
# If the *container* itself stops responding (device truly wedged), `docker rm -f` THIS throwaway
# container only and re-check hl-smi; do not churn other containers (affinity-deadlock risk).
"""


def main(argv) -> int:
    if "--cpu-selftest" in argv:
        return _cpu_selftest()
    if "--print-cmd" in argv:
        print(HUMAN_RUN_CMD)
        return 0
    return _on_card_probe()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
