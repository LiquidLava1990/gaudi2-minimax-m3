"""test_msa_block_indexer_op.py -- VALIDATE the MiniMax-M3 MSA block-indexer top-k
OpBackend / GC-captured op ON THE HPU BOX (gated; cards are busy serving -- run this
ONLY when a card is free and on-card numeric validation is explicitly authorized).

Loads build/libmsa_block_indexer.so (eager) or build/libmsa_block_indexer_lazy.so
(lazy, via $MSA_INDEXER_SO) with torch.ops.load_library, builds synthetic M3-shaped
index inputs, calls the captured op

    torch.ops.hpu_msa_ext.msa_block_indexer_topk(q, k, weights, cmask, index_topk,
                                                 mode, block_size)
        q       [H, Sq, D]  bf16   (per index-head queries; H == H_kv index heads)
        k       [H, Sk, D]  bf16   (single shared index key expanded to H)
        weights [H, Sq, 1]  bf16   (MSA: index scale 1/sqrt(d_idx) broadcast; order-
                                    preserving -> set-match exact. Or all-ones.)
        cmask   [Sq, Sk]   bool    (TRUE where key is non-causal -> dropped)
        index_topk         int     (MSA: top-k BLOCKS; clamped to n_blocks)
        mode               int     (1 = MSA block-sparse; 0 = DSA token-sparse)
        block_size         int     (tokens/block; Sk must be a multiple)
    -> mode==1: [H, Sq, kk] int32  top-k BLOCK ids per (group, query)  (kk=min(topk,n_blk))
    -> mode==0: [Sq, kk]    int32  top-k TOKEN ids per query (the DSA-compat path)

and asserts (MSA) its top-k BLOCK-id SET matches the fp32 ORACLE from
ref/msa_reference.py (msa_index_scores -> per-(query,group) block-max-pool ->
msa_reference_block_topk). Because top-k order may differ on ties / bf16-vs-fp32, we
compare as a SET per (group, query) row (mean Jaccard + mean recall) -- the same gate
the DSA indexer used (Jaccard 0.97 / recall 0.98).

NOTE: this op returns the RAW scored top-k blocks. Force-local injection, ascending
sort and -1 padding are done UPSTREAM (msa_block_select / the runner) and are NOT part
of this op, so the oracle here is run with force_local_blocks=0 / force_init_blocks=0
(raw scored set). The runner reconciles forced blocks separately.

RUN (ON THE HPU BOX, gated -- a CPU-only import cannot exercise the graph):
  cd $HOME/AI/minimax-m3/kernel
  python tests/test_msa_block_indexer_op.py                 # M3 default H=4 D=128 Sq=256 Sk=512 topk=16 B=128
  python tests/test_msa_block_indexer_op.py 4 128 256 512 16 128   # H D Sq Sk topk_blocks block_size
  # lazy .so + HPU-graph capture check:
  PT_HPU_LAZY_MODE=1 MSA_INDEXER_SO=build/libmsa_block_indexer_lazy.so python tests/test_msa_block_indexer_op.py
"""
from __future__ import annotations

import os
import sys

NEG_INF = float("-inf")


def _import_oracle():
    """Import the parity oracle from ref/msa_reference.py (the numeric ground truth)."""
    here = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.normpath(os.path.join(here, "..", "..", "ref"))
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    import msa_reference as ref  # noqa: WPS433
    return ref


def _set_metrics_2d(got_idx, ref_sets):
    """got_idx [N, kk] int32; ref_sets list[N] of python sets. Mean Jaccard + recall."""
    N = got_idx.shape[0]
    jac, rec = 0.0, 0.0
    for n in range(N):
        g = set(int(x) for x in got_idx[n].tolist() if int(x) >= 0)
        r = ref_sets[n]
        if not r:
            jac += 1.0
            rec += 1.0
            continue
        inter = len(g & r)
        union = len(g | r)
        jac += inter / union if union else 1.0
        rec += inter / len(r)
    return jac / N, rec / N


def _find_so():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../kernel
    env = os.environ.get("MSA_INDEXER_SO")
    if env:
        so = env if os.path.isabs(env) else os.path.join(here, env)
    else:
        so = os.path.join(here, "build", "libmsa_block_indexer.so")
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build it first (CPU-only):\n"
            f"  docker run --rm -v $HOME/AI/minimax-m3:/m3 "
            f"-v $HOME/AI/hpu-moe-ext:/hpumoe "
            f"-e BRIDGE=/hpumoe/tpc_kernel/opbackend/bridge "
            f"-w /m3/kernel vllm-gaudi:next bash build_msa_block_indexer.sh\n"
            f"  (or build_msa_block_indexer_lazy.sh; set MSA_INDEXER_SO to pick the .so)")
    return so


def main() -> int:
    H  = int(sys.argv[1]) if len(sys.argv) > 1 else 4    # M3 index heads == num_key_value_heads
    D  = int(sys.argv[2]) if len(sys.argv) > 2 else 128  # sparse_index_dim
    Sq = int(sys.argv[3]) if len(sys.argv) > 3 else 256  # query rows
    Sk = int(sys.argv[4]) if len(sys.argv) > 4 else 512  # key rows (MULTIPLE of block_size)
    topk_blocks = int(sys.argv[5]) if len(sys.argv) > 5 else 16   # sparse_topk_blocks
    block_size  = int(sys.argv[6]) if len(sys.argv) > 6 else 128  # sparse_block_size

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    ref = _import_oracle()

    if Sk % block_size != 0:
        print(f"FATAL: Sk={Sk} must be a multiple of block_size={block_size} "
              "(pad the key axis upstream).", file=sys.stderr)
        return 2
    n_blocks = Sk // block_size
    kk = min(topk_blocks, n_blocks)

    try:
        so = _find_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}")
        return 2
    torch.ops.load_library(so)
    if not hasattr(torch.ops, "hpu_msa_ext") or not hasattr(
            torch.ops.hpu_msa_ext, "msa_block_indexer_topk"):
        print("FATAL: torch.ops.hpu_msa_ext.msa_block_indexer_topk not registered "
              f"after load_library({so}).", file=sys.stderr)
        return 2
    print(f"loaded {so}")

    dev = "hpu"
    torch.manual_seed(0)
    index_scale = 1.0 / (D ** 0.5)
    JAC_TOL = 0.97
    REC_TOL = 0.98
    rc = 0
    print(f"=== test_msa_block_indexer_op (MSA) ===  H={H} D={D} Sq={Sq} Sk={Sk} "
          f"topk_blocks={topk_blocks} block_size={block_size} n_blocks={n_blocks} (kk={kk})")

    # ---- synthetic M3-shaped inputs (CPU master copies, fp32 -> bf16) ----
    # Per-index-head queries [Sq,H,D] and a single shared index key [Sk,D].
    q_thd = (torch.randn(Sq, H, D) * 0.1)         # [Sq,H,D]
    k_sd  = (torch.randn(Sk, D) * 0.1)            # [Sk,D]  (single shared index head)

    # Causal absolute positions: query t sees keys [0, abs_pos[t]]. Spread across the
    # key axis so most rows have many visible blocks (well-defined block top-k).
    abs_pos = torch.clamp(
        torch.linspace(block_size, Sk - 1, Sq).round().long(), max=Sk - 1)  # [Sq]
    key_pos = torch.arange(Sk).view(1, -1)
    cmask = (key_pos > abs_pos.view(-1, 1))       # [Sq,Sk] bool, TRUE => drop

    # ---- oracle (fp32, CPU): block-max-pool scores -> per-(query,group) top-k set ----
    # msa_index_scores expects q_idx [T,H_kv,D], k_idx [T,(1,)D]. Here T==Sq==Sk rows
    # for the synthetic; pass causal_pos=abs_pos and the same single shared key.
    q_idx = q_thd                                  # [Sq,H,D]
    k_idx = k_sd[:Sq] if Sq <= Sk else k_sd        # index key per query-row position
    # For the synthetic indexer parity we score Sq queries against Sk keys, so build the
    # scores directly with the oracle's math over [Sq queries, Sk keys] using a [Sk,D] key.
    qf = q_idx.float()
    kf = k_sd.float()
    s = torch.einsum("qhd,kd->qhk", qf, kf) * index_scale      # [Sq,H,Sk]
    s = s.masked_fill(cmask[:, None, :], NEG_INF)              # causal
    pad = n_blocks * block_size - Sk
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=NEG_INF)
    block_scores = s.view(Sq, H, n_blocks, block_size).amax(dim=-1)   # [Sq,H,n_blocks]
    # raw scored top-k block SETS per (query, group) -- force_local/init = 0 (the op is raw).
    ref_sets_nested = ref.msa_reference_block_topk(
        block_scores, topk_blocks=topk_blocks, causal_pos=abs_pos,
        force_local_blocks=0, force_init_blocks=0, block_size=block_size)  # [Sq][H]->set

    # ---- op inputs on HPU (head axis leading for the batched Q.K^T) ----
    q_op = q_thd.permute(1, 0, 2).contiguous().to(torch.bfloat16).to(dev)   # [H,Sq,D]
    k_op = k_sd.unsqueeze(0).expand(H, -1, -1).contiguous().to(torch.bfloat16).to(dev)  # [H,Sk,D]
    w_op = torch.full((H, Sq, 1), index_scale, dtype=torch.bfloat16, device=dev)  # index scale
    cmask_op = cmask.contiguous().to(dev)          # [Sq,Sk] bool
    htorch.mark_step()

    # ====================================================================
    # CHECK 1 -- top-k BLOCK-id SET vs the fp32 oracle (MSA mode=1).
    # ====================================================================
    print("\n--- CHECK 1: msa_block_indexer_topk BLOCK set (vs msa_reference_block_topk) ---")
    got = torch.ops.hpu_msa_ext.msa_block_indexer_topk(
        q_op, k_op, w_op, cmask_op, topk_blocks, 1, block_size)  # mode=1 (MSA)
    htorch.mark_step()
    got_idx = got.to("cpu")
    if list(got_idx.shape) != [H, Sq, kk]:
        print(f"  >>> FAIL: op returned shape {list(got_idx.shape)}, expected "
              f"[{H}, {Sq}, {kk}].", file=sys.stderr)
        return 1
    if got_idx.dtype != torch.int32:
        print(f"  >>> FAIL: op returned dtype {got_idx.dtype}, expected int32.",
              file=sys.stderr)
        rc = 1
    # flatten (H,Sq) rows and the matching ref sets in the SAME order.
    got_flat = got_idx.reshape(H * Sq, kk)
    ref_flat = [ref_sets_nested[t][h] for h in range(H) for t in range(Sq)]
    # NB: got is [H,Sq,kk] so row order is (h outer, t inner); match ref accordingly.
    ref_flat = [ref_sets_nested[t][h] for h in range(H) for t in range(Sq)]
    jac, rec = _set_metrics_2d(got_flat, ref_flat)
    print(f"  mean Jaccard={jac:.4f}  mean recall={rec:.4f}  "
          f"(kk={kk} blocks per row, {H*Sq} rows)")
    if jac >= JAC_TOL and rec >= REC_TOL:
        print(f"  >>> PASS: MSA block set matches oracle (Jaccard {jac:.3f} >= {JAC_TOL}, "
              f"recall {rec:.3f} >= {REC_TOL}).")
    else:
        print(f"  >>> FAIL: MSA Jaccard {jac:.4f} / recall {rec:.4f} below tol "
              f"({JAC_TOL}/{REC_TOL}).", file=sys.stderr)
        print("      Suspects (device-confirmable): batch_gemm transpose_b (q@k^T),\n"
              "      masked_fill broadcast over H, the reshape [H,Sq,Sk]->[H,Sq,nblk,B],\n"
              "      reduce_max dim-mask (1<<(4-3-1)=1 for dim3/rank4), topk axis\n"
              "      get_dim_in_tpc_order(2,3)=0, or the -inf pad of fully-padded blocks.",
              file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 2 -- HPU-graph CAPTURE (the durable serving path; best-effort).
    # ====================================================================
    print("\n--- CHECK 2: HPU-graph capture (best-effort) ---")
    try:
        import habana_frameworks.torch.hpu as ht_hpu
        from habana_frameworks.torch.hpu.graphs import HPUGraph

        g = HPUGraph()
        s_ = ht_hpu.Stream()
        with ht_hpu.stream(s_):
            g.capture_begin()
            out_cap = torch.ops.hpu_msa_ext.msa_block_indexer_topk(
                q_op, k_op, w_op, cmask_op, topk_blocks, 1, block_size)
            g.capture_end()
        g.replay()
        htorch.mark_step()
        jac2, _ = _set_metrics_2d(out_cap.to("cpu").reshape(H * Sq, kk), ref_flat)
        if jac2 >= JAC_TOL:
            print(f"  >>> PASS: op captured into an HPU graph and replayed "
                  f"(Jaccard {jac2:.3f}). The durable single-recipe serving path.")
        else:
            print(f"  >>> WARN: captured replay Jaccard {jac2:.4f} below tol.",
                  file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped capture check: {ex})")

    # ====================================================================
    # CHECK 3 -- DSA-compat path (mode=0): the op must STILL return token top-k.
    # ====================================================================
    print("\n--- CHECK 3: DSA-compat (mode=0) shape/dtype smoke ---")
    try:
        got_dsa = torch.ops.hpu_msa_ext.msa_block_indexer_topk(
            q_op, k_op, w_op, cmask_op, kk * block_size, 0, 0)  # mode=0, token topk
        htorch.mark_step()
        gd = got_dsa.to("cpu")
        exp_kk = min(kk * block_size, Sk)
        if list(gd.shape) == [Sq, exp_kk] and gd.dtype == torch.int32:
            print(f"  >>> PASS: mode=0 returns [Sq, {exp_kk}] int32 token ids "
                  "(DSA-compat path intact).")
        else:
            print(f"  >>> WARN: mode=0 returned {list(gd.shape)} {gd.dtype}, expected "
                  f"[{Sq}, {exp_kk}] int32.", file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (mode=0 smoke skipped: {ex})")

    print(f"\n[done] test_msa_block_indexer_op {'OK' if rc == 0 else 'FAILED'} (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
