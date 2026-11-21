"""STRONGEST CARD-FREE GATE: corrected oracle (ref/msa_reference.py) vs the REAL transformers
5.12.0 ``MiniMaxM3VLIndexer`` (the AUTHORITATIVE modeling, installed at $HOME/.venvs/m3calib).

This pins the index branch to the OFFICIAL PyTorch code, not the paper. It instantiates the real
``MiniMaxM3VLIndexer`` + ``MiniMaxM3VLRotaryEmbedding`` on CPU with small synthetic inputs and random
small dims, drives it, and asserts our corrected oracle reproduces it:

  (1) PRE-TOPK numeric: our ``msa_collapse_index_heads(msa_index_scores(...))`` == the official
      head-collapsed ``block_scores`` (``scores.amax(-1).amax(dim=1)``), exact modulo fp (atol 1e-4).
      This is the load-bearing check: it isolates Gemma-norm + partial-RoPE + raw-matmul + the
      head-collapse, BEFORE any topk tie-break can hide a discrepancy.
  (2) SELECTION SETS: our ``msa_block_select`` (force-local + topk) block SET per query == the
      official ``indexer.forward`` block-index SET per query (Jaccard == 1.0 over every query).

It ALSO proves each correction is load-bearing (the test has teeth): three CONTROLS that each undo
ONE fix (no rope / plain x*w norm / add 1/sqrt(d_idx) scale) must DIVERGE from the official scores.

RUN WITH THE m3calib VENV (it has transformers 5.12.0 + torch):
    $HOME/.venvs/m3calib/bin/python tests/test_msa_hf_parity.py

If transformers' minimax_m3_vl is unavailable the test SKIPS loudly (never silently passes).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (os.path.join(_ROOT, "ref"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import msa_reference as ref  # noqa: E402  the corrected numeric ground truth (under test here)

torch.set_printoptions(precision=4, sci_mode=False)
NEG_INF = float("-inf")

_FAILS: list[str] = []


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not ok:
        _FAILS.append(f"{name}: {detail}")
    return ok


def _import_hf():
    """Import the official modeling; return (modeling, config) modules or (None, None) to SKIP."""
    try:
        from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as M
        from transformers.models.minimax_m3_vl import configuration_minimax_m3_vl as C
        return M, C
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] transformers minimax_m3_vl unavailable: {e!r}")
        print("        run with $HOME/.venvs/m3calib/bin/python (transformers 5.12.0).")
        return None, None


# Small M3-ratio config (shrunk; same RATIOS as the real M3): head_dim=16, rotary_dim=8
# (partial_rotary_factor 0.5 like M3's 64/128), index_head_dim=16, index_n_heads=2, block_size=4,
# topk=3, local=1. Config-driven; NO M3 literals leak.
_HID = 96
_HEAD_DIM = 16
_ROTARY = 8
_IDX_DIM = 16
_IDX_HEADS = 2
_KV_HEADS = 2
_Q_HEADS = 8
_BLOCK = 4
_TOPK = 3
_LOCAL = 1
_THETA = 5000000.0


def _build_hf_cfg(C):
    return C.MiniMaxM3VLTextConfig(
        vocab_size=128, hidden_size=_HID, intermediate_size=128, num_hidden_layers=6,
        num_attention_heads=_Q_HEADS, num_key_value_heads=_KV_HEADS, head_dim=_HEAD_DIM,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rotary_dim=_ROTARY,
        rope_parameters={"rope_type": "default", "rope_theta": _THETA, "partial_rotary_factor": 0.5},
        index_n_heads=_IDX_HEADS, index_head_dim=_IDX_DIM, index_block_size=_BLOCK,
        index_topk_blocks=_TOPK, index_local_blocks=_LOCAL,
        layer_types=["full_attention"] * 3 + ["minimax_m3_sparse"] * 3,
    )


def _ref_dims():
    return ref.MsaDims(
        hidden_size=_HID, num_attention_heads=_Q_HEADS, num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM, rotary_dim=_ROTARY, rope_theta=_THETA,
        sparse_index_dim=_IDX_DIM, sparse_num_index_heads=_IDX_HEADS, sparse_topk_blocks=_TOPK,
        sparse_block_size=_BLOCK, sparse_score_type="max", sparse_init_block=0,
        sparse_local_block=_LOCAL,
    )


def _official_block_scores(M, idx, hidden, cos, sin, position_ids):
    """Re-derive the OFFICIAL pre-topk collapsed block_scores VERBATIM from
    MiniMaxM3VLIndexer.forward (modeling lines 558-580), using ``idx``'s own weights. Returns
    block_scores [B, S_q, n_blk] (head-collapsed) — the exact tensor topk runs on."""
    b, q_len, _ = hidden.shape
    idx_q = idx.q_proj(hidden).view(b, q_len, -1, idx.head_dim)
    idx_q = idx.q_norm(idx_q).transpose(1, 2)                                   # Gemma norm, then heads
    idx_k = idx.k_proj(hidden).view(b, q_len, 1, idx.head_dim)
    idx_k = idx.k_norm(idx_k).transpose(1, 2)
    iq, ik = M.apply_rotary_pos_emb(idx_q, idx_k, cos[..., :idx.head_dim], sin[..., :idx.head_dim])
    k_len = ik.shape[2]
    nkb = -(-k_len // idx.block_size)
    pad = nkb * idx.block_size - k_len
    scores = torch.matmul(iq.float(), ik.float().transpose(-1, -2))            # RAW (no scale)
    kpos = torch.arange(k_len)
    future = kpos[None, None, None, :] > position_ids[:, None, :, None]
    scores = scores.masked_fill(future, NEG_INF)
    if pad:
        scores = F.pad(scores, (0, pad), value=NEG_INF)
    scores = scores.view(b, idx.num_heads, q_len, nkb, idx.block_size)
    return scores.amax(dim=-1).amax(dim=1)                                     # [B, S_q, n_blk]


def _raw_qk_from_hf(idx, hidden):
    """The RAW projected index q/k (pre-norm, pre-rope) from the SAME q_proj/k_proj weights the HF
    indexer uses, reshaped to our oracle's [T, H_idx, D] / [T, 1, D]. The oracle applies the Gemma
    norm + partial rope itself (given idx's norm weights)."""
    T = hidden.shape[1]
    q_idx = idx.q_proj(hidden)[0].view(T, idx.num_heads, idx.head_dim)         # [T, H_idx, D]
    k_idx = idx.k_proj(hidden)[0].view(T, 1, idx.head_dim)                     # [T, 1, D]
    return q_idx, k_idx


def _sets_from_padded(block_idx_2d):
    """[S, k] int (-1 pad) -> list[S] of frozensets of valid block ids (order/pad-invariant)."""
    return [frozenset(int(x) for x in row.tolist() if x >= 0) for row in block_idx_2d]


def test_1_pretopk_block_scores(M, C):
    print("test (1) corrected oracle block_scores == OFFICIAL MiniMaxM3VLIndexer (pre-topk, numeric)")
    cfg = _build_hf_cfg(C)
    rd = _ref_dims()
    worst = 0.0
    nan_ok = True
    for S in (1, 3, 4, 7, 8, 11, 16):
        for seed in (0, 1, 5):
            torch.manual_seed(seed)
            rot = M.MiniMaxM3VLRotaryEmbedding(cfg)
            idx = M.MiniMaxM3VLIndexer(cfg, layer_idx=3)
            with torch.no_grad():
                # nontrivial Gemma weights (near 0, some NEGATIVE) -> per-element transform like the
                # real checkpoint (mean ~0.2). If norm were plain x*w this would be near-identity-0.
                idx.q_norm.weight.copy_(0.25 * torch.randn(idx.head_dim))
                idx.k_norm.weight.copy_(0.25 * torch.randn(idx.head_dim))
            pos = torch.arange(S).unsqueeze(0)
            cos, sin = rot(torch.zeros(1, S, _HID), pos)
            hidden = torch.randn(1, S, _HID)

            off = _official_block_scores(M, idx, hidden, cos, sin, pos)[0]      # [S, n_blk]

            # our corrected oracle: gemma-norm -> msa_index_scores (with rope) -> collapse heads.
            q_idx, k_idx = _raw_qk_from_hf(idx, hidden)
            wq = idx.q_norm.weight.detach()
            wk = idx.k_norm.weight.detach()
            qn = ref.gemma_rms_norm(q_idx, wq, eps=1e-6)
            kn = ref.gemma_rms_norm(k_idx, wk, eps=1e-6)
            cpos = torch.arange(S)
            c2, s2 = ref.build_index_cos_sin(cpos, rd.rotary_dim, rd.rope_theta)
            bs = ref.msa_index_scores(qn, kn, rd.index_scale, rd.sparse_block_size, cpos,
                                      cos=c2, sin=s2)                            # [S, H_idx, n_blk]
            ours = ref.msa_collapse_index_heads(bs)                            # [S, n_blk]

            # compare finite entries numerically; -inf must align on both sides.
            offf, oursf = off.float(), ours.float()
            both_ninf = torch.isneginf(offf) & torch.isneginf(oursf)
            mism_ninf = torch.isneginf(offf) ^ torch.isneginf(oursf)
            if bool(mism_ninf.any()):
                nan_ok = False
                _check(f"-inf mask aligns S={S} seed={seed}", False,
                       f"{int(mism_ninf.sum())} mismatched -inf cells")
            fin = ~(torch.isneginf(offf) | torch.isneginf(oursf))
            if bool(fin.any()):
                d = (offf[fin] - oursf[fin]).abs().max().item()
                worst = max(worst, d)
                if d >= 1e-4:
                    _check(f"block_scores match S={S} seed={seed}", False, f"max|Δ|={d:.3e}")
            _ = both_ninf
    _check("oracle block_scores == official (all S x seed, atol 1e-4)", worst < 1e-4 and nan_ok,
           f"worst finite |Δ|={worst:.3e}")


def test_2_selection_sets(M, C):
    print("test (2) corrected oracle SELECTION sets == OFFICIAL indexer.forward (Jaccard==1.0)")
    cfg = _build_hf_cfg(C)
    rd = _ref_dims()
    all_ok = True
    worst_j = 1.0
    bad = None
    for S in (1, 3, 4, 7, 8, 11, 16, 20):
        for seed in (0, 1, 2, 5):
            torch.manual_seed(seed)
            rot = M.MiniMaxM3VLRotaryEmbedding(cfg)
            idx = M.MiniMaxM3VLIndexer(cfg, layer_idx=3)
            with torch.no_grad():
                idx.q_norm.weight.copy_(0.25 * torch.randn(idx.head_dim))
                idx.k_norm.weight.copy_(0.25 * torch.randn(idx.head_dim))
            pos = torch.arange(S).unsqueeze(0)
            cos, sin = rot(torch.zeros(1, S, _HID), pos)
            hidden = torch.randn(1, S, _HID)

            # OFFICIAL selection: indexer.forward -> [B, S_q, topk] block ids, -1 padded.
            off_idx = idx(hidden, (cos, sin), None, pos)[0]                     # [S, topk]
            off_sets = _sets_from_padded(off_idx)

            # OUR corrected oracle selection (msa_forward's selection path).
            q_idx, k_idx = _raw_qk_from_hf(idx, hidden)
            cpos = torch.arange(S)
            _, b_idx, _ = ref.msa_forward(
                torch.randn(S, _Q_HEADS, _HEAD_DIM),    # q/k/v values irrelevant to SELECTION
                torch.randn(S, _KV_HEADS, _HEAD_DIM),
                torch.randn(S, _KV_HEADS, _HEAD_DIM),
                q_idx, k_idx, rd, causal_pos=cpos, return_selection=True,
                index_q_norm_weight=idx.q_norm.weight.detach(),
                index_k_norm_weight=idx.k_norm.weight.detach(),
                index_rope=True,
            )
            our_sets = _sets_from_padded(b_idx)

            for t in range(S):
                a, b2 = off_sets[t], our_sets[t]
                union = a | b2
                jac = 1.0 if not union else len(a & b2) / len(union)
                worst_j = min(worst_j, jac)
                if jac != 1.0:
                    all_ok = False
                    bad = (S, seed, t, sorted(a), sorted(b2))
    _check("selection sets identical (Jaccard==1.0 every query)", all_ok,
           "" if all_ok else f"(S,seed,t,official,ours)={bad}")
    _check("worst-case Jaccard over the whole sweep", worst_j == 1.0, f"min Jaccard={worst_j:.4f}")


def test_3_controls_have_teeth(M, C):
    print("test (3) CONTROLS: undoing ANY single fix DIVERGES from the official (the fixes matter)")
    cfg = _build_hf_cfg(C)
    rd = _ref_dims()
    # one moderately long case where blocks compete (selection is sensitive).
    S = 16
    torch.manual_seed(3)
    rot = M.MiniMaxM3VLRotaryEmbedding(cfg)
    idx = M.MiniMaxM3VLIndexer(cfg, layer_idx=3)
    with torch.no_grad():
        idx.q_norm.weight.copy_(0.6 * torch.randn(idx.head_dim) + 0.2)   # bigger so x*w vs x*(1+w) bites
        idx.k_norm.weight.copy_(0.6 * torch.randn(idx.head_dim) + 0.2)
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = rot(torch.zeros(1, S, _HID), pos)
    hidden = torch.randn(1, S, _HID)
    off = _official_block_scores(M, idx, hidden, cos, sin, pos)[0]              # [S, n_blk]
    offf = off.float()
    fin = ~torch.isneginf(offf)

    q_idx, k_idx = _raw_qk_from_hf(idx, hidden)
    cpos = torch.arange(S)
    wq, wk = idx.q_norm.weight.detach(), idx.k_norm.weight.detach()
    c2, s2 = ref.build_index_cos_sin(cpos, rd.rotary_dim, rd.rope_theta)

    def collapsed(qn, kn, scale, cos_, sin_):
        bs = ref.msa_index_scores(qn, kn, scale, rd.sparse_block_size, cpos, cos=cos_, sin=sin_)
        return ref.msa_collapse_index_heads(bs).float()

    # CONTROL A — NO rope (the old paper-derived assumption). Must differ.
    qn = ref.gemma_rms_norm(q_idx, wq); kn = ref.gemma_rms_norm(k_idx, wk)
    a = collapsed(qn, kn, 1.0, None, None)
    dA = (offf[fin] - a[fin]).abs().max().item()
    _check("CONTROL no-rope DIVERGES from official", dA > 1e-3, f"max|Δ|={dA:.3e} (want > 1e-3)")

    # CONTROL B — plain x*w norm (the old wire form), rope ON. Must differ.
    qn_b = (q_idx.float() * torch.rsqrt(q_idx.float().pow(2).mean(-1, keepdim=True) + 1e-6)) * wq
    kn_b = (k_idx.float() * torch.rsqrt(k_idx.float().pow(2).mean(-1, keepdim=True) + 1e-6)) * wk
    b = collapsed(qn_b, kn_b, 1.0, c2, s2)
    dB = (offf[fin] - b[fin]).abs().max().item()
    _check("CONTROL plain-x*w-norm DIVERGES from official", dB > 1e-3, f"max|Δ|={dB:.3e} (want > 1e-3)")

    # CONTROL C — correct norm + rope but WITH the 1/sqrt(d_idx) scale. Scores differ by that factor.
    import math
    c = collapsed(qn, kn, 1.0 / math.sqrt(_IDX_DIM), c2, s2)
    dC = (offf[fin] - c[fin]).abs().max().item()
    _check("CONTROL 1/sqrt(d_idx)-scale DIVERGES from official scores", dC > 1e-3,
           f"max|Δ|={dC:.3e} (want > 1e-3)")

    # SANITY — the corrected (all fixes) path MATCHES (re-assert at this harsher config).
    full = collapsed(qn, kn, 1.0, c2, s2)
    dF = (offf[fin] - full[fin]).abs().max().item()
    _check("corrected (all fixes) MATCHES official at the control config", dF < 1e-4,
           f"max|Δ|={dF:.3e}")


def test_4_rope_width_and_headcollapse(M, C):
    print("test (4) structural: cos/sin width == rotary_dim (partial rope) + official IS head-collapsed")
    cfg = _build_hf_cfg(C)
    rot = M.MiniMaxM3VLRotaryEmbedding(cfg)
    S = 8
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = rot(torch.zeros(1, S, _HID), pos)
    _check("main cos/sin last-dim == rotary_dim (partial rope, indexer slice is a no-op)",
           cos.shape[-1] == _ROTARY and sin.shape[-1] == _ROTARY,
           f"cos last-dim={cos.shape[-1]} rotary_dim={_ROTARY}")

    # head-collapse: the official returns ONE block set per query [B, S_q, topk] (no head axis),
    # confirming amax over the index-head dim BEFORE topk -> a single shared selection.
    idx = M.MiniMaxM3VLIndexer(cfg, layer_idx=3)
    out = idx(torch.randn(1, S, _HID), (cos, sin), None, pos)
    # official: topk = min(topk_blocks, num_key_blocks); the LOAD-BEARING fact is the shape has NO
    # H_idx/H_kv axis -> ONE shared set per query (amax over the index-head dim before topk).
    n_blk = -(-S // _BLOCK)
    exp_k = min(_TOPK, n_blk)
    _check("official indexer emits ONE set/query [B,S_q,k] (head-collapsed, not per-group)",
           out.dim() == 3 and out.shape == (1, S, exp_k),
           f"shape={tuple(out.shape)} (expect (1,{S},{exp_k}) — no H_idx/H_kv axis -> one shared set)")


def main():
    print("=" * 96)
    print("MiniMax-M3 MSA — corrected oracle (ref/msa_reference.py) vs REAL transformers 5.12.0 "
          "MiniMaxM3VLIndexer")
    print("=" * 96)
    M, C = _import_hf()
    if M is None:
        print("-" * 96)
        print("SKIPPED (no transformers minimax_m3_vl). Run with the m3calib venv.")
        return
    test_1_pretopk_block_scores(M, C)
    test_2_selection_sets(M, C)
    test_3_controls_have_teeth(M, C)
    test_4_rope_width_and_headcollapse(M, C)
    print("-" * 96)
    if _FAILS:
        print(f"{len(_FAILS)} assertion(s) FAILED:")
        for f in _FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL HF-PARITY ASSERTIONS PASSED (corrected oracle == official MiniMaxM3VLIndexer)")


# pytest entry points
def test_pretopk():
    M, C = _import_hf()
    if M is None:
        import pytest
        pytest.skip("transformers minimax_m3_vl unavailable")
    test_1_pretopk_block_scores(M, C); assert not _FAILS


def test_selection():
    M, C = _import_hf()
    if M is None:
        import pytest
        pytest.skip("transformers minimax_m3_vl unavailable")
    test_2_selection_sets(M, C); assert not _FAILS


def test_controls():
    M, C = _import_hf()
    if M is None:
        import pytest
        pytest.skip("transformers minimax_m3_vl unavailable")
    test_3_controls_have_teeth(M, C); assert not _FAILS


if __name__ == "__main__":
    main()
