"""Synthetic parity tests for the MiniMax-M3 MSA pure-torch reference (ref/msa_reference.py).

Mirrors the DSA synthetic harness style (hpu-dsa-ext tests): tiny CPU tensors, fp32, asserts the
selection matches a brute-force oracle, the gather-consume matches the dense oracle in the
all-blocks-fit limit, the sparse path is NOT silently dense, partial-last-block / forced-local
semantics hold, and causality holds.

Runs as a plain script (no pytest needed in the image):
    python tests/test_msa_synthetic.py
or, if pytest is present:
    python -m pytest tests/test_msa_synthetic.py -x -q

NO device flags / NO HPU — pure CPU. Every assertion prints PASS/FAIL with the metric.
"""
import math
import os
import sys

import torch

# Make ref/ importable whether run from repo root, tests/, or via pytest.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (os.path.join(_ROOT, "ref"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from msa_reference import (  # noqa: E402
    MsaDims,
    msa_index_scores,
    msa_block_select,
    msa_expand_blocks_to_tokens,
    msa_gqa_gather_attention,
    msa_dense_oracle,
    msa_forward,
    msa_reference_block_topk,
)

torch.manual_seed(0)

# Small synthetic config (M3 ratios, shrunk): H_q=8, H_kv=2 (G=4), head_dim=16, d_idx=16,
# block_size=4, topk=3. All config-driven through MsaDims (no literals leak into the reference).
SMALL = MsaDims(
    hidden_size=64,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=16,
    sparse_index_dim=16,
    sparse_num_index_heads=2,
    sparse_topk_blocks=3,
    sparse_block_size=4,
    sparse_score_type="max",
    sparse_init_block=0,
    sparse_local_block=1,
)

_FAILS = []


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{(' :: ' + detail) if detail else ''}")
    if not ok:
        _FAILS.append(f"{name}: {detail}")
    return ok


def _make_inputs(S, dims: MsaDims, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(S, dims.num_attention_heads, dims.head_dim, generator=g)
    k = torch.randn(S, dims.num_key_value_heads, dims.head_dim, generator=g)
    v = torch.randn(S, dims.num_key_value_heads, dims.head_dim, generator=g)
    q_idx = torch.randn(S, dims.num_key_value_heads, dims.sparse_index_dim, generator=g)
    k_idx = torch.randn(S, 1, dims.sparse_index_dim, generator=g)
    return q, k, v, q_idx, k_idx


# ================================================================================================
# (1) block-select == brute-force argmax-topk over block-max-pooled scores.
#     Pure top-k branch (no forced blocks) so the brute-force is exactly argmax-topk.
# ================================================================================================
def test_block_select_matches_bruteforce():
    print("test (1) block-select == brute-force argmax-topk over block-max-pooled scores")
    S = 40  # 10 blocks of size 4; topk=3
    dims = SMALL
    _, _, _, q_idx, k_idx = _make_inputs(S, dims, seed=1)
    causal_pos = torch.arange(S)

    iscores = msa_index_scores(q_idx, k_idx, dims.index_scale, dims.sparse_block_size, causal_pos)
    # HEAD-COLLAPSE (official): selection is ONE shared set per query [T, topk] (no head axis); both
    # msa_block_select and msa_reference_block_topk collapse the index heads (amax) internally.
    # reference selection: NO forced blocks -> pure argmax-topk over the (causal) block-max scores.
    block_idx = msa_block_select(
        iscores, dims.sparse_block_size, dims.sparse_topk_blocks, causal_pos,
        force_local=False, force_local_blocks=0, force_init_blocks=0,
    )
    ref_sets = msa_reference_block_topk(
        iscores, dims.sparse_topk_blocks, causal_pos,
        force_local_blocks=0, force_init_blocks=0, block_size=dims.sparse_block_size,
    )

    T = iscores.shape[0]
    _check("block-select returns ONE set [T,topk] (head-collapsed, no head axis)",
           block_idx.dim() == 2, f"shape={tuple(block_idx.shape)}")
    all_eq = True
    mism = None
    for t in range(T):
        got = set(int(x) for x in block_idx[t].tolist() if x >= 0)
        ref = ref_sets[t]
        if got != ref:
            all_eq = False
            mism = (t, sorted(got), sorted(ref))
            break
    _check("block-select set == brute-force argmax-topk (one set)", all_eq,
           "" if all_eq else f"mismatch (t,got,ref)={mism}")

    # also assert ascending + -1-padded tail invariant (api.py:1222-1223).
    asc = True
    for t in range(T):
        row = block_idx[t].tolist()
        valid = [x for x in row if x >= 0]
        pads = [x for x in row if x < 0]
        if valid != sorted(valid) or any(p != -1 for p in pads):
            asc = False
        # all valid must precede all pads
        seen_pad = False
        for x in row:
            if x < 0:
                seen_pad = True
            elif seen_pad:
                asc = False
    _check("selection ascending + -1 padded to tail", asc)


# ================================================================================================
# (2) gather-consume == dense_oracle when ALL blocks fit (topk*block_size >= S).
# ================================================================================================
def test_gather_equals_dense_when_all_fit():
    print("test (2) gather-consume == dense_oracle when all blocks fit (topk*block_size >= S)")
    dims = SMALL
    # choose S so that ceil(S/block) <= topk -> every causal block is selectable.
    S = dims.sparse_topk_blocks * dims.sparse_block_size  # 3*4 = 12, n_blk=3 == topk
    q, k, v, q_idx, k_idx = _make_inputs(S, dims, seed=2)
    n_blk = (S + dims.sparse_block_size - 1) // dims.sparse_block_size
    assert n_blk <= dims.sparse_topk_blocks, "test setup: blocks must all fit"

    out_sparse, block_idx, token_idx = msa_forward(
        q, k, v, q_idx, k_idx, dims, return_selection=True)
    out_dense = msa_dense_oracle(q, k, v, dims.main_scale)

    diff = (out_sparse - out_dense).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out_sparse.reshape(-1), out_dense.reshape(-1), dim=0).item()
    _check("sparse == dense (all blocks fit)", diff < 1e-5 and cos > 0.99999,
           f"max_diff={diff:.2e}, cos={cos:.8f}")


# ================================================================================================
# (3) sparse OUTPUT != dense when blocks < total  (proves NOT silently dense).
# ================================================================================================
def test_sparse_differs_from_dense_when_pruned():
    print("test (3) sparse OUTPUT != dense when fewer blocks than total (not silently dense)")
    dims = SMALL
    # many more blocks than topk so most are pruned; use a longer seq.
    S = 64  # 16 blocks, topk=3 -> 13 blocks pruned for late queries
    q, k, v, q_idx, k_idx = _make_inputs(S, dims, seed=3)
    n_blk = (S + dims.sparse_block_size - 1) // dims.sparse_block_size
    assert n_blk > dims.sparse_topk_blocks, "test setup: must prune blocks"

    out_sparse = msa_forward(q, k, v, q_idx, k_idx, dims)
    out_dense = msa_dense_oracle(q, k, v, dims.main_scale)

    diff = (out_sparse - out_dense).abs().max().item()
    # The last query sees n_blk>topk blocks; pruning MUST change its output materially.
    _check("sparse != dense (blocks pruned)", diff > 1e-3, f"max_diff={diff:.4f} (want > 1e-3)")

    # sanity: gather form agrees with an independent dense-but-masked recompute of the SAME
    # selection (guards against the gather geometry silently dropping/adding keys).
    out_sparse2, block_idx, token_idx = msa_forward(
        q, k, v, q_idx, k_idx, dims, return_selection=True)
    # recompute consume from the SAME token_idx via a masked dense pass over all S keys.
    ref = _masked_consume_from_token_idx(q, k, v, token_idx, dims.main_scale)
    d2 = (out_sparse2 - ref).abs().max().item()
    _check("gather form == masked-dense over same selection", d2 < 1e-5, f"max_diff={d2:.2e}")


def _masked_consume_from_token_idx(q, k, v, token_idx, scale):
    """Independent oracle: build a [T,S] additive bias from token_idx (selected & causal) and run
    dense SDPA over ALL S keys. Equivalent to the gather form but computed differently.

    HEAD-COLLAPSE: token_idx is ONE shared set per query [T, W] (no head axis); every group uses the
    SAME selected tokens, so the [T,S] selection mask is built once and reused across groups."""
    T, Hq, D = q.shape
    S, Hkv, _ = k.shape
    Dv = v.shape[-1]
    G = Hq // Hkv
    qpos = torch.arange(T)
    key_pos = torch.arange(S)
    out = torch.zeros(T, Hq, Dv, dtype=torch.float32)
    NEG = float("-inf")
    # one shared selection mask [T,S] (token_idx [T,W] is the same set for all groups).
    sel_shared = torch.zeros(T, S, dtype=torch.bool)
    for t in range(T):
        for tok in token_idx[t].tolist():
            if 0 <= tok < S:
                sel_shared[t, tok] = True
    for r in range(Hkv):
        sel = sel_shared
        causal = key_pos[None, :] <= qpos[:, None]
        keep = sel & causal
        bias = torch.where(keep, torch.zeros(T, S), torch.full((T, S), NEG))
        kf = k[:, r, :].float()
        vf = v[:, r, :].float()
        for g in range(G):
            h = r * G + g
            s = torch.einsum("td,sd->ts", q[:, h, :].float(), kf) * scale + bias
            p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
            out[:, h, :] = torch.einsum("ts,sd->td", p, vf)
    return out


# ================================================================================================
# (4) seq NOT a multiple of block_size: pad block must score -inf & never be selected;
#     forced LOCAL block always present.
# ================================================================================================
def test_partial_block_and_forced_local():
    print("test (4) non-multiple seqlen: pad scores -inf / never selected; forced-local present")
    dims = SMALL
    S = 30  # 30 / 4 -> 8 blocks, last block (idx 7) holds only tokens 28,29 (partial)
    q, k, v, q_idx, k_idx = _make_inputs(S, dims, seed=4)
    causal_pos = torch.arange(S)
    n_blk = (S + dims.sparse_block_size - 1) // dims.sparse_block_size
    assert n_blk * dims.sparse_block_size != S, "test setup: S must NOT be a multiple of block"

    iscores = msa_index_scores(q_idx, k_idx, dims.index_scale, dims.sparse_block_size, causal_pos)

    # (4a) the padded columns of the LAST block (positions 30,31) never enter the block score:
    #      verify by checking the block-max equals the max over ONLY the real causal tokens.
    last = n_blk - 1
    real_tokens = list(range(last * dims.sparse_block_size, S))  # 28,29
    ok_pad = True
    for r in range(dims.num_key_value_heads):
        # for query S-1 (sees all), recompute the last block's max over real tokens only.
        qf = q_idx[S - 1, r].float()
        kf = k_idx[:, 0, :].float()
        tok_scores = (kf[real_tokens] @ qf) * dims.index_scale
        ref_max = tok_scores.max().item()
        got = float(iscores[S - 1, r, last])
        if not math.isfinite(got) or abs(got - ref_max) > 1e-4:
            ok_pad = False
    _check("partial last block scored over REAL tokens only (pad excluded)", ok_pad)

    # (4b) a fully-padded / future block never selected. For an EARLY query (pos 2, block 0), every
    # block b>=1 is non-causal -> must be -inf and absent from selection.
    block_idx = msa_block_select(
        iscores, dims.sparse_block_size, dims.sparse_topk_blocks, causal_pos,
        force_local=True, force_local_blocks=dims.sparse_local_block,
        force_init_blocks=dims.sparse_init_block,
    )
    # ONE shared set per query [T, topk] (head-collapse) -> index by query only.
    qearly = 2  # in block 0; only block 0 is causally visible
    picked_early = set(int(x) for x in block_idx[qearly].tolist() if x >= 0)
    no_future = not any(b > 0 for b in picked_early)
    _check("future/non-causal blocks never selected (early query)", no_future,
           f"picked={sorted(picked_early)}")

    # (4c) forced LOCAL block (block containing the query) ALWAYS present, every query (one shared set).
    forced_ok = True
    bad = None
    for t in range(S):
        lb = t // dims.sparse_block_size
        picked = set(int(x) for x in block_idx[t].tolist() if x >= 0)
        if lb not in picked:
            forced_ok = False
            bad = (t, lb, sorted(picked))
            break
    _check("forced LOCAL block always present", forced_ok,
           "" if forced_ok else f"missing local block (t,local,picked)={bad}")

    # (4d) the -1 token expansion for a partial block produces no token >= S.
    token_idx = msa_expand_blocks_to_tokens(block_idx, dims.sparse_block_size, seqlen=S)
    max_tok = int(token_idx.max().item())
    _check("block->token expansion never yields token >= S", max_tok < S,
           f"max_token={max_tok} (S={S})")


# ================================================================================================
# (5) causal: query i never attends j > i.
# ================================================================================================
def test_causality():
    print("test (5) causal: query i never attends key j > i")
    dims = SMALL
    S = 48
    q, k, v, q_idx, k_idx = _make_inputs(S, dims, seed=5)

    # Probe: perturb a FUTURE key/value and assert outputs for earlier queries are unchanged.
    out_a, block_idx, token_idx = msa_forward(q, k, v, q_idx, k_idx, dims, return_selection=True)
    j = S - 1  # last position; perturbing it must not affect any query i < j
    k2 = k.clone(); v2 = v.clone()
    k2[j] += 7.0
    v2[j] += 7.0
    # keep the SAME selection inputs (q_idx/k_idx unchanged) so selection is identical;
    # only the main K/V at the future position change.
    out_b = msa_forward(q, k2, v2, q_idx, k_idx, dims)
    delta_earlier = (out_a[:j] - out_b[:j]).abs().max().item()
    _check("perturbing key j leaves all queries i<j unchanged", delta_earlier < 1e-6,
           f"max_delta(i<j)={delta_earlier:.2e}")

    # Block-granular selection legitimately OVER-gathers: the forced local block (and any selected
    # block straddling the diagonal) contains tokens > t. The MSA reference does exactly this and
    # masks them in the attention (test_sparse_attn.py:67-70 `masked_fill_(qi<ki,-inf)`). So the
    # correct causal invariant is: any selected token tok > t lies in the SAME block as t (i.e. is a
    # within-block future token that the consume mask drops) — NEVER a token from a strictly future
    # block. (The "perturbing key j" check above already proved such tokens do not contribute.)
    ok = True
    bad = None
    bsz = dims.sparse_block_size
    for t in range(S):                              # ONE shared token set per query [T, W]
        for tok in token_idx[t].tolist():
            if tok >= 0 and tok > t and (tok // bsz) != (t // bsz):
                ok = False  # a future token from a DIFFERENT (later) block => real leak
                bad = (t, tok, tok // bsz, t // bsz)
                break
    _check("future tokens confined to the query's own block (masked, never a future block)", ok,
           "" if ok else f"future-block leak (t,tok,tokblk,qblk)={bad}")

    # And prove the within-block future tokens are causally masked at the OUTPUT level: rerun the
    # consume with the within-block future K/V perturbed and assert earlier-in-block queries are
    # unchanged (complements the cross-block perturbation check above).
    out_c = msa_forward(q, k2, v2, q_idx, k_idx, dims)
    # query S-2 shares block with the perturbed token S-1; its output must be unchanged (S-1 > S-2).
    delta_inblock = (out_a[S - 2] - out_c[S - 2]).abs().max().item()
    _check("within-block future key masked at output (query S-2 unaffected by key S-1)",
           delta_inblock < 1e-6, f"max_delta={delta_inblock:.2e}")


def main():
    print("=" * 90)
    print("MiniMax-M3 MSA reference — synthetic parity (CPU, fp32)")
    print("=" * 90)
    test_block_select_matches_bruteforce()
    test_gather_equals_dense_when_all_fit()
    test_sparse_differs_from_dense_when_pruned()
    test_partial_block_and_forced_local()
    test_causality()
    print("-" * 90)
    if _FAILS:
        print(f"{len(_FAILS)} assertion(s) FAILED:")
        for f in _FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL ASSERTIONS PASSED")


# pytest entry points (so `pytest` also works if installed).
def test_1_block_select():
    test_block_select_matches_bruteforce(); assert not _FAILS


def test_2_gather_all_fit():
    test_gather_equals_dense_when_all_fit(); assert not _FAILS


def test_3_not_silently_dense():
    test_sparse_differs_from_dense_when_pruned(); assert not _FAILS


def test_4_partial_and_forced():
    test_partial_block_and_forced_local(); assert not _FAILS


def test_5_causal():
    test_causality(); assert not _FAILS


if __name__ == "__main__":
    main()
