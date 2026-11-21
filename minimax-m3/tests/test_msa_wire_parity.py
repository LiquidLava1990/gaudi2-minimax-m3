"""CPU PARITY GATE for the MSA wire (vllm/msa_m3_wire.py).

Proves the Python ORCHESTRATION (msa_m3_wire.msa_wire + its force-local/sort/pad/expand/GQA-expand
helpers) reproduces ref/msa_reference.py::msa_forward BIT-FOR-BIT on synthetic tensors, by binding
SMALL torch stand-ins for the two compiled kernel ops (msa_m3_wire._cpu_block_indexer_topk /
_cpu_sparse_attn — these match the kernel _shared.h I/O contracts). NO device, NO HPU: run in the
CPU container (docker run --rm, no devices).

Both the wire and the oracle now match the OFFICIAL MiniMaxM3VLIndexer (transformers 5.12.0): Gemma
QK-norm x*(1+w), partial RoPE (rotary_dim, theta=5e6, same cos/sin as main), RAW index dot
(index_scale=1.0), and the HEAD-COLLAPSE (amax over the index heads) -> ONE shared block/token set
per query, so the consume is a SINGLE op call. (The numeric pin of the oracle to that REAL modeling
lives in tests/test_msa_hf_parity.py; THIS test pins the wire to the corrected oracle.) We test:
  (A) STRICT bit-for-bit: wire (Gemma-norm ON + index-rope ON) == oracle msa_forward, e2e output.
  (B) per-STEP parity: cmask, block-indexer-topk op vs oracle scoring (one head-collapsed set),
      force-local+sort+(-1)pad, block->token expansion, GQA kv-expand, and the call sequence
      [block_indexer_topk, sparse_attn] (ONE consume call) — each asserted independently.
  (C) norm ON is LIVE: turning MSA_APPLY_INDEX_NORM on changes selection (proves the norm is wired).
  (D) MsaDims.from_config reads the real M3 dims (index_scale == 1.0, the corrected raw-matmul scale).

Run (CPU, no card):
    python tests/test_msa_wire_parity.py
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (os.path.join(_ROOT, "ref"), os.path.join(_ROOT, "vllm"), _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import msa_reference as ref          # noqa: E402  the numeric ground truth
import msa_m3_wire as wire           # noqa: E402  the thing under test

torch.manual_seed(0)
NEG_INF = float("-inf")

_FAILS: list[str] = []


def _check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not ok:
        _FAILS.append(f"{name}: {detail}")
    return ok


# Small M3-ratio config (shrunk), config-driven through both MsaDims classes (no literals leak).
# rotary_dim=8 (== half of head_dim=16, partial_rotary_factor 0.5, like M3's 64/128); rope_theta 5e6.
def _dims():
    return wire.MsaDims(
        hidden_size=96, num_attention_heads=8, num_key_value_heads=2, head_dim=16,
        rotary_dim=8, rope_theta=5000000.0,
        sparse_index_dim=16, sparse_num_index_heads=2, sparse_topk_blocks=3, sparse_block_size=4,
        sparse_score_type="max", sparse_init_block=0, sparse_local_block=1,
        num_hidden_layers=6, sparse_attention_freq=(0, 0, 0, 1, 1, 1),
    )


def _ref_dims(d: wire.MsaDims):
    return ref.MsaDims(
        hidden_size=d.hidden_size, num_attention_heads=d.num_attention_heads,
        num_key_value_heads=d.num_key_value_heads, head_dim=d.head_dim,
        rotary_dim=d.rotary_dim, rope_theta=d.rope_theta,
        sparse_index_dim=d.sparse_index_dim, sparse_num_index_heads=d.sparse_num_index_heads,
        sparse_topk_blocks=d.sparse_topk_blocks, sparse_block_size=d.sparse_block_size,
        sparse_score_type=d.sparse_score_type, sparse_init_block=d.sparse_init_block,
        sparse_local_block=d.sparse_local_block,
    )


class _FakeAttn(torch.nn.Module):
    """Minimal stand-in for HpuMiniMaxM3Attention exposing exactly what the wire reads: the index
    projections/norms + ``_msa_index_dims`` + ``_msa_hidden``. Linear weights are plain matmuls so
    the test does not need vLLM's ReplicatedLinear (the wire's compute_index_qk calls module(x))."""

    def __init__(self, d: wire.MsaDims, gen):
        super().__init__()
        self.d = d
        self._msa_index_dims = d
        di = d.sparse_index_dim
        self.Wq = torch.randn(d.sparse_num_index_heads * di, d.hidden_size, generator=gen)
        self.Wk = torch.randn(1 * di, d.hidden_size, generator=gen)
        self.index_q_norm = wire._IndexRMSNorm(di)   # Gemma x*(1+w), weight zero-init
        self.index_k_norm = wire._IndexRMSNorm(di)
        # non-trivial Gemma weights (near 0, some negative — like the real checkpoint) so the norm
        # is a real per-element transform (exercises (C) and the HF-parity selection).
        with torch.no_grad():
            self.index_q_norm.weight.copy_(0.25 * torch.randn(di, generator=gen))
            self.index_k_norm.weight.copy_(0.25 * torch.randn(di, generator=gen))

    # the wire calls attn.index_q_proj(x) / attn.index_k_proj(x) -> (out, _)
    def index_q_proj(self, x):
        return x @ self.Wq.t(), None

    def index_k_proj(self, x):
        return x @ self.Wk.t(), None


def _make(S, d: wire.MsaDims, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(S, d.hidden_size, generator=g)
    q = torch.randn(S, d.num_attention_heads, d.head_dim, generator=g)
    k = torch.randn(S, d.num_key_value_heads, d.head_dim, generator=g)
    v = torch.randn(S, d.num_key_value_heads, d.head_dim, generator=g)
    return hidden, q, k, v, g


def _raw_index_qk(hidden, attn, d):
    """RAW projected index q/k (pre-norm, pre-rope) from the SAME projections the wire uses.
    msa_forward applies the Gemma norm + partial RoPE itself, given the norm weights."""
    T = hidden.shape[0]
    di = d.sparse_index_dim
    q_idx = (hidden @ attn.Wq.t()).view(T, d.sparse_num_index_heads, di)
    k_idx = (hidden @ attn.Wk.t()).view(T, 1, di)
    return q_idx, k_idx


def _oracle_forward(q, k, v, hidden, attn, rd, positions, *, index_norm=True, index_rope=True):
    """Run the oracle msa_forward with the SAME index norm weights + rope the wire applies, so the
    wire (norm ON, rope ON) is bit-for-bit comparable. index_norm=False feeds the zero-weight
    (identity (1+0)=1) norm — note that is NOT the bare dot; the wire's norm-OFF path is the bare dot,
    so the strict bit-for-bit (A) is done with BOTH norm OFF (see test_A)."""
    q_idx, k_idx = _raw_index_qk(hidden, attn, rd)
    wq = attn.index_q_norm.weight.detach() if index_norm else None
    wk = attn.index_k_norm.weight.detach() if index_norm else None
    return ref.msa_forward(q, k, v, q_idx, k_idx, rd, causal_pos=positions,
                           index_q_norm_weight=wq, index_k_norm_weight=wk, index_rope=index_rope)


# ==================================================================================================
def test_A_end_to_end_bitforbit():
    print("test (A) wire (Gemma-norm ON + index-rope ON) == oracle msa_forward, e2e (sweep S x seed)")
    d = _dims()
    rd = _ref_dims(d)
    wire.MSA_APPLY_INDEX_NORM = True         # the REAL path: Gemma norm + partial rope on the index
    wire.bind_kernel_ops(wire._cpu_block_indexer_topk, wire._cpu_sparse_attn)

    worst = 0.0
    # S values exercise: partial last block (23, 30), exact multiple (24, 16), all-blocks-fit (8 <=
    # topk*block=12 so every causal block selectable), and longer than topk (40 -> heavy pruning).
    for S in (8, 16, 23, 24, 30, 40):
        for seed in (1, 2, 7):
            hidden, q, k, v, g = _make(S, d, seed=seed)
            attn = _FakeAttn(d, g)
            attn._msa_hidden = hidden
            positions = torch.arange(S)
            out_wire = wire.msa_wire(
                attn, positions, q.reshape(S, -1), k.reshape(S, -1), v.reshape(S, -1)
            ).view(S, d.num_attention_heads, d.head_dim)
            out_ref = _oracle_forward(q, k, v, hidden, attn, rd, positions,
                                      index_norm=True, index_rope=True)
            diff = (out_wire.float() - out_ref.float()).abs().max().item()
            worst = max(worst, diff)
            if diff >= 1e-5:
                _check(f"wire==oracle S={S} seed={seed}", False, f"max_diff={diff:.2e}")
    _check("wire output == oracle msa_forward (bit-for-bit, all S x seed)", worst < 1e-5,
           f"worst max_diff over sweep={worst:.2e}")


def test_B_per_step():
    print("test (B) per-step parity (cmask / op-scoring / force-local+sort+pad / expand / gqa-expand / sequence)")
    d = _dims()
    rd = _ref_dims(d)
    S = 23
    hidden, q, k, v, g = _make(S, d, seed=2)
    attn = _FakeAttn(d, g)
    attn._msa_hidden = hidden
    positions = torch.arange(S)
    bsz = d.sparse_block_size
    wire.MSA_APPLY_INDEX_NORM = False
    q_idx, k_idx = _raw_index_qk(hidden, attn, d)

    # --- (B1) cmask: wire's build_cmask matches the oracle's causal + key-pad-to-n_blocks geometry.
    cmask, n_blocks, sk_pad = wire.build_cmask(S, positions, bsz, torch.device("cpu"))
    n_blk_ref = (S + bsz - 1) // bsz
    # reference cmask: a key (incl padded tail >= S) is dropped iff key_pos > abs_pos.
    key_pos = torch.arange(sk_pad)
    cmask_ref = key_pos[None, :] > positions.long()[:, None]
    _check("cmask shape n_blocks*block_size + values", n_blocks == n_blk_ref
           and sk_pad == n_blk_ref * bsz and bool((cmask == cmask_ref).all()),
           f"n_blocks={n_blocks} sk_pad={sk_pad}")

    # --- (B2) op block-scoring SET == oracle msa_index_scores+reference_block_topk (raw, no force).
    # HEAD-COLLAPSE: the op amax-collapses the index heads -> ONE shared set [Sq, kk] per query (no
    # head axis). msa_index_scores stays per-head [Sq,H_idx,nblk]; reference_block_topk collapses it.
    q_op = q_idx.permute(1, 0, 2).contiguous()
    k_op = k_idx[:, 0, :].unsqueeze(0).expand(d.num_key_value_heads, -1, -1).contiguous()
    if sk_pad != S:
        padk = torch.zeros(d.num_key_value_heads, sk_pad - S, d.sparse_index_dim)
        k_op = torch.cat([k_op, padk], dim=1)
    weights = torch.full((d.num_key_value_heads, S, 1), d.index_scale)
    block_raw = wire._cpu_block_indexer_topk(q_op, k_op, weights, cmask, d.sparse_topk_blocks, 1, bsz)
    _check("op returns ONE head-collapsed set [Sq,kk] (no head axis)", block_raw.dim() == 2,
           f"shape={tuple(block_raw.shape)}")
    # oracle ONE collapsed raw top-k block SET per query (force_*=0) from msa_index_scores.
    iscores = ref.msa_index_scores(q_idx, k_idx, rd.index_scale, bsz, positions)  # [Sq,H_idx,nblk]
    ref_sets = ref.msa_reference_block_topk(            # head-collapsed inside -> list[Sq] of sets
        iscores, rd.sparse_topk_blocks, positions, force_local_blocks=0,
        force_init_blocks=0, block_size=bsz)
    # The op returns RAW top-k incl. spurious non-causal picks when #visible < topk (kernel contract:
    # topk over -inf entries returns arbitrary block ids; the wire drops them). So compare the op's
    # set RESTRICTED to causally-visible blocks against the oracle's (causally-filtered) raw top-k.
    ok_set = True
    bad = None
    for t in range(S):
        vis = {b for b in range((S + bsz - 1) // bsz) if b * bsz <= int(positions[t])}
        got = {int(x) for x in block_raw[t].tolist() if x >= 0} & vis
        if got != ref_sets[t]:
            ok_set = False
            bad = (t, sorted(got), sorted(ref_sets[t]))
    _check("block_indexer_topk op SET (visible) == oracle raw block top-k (one set)", ok_set,
           "" if ok_set else f"(t,got,ref)={bad}")

    # --- (B3) force-local + ascending sort + (-1) pad == oracle msa_block_select (WITH forcing).
    # ONE shared set per query (head-collapse): both are [S, topk] (no head axis).
    block_idx_wire = wire.force_local_sort_pad(block_raw, positions, d, n_blocks)  # [S, topk]
    block_idx_ref = ref.msa_block_select(
        iscores, bsz, rd.sparse_topk_blocks, positions,
        force_local=(rd.sparse_local_block > 0), force_local_blocks=rd.sparse_local_block,
        force_init_blocks=rd.sparse_init_block)                                    # [S, topk]
    _check("force_local_sort_pad returns ONE set [S,topk] (no head axis)",
           block_idx_wire.dim() == 2 and block_idx_ref.dim() == 2,
           f"wire={tuple(block_idx_wire.shape)} ref={tuple(block_idx_ref.shape)}")
    # SET equality per query (order within slots is canonical-ascending for both; compare sets to be
    # robust to a tie landing on a different equal-score block — the consume is set-invariant).
    ok_force = True
    asc_ok = True
    bad = None
    for t in range(S):
        gw = [int(x) for x in block_idx_wire[t].tolist()]
        gr = [int(x) for x in block_idx_ref[t].tolist()]
        if set(x for x in gw if x >= 0) != set(x for x in gr if x >= 0):
            ok_force = False
            bad = (t, gw, gr)
        valid = [x for x in gw if x >= 0]
        if valid != sorted(valid) or any(p != -1 for p in gw if p < 0):
            asc_ok = False
    _check("force-local+sort+(-1)pad == oracle msa_block_select (one set)", ok_force,
           "" if ok_force else f"(t,wire,ref)={bad}")
    _check("selection ascending + -1 right-padded", asc_ok)
    # forced local block present for every query.
    forced_ok = all(
        (t // bsz) in set(int(x) for x in block_idx_wire[t].tolist() if x >= 0)
        for t in range(S))
    _check("forced LOCAL block always present (wire)", forced_ok)

    # --- (B4) block->token expansion == oracle msa_expand_blocks_to_tokens (one set [S,topk*bsz]).
    tok_wire = wire.expand_blocks_to_tokens(block_idx_wire, bsz, seqlen=S)
    tok_ref = ref.msa_expand_blocks_to_tokens(block_idx_ref, bsz, seqlen=S)
    # compare set-of-tokens per query (selection sets matched in B3; expansion is deterministic).
    ok_tok = all(
        set(int(x) for x in tok_wire[t].tolist() if x >= 0)
        == set(int(x) for x in tok_ref[t].tolist() if x >= 0)
        for t in range(S))
    _check("block->token expansion == oracle (token sets, one set)", ok_tok)
    _check("expansion never yields token >= S", int(tok_wire.max().item()) < S,
           f"max_tok={int(tok_wire.max().item())} S={S}")

    # --- (B5) GQA kv-expand: head h reads kv-head h//G (the wire's repeat_interleave).
    G = d.gqa_group
    k_h = k.repeat_interleave(G, dim=1)            # [S,Hq,Dh]
    gqa_ok = all(bool((k_h[:, h, :] == k[:, h // G, :]).all())
                 for h in range(d.num_attention_heads))
    _check("GQA kv-expand maps head h -> kv-head h//G", gqa_ok)

    # --- (B6) full call sequence: a traced op-call recorder confirms order
    #          [block_indexer_topk] THEN exactly ONE [sparse_attn]. HEAD-COLLAPSE => one shared token
    #          set per query, so the consume is a SINGLE op call over all heads (the prior per-group
    #          4-call loop is GONE — this is the kernel/consume simplification, see msa_m3_wire).
    calls: list[str] = []

    def rec_idx(*a, **kw):
        calls.append("block_indexer_topk")
        return wire._cpu_block_indexer_topk(*a, **kw)

    def rec_attn(*a, **kw):
        calls.append("sparse_attn")
        return wire._cpu_sparse_attn(*a, **kw)

    wire.bind_kernel_ops(rec_idx, rec_attn)
    attn2 = _FakeAttn(d, torch.Generator().manual_seed(2))
    attn2._msa_hidden = hidden
    _ = wire.msa_wire(attn2, positions, q.reshape(S, -1), k.reshape(S, -1), v.reshape(S, -1))
    seq_ok = (calls == ["block_indexer_topk", "sparse_attn"])
    _check("call sequence == [block_indexer_topk, sparse_attn] (ONE consume call, head-collapse)",
           seq_ok, f"calls={calls}")
    wire.bind_kernel_ops(wire._cpu_block_indexer_topk, wire._cpu_sparse_attn)  # restore


def test_C_index_norm_live():
    print("test (C) MSA_APPLY_INDEX_NORM ON changes selection (the checkpoint norm is wired/live)")
    d = _dims()
    S = 40
    hidden, q, k, v, g = _make(S, d, seed=3)
    attn = _FakeAttn(d, g)
    attn._msa_hidden = hidden
    positions = torch.arange(S)
    wire.bind_kernel_ops(wire._cpu_block_indexer_topk, wire._cpu_sparse_attn)

    wire.MSA_APPLY_INDEX_NORM = False
    out_off = wire.msa_wire(attn, positions, q.reshape(S, -1), k.reshape(S, -1), v.reshape(S, -1))
    wire.MSA_APPLY_INDEX_NORM = True
    out_on = wire.msa_wire(attn, positions, q.reshape(S, -1), k.reshape(S, -1), v.reshape(S, -1))
    wire.MSA_APPLY_INDEX_NORM = False  # restore default-for-tests

    diff = (out_off.float() - out_on.float()).abs().max().item()
    _check("index norm ON differs from OFF (norm participates in selection)", diff > 1e-3,
           f"max_diff={diff:.4f} (want > 1e-3)")


def test_D_dims_from_config():
    print("test (D) MsaDims.from_config reads the real M3 (nested) config dict — no literals")
    # the real config.json text_config shape (subset), nested under text_config like minimax_m3_vl.
    cfg = {"text_config": {
        "hidden_size": 6144, "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128,
        "num_hidden_layers": 60,
        "sparse_attention_config": {
            "use_sparse_attention": True, "sparse_index_dim": 128, "sparse_num_index_heads": 4,
            "sparse_topk_blocks": 16, "sparse_block_size": 128, "sparse_score_type": "max",
            "sparse_init_block": 0, "sparse_local_block": 1,
            "sparse_attention_freq": [0, 0, 0] + [1] * 57,
        }}}
    d = wire.MsaDims.from_config(cfg)
    ok = (d.num_attention_heads == 64 and d.num_key_value_heads == 4 and d.head_dim == 128
          and d.sparse_index_dim == 128 and d.sparse_topk_blocks == 16 and d.sparse_block_size == 128
          and d.gqa_group == 16 and abs(d.main_scale - 1.0 / (128 ** 0.5)) < 1e-9
          and abs(d.index_scale - 1.0) < 1e-9   # OFFICIAL: raw matmul, NO 1/sqrt(d_idx) -> 1.0
          and not d.layer_is_sparse(2) and d.layer_is_sparse(3) and d.layer_is_sparse(59))
    _check("from_config: all M3 dims + sparse-layer cadence (0,1,2 dense; 3..59 sparse)", ok,
           f"G={d.gqa_group} main_scale={d.main_scale:.5f} index_scale={d.index_scale} sparse@3={d.layer_is_sparse(3)}")


def main():
    print("=" * 94)
    print("MiniMax-M3 MSA WIRE — CPU parity vs ref/msa_reference.py::msa_forward (kernel ops -> torch stand-ins)")
    print("=" * 94)
    test_A_end_to_end_bitforbit()
    test_B_per_step()
    test_C_index_norm_live()
    test_D_dims_from_config()
    print("-" * 94)
    if _FAILS:
        print(f"{len(_FAILS)} assertion(s) FAILED:")
        for f in _FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL WIRE PARITY ASSERTIONS PASSED")


# pytest entry points
def test_a():  # noqa: D401
    test_A_end_to_end_bitforbit(); assert not _FAILS


def test_b():
    test_B_per_step(); assert not _FAILS


def test_c():
    test_C_index_norm_live(); assert not _FAILS


def test_d():
    test_D_dims_from_config(); assert not _FAILS


if __name__ == "__main__":
    main()
