"""MiniMax-M3 MSA index K-cache — allocation hook + insert/gather, scoped PREFILL-FIRST.

The MSA index branch needs, per sparse layer, a K-only buffer holding the single SHARED index-key
head (1 head x sparse_index_dim=128) for every token, so the block indexer can score causal blocks.
This is the GQA/MSA analog of hpu-dsa-ext/hpu_dsa_ext/indexer_cache_hpu.py (which caches the MLA
lightning-indexer key). Only ONE extra K stream (no V) — MSA_SPEC.md §3 "keeps a K-only index
buffer alongside the main KV cache".

SCOPE (MVP, PREFILL-FIRST — like DSA's MVP):
  * PREFILL with --no-enable-prefix-caching: the whole sequence is present in one pass (T == S,
    context_lens == 0), so the index K for scoring is computed in-pass from the hidden states and
    NO cache read is needed. The cache here is the bf16 flat buffer + insert (write current step)
    + the runner spec/bind hooks so the buffer EXISTS and is paged-allocated; it is populated for
    correctness/decode-readiness but the prefill wire does not read history from it.
  * PAGED DECODE (read past index-K from the cache to score blocks across the paged history) is a
    documented FOLLOW-UP — see "DECODE (follow-up)" below. Don't over-build: the prefill MVP needs
    only insert + the spec/bind hooks; decode adds gather_blocks-driven scoring.

WHY a spec/bind hook at all (same HPU runner gap DSA hit): the HPU model runner's
``get_kv_cache_spec`` only recognises Attention/MLAAttention/Mamba/FusedMoE by isinstance and DROPS
any other layer needing a cache, so a bare index-cache module never gets memory. We add a spec for
each registered MSA index-cache module (a flat bf16 (num_blocks*block_size, d_idx) buffer) and
tolerate the index/attn same-layer-index collision in ``bind_kv_cache`` on HPU — both mirror the
DSA patches. CARD-FREE / out-of-tree: monkeypatch only, no /opt edit; the insert/gather helpers are
plain torch (CPU-testable).
"""
from __future__ import annotations

import torch

INDEX_CACHE_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Runtime view over one sparse layer's index K-cache (mirrors DSA IndexerKCache,
# trimmed: single shared index-K head, bf16, flat slot convention).
# ---------------------------------------------------------------------------
class IndexerKCache:
    """Thin runtime view over a bound flat index-K cache ``[num_blocks*block_size, d_idx]`` bf16.

    Slot convention ``slot = block_id*block_size + offset`` — identical to the main KV cache, so the
    SAME ``attn_metadata.slot_mapping`` / ``block_list`` drive both. Stores the single shared index
    key head (MSA: K_idx in R^{n x 1 x d_idx}); the wire reads it directly (bf16, no quant)."""

    __slots__ = ("key", "block_size", "head_dim")

    def __init__(self, bound_cache, block_size: int, index_head_dim: int | None = None):
        self.key = _as_key_tensor(bound_cache, index_head_dim)   # [nblk*block_size, d_idx]
        self.block_size = int(block_size)
        self.head_dim = self.key.shape[-1]

    def insert(self, k: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:
        """Write current-step index keys into the cache at flat slots. k: [T,d_idx] or [T,1,d_idx]
        bf16 (single shared head). Mirrors VLLMKVCache.forward (index_copy_(0, slots, k))."""
        if k.dim() == 3:
            k = k.reshape(-1, k.shape[-1])
        if k.dtype != self.key.dtype:
            k = k.to(self.key.dtype)
        slots = slot_mapping.reshape(-1).to(torch.long)
        n = slots.shape[0]
        if k.shape[0] != n:
            k = k[:n]
        self.key.index_copy_(0, slots, k)
        return self.key

    def gather_blocks(self, block_list: torch.Tensor) -> torch.Tensor:
        """Gather all index keys in ``block_list`` -> flat [len*block_size, d_idx] in block order.
        DECODE path (follow-up): assemble a request's paged index-K history for block scoring."""
        if block_list is None or block_list.numel() == 0:
            return self.key.new_zeros((0, self.head_dim))
        blocks = block_list.reshape(-1).to(torch.long)
        unflat = self.key.unflatten(0, (-1, self.block_size))    # [num_blocks, bs, d_idx]
        sel = unflat.index_select(0, blocks)
        return sel.reshape(-1, self.head_dim)


def _as_key_tensor(bound_cache, index_head_dim: int | None = None) -> torch.Tensor:
    """Normalise a bound cache (4-tuple / tensor / list) to the flat 2D [N, d_idx] key tensor,
    narrowing an over-wide (page-uniform) backing to d_idx. Mirrors DSA as_key_tensor."""
    c = bound_cache
    if isinstance(c, (tuple, list)):
        c = c[0]
    if c is None or (isinstance(c, torch.Tensor) and c.numel() == 0):
        raise RuntimeError(
            "[hpu-msa-ext] index K-cache is empty (tensor([])) — the get_kv_cache_spec patch did "
            "not run or did not include this index layer. Call index_cache_msa.apply() in register().")
    if c.dim() == 3 and c.shape[1] == 1:
        c = c.squeeze(1)
    if index_head_dim is not None and 0 < index_head_dim < c.shape[-1]:
        c = c.narrow(c.dim() - 1, 0, index_head_dim)
    return c


# ---------------------------------------------------------------------------
# Registry of MSA index-cache modules (the plugin registers each sparse layer's
# index-cache so the spec hook can find them in the static forward context). Kept
# minimal: a name -> head_dim map the spec hook reads.
# ---------------------------------------------------------------------------
_MSA_INDEX_CACHE_LAYERS: dict[str, int] = {}


def register_index_cache_layer(layer_name: str, index_head_dim: int) -> None:
    """Record that ``layer_name`` (e.g. ``model.layers.N.self_attn.index_kcache``) needs a flat
    bf16 [.., index_head_dim] cache. The model/plugin calls this per sparse layer at build time."""
    _MSA_INDEX_CACHE_LAYERS[layer_name] = int(index_head_dim)


def make_index_spec(block_size: int, head_dim: int, *, alloc_head_size: int | None = None,
                    cache_dtype_str: str | None = None):
    """Flat bf16 spec for one index cache. M3 is GQA (not MLA), but the FullAttentionSpec flat
    allocation path still yields the (num_blocks*block_size, head_size) layout we want; we use a
    plain FullAttentionSpec (bf16, num_kv_heads=1, head_size=d_idx). If a page-uniform alloc width
    is needed to match the main GQA KV spec, ``alloc_head_size`` over-allocates and IndexerKCache
    narrows to d_idx (same trick as DSA)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=alloc_head_size if alloc_head_size is not None else head_dim,
        dtype=INDEX_CACHE_DTYPE,
        cache_dtype_str=cache_dtype_str,
    )


def patch_get_kv_cache_spec() -> None:
    """Wrap HPUModelRunner.get_kv_cache_spec to add a flat bf16 spec for every registered MSA
    index-cache layer in the static forward context. Idempotent. (DSA-analog; for GQA the merge is
    simpler — we mirror the main GQA KV spec's cache_dtype_str so the dict stays merge-uniform.)"""
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    runner_cls = _find_runner_class(hmr)
    if getattr(runner_cls.get_kv_cache_spec, "_msa_index_patched", False):
        return
    _orig = runner_cls.get_kv_cache_spec

    def get_kv_cache_spec(self):  # noqa: ANN001
        spec = _orig(self)
        if not _MSA_INDEX_CACHE_LAYERS:
            return spec
        block_size = self.vllm_config.cache_config.block_size
        # Mirror the main attention spec's cache_dtype_str so the spec dict stays merge-uniform.
        ref_cds = None
        ref_alloc = None
        for s in spec.values():
            cds = getattr(s, "cache_dtype_str", None)
            if cds is not None and getattr(s, "block_size", None) == block_size:
                ref_cds = cds
                ref_alloc = getattr(s, "head_size", None)
                break
        for layer_name, hd in _MSA_INDEX_CACHE_LAYERS.items():
            if layer_name in spec:
                continue
            spec[layer_name] = make_index_spec(
                block_size, hd, alloc_head_size=ref_alloc, cache_dtype_str=ref_cds)
        return spec

    get_kv_cache_spec._msa_index_patched = True
    get_kv_cache_spec._msa_orig = _orig
    runner_cls.get_kv_cache_spec = get_kv_cache_spec


def patch_bind_kv_cache() -> None:
    """Tolerate the index/attn same-layer-index collision on HPU (the DSA bind_kv_cache patch). The
    index cache and the attn of the same decoder layer share a layer index; once the index cache is
    a real KV layer they collide and stock bind_kv_cache raises NotImplementedError on HPU. We do
    the cuda_alike ``pass`` (append all) then bind to the forward context, exactly as DSA does."""
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    if getattr(hmr, "_msa_bind_kv_cache_patched", False):
        return
    _orig_bind = hmr.bind_kv_cache

    def bind_kv_cache(kv_caches, forward_context, runner_kv_caches, num_attn_module=1):  # noqa: ANN001
        try:
            return _orig_bind(kv_caches, forward_context, runner_kv_caches, num_attn_module)
        except NotImplementedError:
            return _bind_tolerant(kv_caches, forward_context, runner_kv_caches, num_attn_module)

    bind_kv_cache._msa_orig = _orig_bind
    hmr.bind_kv_cache = bind_kv_cache
    hmr._msa_bind_kv_cache_patched = True
    try:
        from vllm_gaudi.v1.worker import hpu_worker as _hw
        _hw.bind_kv_cache = bind_kv_cache   # the worker imports its own copy (DSA note)
    except Exception:  # noqa: BLE001
        pass


def _bind_tolerant(kv_caches, forward_context, runner_kv_caches, num_attn_module=1):
    from collections import defaultdict
    from vllm.model_executor.models.utils import extract_layer_index

    runner_kv_caches.clear()
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)
    for layer_index in sorted(index2name.keys()):
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def _find_runner_class(hmr):
    for name in ("HPUModelRunner", "HpuModelRunner", "HPUModelRunnerBase", "HpuModelRunnerBase"):
        cls = getattr(hmr, name, None)
        if cls is not None and "get_kv_cache_spec" in cls.__dict__:
            return cls
    import inspect
    for _n, cls in inspect.getmembers(hmr, inspect.isclass):
        if cls.__module__ == hmr.__name__ and "get_kv_cache_spec" in cls.__dict__:
            return cls
    raise RuntimeError(
        "[hpu-msa-ext] could not locate the HPU model-runner class defining get_kv_cache_spec")


def apply() -> None:
    """Install both runner patches. Call once from the plugin register() BEFORE the runner builds
    the KV cache. Idempotent. PREFILL-FIRST: only the spec+bind hooks + insert are needed; decode
    gather is a follow-up."""
    patch_get_kv_cache_spec()
    patch_bind_kv_cache()


# ==================================================================================================
# DECODE (follow-up — documented, NOT built in the MVP)
# ==================================================================================================
# For paged decode the wire would, per request: gather the request's index-K history from this cache
# (``IndexerKCache.gather_blocks(block_list_r)`` -> [ctx, d_idx]), build the index Q for the current
# step, score blocks over [0, ctx), and feed block_indexer_topk with abs_pos == ctx-1. The main KV
# is gathered from the standard GQA paged cache (no MLA materialization). This mirrors DSA's
# sparse_mla_wire decode branch (per_request_block_ids + _gather_request_latent) but over real GQA
# K/V instead of MLA latent. The prefill MVP above is the validated-first path; decode is gated
# behind the same MSA_READY flag and added once prefill is on-card-green.
