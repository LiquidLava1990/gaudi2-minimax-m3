"""Dump the (final_hidden, target_id, position) triples that train + evaluate the
M3 depth-1 draft head, by running the REAL M3 model over a calibration corpus.

WHAT A SHARD CONTAINS (one .pt per sequence-chunk, written to dumps/):
    {
      "hidden":      [S, 6144] bf16   # the POST-final-norm stream feeding lm_head
      "target_ids":  [S]      int64   # argmax(main_logits[t]) == the model's greedy
                                       #   next token == the token at position t+1
      "positions":   [S]      int64   # absolute position of row t in its sequence
    }

THE ALIGNMENT (the one bug that silently kills accept-rate — asserted in the CPU
self-test): row t holds hidden[t] (post-norm hidden at position t) and
target_ids[t] = argmax(main_logits[t]). The main model's greedy argmax at position
t IS the token it would emit at position t+1, so target_ids[t] is "the token at
position t+1". The trainer then fuses prev_hidden[t] with embed(target_ids[t]) and
the draft predicts target_ids[t+1] — exactly the GLM MTP oracle's
draft(hidden[t], tok[t+1]) -> tok[t+2] shape, re-indexed (see
hpu-moe-ext/tests/glm_cpu_mtp_bf16.py:111-118). We dump the raw material; the
shift-by-one fusion lives in the trainer/oracle, NOT here.

WHICH HIDDEN WE DUMP, AND WHY. The serving model's HpuMiniMaxM3Model.forward
returns `self.norm(hidden_states, residual)` (minimax_m3.py:867) — the POST-final-
norm stream — and feeds THAT straight into compute_logits ->
logits_processor(lm_head, hidden) (minimax_m3.py:1051). That post-norm tensor is
what we capture (it is exactly the compute_logits input; the probe taps it at zero
extra compute).

  Note on the draft head's own norm: m3_cpu_mtp.M3SharedHead.forward =
  head(gemma_rms(hidden, norm_w)) — the draft applies its OWN final norm to its
  block's (pre-norm) residual output. So at *draft inference* the draft never
  consumes this dumped tensor as input; the draft's block produces its own hidden.
  This dumped post-norm tensor is a TRAINING REFERENCE, not a draft input. Two
  reasons it is the right thing to store:
    (1) target_ids = argmax(main_logits) is the only label the draft is graded on
        (accept iff draft argmax == target argmax). argmax is invariant to the final
        RMSNorm's positive scale, so the *label* is identical whether the reference
        hidden is pre- or post-norm — storing post-norm changes nothing for the
        primary (cross-entropy-on-target) objective.
    (2) it is the cheap exact tap (the compute_logits input), and it gives the
        optional auxiliary signal a trainer may add: regress the draft's shared_head
        pre-logits hidden onto this post-norm reference (same space the lm_head sees).
  If a trainer instead wants the target's PRE-final-norm residual stream (to regress
  the draft block's pre-norm output directly), capture that via the model's
  M3_LAYER_PROBE / the L59_out hook in minimax_m3.py — NOT this path. Default +
  validated dump = post-norm hidden + target_ids. Stated, not buried.

TWO RUN MODES (BOTH RUN ON CARDS — self_test = NA(card-gated)):
  A) PIGGYBACK an existing M3 HPU serve. Set M3_MTP_DUMP=1 in the serve env; this
     module's install_dump_hooks() wraps the model's forward (to capture positions)
     + compute_logits (to capture the post-norm hidden + argmax target) and tees a
     shard per generate step. Wire it from hpu_m3_ext.register() OR import+install
     in a tiny offline driver. The probe is @torch.compiler.disable + host-sync; it
     breaks lazy/compiled capture, so use it on an EAGER dump pass, never the hot
     lazy serve.
  B) STANDALONE vLLM offline: run_offline_dump() builds LLM(MODEL_PATH, ...),
     install_dump_hooks() onto the loaded model, and .generate()s the corpus. The
     SAME compute_logits tap fires. This is the recommended dump path (no live serve
     to perturb; deterministic greedy). Exact command in HOW_TO_RUN below.

CPU SELF-TEST (card-free, the part this file proves):
    $HOME/.venvs/m3calib/bin/python $HOME/AI/minimax-m3/mtp/dump_m3_hidden.py --self-test
  checks: shard write/read round-trips byte-for-byte; the off-by-one alignment on a
  synthetic (logits, hidden) pair (target_ids[t] == argmax(logits[t]) and it equals
  the planted token[t+1]); finite-value guards reject NaN/Inf hidden; dtype/shape.

CARD COMMAND (mode B, offline — the human runs this; you must NOT):
  see HOW_TO_RUN at the bottom / the StructuredOutput.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Constants imported from the validated CPU core — do NOT redefine (mission rule).
# This file lives next to m3_cpu_mtp.py; make that importable without install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_cpu_mtp import H, VOCAB  # noqa: E402  (H=6144, VOCAB=200064)

DUMP_DIR = os.environ.get(
    "M3_MTP_DUMP_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps")
)

# Default M3 int4 checkpoint (mirrors models/minimax-m3-int4.env MODEL_PATH, under
# MODELS_DIR mounted at /models in the serve container). Overridable for the bf16 dir.
DEFAULT_MODEL_PATH = os.environ.get(
    "M3_MTP_MODEL_PATH", "/models/MiniMax-M3-int4-experts-AutoRound"
)


def _log(msg: str) -> None:
    print(f"[m3-mtp-dump] {msg}", file=sys.stderr, flush=True)


# ======================================================================================
# Pure shard I/O + the alignment/finite logic (CPU-testable; no vllm, no HPU).
# ======================================================================================
def build_shard(
    hidden: torch.Tensor,
    main_logits: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Turn one (hidden, main_logits, positions) batch into a training shard.

    hidden:      [S, H]      (any float dtype; stored as bf16)
    main_logits: [S, VOCAB]  the target model's logits at each row
    positions:   [S]         absolute position of each row in its sequence

    target_ids[t] = argmax(main_logits[t]) — the greedy next token (== token t+1).
    LOUD on shape mismatch / non-finite hidden (never write a silently-garbage shard
    that poisons training — the DSA build-quality bar).
    """
    if hidden.dim() != 2 or hidden.shape[-1] != H:
        raise ValueError(f"hidden must be [S, {H}], got {tuple(hidden.shape)}")
    S = hidden.shape[0]
    if main_logits.dim() != 2 or main_logits.shape[0] != S:
        raise ValueError(
            f"main_logits must be [S={S}, vocab], got {tuple(main_logits.shape)}"
        )
    if main_logits.shape[-1] != VOCAB:
        # Not fatal (a future vocab change), but loud — the trainer assumes M3 VOCAB.
        _log(
            f"WARN main_logits vocab={main_logits.shape[-1]} != expected {VOCAB} "
            "(continuing; verify the checkpoint)."
        )
    if positions.dim() != 1 or positions.shape[0] != S:
        raise ValueError(f"positions must be [S={S}], got {tuple(positions.shape)}")

    h = hidden.detach().to(torch.bfloat16).contiguous().cpu()
    if not torch.isfinite(h.float()).all():
        n_bad = int((~torch.isfinite(h.float())).sum())
        raise ValueError(f"hidden has {n_bad} non-finite values — refusing to write shard")

    target_ids = main_logits.detach().float().argmax(dim=-1).to(torch.int64).cpu()
    pos = positions.detach().to(torch.int64).cpu()
    return {"hidden": h, "target_ids": target_ids, "positions": pos}


def write_shard(shard: dict[str, torch.Tensor], path: str) -> str:
    """Atomically write a shard (.pt). Returns the final path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(shard, tmp)
    os.replace(tmp, path)
    return path


def read_shard(path: str) -> dict[str, torch.Tensor]:
    """Read + validate a shard. Raises on a malformed/corrupt shard."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("hidden", "target_ids", "positions"):
        if k not in d:
            raise ValueError(f"shard {path!r} missing key {k!r} (keys={list(d.keys())})")
    S = d["hidden"].shape[0]
    if d["hidden"].dim() != 2 or d["hidden"].shape[-1] != H:
        raise ValueError(f"shard {path!r}: hidden shape {tuple(d['hidden'].shape)} != [S,{H}]")
    if d["target_ids"].shape != (S,) or d["positions"].shape != (S,):
        raise ValueError(
            f"shard {path!r}: ragged rows hidden[{S}] "
            f"target_ids{tuple(d['target_ids'].shape)} positions{tuple(d['positions'].shape)}"
        )
    return d


def _next_shard_path(rank: int, counter: int) -> str:
    return os.path.join(DUMP_DIR, f"m3mtp_r{rank}_{counter:06d}.pt")


# ======================================================================================
# Mode A/B shared: the monkeypatch that tees the post-norm hidden + target during a
# real M3 generate. Wraps BOTH the model's forward (to learn `positions`, which
# compute_logits is not handed) and compute_logits (which IS handed the exact post-
# norm hidden + lets us compute the target logits). Idempotent, LOUD on bad wiring.
# ======================================================================================
_DUMP_STATE = {
    "installed": False,
    "counter": 0,
    "rows": 0,
    "last_positions": None,   # threaded from forward -> compute_logits
    "rank": 0,
}


def _enabled() -> bool:
    return os.environ.get("M3_MTP_DUMP", "0") in ("1", "true", "True")


def _resolve_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank as _tpr

        return int(_tpr())
    except Exception:  # noqa: BLE001
        return 0


def install_dump_hooks(model) -> bool:
    """Wrap `model.forward` + `model.compute_logits` to tee training shards.

    `model` is an HpuMiniMaxM3ForConditionalGeneration instance. Returns True if the
    hooks were installed (M3_MTP_DUMP enabled and not already installed). Idempotent.
    Raises (LOUD) if the model lacks the expected methods (mis-wired tap).
    """
    if not _enabled():
        _log("M3_MTP_DUMP not set -> dump hooks NOT installed (no-op).")
        return False
    if _DUMP_STATE["installed"]:
        return True
    if not (hasattr(model, "forward") and hasattr(model, "compute_logits")):
        raise RuntimeError(
            "install_dump_hooks: model lacks forward/compute_logits — wrong object? "
            f"(type={type(model)!r})"
        )

    os.makedirs(DUMP_DIR, exist_ok=True)
    _DUMP_STATE["rank"] = _resolve_rank()
    _orig_forward = model.forward
    _orig_compute_logits = model.compute_logits

    only_prefill = os.environ.get("M3_MTP_DUMP_PREFILL_ONLY", "0") in ("1", "true", "True")
    max_rows = int(os.environ.get("M3_MTP_DUMP_MAX_ROWS", "0") or "0")  # 0 = unbounded

    # EAGLE-3 AUX CAPTURE (M3_MTP_DUMP_AUX=1): tee 3 decoder layers' residual-stream outputs
    # (low/mid/high) -> shard 'hidden_aux' [S, 3H], the richer multi-depth feature the n_aux=3 head
    # fuses (fc 3H->H) and feeds back during autoregressive drafting.
    _DUMP_STATE.setdefault("aux_latest", {})
    if os.environ.get("M3_MTP_DUMP_AUX", "0") in ("1", "true", "True"):
        _layers = None
        for _path in ("model.layers", "model.model.layers", "model.model.model.layers", "language_model.model.layers"):
            _obj = model
            try:
                for _a in _path.split("."):
                    _obj = getattr(_obj, _a)
                if _obj is not None and hasattr(_obj, "__len__") and len(_obj) > 4:
                    _layers = _obj
                    break
            except Exception:  # noqa: BLE001
                continue
        if _layers is None:
            _log("AUX capture: could NOT find decoder layers -> aux disabled (check model structure)")
        else:
            _L = len(_layers)
            _aux_idx = [max(0, _L // 4), _L // 2, _L - 1]  # low / mid / high
            _DUMP_STATE["aux_idx"] = _aux_idx

            def _mk_aux_hook(_slot):
                def _hook(_module, _inp, _out):
                    try:
                        _h = _out[0] if isinstance(_out, tuple) else _out
                        if _h is not None:
                            _hh = _h.detach()
                            if _hh.dim() == 3:
                                _hh = _hh.reshape(-1, _hh.shape[-1])
                            _DUMP_STATE["aux_latest"][_slot] = _hh
                    except Exception:  # noqa: BLE001
                        pass
                return _hook

            for _slot, _li in enumerate(_aux_idx):
                _layers[_li].register_forward_hook(_mk_aux_hook(_slot))
            _log(f"EAGLE-3 aux capture ENABLED on layers {_aux_idx} of {_L}")

    @torch.compiler.disable
    def forward_tap(*args, **kwargs):
        # REDESIGNED for current vLLM (compute_logits runs only the LAST prefill row, so we
        # capture FORWARD's full hidden [S,H] instead). signature:
        #   forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw)
        input_ids = kwargs.get("input_ids", args[0] if len(args) >= 1 else None)
        positions = kwargs.get("positions", args[1] if len(args) >= 2 else None)
        hidden = _orig_forward(*args, **kwargs)
        if max_rows and _DUMP_STATE["rows"] >= max_rows:
            return hidden
        try:
            _capture_forward(input_ids, positions, hidden)
        except Exception as ex:  # noqa: BLE001
            import traceback
            _log(f"forward capture error (continuing serve): {ex!r}\n{traceback.format_exc()}")
        return hidden

    # SOFT-KL capture (M3_MTP_DUMP_SOFTK=K>0): tee the target's top-K next-token distribution per decode
    # step (aligned to the gen buffer's latest entry) -> shard 'topk_ids'/'topk_logits' for EAGLE soft-KL
    # distillation. Soft labels GENERALIZE (the head fits the target's distribution); hard-CE argmax
    # MEMORIZES (the 94%-val/30%-serve overfit). train.py reads topk_ids/topk_logits directly.
    _softk = int(os.environ.get("M3_MTP_DUMP_SOFTK", "0") or "0")
    if _softk > 0:
        _orig_compute_logits = model.compute_logits

        @torch.compiler.disable
        def compute_logits_tap(hidden_states, *a, **k):
            logits = _orig_compute_logits(hidden_states, *a, **k)
            try:
                buf = _DUMP_STATE.get("gen_buffer", [])
                if buf and buf[-1][4] is None and logits is not None:
                    lg = logits.detach()
                    if lg.dim() == 3:
                        lg = lg.reshape(-1, lg.shape[-1])
                    if lg.shape[0] >= 1:
                        tv, ti = lg[-1].topk(min(_softk, lg.shape[-1]))   # top-K of the decode row
                        buf[-1][4] = (ti.to(torch.int32).cpu(), tv.to(torch.bfloat16).cpu())
            except Exception:  # noqa: BLE001
                pass
            return logits

        model.compute_logits = compute_logits_tap
        _log(f"SOFT-KL capture ENABLED (top-{_softk} per decode step).")

    model.forward = forward_tap
    _DUMP_STATE["installed"] = True
    _log(
        f"dump hooks INSTALLED on {type(model).__name__} (rank={_DUMP_STATE['rank']}, "
        f"dir={DUMP_DIR}, prefill_only={only_prefill}, max_rows={max_rows or 'inf'})."
    )
    return True


def _capture_forward(input_ids, positions, hidden) -> None:
    """GENERATION capture (matches the SERVE distribution — the 6th bug fix). Accumulate DECODE
    rows (S==1) of one single-stream sequence into a buffer; on a PREFILL (S>1 = a NEW sequence
    started) flush the buffer to a shard. Pairs hidden_t with the NEXT generated token (the
    input_ids of the following decode step). This is the autoregressive distribution the eagle
    proposer actually feeds the head, NOT the teacher-forced prompt text (which gave 97.5% val /
    6.4% serve). REQUIRES single-stream (S==1 decode) + eager (every forward fires)."""
    if input_ids is None or hidden is None:
        return
    ii = input_ids.detach().reshape(-1)
    h = hidden.detach()
    if h.dim() == 3:
        h = h.reshape(-1, h.shape[-1])
    S = h.shape[0]
    buf = _DUMP_STATE.setdefault("gen_buffer", [])
    if S > 1:
        _flush_gen_buffer()           # a new sequence's prefill -> flush the previous generation
        return
    if ii.numel() < 1:
        return
    try:
        p = int(positions.detach().reshape(-1)[0].item()) if positions is not None else len(buf)
    except Exception:  # noqa: BLE001
        p = len(buf)
    aux_row = None
    _al = _DUMP_STATE.get("aux_latest", {})
    if len(_al) >= 3:
        try:
            aux_row = torch.cat([_al[k][0].to(torch.bfloat16).cpu() for k in sorted(_al)], dim=-1)  # [3H]
        except Exception:  # noqa: BLE001
            aux_row = None
    buf.append([h[0].to(torch.bfloat16).contiguous().cpu(), int(ii[0].item()), p, aux_row, None])
    if len(buf) >= 512:
        _flush_gen_buffer()


def _flush_gen_buffer() -> None:
    buf = _DUMP_STATE.get("gen_buffer", [])
    _DUMP_STATE["gen_buffer"] = []
    if len(buf) < 3:                  # need >=3 decode rows for one (hidden_t, tok_{t+1}) pair
        return
    n = len(buf)
    hh = torch.stack([buf[t][0] for t in range(n - 1)])                              # hidden_t
    tgt = torch.tensor([buf[t + 1][1] for t in range(n - 1)], dtype=torch.int64)     # token_{t+1}
    pos = torch.tensor([buf[t][2] for t in range(n - 1)], dtype=torch.int64)
    if not torch.isfinite(hh.float()).all():
        _log("gen flush: non-finite hidden -> skip")
        return
    shard = {"hidden": hh, "target_ids": tgt, "positions": pos}
    # EAGLE-3 aux stack [S-1, 3H] if every row captured the 3 low/mid/high layers
    if all(len(buf[t]) > 3 and buf[t][3] is not None for t in range(n - 1)):
        try:
            shard["hidden_aux"] = torch.stack([buf[t][3] for t in range(n - 1)])
        except Exception:  # noqa: BLE001
            pass
    # SOFT-KL teacher top-K [S-1, K] if captured (topk[t] = topk(compute_logits(hidden_t)), aligned to hidden_t)
    if all(len(buf[t]) > 4 and buf[t][4] is not None for t in range(n - 1)):
        try:
            shard["topk_ids"] = torch.stack([buf[t][4][0] for t in range(n - 1)])
            shard["topk_logits"] = torch.stack([buf[t][4][1] for t in range(n - 1)])
        except Exception:  # noqa: BLE001
            pass
    path = _next_shard_path(_DUMP_STATE["rank"], _DUMP_STATE["counter"])
    write_shard(shard, path)
    _DUMP_STATE["counter"] += 1
    _DUMP_STATE["rows"] += int(hh.shape[0])


def _capture_step(hidden_states: torch.Tensor, logits, only_prefill: bool) -> None:
    """One generate step's worth of rows -> a shard. Skips decode steps if
    only_prefill (a single decode row carries little signal / many tiny shards)."""
    if logits is None:
        return
    h = hidden_states.detach()
    if h.dim() == 3:  # [B, T, H] -> [B*T, H]
        h = h.reshape(-1, h.shape[-1])
    lg = logits.detach()
    if lg.dim() == 3:
        lg = lg.reshape(-1, lg.shape[-1])
    S = h.shape[0]
    if lg.shape[0] != S:
        # compute_logits sometimes runs only the last row of a prefill; align by tail.
        if lg.shape[0] < S:
            h = h[-lg.shape[0]:]
            S = h.shape[0]
        else:
            lg = lg[-S:]

    positions = _DUMP_STATE["last_positions"]
    if positions is None or positions.numel() < S:
        # No positions for these rows (decode after a capture we couldn't tag, or a
        # mismatch) -> synthesize an arange so the shard is still well-formed + loud.
        _log(f"capture: positions unavailable/short for {S} rows -> using arange placeholder")
        positions = torch.arange(S, dtype=torch.int64)
    else:
        positions = positions[-S:]  # last-S rows correspond to the compute_logits rows

    if only_prefill and S <= 1:
        return  # a single decode row — skip in prefill-only mode

    # Drop padded/zero rows (HPU pads prefill batches to a bucket): a genuinely zero
    # hidden row is a pad, not a real token. Keep rows with any signal.
    hf = h.float()
    real = hf.abs().amax(dim=-1) > 0
    if not bool(real.all()):
        keep = real
        h = h[keep]
        lg = lg[keep]
        positions = positions[keep]
        S = h.shape[0]
        if S == 0:
            return  # all pad (warmup/dummy) — nothing to dump

    shard = build_shard(h, lg, positions)
    path = _next_shard_path(_DUMP_STATE["rank"], _DUMP_STATE["counter"])
    write_shard(shard, path)
    _DUMP_STATE["counter"] += 1
    _DUMP_STATE["rows"] += S
    if _DUMP_STATE["counter"] % 25 == 1:
        _log(
            f"wrote {os.path.basename(path)} (S={S}, total_rows={_DUMP_STATE['rows']}, "
            f"shards={_DUMP_STATE['counter']})"
        )


# ======================================================================================
# Mode B driver: standalone vLLM offline dump. RUNS ON CARDS.
# ======================================================================================
_FALLBACK_CORPUS = [
    "The quick brown fox jumps over the lazy dog. ",
    "In a hole in the ground there lived a hobbit. ",
    "List the planets of the solar system in order: ",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "The mitochondrion is the powerhouse of the cell because ",
    "Question: What is the capital of France? Answer: ",
    "Translate to French: 'Good morning, how are you today?'\n",
    "Once upon a time, in a kingdom far away, there was a ",
]


def _load_corpus(path: str | None, limit: int) -> list[str]:
    if path and os.path.isfile(path):
        if path.endswith(".jsonl"):
            # JSONL: one JSON object per line; use its "text" (or "prompt") field.
            # Preserves multi-line prompts (code samples) a raw line-split would shatter.
            import json as _json
            prompts = []
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    obj = _json.loads(ln)
                    t = obj.get("text") or obj.get("prompt")
                    if t and t.strip():
                        prompts.append(t)
            _log(f"corpus: {len(prompts)} prompts from JSONL {path}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            # one prompt per non-empty line; fall back to whole-file if single block.
            prompts = [ln for ln in txt.splitlines() if ln.strip()]
            if len(prompts) <= 1:
                prompts = [txt]
            _log(f"corpus: {len(prompts)} prompts from {path}")
    else:
        prompts = list(_FALLBACK_CORPUS)
        _log(f"corpus: {len(prompts)} built-in fallback prompts (no --corpus file)")
    if limit and limit < len(prompts):
        prompts = prompts[:limit]
    return prompts


def run_offline_dump(
    model_path: str,
    corpus_path: str | None,
    max_new_tokens: int,
    prompt_limit: int,
    vllm_kwargs: dict | None = None,
) -> None:
    """Build an offline vLLM LLM over the M3 checkpoint, install the dump hooks, and
    greedily generate the corpus so the compute_logits tap tees shards. CARD OP."""
    os.environ.setdefault("M3_MTP_DUMP", "1")  # the tap is gated on this
    from vllm import LLM, SamplingParams  # local import: heavy + card-only

    kw = dict(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=int(os.environ.get("M3_MTP_TP", "8")),
        enable_expert_parallel=True,
        max_model_len=int(os.environ.get("M3_MTP_MAXLEN", "8192")),
        gpu_memory_utilization=float(os.environ.get("M3_MTP_GPU_UTIL", "0.9")),
        enforce_eager=True,  # the tap is @compiler.disable + host-sync -> eager
        kv_cache_dtype="fp8_inc",
        block_size=128,
        max_num_seqs=1,  # one sequence per prefill -> each compute_logits tee call is one ordered seq
    )
    if vllm_kwargs:
        kw.update(vllm_kwargs)
    _log(f"building LLM({model_path!r}) kw={ {k: kw[k] for k in ('tensor_parallel_size','max_model_len','enforce_eager')} }")
    llm = LLM(**kw)

    # The tap is installed WORKER-SIDE: minimax_m3._m3_mtp_dump_tee fires inside compute_logits in
    # each TP worker (gated on M3_MTP_DUMP=1) — the only place the per-position post-norm hidden
    # exists under vLLM-V1 multiproc. No reach-into-the-LLM needed (that path cannot see the worker
    # model). _ = _install_on_engine kept in file for reference but intentionally NOT called.
    prompts = _load_corpus(corpus_path, prompt_limit)
    # prompt_logprobs forces logits over EVERY prompt position -> compute_logits receives the full
    # [S,H] prefill hidden -> the worker tee writes one shard per prompt. max_tokens=1 = minimal
    # (the prefill positions ARE the training data; greedy = deterministic argmax target).
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    # This vLLM build's SamplingParams has NO truncate_prompt_tokens -> truncate at the TOKENIZER
    # level and feed token IDs (TokensPrompt dict) so any-length prompt fits under max_model_len.
    _trunc = max(256, int(os.environ.get("M3_MTP_MAXLEN", "8192")) - 64)
    tok = llm.get_tokenizer()
    inputs = []
    for p in prompts:
        ids = tok.encode(p)
        if len(ids) > _trunc:
            ids = ids[:_trunc]
        inputs.append({"prompt_token_ids": ids})
    t0 = time.time()
    _log(f"generating {len(inputs)} prompts (prefill capture, prompt_logprobs=1, max_num_seqs=1, trunc={_trunc})...")
    llm.generate(inputs, sp)
    import glob as _glob
    nshards = len(_glob.glob(os.path.join(DUMP_DIR, "shard_*.pt")))
    _log(f"DONE in {time.time()-t0:.0f}s: {nshards} shards on disk -> {DUMP_DIR}")


def _install_on_engine(llm) -> bool:
    """Best-effort reach into a vLLM v1 LLM to get the model_runner's model + install.
    Returns True on success. (Layout differs across vLLM versions — try the known
    paths; if all fail, the human can install via hpu_m3_ext.register() instead.)"""
    candidates = []
    try:
        # vLLM v1 in-process executor path.
        eng = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        for attr in ("engine_core", "_engine_core"):
            ec = getattr(eng, attr, eng)
            mr = getattr(getattr(ec, "model_executor", None), "driver_worker", None)
            if mr is not None:
                candidates.append(mr)
    except Exception:  # noqa: BLE001
        pass

    # Generic: walk for an object exposing .model_runner.model.
    def _find(obj, depth=0):
        if depth > 6 or obj is None:
            return None
        mr = getattr(obj, "model_runner", None)
        m = getattr(mr, "model", None) if mr is not None else None
        if m is not None and hasattr(m, "compute_logits"):
            return m
        for a in ("engine", "llm_engine", "engine_core", "model_executor",
                  "driver_worker", "worker", "collective_rpc"):
            child = getattr(obj, a, None)
            if child is not None and not callable(child):
                found = _find(child, depth + 1)
                if found is not None:
                    return found
        return None

    m = _find(llm)
    if m is not None:
        return install_dump_hooks(m)
    for c in candidates:
        m = getattr(getattr(c, "model_runner", None), "model", None)
        if m is not None:
            return install_dump_hooks(m)
    return False


# ======================================================================================
# CPU SELF-TEST (card-free) — proves the pure logic this file owns.
# ======================================================================================
def _self_test() -> bool:
    import tempfile

    torch.manual_seed(0)
    checks: list[tuple[str, bool]] = []

    # ---- 1. round-trip: build_shard -> write -> read, byte-for-byte ----
    S = 7
    hidden = torch.randn(S, H, dtype=torch.float32)
    logits = torch.randn(S, VOCAB, dtype=torch.float32)
    positions = torch.arange(S, dtype=torch.int64)
    shard = build_shard(hidden, logits, positions)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "rt.pt")
        write_shard(shard, p)
        back = read_shard(p)
    checks.append(("round-trip hidden bf16 exact",
                   torch.equal(back["hidden"], hidden.to(torch.bfloat16))))
    checks.append(("round-trip target_ids exact",
                   torch.equal(back["target_ids"], logits.argmax(-1).to(torch.int64))))
    checks.append(("round-trip positions exact", torch.equal(back["positions"], positions)))
    checks.append(("shard dtypes (bf16/int64/int64)",
                   back["hidden"].dtype == torch.bfloat16
                   and back["target_ids"].dtype == torch.int64
                   and back["positions"].dtype == torch.int64))
    checks.append(("shard shapes [S,H]/[S]/[S]",
                   back["hidden"].shape == (S, H)
                   and back["target_ids"].shape == (S,)
                   and back["positions"].shape == (S,)))

    # ---- 2. THE off-by-one: plant token[t+1] as the unique argmax of logits[t] ----
    # Construct logits so that argmax(logits[t]) == planted[t]; assert that equals the
    # "token at position t+1" of a synthetic sequence. This is the alignment the
    # trainer relies on (hidden[t] + embed(target[t]) -> predict target[t+1]).
    seq_len = 6
    token_seq = torch.randint(0, VOCAB, (seq_len,), dtype=torch.int64)  # the realized tokens 0..seq_len-1
    # For each row t (0..seq_len-2) the model's greedy at t IS token[t+1].
    rows = seq_len - 1  # rows 0..seq_len-2 have a defined "next token"
    aln_logits = torch.full((rows, VOCAB), -10.0)
    next_tokens = token_seq[1:]  # token[t+1] for t=0..seq_len-2
    aln_logits[torch.arange(rows), next_tokens] = 10.0  # unique max at the next token
    aln_hidden = torch.randn(rows, H)
    aln_pos = torch.arange(rows, dtype=torch.int64)
    aln_shard = build_shard(aln_hidden, aln_logits, aln_pos)
    checks.append(("target_ids[t] == argmax(logits[t])",
                   torch.equal(aln_shard["target_ids"], aln_logits.argmax(-1).to(torch.int64))))
    checks.append(("target_ids[t] == token[t+1] (the off-by-one)",
                   torch.equal(aln_shard["target_ids"], next_tokens)))
    # And the NEGATIVE: target_ids must NOT equal token[t] (would be the off-by-one bug).
    checks.append(("target_ids[t] != token[t] (no zero-shift bug)",
                   not torch.equal(aln_shard["target_ids"], token_seq[:rows])))

    # ---- 3. finite guards: NaN/Inf hidden is rejected LOUDLY ----
    bad = torch.randn(3, H)
    bad[1, 0] = float("nan")
    try:
        build_shard(bad, torch.randn(3, VOCAB), torch.arange(3))
        rejected_nan = False
    except ValueError:
        rejected_nan = True
    checks.append(("rejects NaN hidden", rejected_nan))

    bad2 = torch.randn(3, H)
    bad2[2, 5] = float("inf")
    try:
        build_shard(bad2, torch.randn(3, VOCAB), torch.arange(3))
        rejected_inf = False
    except ValueError:
        rejected_inf = True
    checks.append(("rejects Inf hidden", rejected_inf))

    # ---- 4. shape guards: wrong hidden width / ragged rows rejected ----
    try:
        build_shard(torch.randn(4, H + 1), torch.randn(4, VOCAB), torch.arange(4))
        bad_width_caught = False
    except ValueError:
        bad_width_caught = True
    checks.append(("rejects wrong hidden width", bad_width_caught))

    try:
        build_shard(torch.randn(4, H), torch.randn(5, VOCAB), torch.arange(4))
        ragged_caught = False
    except ValueError:
        ragged_caught = True
    checks.append(("rejects ragged hidden/logits rows", ragged_caught))

    # ---- 5. read_shard rejects a malformed shard ----
    with tempfile.TemporaryDirectory() as td:
        bp = os.path.join(td, "bad.pt")
        torch.save({"hidden": torch.randn(3, H), "target_ids": torch.arange(3)}, bp)  # missing positions
        try:
            read_shard(bp)
            missing_key_caught = False
        except ValueError:
            missing_key_caught = True
    checks.append(("read_shard rejects missing key", missing_key_caught))

    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(
        f"\n{'='*52}\nM3 MTP DUMP CPU SELF-TEST: "
        f"{'GREEN — all pass' if ok else 'RED — FIX ABOVE'}\n{'='*52}"
    )
    return ok


# ======================================================================================
# CLI
# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="M3 MTP hidden-state dump (CPU self-test or card dump)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the card-free CPU self-test (shard I/O + alignment + guards)")
    ap.add_argument("--offline", action="store_true",
                    help="MODE B: build an offline vLLM LLM and dump the corpus (RUNS ON CARDS)")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--corpus", default=None, help="text file, one prompt per line (else built-in)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-limit", type=int, default=0, help="cap #prompts (0=all)")
    args = ap.parse_args()

    if args.self_test or not (args.offline):
        # default action = self-test (card-free, safe to run anywhere)
        return 0 if _self_test() else 1

    # MODE B — card op.
    run_offline_dump(
        model_path=args.model_path,
        corpus_path=args.corpus,
        max_new_tokens=args.max_new_tokens,
        prompt_limit=args.prompt_limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
