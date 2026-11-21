"""Build the M3 MTP calibration corpus for the hidden-state dump.

The draft head trains on M3's REAL interactive distribution: it sees the target's
hidden states at every position and learns to predict the next token. So the
corpus must look like what M3 actually decodes in this project's use:
  - CODE      completion (plain + FIM <fim_prefix>/<fim_suffix>/<fim_middle>),
  - AGENTIC   tool-call traces (<tool_call> ... </tool_call> + JSON args + results),
  - REASONING chat with <think> ... </think> then an answer.

Why a *mixed code+reasoning* corpus (not a generic web pile): a depth-1 MTP draft
only pays off on the tokens you actually generate. M3's single-stream path here is
agentic coding + reasoning, so we over-sample exactly that. The accept-rate of the
trained head is bounded by how well the calibration distribution matches serve-time
traffic — a mismatched corpus trains a draft that the verifier rejects, killing the
~1.70x. Hence: real local source (authentic M3-tokenizer code statistics) + a
deterministic synthetic scaffold to guarantee count, balance, and length coverage.

Output: $HOME/AI/minimax-m3/mtp/calib/calib.jsonl   (one JSON obj per line)
  {"id","kind","n_tokens","text"}     kind in {code, fim, agentic, reasoning}

Lengths are measured/trimmed with the REAL M3 tokenizer
(/mnt/data2/minimax-m3-bf16, AutoTokenizer trust_remote_code). Fully deterministic
(fixed seed). No network, no large dataset download — only local files + templates.

CPU self-test (card-free, ~a few s after tokenizer load):
  $HOME/.venvs/m3calib/bin/python $HOME/AI/minimax-m3/mtp/build_calib.py --self-test

Normal build:
  $HOME/.venvs/m3calib/bin/python $HOME/AI/minimax-m3/mtp/build_calib.py
"""
from __future__ import annotations

import os

# Pin tokenizer threading BEFORE transformers/tokenizers is imported anywhere.
# The HF fast tokenizer (tokenizers Rust + rayon) can wedge a worker thread in a
# futex when its thread pool contends with other threads / forked siblings; on a
# CPU-only, batch-encode workload we want none of that parallelism. This MUST be
# set before the first `import transformers`. (We import transformers lazily in
# load_tokenizer, so setting it here at module top is early enough.)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never hit the network for a local dir

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

# --- paths / knobs ---------------------------------------------------------
M3_MODEL_DIR = "/mnt/data2/minimax-m3-bf16"
MTP_DIR = "$HOME/AI/minimax-m3/mtp"
CALIB_DIR = os.path.join(MTP_DIR, "calib")
CALIB_PATH = os.path.join(CALIB_DIR, "calib.jsonl")

# Real local trees we are allowed to read as authentic code/prose material.
SOURCE_ROOTS = [
    "$HOME/AI/minimax-m3",
    "$HOME/AI/hpu-m3-ext",
    "$HOME/AI/hpu-moe-ext",
    "$HOME/AI/hpu-dsa-ext",
    "$HOME/AI/_src/vllm_models",
]
CODE_EXTS = (".py",)
PROSE_EXTS = (".md",)

SEED = 1337
TARGET_N = 320          # >= 256 required; we aim higher then balance/trim
MAX_TOKENS = 8192       # hard cap per sample (M3 interactive window we target)
MIN_TOKENS = 24         # drop trivially short fragments
# We want a real spread, not all-short and not all-max. Soft bucket targets
# (short = code-completion-ish, long = reasoning/agentic transcripts).
LEN_BUCKETS = [
    ("short", 24, 256),
    ("mid", 256, 1024),
    ("long", 1024, 4096),
    ("xlong", 4096, MAX_TOKENS),
]

# M3 special tokens (ids verified against /mnt/data2/minimax-m3-bf16 tokenizer).
T_FIM_PREFIX = "<fim_prefix>"
T_FIM_MIDDLE = "<fim_middle>"
T_FIM_SUFFIX = "<fim_suffix>"
T_FILENAME = "<filename>"
T_THINK = "<think>"
T_THINK_END = "</think>"
T_TOOLCALL = "<tool_call>"
T_TOOLCALL_END = "</tool_call>"


# ---------------------------------------------------------------------------
# Tokenizer (real M3 tokenizer on the CPU venv).
# ---------------------------------------------------------------------------
def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(M3_MODEL_DIR, trust_remote_code=True)
    return tok


def n_tokens(tok, text: str) -> int:
    # No special tokens added: we count the *content* the model would actually
    # see mid-stream (the dump prepends BOS itself / the head trains per-position).
    return len(tok(text, add_special_tokens=False)["input_ids"])


def trim_to_tokens(tok, text: str, max_tokens: int) -> str:
    """Trim text to <= max_tokens by decoding a truncated id slice. Cheap and
    exact in token count (decode of a prefix slice). Single-sample helper —
    the bulk build uses encode_batch() instead (see below)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens])


def encode_batch(tok, texts: list[str]) -> list[list[int]]:
    """Tokenize a whole list in ONE Rust call.

    Why batch (not a Python for-loop of tok(text)): on this box (Python 3.14 +
    tokenizers 0.22.2) a long tight Python loop that calls the fast tokenizer
    per-sample can intermittently wedge a rayon worker thread in a futex and the
    encode never returns. A single batch call does all the work inside one Rust
    invocation and is both faster and deterministic. Verified: 359 samples in
    ~0.7s vs an intermittent multi-minute hang for the per-sample loop."""
    if not texts:
        return []
    enc = tok(texts, add_special_tokens=False)
    return enc["input_ids"]


def trim_overcap(tok, texts: list[str], ids_list: list[list[int]], max_tokens: int):
    """Given texts + their (batch) ids, return (trimmed_texts, exact_n_tokens).

    Only the over-cap samples are decoded+re-encoded (a handful), and that tiny
    set is re-encoded as ONE batch so we never re-enter the per-sample loop."""
    trimmed = list(texts)
    counts = [len(ids) for ids in ids_list]
    over_idx = [i for i, ids in enumerate(ids_list) if len(ids) > max_tokens]
    if over_idx:
        decoded = [tok.decode(ids_list[i][:max_tokens]) for i in over_idx]
        recounts = encode_batch(tok, decoded)  # exact post-trim token counts
        for k, i in enumerate(over_idx):
            trimmed[i] = decoded[k]
            counts[i] = len(recounts[k])
    return trimmed, counts


# ---------------------------------------------------------------------------
# Real-source harvesting.
# ---------------------------------------------------------------------------
def _iter_source_files(roots: Iterable[str], exts: tuple) -> list[str]:
    out: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # skip noise
            dirnames[:] = [
                d for d in dirnames
                if d not in (".git", "__pycache__", "node_modules", ".venv", "calib")
            ]
            for fn in filenames:
                if fn.endswith(exts):
                    out.append(os.path.join(dirpath, fn))
    out.sort()  # deterministic order before shuffle
    return out


def _read_text(path: str, cap_chars: int = 60_000) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            data = f.read(cap_chars + 1)
    except (OSError, UnicodeDecodeError):
        return None
    if not data.strip():
        return None
    return data[:cap_chars]


@dataclass
class Sample:
    kind: str
    text: str


def harvest_code(rng: random.Random, max_files: int) -> list[Sample]:
    """Real .py files as plain code-completion material (the model's code stats)."""
    files = _iter_source_files(SOURCE_ROOTS, CODE_EXTS)
    rng.shuffle(files)
    samples: list[Sample] = []
    for path in files:
        if len(samples) >= max_files:
            break
        txt = _read_text(path)
        if txt is None or len(txt) < 80:
            continue
        rel = os.path.basename(path)
        # Prepend a filename banner the way code models see it (helps the draft
        # condition on "this is python file X").
        body = f"{T_FILENAME}{rel}\n{txt}"
        samples.append(Sample("code", body))
    return samples


def harvest_fim(rng: random.Random, max_files: int) -> list[Sample]:
    """Build FIM (fill-in-the-middle) examples from real code: cut a middle span
    and present prefix/suffix/middle the way the model's FIM tokens expect. This
    is the dominant shape of *editor* code completion."""
    files = _iter_source_files(SOURCE_ROOTS, CODE_EXTS)
    rng.shuffle(files)
    samples: list[Sample] = []
    for path in files:
        if len(samples) >= max_files:
            break
        txt = _read_text(path, cap_chars=8_000)
        if txt is None:
            continue
        lines = txt.splitlines(keepends=True)
        if len(lines) < 12:
            continue
        # pick a contiguous middle span [a, b)
        a = rng.randint(3, max(3, len(lines) // 2))
        span = rng.randint(2, 6)
        b = min(len(lines), a + span)
        prefix = "".join(lines[:a])
        middle = "".join(lines[a:b])
        suffix = "".join(lines[b:])
        if not middle.strip():
            continue
        body = f"{T_FIM_PREFIX}{prefix}{T_FIM_SUFFIX}{suffix}{T_FIM_MIDDLE}{middle}"
        samples.append(Sample("fim", body))
    return samples


def harvest_reasoning_prose(rng: random.Random, max_files: int) -> list[Sample]:
    """Real .md files -> reasoning/chat shape: a user question, a <think> block
    grounded in the doc text, then a concise answer. The doc supplies authentic
    technical prose the model reasons over (matches this project's traffic)."""
    files = _iter_source_files(SOURCE_ROOTS, PROSE_EXTS)
    rng.shuffle(files)
    samples: list[Sample] = []
    qtemplates = [
        "Summarize the key technical decision in this note and why it was made.",
        "What is the failure mode described here, and what fixes it?",
        "Explain the design tradeoff this document is weighing.",
        "Walk through the steps this runbook prescribes, in order.",
        "What is the current state, and what is the remaining blocker?",
    ]
    for path in files:
        if len(samples) >= max_files:
            break
        txt = _read_text(path, cap_chars=12_000)
        if txt is None or len(txt) < 200:
            continue
        q = rng.choice(qtemplates)
        # The "think" is the source prose (authentic reasoning substrate); the
        # answer is a short deterministic lead. We are training the draft to track
        # the target's hidden state through real technical reasoning text, not to
        # produce a gold answer — so using the doc as the think body is correct.
        head = txt.strip()
        body = (
            f"User: {q}\n\n"
            f"{T_THINK}\n{head}\n{T_THINK_END}\n\n"
            f"Assistant: Based on the note above, here is the summary:\n"
        )
        samples.append(Sample("reasoning", body))
    return samples


# ---------------------------------------------------------------------------
# Synthetic-but-realistic scaffolding (guarantees count + balance + length).
# These are deterministic templates parameterized by an index so we can emit any
# number without a dataset. They are intentionally realistic for THIS workload.
# ---------------------------------------------------------------------------
_FUNC_NAMES = [
    "load_shard", "fuse_experts", "build_rope_cache", "quantize_block",
    "route_topk", "apply_swigluoai", "rms_norm", "spec_decode_step",
    "pack_int4", "dequantize", "gather_kv", "scatter_add_routed",
    "verify_logits", "accept_reject", "compact_kv", "draft_forward",
    "norm_topk", "shard_embed", "fold_bias", "clamp_act",
]
_TOPICS = [
    "speculative decoding acceptance rate", "MoE expert routing under load",
    "RoPE partial-rotary correctness", "int4 dequant numerics on Gaudi",
    "GQA KV-cache layout", "draft-head training stability",
    "tokenizer roundtrip edge cases", "swigluoai clamp behavior",
    "KV-cache eviction policy", "fused MoE down-projection",
    "draft acceptance under temperature", "qk-norm per-head scaling",
    "rotary theta=5e6 long-context drift", "shared-expert residual weighting",
    "top-k gather load-balance", "bf16 vs fp8 logit divergence",
    "router bias correction term", "depth-1 vs depth-2 draft tradeoff",
]
_TOOLS = ["read_file", "run_bash", "grep_repo", "edit_file", "list_dir",
          "write_file", "test_run", "git_diff"]
# Distinct opener phrasings so two synthetic samples that share a topic still
# differ from their very first characters (full-text de-dup + prefix diversity).
_REASON_OPENERS = [
    "Analyze", "Work through", "Derive the numeric bound for",
    "Estimate the impact of", "Reason step by step about",
    "Quantify", "Trace the control flow of", "Evaluate the correctness of",
]
_AGENT_OPENERS = [
    "Please investigate", "Track down the bug in", "Audit",
    "Profile and fix", "Reproduce the failure in", "Bisect",
    "Root-cause", "Harden",
]


def synth_code(rng: random.Random, i: int) -> Sample:
    name = _FUNC_NAMES[i % len(_FUNC_NAMES)]
    topic = _TOPICS[(i * 7) % len(_TOPICS)]  # decorrelate topic from name cycle
    n_helpers = 1 + (i % 5)
    lines = [f"{T_FILENAME}synth_{name}_{i}.py",
             f"# module #{i}: {topic}",
             "import torch", "import torch.nn.functional as F", ""]
    lines.append(f"def {name}_{i}(x, w, *, eps={10.0 ** -(3 + (i % 4))}):")
    lines.append(f'    """{topic} (variant {i})."""')
    lines.append("    d = x.dtype")
    lines.append("    x = x.float()")
    lines.append("    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)")
    lines.append("    return (x * (1.0 + w.float())).to(d)")
    lines.append("")
    for h in range(n_helpers):
        lines.append(f"def {name}_helper_{h}(t, scale={1.0 + h * 0.5}):")
        lines.append("    out = t")
        for k in range(2 + (h % 3)):
            lines.append(f"    out = F.silu(out) * scale + {k}.0")
        lines.append("    return out")
        lines.append("")
    return Sample("code", "\n".join(lines))


def synth_agentic(rng: random.Random, i: int) -> Sample:
    topic = _TOPICS[(i * 5 + 2) % len(_TOPICS)]
    opener = _AGENT_OPENERS[i % len(_AGENT_OPENERS)]
    n_turns = 2 + (i % 5)
    parts = [f"User: [task #{i}] {opener} {topic} in the repo and propose a fix.\n"]
    for t in range(n_turns):
        tname = _TOOLS[(i + t) % len(_TOOLS)]
        args = {
            "read_file": {"path": f"$HOME/AI/minimax-m3/mod_{t}.py"},
            "run_bash": {"command": f"grep -rn 'def step_{t}' $HOME/AI"},
            "grep_repo": {"pattern": f"topk_{t}", "glob": "*.py"},
            "edit_file": {"path": f"mod_{t}.py", "old": f"x{t}", "new": f"y{t}"},
            "list_dir": {"path": f"$HOME/AI/minimax-m3/sub_{t}"},
            "write_file": {"path": f"out_{t}.py", "content": f"# step {t}\\nreturn {t}"},
            "test_run": {"target": f"tests/test_step_{t}.py", "k": f"case_{t}"},
            "git_diff": {"ref": f"HEAD~{t + 1}", "path": f"mod_{t}.py"},
        }.get(tname, {"arg": f"value_{t}"})
        parts.append(
            f"Assistant: I'll inspect step {t}.\n"
            f"{T_TOOLCALL}\n"
            + json.dumps({"name": tname, "arguments": args})
            + f"\n{T_TOOLCALL_END}\n"
        )
        parts.append(
            f"Tool ({tname}): returned {12 * (t + 1)} lines; found {2 + t} matches near `step_{t}`.\n"
        )
    parts.append(
        "Assistant: Root cause is an off-by-one in the "
        f"{topic} path; the fix normalizes before the top-k gather. Applying now.\n"
    )
    return Sample("agentic", "".join(parts))


def synth_reasoning(rng: random.Random, i: int) -> Sample:
    topic = _TOPICS[(i * 3 + 1) % len(_TOPICS)]
    opener = _REASON_OPENERS[i % len(_REASON_OPENERS)]
    n_steps = 3 + (i % 7)
    base = 1 + (i % 9)  # vary the numeric scenario per-i so steps differ in content
    think = [f"Problem #{i}: we need to reason carefully about {topic}."]
    for s in range(n_steps):
        mag = base * (2 ** s)
        think.append(
            f"Step {s + 1}: consider the case where the input magnitude is "
            f"{mag}; then the clamp at limit=7.0 changes the gradient because "
            f"sigmoid(1.702 * 7.0) saturates, so the {s}-th term contributes "
            f"approximately {round(7.0 * (1.0 / (1.0 + 2.718 ** (-1.702 * 7.0))) * (mag / (mag + 1)), 4)}."
        )
    answer = (
        f"Therefore for {topic} the acceptance rate is dominated by the "
        f"high-frequency tokens (scenario base={base}), and a depth-1 draft "
        "captures most of the win at low inference cost."
    )
    body = (
        f"User: [{i}] {opener} {topic} and give a numeric conclusion.\n\n"
        f"{T_THINK}\n" + "\n".join(think) + f"\n{T_THINK_END}\n\n"
        f"Assistant: {answer}\n"
    )
    return Sample("reasoning", body)


# ---------------------------------------------------------------------------
# Assembly: harvest real material, top up with synthetic, balance by length.
# ---------------------------------------------------------------------------
def _bucket_of(n: int) -> Optional[str]:
    for name, lo, hi in LEN_BUCKETS:
        if lo <= n < hi:
            return name
    if n >= MAX_TOKENS:
        return LEN_BUCKETS[-1][0]
    return None  # below MIN handled by caller


def build_corpus(tok, target_n: int = TARGET_N, verbose: bool = True) -> list[dict]:
    rng = random.Random(SEED)

    raw: list[Sample] = []
    # Real material first (authentic token statistics). Caps keep it balanced and fast.
    raw += harvest_code(rng, max_files=70)
    raw += harvest_fim(rng, max_files=55)
    raw += harvest_reasoning_prose(rng, max_files=55)
    if verbose:
        print(f"[harvest] real samples: {len(raw)}", file=sys.stderr)

    # Synthetic scaffold — always added so the four kinds + all length buckets are
    # represented even on a thin checkout. Deterministic, parameterized by index.
    for i in range(60):
        raw.append(synth_code(rng, i))
    for i in range(60):
        raw.append(synth_agentic(rng, i))
    for i in range(60):
        raw.append(synth_reasoning(rng, i))
    if verbose:
        print(f"[scaffold] total raw (real+synth): {len(raw)}", file=sys.stderr)

    # Tokenize (ONE batch call), trim over-cap, drop-too-short, de-dup, bucket.
    ids_list = encode_batch(tok, [s.text for s in raw])
    trimmed_texts, counts = trim_overcap(tok, [s.text for s in raw], ids_list, MAX_TOKENS)
    if verbose:
        print(f"[tokenize] batch-encoded {len(raw)} samples", file=sys.stderr)

    seen: set[str] = set()
    scored: list[dict] = []
    for s, text, nt in zip(raw, trimmed_texts, counts):
        if nt < MIN_TOKENS:
            continue
        key = text  # de-dup on the FULL text (a prefix key collapsed distinct samples)
        if key in seen:
            continue
        seen.add(key)
        bucket = _bucket_of(nt)
        if bucket is None:
            continue
        scored.append({"kind": s.kind, "n_tokens": nt, "bucket": bucket, "text": text})

    rng.shuffle(scored)

    # Balance: ensure every (kind) and every (length bucket) is non-trivially
    # populated, then fill up to target_n preferring under-represented buckets.
    by_bucket: dict[str, list[dict]] = {b[0]: [] for b in LEN_BUCKETS}
    for r in scored:
        by_bucket[r["bucket"]].append(r)

    chosen: list[dict] = []
    # round-robin across buckets so the length distribution stays spread
    cursors = {b: 0 for b in by_bucket}
    order = [b[0] for b in LEN_BUCKETS]
    progress = True
    while len(chosen) < target_n and progress:
        progress = False
        for b in order:
            if cursors[b] < len(by_bucket[b]):
                chosen.append(by_bucket[b][cursors[b]])
                cursors[b] += 1
                progress = True
                if len(chosen) >= target_n:
                    break

    # If we still fell short (thin local tree), top up with extra synthetic
    # reasoning/agentic (cheap, deterministic) until we clear target_n. Generate
    # in batches and tokenize each batch in ONE call (same anti-hang discipline).
    extra_i = 10_000
    while len(chosen) < target_n:
        need = target_n - len(chosen)
        batch_n = max(need * 2, 32)  # over-generate to survive de-dup drops
        gens = [
            synth_reasoning(rng, extra_i + k) if (extra_i + k) % 2 else synth_agentic(rng, extra_i + k)
            for k in range(batch_n)
        ]
        extra_i += batch_n
        gids = encode_batch(tok, [g.text for g in gens])
        gtexts, gcounts = trim_overcap(tok, [g.text for g in gens], gids, MAX_TOKENS)
        advanced = False
        for g, text, nt in zip(gens, gtexts, gcounts):
            if nt < MIN_TOKENS:
                continue
            key = text  # full-text de-dup (see note above)
            if key in seen:
                continue
            seen.add(key)
            chosen.append({"kind": g.kind, "n_tokens": nt,
                           "bucket": _bucket_of(nt) or "short", "text": text})
            advanced = True
            if len(chosen) >= target_n:
                break
        if not advanced:  # safety: never spin forever if everything de-dups
            break

    # finalize: stable ids, strip helper 'bucket' key from output (keep n_tokens/kind)
    out: list[dict] = []
    for idx, r in enumerate(chosen[:max(target_n, len(chosen))]):
        out.append({
            "id": idx,
            "kind": r["kind"],
            "n_tokens": r["n_tokens"],
            "text": r["text"],
        })
    return out


def write_corpus(records: list[dict], path: str = CALIB_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    import statistics

    kinds: dict[str, int] = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    lens = [r["n_tokens"] for r in records]
    buckets = {b[0]: 0 for b in LEN_BUCKETS}
    for n in lens:
        buckets[_bucket_of(n) or "short"] += 1
    return {
        "n": len(records),
        "kinds": kinds,
        "buckets": buckets,
        "tok_min": min(lens),
        "tok_max": max(lens),
        "tok_mean": round(statistics.mean(lens), 1),
        "tok_median": int(statistics.median(lens)),
        "tok_total": sum(lens),
    }


# ---------------------------------------------------------------------------
# CPU self-test (card-free). Builds a small corpus and asserts the contract.
# ---------------------------------------------------------------------------
def self_test() -> bool:
    checks: list[tuple[str, bool]] = []
    tok = load_tokenizer()

    # quick tokenizer sanity (roundtrip on a code line + a special-token line)
    s_code = "def add(a, b):\n    return a + b\n"
    rt = tok.decode(tok(s_code, add_special_tokens=False)["input_ids"])
    checks.append(("tokenizer roundtrips plain code", rt == s_code))
    # special tokens encode to single ids (they exist in the vocab)
    for t in (T_THINK, T_TOOLCALL, T_FIM_PREFIX, T_FILENAME):
        tid = tok.convert_tokens_to_ids(t)
        checks.append((f"special token {t!r} in vocab", isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id))

    records = build_corpus(tok, target_n=TARGET_N, verbose=True)
    stats = summarize(records)
    print(f"[self-test] stats: {json.dumps(stats)}", file=sys.stderr)

    # --- contract assertions ---
    checks.append((f"N >= 256 (got {stats['n']})", stats["n"] >= 256))

    kinds = stats["kinds"]
    has_code = (kinds.get("code", 0) + kinds.get("fim", 0)) > 0
    has_reasoning = kinds.get("reasoning", 0) > 0
    has_agentic = kinds.get("agentic", 0) > 0
    checks.append(("has code samples (code|fim)", has_code))
    checks.append(("has reasoning samples", has_reasoning))
    checks.append(("has agentic/tool-call samples", has_agentic))
    # balance: code-side and reasoning-side each >= 20% of corpus
    code_side = kinds.get("code", 0) + kinds.get("fim", 0)
    reason_side = kinds.get("reasoning", 0) + kinds.get("agentic", 0)
    checks.append((f"code-side >= 20% (got {code_side})", code_side >= 0.20 * stats["n"]))
    checks.append((f"reasoning-side >= 20% (got {reason_side})", reason_side >= 0.20 * stats["n"]))

    # length distribution covers code (short) AND reasoning (long) ranges
    b = stats["buckets"]
    checks.append((f"short bucket populated (got {b['short']})", b["short"] >= 1))
    long_total = b["long"] + b["xlong"]
    checks.append((f"long/xlong populated (got {long_total})", long_total >= 1))
    checks.append((f"max tokens <= {MAX_TOKENS} (got {stats['tok_max']})", stats["tok_max"] <= MAX_TOKENS))
    checks.append((f"min tokens >= {MIN_TOKENS} (got {stats['tok_min']})", stats["tok_min"] >= MIN_TOKENS))

    # every record round-trips through the tokenizer (encode->decode->encode stable
    # token count). Batched (one encode call for the probes, one for the re-encode).
    sample_ok = True
    rng = random.Random(SEED)
    probe_idx = rng.sample(range(len(records)), k=min(16, len(records)))
    probe_texts = [records[i]["text"] for i in probe_idx]
    probe_ids = encode_batch(tok, probe_texts)
    # stored n_tokens must match a fresh encode
    for i, ids in zip(probe_idx, probe_ids):
        if len(ids) != records[i]["n_tokens"]:
            sample_ok = False
            break
    if sample_ok:
        decoded = [tok.decode(ids) for ids in probe_ids]
        reids = encode_batch(tok, decoded)
        for ids, ids2 in zip(probe_ids, reids):
            # allow tiny boundary differences but require near-stability
            if abs(len(ids2) - len(ids)) > max(2, int(0.01 * len(ids))):
                sample_ok = False
                break
    checks.append(("records round-trip through tokenizer (stable token count)", sample_ok))

    # determinism: rebuild and compare the produced texts byte-for-byte (reuse the
    # same tokenizer object — avoids a second concurrent fast-tokenizer instance).
    records2 = build_corpus(tok, target_n=TARGET_N, verbose=False)
    same = (len(records) == len(records2)
            and all(a["text"] == b["text"] and a["kind"] == b["kind"]
                    for a, b in zip(records, records2)))
    checks.append(("deterministic rebuild (byte-identical)", same))

    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}")
    print(f"\n{'=' * 56}\nM3 CALIB BUILDER SELF-TEST: "
          f"{'GREEN — all pass' if ok else 'RED — FIX ABOVE'}\n{'=' * 56}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the M3 MTP calibration corpus.")
    ap.add_argument("--self-test", action="store_true", help="run CPU self-test and exit")
    ap.add_argument("--target-n", type=int, default=TARGET_N, help="number of samples to emit")
    ap.add_argument("--out", type=str, default=CALIB_PATH, help="output jsonl path")
    args = ap.parse_args()

    if args.self_test:
        return 0 if self_test() else 1

    tok = load_tokenizer()
    records = build_corpus(tok, target_n=args.target_n, verbose=True)
    write_corpus(records, args.out)
    stats = summarize(records)
    print(f"[build] wrote {len(records)} samples -> {args.out}")
    print(f"[build] {json.dumps(stats)}")
    if stats["n"] < 256:
        print("ERROR: corpus < 256 samples — FAILING LOUD", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
