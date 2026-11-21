#!/usr/bin/env python3
"""
parse_trace.py — single-stream (batch=1) decode breakdown from a vllm-gaudi trace.

Ingests ONE torch.profiler / Kineto chrome-trace (a `*.pt.trace.json[.gz]` produced by
`VLLM_TORCH_PROFILER_DIR` on a vllm-gaudi serve) OR a Perfetto-style chrome-trace JSON,
and prints the per-decode-token breakdown:
  - total wall (sum of decode-step durations) and ms/token
  - device-busy vs host idle-gap ms  (busy = union of device-kernel intervals)
  - % time in HCCL collectives
  - % time in MoE ops
  - top host-gap categories (device self-time bucketed by op family)
  - total op-launch count (device kernels + host op launches), per step
  - the single biggest contributor (op name + bucket)

USAGE
  parse_trace.py <trace.json[.gz]> [trace2 ...]      # one or many (8 TP ranks); rank0 is canonical
  parse_trace.py --glob '$HOME/AI/logs/prof/*.pt.trace.json.gz'
  parse_trace.py <trace> --min-step-ms 5            # threshold for "a real decode step"
  parse_trace.py <trace> --json                     # machine-readable summary to stdout

SCHEMA NOTES (verified against the real on-box traces at $HOME/AI/logs/prof/, 2026-06-08,
and against vllm_gaudi/extension/profiler.py which writes them) — see ASSUMPTIONS at bottom:
  * Top level: {"traceEvents":[...], "baseTimeNanoseconds":..., "displayTimeUnit":"ms", ...}
  * Each event: ph (X=complete/dur, M=metadata, i=instant, C=counter, ...), cat, name, pid, tid,
    ts (float), dur (float, MICROSECONDS), args{}.
  * Device process: ph==M name=="process_name" whose args.name contains "device:HPU"  -> its pid
    owns cat=="kernel" device events. (Real trace: device pid==0, name "/device:HPU:0".)
  * Host worker process: process_name args.name contains "Worker"/"VLLM"; if absent we fall back to
    the pid that owns the ProfilerStep user_annotation events.
  * A decode step = ph=="X", cat=="user_annotation", name startswith "ProfilerStep", dur>threshold.
    (vllm-gaudi's full_trace_handler also injects high-level annotation events on tid 1/2/3 — handled.)

This parser is DEFENSIVE: every field access tolerates absence; it works on a plain
tensorboard_trace_handler trace (no high-level events) and on a VLLM_PROFILER_ENABLED=full
trace (with them); and it degrades gracefully if there are no ProfilerStep markers (it then
treats the whole device-kernel span as one window and says so LOUDLY).
"""
from __future__ import annotations
import argparse
import glob as globmod
import gzip
import io
import json
import os
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# loading (gzip or plain; tolerate a leading/trailing junk byte from concat traces)
# ---------------------------------------------------------------------------
def load_trace(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        raw = gzip.decompress(raw)
    txt = raw.decode("utf-8", "replace").strip()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        # some producers leave a trailing comma / unterminated array; salvage the events list
        m = re.search(r'"traceEvents"\s*:\s*\[', txt)
        if not m:
            raise
        # best-effort: cut to the last "}]" and close the object
        start = m.end() - 1
        end = txt.rfind("}")
        frag = txt[start:end + 1] + "]"
        d = {"traceEvents": json.loads(frag)}
    if isinstance(d, list):              # bare event array (some perfetto exports)
        d = {"traceEvents": d}
    d.setdefault("traceEvents", [])
    return d


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def union_busy(intervals):
    """Total covered length of possibly-overlapping (start,end) intervals (device queues overlap)."""
    iv = sorted(i for i in intervals if i[0] is not None and i[1] is not None and i[1] >= i[0])
    tot = 0.0
    cs = ce = None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot


# bucket regexes — applied to LOWERCASED op name. Order matters (first match wins).
# Grounded on the real op names seen on-box: MoE = fused_kernel_*_bf16 / convert_from_uint*;
# collectives = hcclAllReduce / all_reduce; attention = sdpa/fmha/flash/rotary; etc.
BUCKETS = [
    ("HCCL/collective",  r"hccl|all_?reduce|reduce_?scatter|all_?gather|broadcast|send_?recv|sendrecv|"
                         r"collective|nccl|alltoall|all_to_all|p2p"),
    ("MoE",              r"\bmoe\b|expert|mixture_of_experts|convert_from_uint|convert_from_int|"
                         r"fused_kernel|dequant|moe_gemm|grouped_gemm|wna16|gptq|awq"),
    ("attention/MLA",    r"attn|attention|sdpa|flash|fmha|softmax|rotary|\brope\b|paged|flat_pa|mla|"
                         r"indexer|sparse_attn|topk"),
    ("matmul/GEMM",      r"\bgemm\b|matmul|\bmm\b|\bbmm\b|\bdot\b|batch_gemm|linear|fp8_gemm|addmm"),
    ("copy/cast/memcpy", r"memcpy|\bcopy\b|d2h|h2d|cast|reinterpret|to_copy|contiguous|\bclone\b|"
                         r"index_select|gather|scatter|slice|reshape|transpose|permute|view|strided"),
    ("norm/elementwise/act",
                         r"\badd\b|\bmul\b|\bsub\b|\bdiv\b|norm|rms|layernorm|silu|gelu|swiglu|swigluoai|"
                         r"sigmoid|relu|elementwise|binary|unary|\bexp\b|\bneg\b|fill|memset|where|clamp"),
    ("sync/wait",        r"synchronize|synchronizeevent|wait|barrier|stream|event|record_event|"
                         r"mark_step|drain"),
]
_BUCKET_RE = [(name, re.compile(pat)) for name, pat in BUCKETS]


def bucket_of(name):
    low = (name or "").lower()
    for bname, rx in _BUCKET_RE:
        if rx.search(low):
            return bname
    return "other"


# ---------------------------------------------------------------------------
# per-file analysis
# ---------------------------------------------------------------------------
def analyze(path, min_step_ms=1.0):
    d = load_trace(path)
    ev = d["traceEvents"]
    out = {"path": path, "n_events": len(ev), "warnings": []}

    # ---- resolve device pid and host pid from process_name metadata --------------------------
    proc_name = {}
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid = e.get("pid")
            nm = (e.get("args") or {}).get("name", "")
            proc_name[pid] = nm
    dev_pids = {p for p, nm in proc_name.items() if "device:hpu" in str(nm).lower() or "hpu:" in str(nm).lower()}
    host_pids = {p for p, nm in proc_name.items() if re.search(r"worker|vllm", str(nm), re.I)}

    # ProfilerStep markers (the canonical decode-step windows)
    steps_raw = [e for e in ev
                 if e.get("ph") == "X" and e.get("cat") == "user_annotation"
                 and str(e.get("name", "")).startswith("ProfilerStep")
                 and _num(e.get("dur")) is not None]
    # the host pid is whoever owns the ProfilerStep events, if process_name didn't tell us
    if not host_pids and steps_raw:
        host_pids = {e.get("pid") for e in steps_raw}

    # device pid fallback: the pid that owns the most cat=="kernel" events
    if not dev_pids:
        kc = defaultdict(int)
        for e in ev:
            if e.get("ph") == "X" and e.get("cat") == "kernel":
                kc[e.get("pid")] += 1
        if kc:
            dev_pids = {max(kc, key=kc.get)}
            out["warnings"].append("no 'device:HPU' process_name; inferred device pid from kernel-event count")

    # ---- decode-step windows --------------------------------------------------------
    thr_us = min_step_ms * 1000.0
    steps = sorted([(e["name"], float(e["ts"]), float(e["ts"]) + float(e["dur"]), float(e["dur"]))
                    for e in steps_raw if float(e["dur"]) > thr_us],
                   key=lambda s: s[1])
    if not steps:
        # No ProfilerStep markers above threshold -> degrade to "whole device span = one window".
        dev_all = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"
                   and (not dev_pids or e.get("pid") in dev_pids) and _num(e.get("dur")) is not None]
        if not dev_all:
            out["error"] = "no ProfilerStep markers AND no device kernel events found — unrecognized trace"
            return out
        s0 = min(float(e["ts"]) for e in dev_all)
        s1 = max(float(e["ts"]) + float(e["dur"]) for e in dev_all)
        steps = [("WHOLE-DEVICE-SPAN", s0, s1, s1 - s0)]
        out["warnings"].append(
            "NO ProfilerStep markers > %.1f ms — fell back to one whole-device-span window; "
            "per-token numbers are span-wide, NOT per-decode-step. Capture with VLLM_PROFILE_STEPS "
            "or /start_profile around steady-state decode for true per-token figures." % min_step_ms)

    nstep = len(steps)
    wall_total = sum(s[3] for s in steps)              # sum of step durations (us)

    # interval membership test (steps are sorted, non-trivially few -> linear is fine)
    win = [(a, b) for _, a, b, _ in steps]

    def in_win(ts):
        for a, b in win:
            if a <= ts < b:
                return True
        return False

    # ---- device kernels inside the windows ------------------------------------------
    dev = [e for e in ev
           if e.get("ph") == "X" and e.get("cat") == "kernel"
           and (not dev_pids or e.get("pid") in dev_pids)
           and _num(e.get("dur")) is not None and in_win(float(e["ts"]))]
    dev_intervals = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in dev]
    dev_busy = union_busy(dev_intervals)               # wall-clock device occupancy (no double count)
    dev_self = sum(float(e["dur"]) for e in dev)        # self-time sum (counts overlap)
    n_dev_launch = len(dev)

    # ---- host op launches inside the windows (cpu_op + the runtime dispatch shim) ----
    host_cats = {"cpu_op", "privateuse1_runtime", "cuda_runtime", "hpu_op", "runtime"}
    host = [e for e in ev
            if e.get("ph") == "X" and e.get("cat") in host_cats
            and (not host_pids or e.get("pid") in host_pids)
            and _num(e.get("dur")) is not None and in_win(float(e["ts"]))]
    n_host_launch = len(host)

    idle_gap = wall_total - dev_busy                   # host/dispatch bubble (wall not covered by device)

    # ---- aggregate device kernels by name + by bucket -------------------------------
    byname = defaultdict(lambda: [0.0, 0])             # name -> [self_us, count]
    for e in dev:
        byname[e.get("name", "?")][0] += float(e["dur"])
        byname[e.get("name", "?")][1] += 1
    bybucket = defaultdict(lambda: [0.0, 0])
    for nm, (us, c) in byname.items():
        b = bucket_of(nm)
        bybucket[b][0] += us
        bybucket[b][1] += c

    def bucket_us(name):
        return bybucket.get(name, [0.0, 0])[0]

    hccl_us = bucket_us("HCCL/collective")
    moe_us = bucket_us("MoE")

    # single biggest contributor (by device self-time)
    if byname:
        top_name, (top_us, top_cnt) = max(byname.items(), key=lambda kv: kv[1][0])
    else:
        top_name, top_us, top_cnt = "(none)", 0.0, 0

    # ---- host-side gap categories: which op families dominate the host launches -----
    host_by_bucket = defaultdict(lambda: [0.0, 0])
    for e in host:
        b = bucket_of(e.get("name", ""))
        host_by_bucket[b][0] += float(e["dur"])
        host_by_bucket[b][1] += 1

    out.update({
        "device_pid": sorted(dev_pids) if dev_pids else None,
        "host_pid": sorted(host_pids) if host_pids else None,
        "n_decode_steps": nstep,
        "step_names": [s[0] for s in steps][:12],
        "per_step_wall_ms": [round(s[3] / 1000, 3) for s in steps][:24],
        "wall_total_ms": wall_total / 1000,
        "ms_per_token": wall_total / nstep / 1000,
        "tok_per_s_wall": (1000.0 / (wall_total / nstep / 1000)) if wall_total else 0.0,
        "device_busy_ms": dev_busy / 1000,
        "device_busy_ms_per_tok": dev_busy / nstep / 1000,
        "device_self_ms": dev_self / 1000,
        "idle_gap_ms": idle_gap / 1000,
        "idle_gap_ms_per_tok": idle_gap / nstep / 1000,
        "idle_gap_pct": (100 * idle_gap / wall_total) if wall_total else 0.0,
        "device_busy_pct": (100 * dev_busy / wall_total) if wall_total else 0.0,
        "pct_hccl_of_wall": (100 * hccl_us / wall_total) if wall_total else 0.0,
        "pct_hccl_of_devself": (100 * hccl_us / dev_self) if dev_self else 0.0,
        "pct_moe_of_wall": (100 * moe_us / wall_total) if wall_total else 0.0,
        "pct_moe_of_devself": (100 * moe_us / dev_self) if dev_self else 0.0,
        "n_device_launch": n_dev_launch,
        "n_device_launch_per_step": n_dev_launch / nstep,
        "n_host_launch": n_host_launch,
        "n_host_launch_per_step": n_host_launch / nstep,
        "n_total_launch": n_dev_launch + n_host_launch,
        "n_total_launch_per_step": (n_dev_launch + n_host_launch) / nstep,
        "biggest_contributor": {
            "name": top_name, "bucket": bucket_of(top_name),
            "self_ms": top_us / 1000, "self_ms_per_tok": top_us / nstep / 1000,
            "count": top_cnt, "count_per_step": top_cnt / nstep,
            "pct_of_devself": (100 * top_us / dev_self) if dev_self else 0.0,
        },
        "device_buckets": {b: {"self_ms": v[0] / 1000, "self_ms_per_tok": v[0] / nstep / 1000,
                               "count": v[1], "pct_devself": (100 * v[0] / dev_self) if dev_self else 0.0}
                           for b, v in sorted(bybucket.items(), key=lambda kv: -kv[1][0])},
        "host_buckets": {b: {"self_ms": v[0] / 1000, "count": v[1]}
                         for b, v in sorted(host_by_bucket.items(), key=lambda kv: -kv[1][0])},
        "top_ops": [(nm, us / 1000, us / nstep / 1000, c, c / nstep)
                    for nm, (us, c) in sorted(byname.items(), key=lambda kv: -kv[1][0])[:20]],
    })
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def print_report(r):
    p = r["path"]
    print("=" * 88)
    print("TRACE:", p)
    if "error" in r:
        print("  ERROR:", r["error"])
        return
    for w in r.get("warnings", []):
        print("  ! WARN:", w)
    print("  events: %d | device pid=%s | host pid=%s"
          % (r["n_events"], r["device_pid"], r["host_pid"]))
    print("  decode steps captured: %d %s"
          % (r["n_decode_steps"], r["step_names"]))
    print("  per-step wall (ms): %s" % r["per_step_wall_ms"])
    print("-" * 88)
    print("  PER-DECODE-TOKEN BREAKDOWN  (averaged over %d steps)" % r["n_decode_steps"])
    print("    total wall .................. %9.3f ms/tok   (%.2f tok/s single-stream)"
          % (r["ms_per_token"], r["tok_per_s_wall"]))
    print("    device-busy (union) ......... %9.3f ms/tok   (%.1f%% of wall)"
          % (r["device_busy_ms_per_tok"], r["device_busy_pct"]))
    print("    host/dispatch idle-gap ...... %9.3f ms/tok   (%.1f%% of wall)  <-- the host-bound bubble"
          % (r["idle_gap_ms_per_tok"], r["idle_gap_pct"]))
    # NOTE: bucket figures are device *self-time* (sums kernels that run concurrently across the
    # many HPU queues), so "% of device self-time" is the sound metric and is reported first.
    # "% of wall" can exceed 100% precisely because device self-time > wall when queues overlap;
    # it is shown only as a rough share and clamped-labelled.
    print("    %% time in HCCL collectives .. %8.1f%% of device self-time   (~%.0f%% of wall*)"
          % (r["pct_hccl_of_devself"], r["pct_hccl_of_wall"]))
    print("    %% time in MoE ops ........... %8.1f%% of device self-time   (~%.0f%% of wall*)"
          % (r["pct_moe_of_devself"], r["pct_moe_of_wall"]))
    print("       (* device self-time sums concurrent queues -> 'of wall' can exceed 100%; "
          "the device-busy union above is the true wall occupancy)")
    print("    op launches / step .......... %9.0f   (device %.0f + host %.0f)"
          % (r["n_total_launch_per_step"], r["n_device_launch_per_step"], r["n_host_launch_per_step"]))
    bc = r["biggest_contributor"]
    print("    BIGGEST CONTRIBUTOR ......... %s  [%s]" % (bc["name"][:60], bc["bucket"]))
    print("                                  %.3f ms/tok | %.0f launches/step | %.1f%% of device self-time"
          % (bc["self_ms_per_tok"], bc["count_per_step"], bc["pct_of_devself"]))
    print("-" * 88)
    print("  DEVICE self-time by bucket (ms/tok | %% device self | launches):")
    for b, v in r["device_buckets"].items():
        print("    %-22s %9.3f ms/tok | %5.1f%% | %8d" % (b, v["self_ms_per_tok"], v["pct_devself"], v["count"]))
    print("  TOP HOST-GAP categories (host op launches inside the step, ms | count):")
    if r["host_buckets"]:
        for b, v in list(r["host_buckets"].items())[:8]:
            print("    %-22s %9.3f ms | %8d" % (b, v["self_ms"], v["count"]))
    else:
        print("    (no host cpu_op/runtime events in window)")
    print("  TOP 12 device ops (ms total | ms/tok | count | /step):")
    for nm, tot_ms, ms_tok, c, c_step in r["top_ops"][:12]:
        print("    %9.2f | %7.3f | %8d | %7.0f  %s" % (tot_ms, ms_tok, c, c_step, nm[:58]))


def main():
    ap = argparse.ArgumentParser(description="Per-decode-token breakdown from a vllm-gaudi chrome-trace.")
    ap.add_argument("traces", nargs="*", help="trace file(s) .pt.trace.json[.gz]")
    ap.add_argument("--glob", help="glob of trace files (e.g. '$HOME/AI/logs/prof/*.pt.trace.json.gz')")
    ap.add_argument("--min-step-ms", type=float, default=1.0,
                    help="min ProfilerStep dur (ms) to count as a real decode step (default 1.0)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    args = ap.parse_args()

    paths = list(args.traces)
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    if not paths:
        ap.error("no trace files given (positional args or --glob). "
                 "On-box traces live in $HOME/AI/logs/prof/*.pt.trace.json.gz")

    results = []
    for p in paths:
        if not os.path.exists(p):
            print("SKIP (missing):", p, file=sys.stderr)
            continue
        try:
            r = analyze(p, min_step_ms=args.min_step_ms)
        except Exception as e:  # never die mid-batch on one bad file
            r = {"path": p, "error": "parse failed: %r" % e}
        results.append(r)

    if args.json:
        # strip the verbose top_ops list weight for the JSON dump's readability but keep it
        print(json.dumps(results, indent=2, default=str))
        return

    for r in results:
        print_report(r)

    # cross-rank note: rank0 (the one carrying ProfilerStep, usually the lowest pid) is canonical;
    # other ranks confirm collective balance. Flag big wall spread across ranks.
    walls = [(r["path"], r.get("ms_per_token")) for r in results if r.get("ms_per_token")]
    if len(walls) > 1:
        vals = [w for _, w in walls]
        spread = (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100 if vals else 0
        print("=" * 88)
        print("CROSS-RANK ms/token: min=%.2f max=%.2f spread=%.0f%% over %d ranks "
              "(rank with ProfilerStep markers is canonical; large spread => collective imbalance / straggler)"
              % (min(vals), max(vals), spread, len(vals)))


if __name__ == "__main__":
    main()
