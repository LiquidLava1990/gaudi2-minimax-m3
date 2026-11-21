#!/usr/bin/env python3
# ==================================================================================================
# aggregate_matrix.py — collapse the MiniMax-M3 benchmark MATRIX into ONE markdown table.
#
# This is the DELIVERABLE emitter for run_matrix.sh. It scans the matrix run tree
#   <runs>/<cfg-id>__<regime>/  (e.g.  c1_fp8native__128K_30c/)
# where each such dir holds, per optimization-loop trial, a knob-tagged result JSON written by the
# existing bench client:
#   trial_<knobtag>__c<conc>.json   (the bench_client.py output: throughput_tok_s, ttft_*, mean_tpot_ms, …)
# plus a per-config picked-winner file  best.json  (written by run_matrix.sh after the loop plateaus)
# and a meta.json (the env composition actually used, for the audit column).
#
# For EACH (config, regime) it reports the BEST trial (max CONCURRENT decode tok/s at the regime's
# target concurrency) and renders ROWS = configs, COLS = best --max-num-seqs, SINGLE-STREAM tok/s
# (batch=1, the latency floor carried by that winning trial), CONCURRENT tok/s (aggregate at target
# conc), prefill TTFT(p50), p99 TTFT, TPOT, fit point (ctx x conc), vs-baseline %. Config #0
# (int4-in-HBM x fp8_inc) is the baseline for the vs-% column.
#
# CARD-FREE. Pure file scan + arithmetic — no card, no serve, no torch. `--schema` prints the empty
# table (the schema) with no runs present; `--dry-run-rows` injects the planned config set so the
# table shape is visible before any card run. Real numbers fill in once run_matrix.sh has executed.
# ==================================================================================================
from __future__ import annotations

import argparse
import glob
import json
import os
import re

# The canonical config order + identity. MUST match run_matrix.sh's CONFIGS array (single source of
# truth lives in the shell; this mirror is for the --schema/--dry-run-rows card-free preview and for
# stable row ordering + the human label / weight / kv / gate columns in the table).
#   id        : the cfg-id prefix used in the run dir name (cN_...)
#   label     : human description for the table's first column
#   weight    : weight runtime in HBM
#   kv        : kv-cache scheme
#   baseline  : True for the single vs-% reference row (#0)
# The 3 WEIGHT runtimes x 3 KV modes grid (+ the baseline cell). MUST match run_matrix.sh CONFIGS.
#   weight runtimes : int4-in-HBM (M3_EXPERT_RUNTIME=int4, int4 ckpt)
#                     int4->fp8   (M3_EXPERT_RUNTIME=fp8,  int4 ckpt)
#                     fp8-normal  (stock INC bf16->fp8 at load, bf16 ckpt + QUANT_CONFIG; NO M3_EXPERT_RUNTIME)
#   KV modes        : fp8_inc | TurboQuant k8v4 (4-bit V; k8v2/2-bit DROPPED) | fp8 + MSA-offload
# baseline = c0 = (int4-in-HBM x fp8_inc) — the vs-% reference cell.
MATRIX = [
    # --- KV fp8_inc column ---
    dict(id="c0_int4hbm_fp8inc",   label="#0 int4-in-HBM (fused) | KV fp8_inc",
         weight="int4-in-HBM",       kv="fp8_inc",           baseline=True),
    dict(id="c1_fp8native_fp8inc", label="#1 int4->fp8 (native)  | KV fp8_inc",
         weight="int4->fp8 native",  kv="fp8_inc",           baseline=False),
    dict(id="c2_fp8inc_fp8inc",    label="#2 fp8-normal (INC)    | KV fp8_inc",
         weight="fp8-normal (INC)",  kv="fp8_inc",           baseline=False),
    # --- KV TurboQuant k8v4 column ---
    dict(id="c3_int4hbm_tq",       label="#3 int4-in-HBM (fused) | KV TurboQuant",
         weight="int4-in-HBM",       kv="TurboQuant k8v4",   baseline=False),
    dict(id="c4_fp8native_tq",     label="#4 int4->fp8 (native)  | KV TurboQuant",
         weight="int4->fp8 native",  kv="TurboQuant k8v4",   baseline=False),
    dict(id="c5_fp8inc_tq",        label="#5 fp8-normal (INC)    | KV TurboQuant",
         weight="fp8-normal (INC)",  kv="TurboQuant k8v4",   baseline=False),
    # --- KV fp8 + MSA-offload column (the 1M target) ---
    dict(id="c6_int4hbm_msa",      label="#6 int4-in-HBM (fused) | KV fp8 + MSA-offload",
         weight="int4-in-HBM",       kv="fp8 + MSA-offload", baseline=False),
    dict(id="c7_fp8native_msa",    label="#7 int4->fp8 (native)  | KV fp8 + MSA-offload",
         weight="int4->fp8 native",  kv="fp8 + MSA-offload", baseline=False),
    dict(id="c8_fp8inc_msa",       label="#8 fp8-normal (INC)    | KV fp8 + MSA-offload",
         weight="fp8-normal (INC)",  kv="fp8 + MSA-offload", baseline=False),
]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fmt(v, nd=1):
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def scan_config(runs_dir, cfg):
    """Return the best trial for every (config,regime) dir matching this config id.

    Yields dicts: {regime, ctx, conc, knobtag, max_num_seqs, block_size, ss_tps, decode_tps,
                   ttft_p50, ttft_p99, tpot_ms, fit_ok, meta}. One per regime (the loop winner).
    ss_tps = the single-stream (batch=1) tok/s carried by the winning trial."""
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, f"{cfg['id']}__*"))):
        regime = os.path.basename(d).split("__", 1)[1]
        meta = load(os.path.join(d, "meta.json")) or {}
        ctx = meta.get("ctx", "?")
        target_c = meta.get("target_conc")
        # Prefer the explicit winner the runner picked; else compute it from the trial JSONs.
        best = load(os.path.join(d, "best.json"))
        if best is None:
            best = _pick_best_trial(d, target_c)
        if best is None:
            # config/regime planned but produced no usable result (e.g. OOM at every knob, or
            # card-gated / not yet run). Still emit a row so the matrix shows the gap.
            out.append(dict(regime=regime, ctx=ctx, conc=target_c, knobtag=None,
                            max_num_seqs=None, block_size=None, ss_tps=None, decode_tps=None,
                            ttft_p50=None, ttft_p99=None, tpot_ms=None, fit_ok=False, meta=meta))
            continue
        out.append(dict(regime=regime, ctx=ctx,
                        conc=best.get("conc", target_c),
                        knobtag=best.get("knobtag"),
                        max_num_seqs=best.get("max_num_seqs"),
                        block_size=best.get("block_size"),
                        ss_tps=best.get("single_stream_tok_s"),
                        decode_tps=best.get("throughput_tok_s"),
                        ttft_p50=best.get("ttft_p50_s"),
                        ttft_p99=best.get("ttft_p99_s"),
                        tpot_ms=best.get("mean_tpot_ms"),
                        fit_ok=best.get("fit_ok", best.get("ok", 0) and not best.get("fail", 0)),
                        meta=meta))
    return out


def _pick_best_trial(d, target_c):
    """Fallback winner selection if best.json is absent: among trials at the target concurrency,
    the one with max decode throughput_tok_s. Knob fields are parsed from the trial filename tag."""
    cands = []
    pat = re.compile(r"trial_(?P<tag>.+)__c(?P<c>\d+)\.json$")
    for f in glob.glob(os.path.join(d, "trial_*__c*.json")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        c = int(m.group("c"))
        if target_c is not None and c != int(target_c):
            continue
        j = load(f)
        if not j:
            continue
        j = dict(j)
        j["knobtag"] = m.group("tag")
        j["conc"] = c
        # decode knob values out of the tag (mns<N>_blk<N>_gmem<F>_...); best-effort.
        for key, rx in (("max_num_seqs", r"mns(\d+)"), ("block_size", r"blk(\d+)")):
            mm = re.search(rx, j["knobtag"])
            if mm:
                j[key] = int(mm.group(1))
        cands.append(j)
    if not cands:
        return None
    return max(cands, key=lambda x: (x.get("throughput_tok_s") or -1))


def build_table(rows_by_cfg):
    """rows_by_cfg: list of (cfg, [regime-best dicts]). Render the single deliverable table."""
    # baseline decode tok/s for the vs-% column = the #0 row whose regime is the 32K/30c point.
    base_tps = None
    for cfg, regs in rows_by_cfg:
        if cfg.get("baseline"):
            for r in regs:
                if r.get("decode_tps") is not None:
                    base_tps = r["decode_tps"]
                    break
    out = []
    out.append("## MiniMax-M3 benchmark matrix — best config per (weight x KV) x regime")
    out.append("")
    out.append("3 WEIGHT runtimes {int4-in-HBM, int4->fp8, fp8-normal(INC)} x 3 KV modes "
               "{fp8_inc, TurboQuant k8v4, fp8 + MSA-offload}. EACH config is measured at BOTH "
               "single-stream (batch=1, the latency floor) AND its target concurrency (aggregate). "
               "After a per-config knob optimization loop (max-num-seqs / block-size / "
               "exponential-bucketing bounds / recipe-cache / gpu-mem-util), with prefill TTFT (p50) "
               "+ p99 TTFT + TPOT. vs-base % is CONCURRENT decode tok/s relative to config #0 "
               "(int4-in-HBM, KV fp8_inc).")
    out.append("")
    out.append("| # | config (weight \\| KV) | regime (ctx x conc = fit point) | best max-num-seqs | "
               "best block / bucket / cache / gmem | single-stream tok/s | concurrent tok/s | "
               "TTFT p50 (s) | p99 TTFT (s) | TPOT (ms) | fits? | vs-base % |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg, regs in rows_by_cfg:
        if not regs:
            regs = [dict(regime="(no run)", ctx="?", conc=None, max_num_seqs=None, block_size=None,
                         ss_tps=None, decode_tps=None, ttft_p50=None, ttft_p99=None, tpot_ms=None,
                         fit_ok=False, knobtag=None, meta={})]
        for r in regs:
            fit = f"{r.get('ctx','?')} x {r.get('conc','?')}"
            knobs = r.get("knobtag") or "-"
            vs = "-"
            if cfg.get("baseline"):
                vs = "1.00x (base)"
            elif base_tps and r.get("decode_tps") is not None:
                try:
                    vs = f"{(float(r['decode_tps'])/float(base_tps)):.2f}x"
                except Exception:
                    vs = "-"
            fits = "yes" if r.get("fit_ok") else ("OOM/—" if r.get("decode_tps") is None else "no")
            out.append(
                f"| {cfg['id'].split('_')[0].lstrip('c') or '0'} "
                f"| {cfg['label']} "
                f"| {r.get('regime','?')} ({fit}) "
                f"| {fmt(r.get('max_num_seqs'),0)} "
                f"| {knobs} "
                f"| {fmt(r.get('ss_tps'),1)} "
                f"| {fmt(r.get('decode_tps'),1)} "
                f"| {fmt(r.get('ttft_p50'),3)} "
                f"| {fmt(r.get('ttft_p99'),3)} "
                f"| {fmt(r.get('tpot_ms'),2)} "
                f"| {fits} "
                f"| {vs} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Aggregate the M3 benchmark matrix into one markdown table.")
    ap.add_argument("--runs", default="$HOME/AI/minimax-m3/bench/runs",
                    help="matrix run tree (dirs <cfg-id>__<regime>/).")
    ap.add_argument("--out", default="$HOME/AI/minimax-m3/bench/M3_MATRIX_REPORT.md")
    ap.add_argument("--schema", action="store_true",
                    help="card-free: print the EMPTY table schema (no runs) and exit.")
    ap.add_argument("--dry-run-rows", action="store_true",
                    help="card-free: print the table with one placeholder row per planned config "
                         "(no numbers) so the matrix shape is visible before any run.")
    a = ap.parse_args()

    if a.schema or a.dry_run_rows:
        rows_by_cfg = []
        for cfg in MATRIX:
            if a.dry_run_rows:
                regs = [dict(regime="<pending>", ctx="<ctx>", conc="<c>", max_num_seqs=None,
                             block_size=None, ss_tps=None, decode_tps=None, ttft_p50=None,
                             ttft_p99=None, tpot_ms=None, fit_ok=False, knobtag=None, meta={})]
            else:
                regs = []
            rows_by_cfg.append((cfg, regs))
        print(build_table(rows_by_cfg))
        return

    rows_by_cfg = [(cfg, scan_config(a.runs, cfg)) for cfg in MATRIX]
    txt = build_table(rows_by_cfg)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\nWritten: {a.out}")


if __name__ == "__main__":
    main()
