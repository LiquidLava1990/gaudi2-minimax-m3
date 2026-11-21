#!/usr/bin/env python3
"""Concurrency sweep for the M3 serve. stdlib-only (urllib + threads).
Usage: python m3_concurrency.py <api_key> [host:port] [model]
Measures aggregate tok/s + TTFT-ish at c=1/10/30/50 (code prompt, 256 gen, greedy)."""
import urllib.request, json, time, sys
from concurrent.futures import ThreadPoolExecutor

KEY   = sys.argv[1] if len(sys.argv) > 1 else ""
HP    = sys.argv[2] if len(sys.argv) > 2 else "<SERVER_IP>:8000"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "minimax-m3"
EP    = f"http://{HP}/v1/chat/completions"
PROMPT = ("Write a Python function that merges two sorted lists into one sorted list, "
          "then explain its time and space complexity step by step.")

def one(mt=256):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": mt, "temperature": 0.0,
                       "stream": False}).encode()
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(EP, body, h), timeout=600)
        d = json.load(r)
        dt = time.time() - t0
        return d.get("usage", {}).get("completion_tokens", 0), dt, None
    except Exception as e:
        return 0, time.time() - t0, repr(e)[:120]

def sweep(c, mt=256):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(lambda _: one(mt), range(c)))
    wall = time.time() - t0
    toks = sum(n for n, _, _ in res)
    errs = [e for _, _, e in res if e]
    per = sorted(dt for _, dt, e in res if not e)
    return toks, wall, (toks / wall if wall else 0), per, errs

if __name__ == "__main__":
    # warm
    one(8)
    print(f"endpoint={EP} model={MODEL}")
    for c in [1, 10, 30, 50]:
        toks, wall, agg, per, errs = sweep(c)
        med = per[len(per)//2] if per else 0
        print(f"c={c:>2}: {agg:6.1f} tok/s aggregate | {agg/c:5.2f}/stream | "
              f"{toks} tok / {wall:.1f}s | req-latency p50 {med:.1f}s"
              + (f" | ERRORS {len(errs)}: {errs[0]}" if errs else ""))
