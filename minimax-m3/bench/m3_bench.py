#!/usr/bin/env python3
"""M3 int4 benchmark driver — chat-completions, long-prompt capable.

Extends the stock bench_client.py approach (async OpenAI streaming -> aggregate
tok/s + TTFT p50/p99 + TPOT) with a configurable PROMPT CONTEXT length so we can
measure at a moderate 2K-8K-token prompt (the stock client's prompts are ~20
tokens). Uses the CHAT template (raw completions loop on this base model).

Each request: a long, unique "document" (filler paragraphs sized to ~--prompt-tokens)
followed by a short instruction. temperature=0, stream=True. Reports the same JSON
schema bench_client.py emits (throughput_tok_s, ttft_p50_s, ttft_p99_s, mean_tpot_ms)
so it slots into the existing tooling, plus per-request decode tok/s for the
single-stream latency-floor number.

Usage:
  m3_bench.py --base-url URL --api-key KEY --model NAME \
      --concurrency 10 --num-prompts 40 --max-tokens 256 --prompt-tokens 2048 [--json out.json]
"""
import argparse, asyncio, json, time, statistics, random
from openai import AsyncOpenAI

SYS = "You are a helpful, concise assistant."

# A bank of sentences to build unique long prompts (~ words). ~0.75 tok/word for this
# tokenizer-ish; we oversize then the instruction asks for a SHORT answer so gen length
# is controlled by max_tokens, not the prompt.
_BANK = [
    "The distributed scheduler reconciles desired and observed state on every tick.",
    "Memory bandwidth, not raw flops, bounds single-stream decode on this accelerator.",
    "Each routed expert dequantizes its int4 weights before the matrix multiply.",
    "Tail latency is dominated by the slowest collective in the all-reduce.",
    "Paged attention stores key and value blocks in a non-contiguous pool.",
    "Speculative decoding trades extra compute for fewer sequential steps.",
    "The fabric carries gradient-free activations between tensor-parallel ranks.",
    "Quantization error compounds across layers unless scales are calibrated.",
    "A warm recipe cache replays the compiled graph without recompilation.",
    "Throughput climbs with batch size until the device saturates or KV runs out.",
    "Prefill is compute-bound; decode is bandwidth-bound and host-sensitive.",
    "The sigmoid router assigns each token to its top experts by score.",
    "Block size feeds the matrix engine and changes the memory access pattern.",
    "An idle fabric shows no symbol errors and no local faults on any port.",
    "The KV cache in fp8 halves the per-token memory footprint versus bf16.",
]


def make_prompt(idx: int, approx_tokens: int) -> str:
    """Build a deterministic-but-unique prompt of ~approx_tokens tokens."""
    rng = random.Random(idx)
    # ~0.75 words/token heuristic -> words needed
    n_words_target = int(approx_tokens / 0.75)
    parts = []
    wc = 0
    while wc < n_words_target:
        s = rng.choice(_BANK)
        parts.append(s)
        wc += len(s.split())
    doc = " ".join(parts)
    return (f"Document #{idx} (reference material, do not summarize unless asked):\n{doc}\n\n"
            f"Question: In ONE short sentence, what single resource most limits "
            f"single-stream decode throughput on this accelerator?")


async def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0, stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
        dur = time.perf_counter() - t0
        return {"ok": True, "ttft": ttft or dur, "dur": dur, "ntok": ntok}
    except Exception as e:
        return {"ok": False, "err": str(e)[:300], "dur": time.perf_counter() - t0}


async def worker(client, model, max_tokens, ptoks, queue, out):
    while True:
        try:
            idx = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        prompt = make_prompt(idx, ptoks)
        out.append(await one_request(client, model, prompt, max_tokens))
        queue.task_done()


async def run(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=900)
    queue = asyncio.Queue()
    for i in range(args.num_prompts):
        queue.put_nowait(i)
    out = []
    t0 = time.perf_counter()
    workers = [asyncio.create_task(worker(client, args.model, args.max_tokens, args.prompt_tokens, queue, out))
               for _ in range(args.concurrency)]
    await asyncio.gather(*workers)
    wall = time.perf_counter() - t0

    ok = [r for r in out if r["ok"]]
    fail = [r for r in out if not r["ok"]]
    total_tok = sum(r["ntok"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok)
    # per-request decode tok/s (excludes prefill): ntok / (dur - ttft)
    perreq = [r["ntok"] / (r["dur"] - r["ttft"]) for r in ok if r["ntok"] > 1 and (r["dur"] - r["ttft"]) > 0]
    res = {
        "concurrency": args.concurrency,
        "prompt_tokens_approx": args.prompt_tokens,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "ok": len(ok), "fail": len(fail),
        "wall_s": round(wall, 2),
        "total_output_tokens": total_tok,
        "throughput_tok_s": round(total_tok / wall, 1) if wall else 0,
        "per_req_decode_tok_s_median": round(statistics.median(perreq), 1) if perreq else None,
        "req_per_s": round(len(ok) / wall, 2) if wall else 0,
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_p99_s": round(ttfts[int(len(ttfts)*0.99)-1], 3) if len(ttfts) > 1 else (ttfts[0] if ttfts else None),
        "mean_tpot_ms": round(1000*statistics.mean([(r["dur"]-r["ttft"])/max(1, r["ntok"]-1) for r in ok if r["ntok"] > 1]), 2) if ok else None,
    }
    if fail:
        res["sample_error"] = fail[0].get("err")
    print(json.dumps(res, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--num-prompts", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompt-tokens", type=int, default=2048)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    asyncio.run(run(args))
