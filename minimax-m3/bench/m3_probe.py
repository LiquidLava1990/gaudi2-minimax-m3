#!/usr/bin/env python3
"""M3 coherence + single-stream probe (one request, precise per-token timing).

Sends ONE chat-completion (streaming, temp 0) and prints:
  - the full text (for coherence eyeballing)
  - finish_reason
  - decode tok/s (ntok / (dur - ttft)), ttft, ntok

Used for: coherence gate ("Paris"/"80 mph"/code) AND single-stream decode tok/s on a
specific prompt (code vs prose), incl. under spec-decode.

Usage:
  m3_probe.py --base-url URL --api-key KEY --model NAME --prompt-kind {paris,reason,code,prose,codegen} \
      [--max-tokens 256] [--tag NAME]
"""
import argparse, asyncio, json, time
from openai import AsyncOpenAI

SYS = "You are a helpful, concise assistant."

PROMPTS = {
    "paris": "What is the capital of France, and what river runs through it?",
    "reason": ("A train travels 80 miles in 1 hour at a constant speed. How far does it travel "
               "in 3 hours and 30 minutes? Show your reasoning."),
    "code": "Write a Python function is_prime(n) that returns True iff n is prime. Then call it on 97.",
    # a STRUCTURED/repetitive prompt where n-gram prompt-lookup can actually fire (the model is
    # asked to echo + lightly transform a block that already appears in the prompt):
    "codegen": (
        "Here is a Python function:\n\n"
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n\n"
        "Reproduce the EXACT same function verbatim, then add a one-line docstring as the first line "
        "of the body. Output only the code."),
    # a PROSE prompt (novel tokens, n-gram should NOT help):
    "prose": ("Write a short, original paragraph (about 120 words) describing a quiet morning beside "
              "a mountain lake. Use vivid sensory detail and do not repeat sentences."),
}


async def run(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=900)
    prompt = PROMPTS[args.prompt_kind]
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    chunks = []
    finish = None
    err = None
    try:
        stream = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
            max_tokens=args.max_tokens, temperature=0.0, stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            if ch.finish_reason:
                finish = ch.finish_reason
            delta = ch.delta.content
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
                chunks.append(delta)
    except Exception as e:
        err = str(e)[:500]
    dur = time.perf_counter() - t0
    text = "".join(chunks)
    dec = ntok / (dur - ttft) if (ttft and ntok > 1 and (dur - ttft) > 0) else None
    tag = args.tag or args.prompt_kind
    print(f"===== PROBE [{tag}] kind={args.prompt_kind} =====")
    if err:
        print(f"ERROR: {err}")
    print(text)
    print(f"----- finish={finish} ntok={ntok} ttft={ttft and round(ttft,3)}s "
          f"dur={round(dur,2)}s  DECODE_TOK_S={dec and round(dec,2)} -----")
    res = {"tag": tag, "kind": args.prompt_kind, "ok": err is None, "finish": finish,
           "ntok": ntok, "ttft_s": ttft and round(ttft, 3), "dur_s": round(dur, 2),
           "decode_tok_s": dec and round(dec, 2), "err": err,
           "text_head": text[:200]}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-kind", required=True, choices=list(PROMPTS))
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--tag", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    asyncio.run(run(args))
