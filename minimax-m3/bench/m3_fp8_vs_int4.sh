#!/bin/bash
# Head-to-head bench helper for the M3 fp8-vs-int4 study. Runs against the ALREADY-SERVING endpoint
# (does NOT launch/stop containers — the caller manages the one container). Tag picks the output dir.
#   ./m3_fp8_vs_int4.sh <tag>           tag in {int4, fp8}
# Emits: coherence answers (3 prompts) + single-stream + aggregate c=10/30/50 JSON under runs/<tag>/.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/../../common.env"
TAG="${1:?usage: m3_fp8_vs_int4.sh <int4|fp8>}"
OUT="$HERE/runs/${TAG}_h2h"; mkdir -p "$OUT"
BASE="http://$BIND_IP:$PORT/v1"
AUTH="Authorization: Bearer $API_KEY"
MODEL="minimax-m3"

echo "================ COHERENCE ($TAG) ================"
coh() {  # $1=name $2=prompt
  local ans
  ans=$(curl -sf -m 180 -H "$AUTH" -H 'Content-Type: application/json' "$BASE/chat/completions" -d "$(python3 -c '
import json,sys
print(json.dumps({"model":"'"$MODEL"'","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":320,"temperature":0,"stream":False}))
' "$2")" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print("[finish:",d["choices"][0].get("finish_reason"),"]")' 2>/dev/null)
  echo "--- [$1] ---"; echo "$ans"; echo "$ans" > "$OUT/coh_$1.txt"; echo
}
coh paris    "What is the capital of France, and which river runs through it? Answer in one sentence."
coh reason   "A train travels 80 miles in 1 hour. At the same constant speed, how far does it travel in 3 hours and 30 minutes? Show your reasoning, then give the final answer."
coh code     "Write a Python function is_prime(n) that returns True if n is prime. Just the function."

echo "================ BENCH ($TAG) ================"
PYBENCH="${PYBENCH:-$HOME/.venvs/m3calib/bin/python3}"   # has openai; isolated from Kimi's hf venv
run() {  # $1=concurrency $2=num_prompts
  echo "### concurrency=$1 prompts=$2"
  "$PYBENCH" "$HERE/m3_bench.py" --base-url "$BASE" --api-key "$API_KEY" --model "$MODEL" \
    --concurrency "$1" --num-prompts "$2" --max-tokens 256 --prompt-tokens 2048 \
    --json "$OUT/c$1.json"
}
run 1 4      # single-stream latency floor
run 10 20
run 30 60
run 50 100
echo "================ DONE ($TAG) -> $OUT ================"
