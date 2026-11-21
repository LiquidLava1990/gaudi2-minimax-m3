#!/bin/bash
# Drive one PHASE of the int4-opt measurement against the LIVE server.
# Usage: run_int4_opt.sh <phase-tag> [spec]
#   <phase-tag>  e.g. baseline | fix1 | fix1_ngram
#   [spec]       if "spec", note that the server is running n-gram (just affects which probes we run)
# Reads BIND_IP/PORT/API_KEY from common.env. Uses the m3calib venv (openai; isolated from Kimi's hf venv).
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/../../common.env"
PHASE="${1:-baseline}"
MODE="${2:-nospec}"
PY=$HOME/.venvs/m3calib/bin/python3
URL="http://$BIND_IP:$PORT/v1"
OUT="$HERE/runs/int4_opt"
mkdir -p "$OUT"
A=(--base-url "$URL" --api-key "$API_KEY" --model "$SERVED_NAME")

echo "############ PHASE=$PHASE MODE=$MODE  ($(date +%H:%M:%S)) ############"

echo "===== COHERENCE ====="
$PY "$HERE/m3_probe.py" "${A[@]}" --prompt-kind paris  --max-tokens 128 --tag "${PHASE}_paris"  --json "$OUT/${PHASE}_paris.json"
$PY "$HERE/m3_probe.py" "${A[@]}" --prompt-kind reason --max-tokens 320 --tag "${PHASE}_reason" --json "$OUT/${PHASE}_reason.json"
$PY "$HERE/m3_probe.py" "${A[@]}" --prompt-kind code   --max-tokens 320 --tag "${PHASE}_code"   --json "$OUT/${PHASE}_code.json"

echo "===== SINGLE-STREAM (per-prompt decode tok/s, 256 tok) ====="
# code/structured (codegen — where n-gram prompt-lookup can fire) + prose (novel tokens)
$PY "$HERE/m3_probe.py" "${A[@]}" --prompt-kind codegen --max-tokens 256 --tag "${PHASE}_ss_codegen" --json "$OUT/${PHASE}_ss_codegen.json"
$PY "$HERE/m3_probe.py" "${A[@]}" --prompt-kind prose   --max-tokens 256 --tag "${PHASE}_ss_prose"   --json "$OUT/${PHASE}_ss_prose.json"

echo "===== SINGLE-STREAM (matrix-style 2K-prompt bench, c=1, 256 tok) ====="
$PY "$HERE/m3_bench.py" "${A[@]}" --concurrency 1 --num-prompts 3 --max-tokens 256 --prompt-tokens 2048 --json "$OUT/${PHASE}_bench_c1.json"

echo "############ PHASE=$PHASE done ($(date +%H:%M:%S)) ############"
