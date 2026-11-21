#!/usr/bin/env bash
# profile_decode.sh — capture a batch=1 single-stream DECODE trace from a running vllm-gaudi serve,
# then tell you where it landed for parse_trace.py to read.
#
# PRECONDITION: the serve MUST have been launched with profiling enabled, i.e. with
#   VLLM_TORCH_PROFILER_DIR=/profiler_out  (+ optionally VLLM_PROFILER_ENABLED=full)
# in its docker env — see the "SERVE ENV ADDITIONS" block in the README / handoff. Without it,
# vllm-gaudi's init_profiler() builds NO profiler and POST /start_profile returns 500.
#
# This script uses the SAME request shape as m3_concurrency.py (chat/completions, greedy,
# temperature 0, non-stream) so the captured decode matches the concurrency-bench code path.
# It drives vLLM's HTTP profiler control endpoints:  POST /start_profile -> fire decode reqs ->
# POST /stop_profile  (the trace is flushed on stop, one *.pt.trace.json.gz per TP rank).
#
# USAGE:
#   ./profile_decode.sh [HOST:PORT] [API_KEY] [MODEL] [MAX_TOKENS] [N_REQ]
# defaults (match m3_concurrency.py / common.env):
#   HOST:PORT  <SERVER_IP>:8000
#   API_KEY    (from $API_KEY env, else the common.env key)
#   MODEL      minimax-m3
#   MAX_TOKENS 64     (short: we want STEADY-STATE decode steps, not a long generation)
#   N_REQ      3      (batch=1, sent sequentially -> pure single-stream)
#
# The trace dir is the HOST path that the serve's VLLM_TORCH_PROFILER_DIR was bind-mounted to.
# Override with PROFILER_OUT_HOST=/some/host/dir if you mounted it elsewhere.
set -uo pipefail

HP="${1:-${HOST_PORT:-<SERVER_IP>:8000}}"
KEY="${2:-${API_KEY:-${VLLM_API_KEY}}}"
MODEL="${3:-${MODEL:-minimax-m3}}"
MAX_TOKENS="${4:-${MAX_TOKENS:-64}}"
N_REQ="${5:-${N_REQ:-3}}"

# Host-side directory the serve's VLLM_TORCH_PROFILER_DIR is bind-mounted to. The default matches
# the established on-box layout (logs/prof carries the 2026-06-08 reference traces).
PROFILER_OUT_HOST="${PROFILER_OUT_HOST:-$HOME/AI/logs/prof}"

BASE="http://$HP"
AUTH=(); [ -n "$KEY" ] && AUTH=(-H "Authorization: Bearer $KEY")
HERE="$(cd "$(dirname "$0")" && pwd)"
PROMPT='Write a Python function that merges two sorted lists into one sorted list, then explain its time and space complexity step by step.'

say() { printf '%s\n' "$*" >&2; }
have_curl() { command -v curl >/dev/null 2>&1; }
have_curl || { say "FATAL: curl not found"; exit 2; }

# --- 0. health + auth probe -------------------------------------------------------------
say "== profile_decode: endpoint=$BASE model=$MODEL max_tokens=$MAX_TOKENS n_req=$N_REQ =="
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "${AUTH[@]}" "$BASE/v1/models" || echo 000)
if [ "$code" != "200" ]; then
  say "FATAL: GET /v1/models returned HTTP $code (serve down, wrong host:port, or bad API key). Aborting."
  exit 3
fi
say "   /v1/models OK (HTTP 200)"

# one decode request, batch=1, SAME shape as m3_concurrency.py:one()
fire_one() {
  local mt="$1"
  local body
  body=$(printf '{"model":"%s","messages":[{"role":"user","content":%s}],"max_tokens":%d,"temperature":0.0,"stream":false}' \
         "$MODEL" "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$mt")
  curl -s --max-time 600 "${AUTH[@]}" -H 'Content-Type: application/json' \
       -d "$body" "$BASE/v1/chat/completions"
}

# --- 1. WARM UP (compile decode buckets / populate recipe cache) so the captured steps are
#        steady-state, not first-shape compiles -----------------------------------------
say "-- warmup request (discarded; forces decode-bucket compile so the trace is steady-state) --"
warm=$(fire_one 8)
wtok=$(printf '%s' "$warm" | python3 -c 'import json,sys;
try: print(json.load(sys.stdin).get("usage",{}).get("completion_tokens","?"))
except Exception as e: print("ERR:"+repr(e)[:80])' 2>/dev/null)
say "   warmup completion_tokens=$wtok"
[ "$wtok" = "0" ] || [ -z "$wtok" ] && say "   (warmup produced $wtok tokens — check model name / serve health if this is 0/empty)"

# --- 2. snapshot existing traces so we can report ONLY the new ones ----------------------
shopt -s nullglob
before=("$PROFILER_OUT_HOST"/*.pt.trace.json.gz "$PROFILER_OUT_HOST"/*.pt.trace.json)
declare -A seen; for f in "${before[@]}"; do seen["$f"]=1; done
say "-- pre-existing traces in $PROFILER_OUT_HOST: ${#before[@]} --"

# --- 3. START profiler ------------------------------------------------------------------
say "-- POST /start_profile --"
sc=$(curl -s -o /tmp/pdec_start.$$ -w '%{http_code}' --max-time 60 "${AUTH[@]}" -X POST "$BASE/start_profile" || echo 000)
if [ "$sc" != "200" ]; then
  say "FATAL: /start_profile -> HTTP $sc. The serve was almost certainly NOT launched with"
  say "       VLLM_TORCH_PROFILER_DIR set (init_profiler built no profiler -> RuntimeError 'Profiler is not enabled')."
  say "       Relaunch with the SERVE ENV ADDITIONS, then re-run. Body:"
  say "       $(head -c 300 /tmp/pdec_start.$$ 2>/dev/null)"
  rm -f /tmp/pdec_start.$$
  exit 4
fi
rm -f /tmp/pdec_start.$$
say "   profiler started"

# --- 4. fire N batch=1 decode requests SEQUENTIALLY (pure single-stream) -----------------
ok=0
for i in $(seq 1 "$N_REQ"); do
  t0=$(date +%s.%N)
  resp=$(fire_one "$MAX_TOKENS")
  t1=$(date +%s.%N)
  read -r ctok finish < <(printf '%s' "$resp" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin); u=d.get("usage",{})
    fr=(d.get("choices",[{}])[0] or {}).get("finish_reason","?")
    print(u.get("completion_tokens",0), fr)
except Exception as e:
    print(0, "ERR:"+repr(e)[:60])' 2>/dev/null)
  dt=$(python3 -c "print(f'{$t1-$t0:.1f}')")
  say "   req $i/$N_REQ: completion_tokens=$ctok finish=$finish wall=${dt}s"
  [ "${ctok:-0}" -gt 0 ] 2>/dev/null && ok=$((ok+1))
done
[ "$ok" -eq 0 ] && say "   WARNING: 0 requests produced tokens — the trace may be empty."

# --- 5. STOP profiler (flushes the trace files) -----------------------------------------
say "-- POST /stop_profile (flushes trace) --"
ec=$(curl -s -o /dev/null -w '%{http_code}' --max-time 120 "${AUTH[@]}" -X POST "$BASE/stop_profile" || echo 000)
[ "$ec" = "200" ] && say "   profiler stopped" || say "   WARN: /stop_profile -> HTTP $ec (trace may still flush; check the dir)"

# --- 6. wait for the gz files to appear (the FileWriter/handler flush is async) ----------
say "-- waiting up to 60s for new trace files to land in $PROFILER_OUT_HOST --"
new=()
for _ in $(seq 1 30); do
  sleep 2
  new=()
  for f in "$PROFILER_OUT_HOST"/*.pt.trace.json.gz "$PROFILER_OUT_HOST"/*.pt.trace.json; do
    [ -e "$f" ] || continue
    [ -n "${seen["$f"]:-}" ] && continue
    new+=("$f")
  done
  [ "${#new[@]}" -gt 0 ] && break
done

echo "============================================================================" >&2
if [ "${#new[@]}" -eq 0 ]; then
  say "NO NEW TRACE FILES appeared in $PROFILER_OUT_HOST."
  say "Likely causes (in order):"
  say "  1. The serve's VLLM_TORCH_PROFILER_DIR points at a CONTAINER path NOT bind-mounted to"
  say "     $PROFILER_OUT_HOST. Find it:  docker exec $CONTAINER printenv VLLM_TORCH_PROFILER_DIR"
  say "     then re-run with PROFILER_OUT_HOST=<the host side of that bind mount>."
  say "  2. The trace is still flushing (large). List newest:"
  say "       ls -lt $PROFILER_OUT_HOST/*.pt.trace.json.gz | head"
  exit 5
fi
say "NEW TRACE FILE(S) (${#new[@]} — expect 1 per TP rank; rank0 carries the ProfilerStep markers):"
for f in "${new[@]}"; do printf '  %s\n' "$f" >&2; done
echo >&2
say "PARSE THEM:"
say "  python3 $HERE/parse_trace.py ${new[*]}"
say "or all at once / future runs:"
say "  python3 $HERE/parse_trace.py --glob '$PROFILER_OUT_HOST/*.pt.trace.json.gz'"
# also emit the parse command on STDOUT (so callers can capture it)
echo "python3 $HERE/parse_trace.py ${new[*]}"
