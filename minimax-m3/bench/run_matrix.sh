#!/bin/bash
# ==================================================================================================
# run_matrix.sh — MiniMax-M3 benchmark MATRIX runner + per-config optimization loop.
#
# Drives the EXISTING box tooling (AI/serve.sh + AI/bench/bench_client.py) across the (weight x KV) x
# regime matrix, runs a knob optimization loop per config until decode tok/s plateaus, tears each
# config down on the safe (graceful) path, and emits ONE markdown table via aggregate_matrix.py.
#
# THE GRID = 3 WEIGHT runtimes x 3 KV modes (+ a baseline row).  EVERY config is measured at BOTH:
#     * SINGLE-STREAM  (batch=1) -> the latency-floor tok/s for one request
#     * the regime's TARGET CONCURRENCY -> the saturated aggregate tok/s
# so each row carries a single-stream tok/s column AND a concurrent tok/s column.
#
#   WEIGHT runtimes (3):
#     int4-in-HBM   M3_EXPERT_RUNTIME=int4 (fused int4->bf16 MoE kernel; MAX KV room) — int4 ckpt
#     int4->fp8     M3_EXPERT_RUNTIME=fp8  (int4->fp8 load transform + native fp8 op; MAX speed) — int4 ckpt
#     fp8-normal    stock INC: bf16 ckpt + --quantization inc + QUANT_CONFIG=maxabs_quant_g2.json
#                   (bf16->fp8 at LOAD; NO M3_EXPERT_RUNTIME) — bf16 ckpt
#   KV modes (3):  fp8_inc  |  TurboQuant k8v4 (4-bit V)  |  fp8 + MSA-offload   (k8v2/2-bit DROPPED)
#
# THE TABLE IS THE DELIVERABLE.  rows = configs; cols = best max-num-seqs, SINGLE-STREAM tok/s,
# CONCURRENT tok/s, prefill TTFT(p50), p99 TTFT, TPOT, fit point (ctx x conc), vs-baseline %.
#
# ------------------------------------------------------------------------------------------------
# REUSE, don't reinvent: serve.sh consumes models/minimax-m3.env and honors the override hooks this
# runner sets per config —
#     VLLM_ARGS_OVERRIDE    replaces VLLM_ARGS   (we rebuild the serve flag string per config/knob)
#     MODEL_ENV_EXTRA       appends docker -e    (M3_EXPERT_RUNTIME / PYTHONPATH / gate flags / recipe
#                                                 cache override / bucketing bounds / QUANT_CONFIG)
#     MODEL_PATH_OVERRIDE   swaps the served checkpoint   (int4 ckpt for int4/int4->fp8; bf16 for INC)
#     EXTRA_MOUNTS_OVERRIDE binds that checkpoint into /models
# and the mount toggles USE_DSA / USE_MSA (serve.sh learns USE_MSA when MSA lands — see GATES below).
# bench_client.py is the load generator (async OpenAI client -> decode tok/s + TTFT p50/p99 + TPOT).
# ------------------------------------------------------------------------------------------------
#
# CARD-FREE by default for AUTHORING:  `--dry-run` prints the full config matrix it WILL execute, the
# knob ranges, the exact env composition per config, and a sample empty table — and launches NOTHING.
# A real run requires --run AND the gates below; without them it refuses (loud) so it can never touch
# the cards / vllm-active by accident.
#
# Usage:
#   run_matrix.sh --dry-run [configs...]      # card-free: print matrix + knobs + env + empty table
#   run_matrix.sh --run [configs...]          # CARD-GATED: actually serve+bench (one config at a time)
#       configs... = subset of ids (c0 c1 ... c9); empty => all. e.g.  run_matrix.sh --run c0 c1
# ==================================================================================================
set -uo pipefail

AI=$HOME/AI
BENCH="$AI/minimax-m3/bench"
MODEL_KEY=minimax-m3
PY=$HOME/.venvs/hf/bin/python; [ -x "$PY" ] || PY=python3
M3DIR="$AI/minimax-m3"
RUNS="$BENCH/runs"
REPORT="$BENCH/M3_MATRIX_REPORT.md"

source "$AI/common.env"
URL="http://$BIND_IP:$PORT/v1"

# ------------------------------------------------------------------------------------------------
# CHECKPOINTS — the two on-disk model dirs the grid serves (a weight runtime picks one):
#   INT4 ckpt : the AutoRound int4-expert export. BOTH int4-in-HBM and int4->fp8 serve THIS one
#               (same on-disk weights; only the HBM staging differs — int4_to_fp8_load.py).
#   BF16 ckpt : the original bf16 checkpoint. fp8-normal(INC) serves THIS one and lets INC cast
#               bf16->fp8 at load via QUANT_CONFIG (NOT the int4 export).
# These map to a container-side MODEL_PATH (mounted under /models) via *_MOUNT below; the runner
# hands serve.sh MODEL_PATH_OVERRIDE + EXTRA_MOUNTS_OVERRIDE per config so the right ckpt is served
# without editing models/minimax-m3.env (which defaults to the bf16/INC config).
INT4_CKPT_HOST="/mnt/fastmodels/MiniMax-M3-int4-experts-AutoRound"   # quantize_m3_int4.py OUT_DIR
BF16_CKPT_HOST="/mnt/data2/minimax-m3-bf16"                          # original bf16 checkpoint
INT4_MODEL_PATH="MiniMax-M3-int4-experts-AutoRound"                  # /models/<this> (under MODELS_DIR=/mnt/fastmodels)
BF16_MODEL_PATH="MiniMax-M3"                                         # /models/<this> via the bind mount below
# Always bind the chosen ckpt EXPLICITLY (a non-empty EXTRA_MOUNTS_OVERRIDE -> serve.sh replaces the
# env-file's bf16 bind, so an int4 config never leaves the bf16 bind dangling at /models/MiniMax-M3).
INT4_MOUNT="-v $INT4_CKPT_HOST:/models/$INT4_MODEL_PATH:ro"          # int4 ckpt (under MODELS_DIR; explicit bind = override is non-empty)
BF16_MOUNT="-v $BF16_CKPT_HOST:/models/$BF16_MODEL_PATH:ro"          # bf16 ckpt is off /mnt/fastmodels -> bind it in
# INC scale source (QUANT_CONFIG) for fp8-normal — the QUANTIZE json the calibrate pass writes.
# Host path (audit) + the container path INC reads (QUANT_DIR=/mnt/data1/quant mounted at /quant).
QUANT_CONFIG_HOST="$QUANT_DIR/minimax-m3-fp8-inc/minimax-m3/maxabs_quant_g2.json"
QUANT_CONFIG_CTR="/quant/minimax-m3-fp8-inc/minimax-m3/maxabs_quant_g2.json"

# ------------------------------------------------------------------------------------------------
# MODE
# ------------------------------------------------------------------------------------------------
MODE="dry-run"
WANT=()
for a in "$@"; do
  case "$a" in
    --dry-run) MODE="dry-run" ;;
    --run)     MODE="run" ;;
    c[0-9]|c[0-9]*) WANT+=("$a") ;;
    *) echo "unknown arg: $a (use --dry-run | --run | a config id like c0)"; exit 2 ;;
  esac
done
want() { # is config $1 in the requested subset (empty subset => all)?
  [ ${#WANT[@]} -eq 0 ] && return 0
  local id="$1" w
  for w in "${WANT[@]}"; do [[ "$id" == "$w"* ]] && return 0; done
  return 1
}

# ================================================================================================
# THE MATRIX  (config x regime) = 3 WEIGHT runtimes x 3 KV modes (+ baseline).  ONE row per CONFIG;
# each carries its regime list. Every config is benched at single-stream (batch=1) AND its target conc.
# Fields ('^'-separated — the human label contains '|' (weight | KV), so '^' is the separator):
#   id ^ label ^ WEIGHT ^ KVDTYPE ^ KV_GATE ^ MSA ^ REGIMES
#     WEIGHT  : weight runtime + which checkpoint is served (compose_env maps these):
#                 int4      -> M3_EXPERT_RUNTIME=int4 (fused int4->bf16) on the INT4 ckpt   (max KV)
#                 fp8native -> M3_EXPERT_RUNTIME=fp8  (int4->fp8 native) on the INT4 ckpt   (max speed)
#                 fp8inc    -> stock INC bf16->fp8 (--quantization inc + QUANT_CONFIG) on the BF16 ckpt
#     KVDTYPE : the --kv-cache-dtype value passed to vllm serve (fp8_inc | turboquant_k8v4)
#     KV_GATE : extra -e gate flags for the KV path ("" | KVQ_READY=1 for TurboQuant k8v4)
#     MSA     : 0 | 1  (1 => HPU_MSA_READY=1 + USE_MSA=1 mount; unlocks long ctx + KV offload)
#     REGIMES : space-separated  <ctxTokens>:<targetConc>  points to sweep for THIS config. ctx tokens
#               use K/M suffixes expanded by ctx_tokens(). EACH regime is measured single-stream + @conc.
# ------------------------------------------------------------------------------------------------
# REGIME choice per KV mode (per the brief): fp8-speed KV (fp8_inc, TurboQuant) -> moderate/long ctx;
# int4-in-HBM (max KV room) -> long ctx; MSA-offload -> the 1M target.
# baseline #0 == (int4-in-HBM x fp8_inc) and is the vs-% reference. The other 8 cells complete the
# 3x3 grid. Configs carrying TurboQuant / MSA gates are NOT card-validated yet (see GATES): the
# dry-run shows them; a --run SKIPS a config whose gate is unmet (loud) rather than serve a wrong
# precision / a fake number.
#
#                  GRID  (rows=weight runtime, cols=KV mode):
#                  |  KV fp8_inc   |  KV TurboQuant k8v4 |  KV fp8 + MSA-offload
#   int4-in-HBM    |  c0 (BASE)    |  c3                 |  c6
#   int4->fp8      |  c1           |  c4                 |  c7
#   fp8-normal INC |  c2           |  c5                 |  c8
CONFIGS=(
  # --- KV fp8_inc column ---
  "c0_int4hbm_fp8inc^#0 int4-in-HBM (fused) | KV fp8_inc^int4^fp8_inc^^0^128K:30 256K:30"
  "c1_fp8native_fp8inc^#1 int4->fp8 (native) | KV fp8_inc^fp8native^fp8_inc^^0^32K:30 128K:30"
  "c2_fp8inc_fp8inc^#2 fp8-normal (INC) | KV fp8_inc^fp8inc^fp8_inc^^0^32K:30 128K:30"
  # --- KV TurboQuant k8v4 column (4-bit V; k8v2/2-bit DROPPED) ---
  "c3_int4hbm_tq^#3 int4-in-HBM (fused) | KV TurboQuant^int4^turboquant_k8v4^KVQ_READY=1^0^256K:30"
  "c4_fp8native_tq^#4 int4->fp8 (native) | KV TurboQuant^fp8native^turboquant_k8v4^KVQ_READY=1^0^128K:30"
  "c5_fp8inc_tq^#5 fp8-normal (INC) | KV TurboQuant^fp8inc^turboquant_k8v4^KVQ_READY=1^0^128K:30"
  # --- KV fp8 + MSA-offload column (the 1M target) ---
  "c6_int4hbm_msa^#6 int4-in-HBM (fused) | KV fp8 + MSA-offload^int4^fp8_inc^^1^1M:15"
  "c7_fp8native_msa^#7 int4->fp8 (native) | KV fp8 + MSA-offload^fp8native^fp8_inc^^1^1M:15"
  "c8_fp8inc_msa^#8 fp8-normal (INC) | KV fp8 + MSA-offload^fp8inc^fp8_inc^^1^1M:15"
)

ctx_tokens() { # 32K -> 32768 ; 128K -> 131072 ; 256K -> 262144 ; 1M -> 1048576
  case "$1" in
    *K) echo $(( ${1%K} * 1024 )) ;;
    *M) echo $(( ${1%M} * 1024 * 1024 )) ;;
    *)  echo "$1" ;;
  esac
}

weight_desc() { # human one-liner for a WEIGHT token (used by the dry-run audit)
  case "$1" in
    int4)      echo "int4-in-HBM  (M3_EXPERT_RUNTIME=int4; fused int4->bf16 MoE) on the INT4 ckpt" ;;
    fp8native) echo "int4->fp8    (M3_EXPERT_RUNTIME=fp8; int4->fp8 load transform + native fp8 op) on the INT4 ckpt" ;;
    fp8inc)    echo "fp8-normal   (stock INC: --quantization inc + QUANT_CONFIG; bf16->fp8 at load, NO M3_EXPERT_RUNTIME) on the BF16 ckpt" ;;
    *)         echo "unknown weight runtime '$1'" ;;
  esac
}

# ================================================================================================
# THE OPTIMIZATION LOOP — knob ranges swept per config until decode tok/s plateaus.
# Coordinate-descent (OFAT), NOT cartesian: start from the per-config seed, walk ONE axis at a time,
# keep each axis winner, stop an axis when the next step gains < PLATEAU_PCT over the current best.
# (~9-lever cartesian * minutes/run = days; OFAT captures the levers that move decode tok/s here.)
# ------------------------------------------------------------------------------------------------
SWEEP_MAX_NUM_SEQS=(64 128 192)        # batch capacity  (the headline knob; "saturate the batch")
SWEEP_BLOCK_SIZE=(128 64)              # HPU paged block; 128 feeds the MME (expect it wins on Gaudi 2)
SWEEP_GMEM=(0.90 0.94 0.96)            # gpu-memory-utilization -> KV headroom (watch OOM at long ctx)
SWEEP_RECIPE_CACHE=(off on)            # PT_HPU_RECIPE_CACHE_CONFIG validation flag (warm-replay vs cold)
# exponential-bucketing bounds: VLLM_EXPONENTIAL_BUCKETING on (default) vs hand-tuned linear decode
# block-bucket MAX = max(128, max_num_seqs*max_model_len/block_size). "exp" = default; "tuned" sets
# the linear decode-block knobs around that target for the long-ctx points.
SWEEP_BUCKETING=(exp tuned)
PLATEAU_PCT=3                          # stop walking an axis when a step gains < this % decode tok/s
MAX_TOKENS=256                         # decode length per request (matches run_cycle.sh)
WARM_C=(1 64)                          # warmup concurrencies to pre-compile bs buckets before MEASURE
SINGLE_STREAM_C=1                      # the batch=1 sweep point measured for EVERY config (latency floor)

# ------------------------------------------------------------------------------------------------
# GATES — what must be true on the box for a --run to be HONEST. Checked per config; an unmet gate
# SKIPS that config loudly (never serves a wrong precision / fake number).
#   int4 weight runtimes (int4 / fp8native) need the INT4 export; fp8-normal(INC) needs the BF16
#   checkpoint AND the QUANT_CONFIG scale json (the calibrate MEASURE->QUANTIZE must have run on-card).
# ------------------------------------------------------------------------------------------------
gate_int4_ckpt() { [ -d "$INT4_CKPT_HOST" ]; }                       # int4 / int4->fp8 need the int4 export
gate_fp8inc_ckpt() { [ -d "$BF16_CKPT_HOST" ] && [ -f "$QUANT_CONFIG_HOST" ]; }  # INC: bf16 ckpt + QUANTIZE scales
gate_turboquant() { [ -d "$AI/hpu-kvq-ext/hpu_kvq_ext" ]; }         # KVQ plugin pkg (NOT built yet)
gate_msa() { [ -d "$AI/hpu-msa-ext/hpu_msa_ext" ] && grep -q 'USE_MSA' "$AI/serve.sh"; }  # MSA plugin + serve hook

# ================================================================================================
# ENV COMPOSITION — build the MODEL_ENV_EXTRA (-e flags) + USE_* mounts + VLLM_ARGS_OVERRIDE for a
# given config + a given knob point.  This is the single place precision/KV/gate flags are assembled,
# so the dry-run audit and the real run use IDENTICAL composition.
# ------------------------------------------------------------------------------------------------
# compose_env  <WEIGHT> <KVDTYPE> <KV_GATE> <MSA> <ctxTokens> <mns> <blk> <gmem> <recipe> <bucketing>
#   WEIGHT in {int4, fp8native, fp8inc}
#   -> sets globals:  ENV_EXTRA   (string for MODEL_ENV_EXTRA)
#                     USE_FLAGS   (string of env-var assignments to prefix serve.sh, e.g. "USE_MSA=1")
#                     ARGS_OVR    (string for VLLM_ARGS_OVERRIDE)
#                     MPATH_OVR   (string for MODEL_PATH_OVERRIDE   — which checkpoint to serve)
#                     MOUNT_OVR   (string for EXTRA_MOUNTS_OVERRIDE — the bind for that checkpoint)
compose_env() {
  local rt="$1" kvdtype="$2" kvgate="$3" msa="$4" ctxtok="$5" mns="$6" blk="$7" gmem="$8" recipe="$9" bucketing="${10}"
  ENV_EXTRA=""; USE_FLAGS=""; ARGS_OVR=""; MPATH_OVR=""; MOUNT_OVR=""

  # --- weight runtime (weight format in HBM) + which checkpoint is served ---
  # int4      = the default fused int4->bf16 hpu-moe-ext path (max KV room) on the INT4 ckpt. The
  #             m3_expert_runtime_patch is a no-op at int4.
  # fp8native = the int4->fp8 load-time transform + native per-channel fp8 op (max speed) on the INT4
  #             ckpt. Needs the patch on PYTHONPATH + M3_EXPERT_RUNTIME=fp8 (m3_expert_runtime_patch).
  # fp8inc    = STOCK INC: serve the BF16 checkpoint, let INC cast bf16->fp8 AT LOAD from the measured
  #             maxabs scales (QUANT_CONFIG). This is the models/minimax-m3.env phase-1 path, distinct
  #             from int4->fp8: NO M3_EXPERT_RUNTIME (no int4 export at all). INC also owns the fp8 KV
  #             scales, so --quantization inc is required (kept in ARGS_OVR below for all weights).
  case "$rt" in
    int4)
      ENV_EXTRA+=" -e M3_EXPERT_RUNTIME=int4"
      MPATH_OVR="$INT4_MODEL_PATH"; MOUNT_OVR="$INT4_MOUNT" ;;
    fp8native)
      ENV_EXTRA+=" -e M3_EXPERT_RUNTIME=fp8 -e PYTHONPATH=$M3DIR -e VLLM_HPU_FORCE_CHANNEL_FP8=true"
      MPATH_OVR="$INT4_MODEL_PATH"; MOUNT_OVR="$INT4_MOUNT" ;;
    fp8inc)
      # INC master switch + the lazy/weight-sharing env INC needs (mirrors models/minimax-m3.env
      # MODEL_ENV). NO M3_EXPERT_RUNTIME (this is the bf16 ckpt, not the int4 export).
      ENV_EXTRA+=" -e QUANT_CONFIG=$QUANT_CONFIG_CTR -e PT_HPU_LAZY_MODE=1 -e PT_HPU_WEIGHT_SHARING=0"
      ENV_EXTRA+=" -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e VLLM_DISABLE_MARK_SCALES_AS_CONST=true"
      MPATH_OVR="$BF16_MODEL_PATH"; MOUNT_OVR="$BF16_MOUNT" ;;
    *) echo "compose_env: unknown WEIGHT '$rt' (want int4|fp8native|fp8inc)" >&2; return 2 ;;
  esac

  # --- KV path gate (TurboQuant) ---
  # fp8_inc is the ONLY HPU-native KV dtype today. TurboQuant k8v4 (4-bit V; the k8v2/2-bit mode is
  # DROPPED — V-error too high) is an OUT-OF-TREE OpBackend KV kernel gated KVQ_READY=1 (loud-fallback
  # -to-fp8_inc until on-card-validated) — same discipline as DSA_READY/MSA_READY. Until the kernel +
  # platform-gate extension exist, gate_turboquant() is false and the --run loop skips c3/c4/c5.
  [ -n "$kvgate" ] && ENV_EXTRA+=" -e $kvgate"

  # --- MSA sparse + offload gate ---
  # HPU_MSA_READY=1 flips the platform patch to serve sparse (top-16 block reads) -> 1M-tractable; the
  # USE_MSA=1 mount installs hpu-msa-ext (mirrors USE_DSA). Until both exist the loop skips c6/c7/c8.
  if [ "$msa" = "1" ]; then
    ENV_EXTRA+=" -e HPU_MSA_READY=1"
    USE_FLAGS+=" USE_MSA=1"
  fi

  # --- recipe-cache validation flag ---
  # serve.sh hardcodes PT_HPU_RECIPE_CACHE_CONFIG=/recipe_cache/<key>,false,8192 (validation OFF for
  # warm replay). To A/B recipe-cache we override it via MODEL_ENV_EXTRA (a later -e wins): "on" turns
  # validation ON (cold-correct, recompiles on any change); "off" keeps the warm-replay default.
  if [ "$recipe" = "on" ]; then
    ENV_EXTRA+=" -e PT_HPU_RECIPE_CACHE_CONFIG=/recipe_cache/$MODEL_KEY,true,8192"
  fi

  # --- exponential vs tuned-linear bucketing bounds ---
  # exp = VLLM_EXPONENTIAL_BUCKETING=true (default since 1.22). tuned = linear decode-block knobs sized
  # to this config's (mns x ctx / block) so the steady-state decode shape isn't over-bucketed.
  if [ "$bucketing" = "tuned" ]; then
    local dbmax=$(( mns * ctxtok / blk )); [ "$dbmax" -lt 128 ] && dbmax=128
    ENV_EXTRA+=" -e VLLM_EXPONENTIAL_BUCKETING=false -e VLLM_DECODE_BS_BUCKET_MIN=1 -e VLLM_DECODE_BLOCK_BUCKET_STEP=128 -e VLLM_DECODE_BLOCK_BUCKET_MAX=$dbmax"
  else
    ENV_EXTRA+=" -e VLLM_EXPONENTIAL_BUCKETING=true"
  fi

  # --- the serve flag string (replaces VLLM_ARGS wholesale) ---
  # Mirrors minimax-m3.env's VLLM_ARGS but with the swept knobs + this config's ctx/KV substituted.
  # --quantization inc stays for ALL weight runtimes: INC owns the fp8 KV scales + activation
  # calibration in every mode. For int4/fp8native the EXPERT weights come from the int4 export and the
  # runtime is chosen by M3_EXPERT_RUNTIME (orthogonal to INC); for fp8-normal(INC) the same flag is
  # ALSO what casts the bf16 experts->fp8 at load via QUANT_CONFIG. --no-enable-prefix-caching is kept
  # (required once MSA is on; harmless otherwise).
  ARGS_OVR="--quantization inc --kv-cache-dtype $kvdtype --tensor-parallel-size 8 --enable-expert-parallel"
  ARGS_OVR+=" --block-size $blk --max-num-seqs $mns --gpu-memory-utilization $gmem"
  ARGS_OVR+=" --max-model-len $ctxtok --async-scheduling --no-enable-prefix-caching --trust-remote-code"
}

# ================================================================================================
# DRY-RUN — card-free authoring view: full matrix, knob ranges, per-config env composition, empty table.
# ================================================================================================
if [ "$MODE" = "dry-run" ]; then
  echo "=================================================================================================="
  echo "MiniMax-M3 benchmark matrix — DRY RUN (card-free). NOTHING is launched. No serve, no card touch."
  echo "=================================================================================================="
  echo
  echo "MODEL_KEY=$MODEL_KEY   serve.sh=$AI/serve.sh   bench_client=$AI/bench/bench_client.py"
  echo "endpoint(when run)=$URL   reports -> $REPORT   runs -> $RUNS"
  echo
  echo "--- KNOB RANGES (per-config optimization loop; OFAT, plateau<${PLATEAU_PCT}% stops an axis) ---"
  echo "  --max-num-seqs        : ${SWEEP_MAX_NUM_SEQS[*]}"
  echo "  --block-size          : ${SWEEP_BLOCK_SIZE[*]}"
  echo "  --gpu-memory-util     : ${SWEEP_GMEM[*]}"
  echo "  recipe-cache validate : ${SWEEP_RECIPE_CACHE[*]}   (PT_HPU_RECIPE_CACHE_CONFIG ...,<off=false|on=true>,8192)"
  echo "  bucketing bounds      : ${SWEEP_BUCKETING[*]}      (exp=VLLM_EXPONENTIAL_BUCKETING; tuned=linear decode-block knobs)"
  echo "  decode max_tokens     : $MAX_TOKENS   warmup concurrencies: ${WARM_C[*]}"
  echo "  measured per config   : single-stream (batch=$SINGLE_STREAM_C) tok/s  AND  concurrent (target c) aggregate tok/s"
  echo
  echo "--- CONFIG MATRIX (config x regime) + ENV COMPOSITION it will execute ---"
  for line in "${CONFIGS[@]}"; do
    IFS='^' read -r id label rt kvdtype kvgate msa regimes <<<"$line"
    want "$id" || continue
    echo
    echo "================================================================================================"
    printf "CONFIG %-20s  %s\n" "$id" "$label"
    echo "  weight-runtime [$rt] = $(weight_desc "$rt")"
    echo "  --kv-cache-dtype $kvdtype    KV-gate=[${kvgate:-none}]    MSA=$msa"
    echo "  regimes (each measured at BOTH single-stream batch=1 AND target concurrency):"
    for reg in $regimes; do
      ctxlabel="${reg%%:*}"; conc="${reg##*:}"; ctxtok=$(ctx_tokens "$ctxlabel")
      echo "     - $ctxlabel ($ctxtok tok): single-stream c=$SINGLE_STREAM_C  +  concurrent c=$conc   [fit point]"
    done
    # show the SEED env composition (first knob point) so the precision/gate flags are auditable
    seedreg="${regimes%% *}"; ctxtok=$(ctx_tokens "${seedreg%%:*}")
    compose_env "$rt" "$kvdtype" "$kvgate" "$msa" "$ctxtok" "${SWEEP_MAX_NUM_SEQS[0]}" "${SWEEP_BLOCK_SIZE[0]}" "${SWEEP_GMEM[0]}" "off" "exp"
    echo "  SEED env composition (knob point mns=${SWEEP_MAX_NUM_SEQS[0]} blk=${SWEEP_BLOCK_SIZE[0]} gmem=${SWEEP_GMEM[0]} recipe=off bucketing=exp):"
    echo "     checkpoint served : ${MPATH_OVR}   ($([ "$rt" = fp8inc ] && echo "bf16 ckpt $BF16_CKPT_HOST" || echo "int4 ckpt $INT4_CKPT_HOST"))"
    echo "     MODEL_PATH_OVERRIDE   : $MPATH_OVR"
    echo "     EXTRA_MOUNTS_OVERRIDE : $MOUNT_OVR"
    echo "     USE flags         : ${USE_FLAGS:-<none>}"
    echo "     MODEL_ENV_EXTRA   :${ENV_EXTRA}"
    echo "     VLLM_ARGS_OVERRIDE: $ARGS_OVR"
    # gate verdict (informational in dry-run)
    g_ok="OK"; g_why=""
    if [ "$rt" = "fp8inc" ]; then
      gate_fp8inc_ckpt || { g_ok="GATED"; g_why+=" [fp8-normal(INC) needs bf16 ckpt $BF16_CKPT_HOST + QUANT_CONFIG $QUANT_CONFIG_HOST]"; }
    else
      gate_int4_ckpt || { g_ok="GATED"; g_why+=" [int4-ckpt missing: $INT4_CKPT_HOST]"; }
    fi
    [ -n "$kvgate" ] && { gate_turboquant || { g_ok="GATED"; g_why+=" [TurboQuant kernel/pkg hpu-kvq-ext missing]"; }; }
    [ "$msa" = "1" ] && { gate_msa || { g_ok="GATED"; g_why+=" [MSA plugin hpu-msa-ext + serve USE_MSA hook missing]"; }; }
    echo "  GATE STATUS (for --run): $g_ok$g_why"
  done
  echo
  echo "================================================================================================"
  echo "--- SAMPLE EMPTY TABLE (schema of the single deliverable) ---"
  echo "================================================================================================"
  "$PY" "$BENCH/aggregate_matrix.py" --dry-run-rows
  echo
  echo "DRY RUN complete. To execute (CARD-GATED, one config at a time):  $0 --run [configs...]"
  exit 0
fi

# ================================================================================================
# REAL RUN  (CARD-GATED).  Refuse unless explicitly --run.  One config at a time (serve.sh enforces
# the single vllm-active container; we always tear down on the safe graceful path between configs).
# ================================================================================================
mkdir -p "$RUNS"
echo "MiniMax-M3 benchmark matrix — REAL RUN. Serving one config at a time on $URL."
echo "Each config measured at single-stream (c=$SINGLE_STREAM_C) AND its target concurrency."
echo "Gates: int4-ckpt=$INT4_CKPT_HOST  fp8-normal(INC)=bf16 ckpt+QUANT_CONFIG  TurboQuant=hpu-kvq-ext  MSA=hpu-msa-ext+USE_MSA"

# one bench measurement at a fixed concurrency -> JSON via the existing client. echoes the file.
bench_at() { # <served> <conc> <jsonout>
  local served="$1" c="$2" out="$3" np=$(( c * 4 )); [ "$np" -lt 8 ] && np=8
  "$PY" "$AI/bench/bench_client.py" --base-url "$URL" --api-key "$API_KEY" --model "$served" \
      --concurrency "$c" --num-prompts "$np" --max-tokens "$MAX_TOKENS" --json "$out" >/dev/null 2>&1 || true
}
jget() { "$PY" -c "import json,os;p='$1';print(json.load(open(p)).get('$2','') if os.path.exists(p) else '')" 2>/dev/null; }

# bring up ONE config at a knob point, wait for health, return 0 if ready (else dump + teardown).
# Threads the per-config checkpoint swap (MODEL_PATH_OVERRIDE / EXTRA_MOUNTS_OVERRIDE) so int4/fp8
# configs serve the int4 export and fp8-normal(INC) serves the bf16 checkpoint — see compose_env.
serve_up() { # <ENV_EXTRA> <USE_FLAGS> <ARGS_OVR> <MPATH_OVR> <MOUNT_OVR>
  local env_extra="$1" use_flags="$2" args_ovr="$3" mpath_ovr="$4" mount_ovr="$5"
  # shellcheck disable=SC2086
  env $use_flags VLLM_ARGS_OVERRIDE="$args_ovr" MODEL_ENV_EXTRA="$env_extra" \
      MODEL_PATH_OVERRIDE="$mpath_ovr" EXTRA_MOUNTS_OVERRIDE="$mount_ovr" DOCKER="sudo docker" \
      "$AI/serve.sh" "$MODEL_KEY" >/tmp/m3_matrix_launch.log 2>&1 || { echo "  LAUNCH FAILED"; sudo docker logs --tail 30 "$CONTAINER" 2>&1 | sed 's/^/    /'; return 1; }
  for i in $(seq 1 300); do          # up to ~50 min: long-ctx warmup compiles many buckets
    curl -sf -H "Authorization: Bearer $API_KEY" "$URL/models" >/dev/null 2>&1 && return 0
    sudo docker ps --filter "name=$CONTAINER" --format '{{.Names}}' | grep -q "$CONTAINER" || { echo "  CONTAINER EXITED during warmup"; sudo docker logs --tail 30 "$CONTAINER" 2>&1 | sed 's/^/    /'; return 1; }
    sleep 10
  done
  echo "  TIMED OUT waiting for health"; return 1
}
serve_down() { sudo DOCKER="sudo docker" bash -c "source $AI/gaudi_guard.sh; safe_stop $CONTAINER" >/dev/null 2>&1 || sudo docker stop -t 90 "$CONTAINER" >/dev/null 2>&1 || true; }

# ---- per-config optimization loop ----
# OFAT coordinate descent. Seed = first value of every axis. Walk one axis, keep its best by decode
# tok/s at the regime's target concurrency; advance only if the gain >= PLATEAU_PCT. Carries the best
# of each axis into the next. Each trial ALSO records a single-stream (batch=1) measurement (the
# winning trial's single-stream number is what the table reports). Writes per-trial JSON (knob-tagged)
# + a best.json the aggregator reads.  ($2 = WEIGHT token int4|fp8native|fp8inc.)
optimize_config() { # <rundir> <weight> <kvdtype> <kvgate> <msa> <served> <ctxtok> <conc>
  local rundir="$1" rt="$2" kvdtype="$3" kvgate="$4" msa="$5" served="$6" ctxtok="$7" conc="$8"
  local b_mns="${SWEEP_MAX_NUM_SEQS[0]}" b_blk="${SWEEP_BLOCK_SIZE[0]}" b_gmem="${SWEEP_GMEM[0]}" b_recipe="off" b_buck="exp"
  local best_tps=-1 best_json="" best_tag=""

  trial() { # tag mns blk gmem recipe bucketing -> echoes CONCURRENT decode tok/s (empty on fail); updates best_*
    local tag="$1" mns="$2" blk="$3" gmem="$4" recipe="$5" buck="$6"
    local jf="$rundir/trial_${tag}__c${conc}.json"
    local ssf="$rundir/trial_${tag}__ss${SINGLE_STREAM_C}.json"   # single-stream (batch=1) result
    compose_env "$rt" "$kvdtype" "$kvgate" "$msa" "$ctxtok" "$mns" "$blk" "$gmem" "$recipe" "$buck"
    # progress goes to STDERR so the only thing on STDOUT (captured by `cur=$(trial ...)`) is the tps.
    echo "    [trial $tag] mns=$mns blk=$blk gmem=$gmem recipe=$recipe buck=$buck" >&2
    serve_up "$ENV_EXTRA" "$USE_FLAGS" "$ARGS_OVR" "$MPATH_OVR" "$MOUNT_OVR" || { serve_down; echo ""; return; }
    for w in "${WARM_C[@]}"; do bench_at "$served" "$w" "/tmp/m3_warm_$w.json"; done   # pre-compile bs buckets
    # measure BOTH operating points while the server is up: single-stream (latency floor) + concurrent.
    bench_at "$served" "$SINGLE_STREAM_C" "$ssf"
    bench_at "$served" "$conc" "$jf"
    serve_down
    local tps ss_tps; tps=$(jget "$jf" throughput_tok_s); ss_tps=$(jget "$ssf" throughput_tok_s)
    # annotate the trial json with the knobs + single-stream tok/s (+ its ttft/tpot) + fit flag for the
    # aggregator. fit_ok = the CONCURRENT run completed (ok>0 && fail==0).
    "$PY" - "$jf" "$mns" "$blk" "$gmem" "$recipe" "$buck" "$tag" "$conc" "$ssf" "$SINGLE_STREAM_C" <<'PYEOF' 2>/dev/null || true
import json,sys
p,mns,blk,gmem,recipe,buck,tag,conc,ssf,ssc=sys.argv[1:11]
try: d=json.load(open(p))
except Exception: d={}
try: s=json.load(open(ssf))
except Exception: s={}
d.update(max_num_seqs=int(mns),block_size=int(blk),gpu_mem_util=float(gmem),recipe_cache=recipe,
         bucketing=buck,knobtag=tag,conc=int(conc),
         single_stream_conc=int(ssc),
         single_stream_tok_s=s.get("throughput_tok_s"),
         single_stream_ttft_p50_s=s.get("ttft_p50_s"),
         single_stream_tpot_ms=s.get("mean_tpot_ms"),
         fit_ok=bool(d.get("ok") and not d.get("fail")))
json.dump(d,open(p,"w"),indent=2)
PYEOF
    # winner is picked by CONCURRENT throughput (the saturation headline); the winning trial carries
    # its own single-stream number alongside.
    if [ -n "$tps" ]; then
      awk -v a="$tps" -v b="$best_tps" 'BEGIN{exit !(a>b)}' && { best_tps="$tps"; best_json="$jf"; best_tag="$tag"; }
    fi
    echo "    -> single-stream=${ss_tps:-FAIL} tok/s   concurrent(c=$conc)=${tps:-FAIL} tok/s" >&2
    echo "$tps"
  }
  improved() { awk -v a="$1" -v b="$2" -v p="$PLATEAU_PCT" 'BEGIN{exit !(b>0 && (a-b)/b*100>=p)}'; }

  # seed
  local cur; cur=$(trial "mns${b_mns}_blk${b_blk}_gmem${b_gmem}_recipe${b_recipe}_buck${b_buck}" "$b_mns" "$b_blk" "$b_gmem" "$b_recipe" "$b_buck")
  local prev="${cur:-0}"

  # axis 1: max-num-seqs (the headline knob)
  for v in "${SWEEP_MAX_NUM_SEQS[@]:1}"; do
    cur=$(trial "mns${v}_blk${b_blk}_gmem${b_gmem}_recipe${b_recipe}_buck${b_buck}" "$v" "$b_blk" "$b_gmem" "$b_recipe" "$b_buck")
    [ -z "$cur" ] && break
    if improved "$cur" "$prev"; then b_mns="$v"; prev="$cur"; else break; fi
  done
  # axis 2: block-size
  for v in "${SWEEP_BLOCK_SIZE[@]:1}"; do
    cur=$(trial "mns${b_mns}_blk${v}_gmem${b_gmem}_recipe${b_recipe}_buck${b_buck}" "$b_mns" "$v" "$b_gmem" "$b_recipe" "$b_buck")
    [ -z "$cur" ] && break
    if improved "$cur" "$prev"; then b_blk="$v"; prev="$cur"; else break; fi
  done
  # axis 3: gpu-memory-utilization (KV headroom; long ctx may OOM at the top -> trial returns empty)
  for v in "${SWEEP_GMEM[@]:1}"; do
    cur=$(trial "mns${b_mns}_blk${b_blk}_gmem${v}_recipe${b_recipe}_buck${b_buck}" "$b_mns" "$b_blk" "$v" "$b_recipe" "$b_buck")
    [ -z "$cur" ] && break
    if improved "$cur" "$prev"; then b_gmem="$v"; prev="$cur"; else break; fi
  done
  # axis 4: bucketing bounds (exp vs tuned-linear)
  for v in "${SWEEP_BUCKETING[@]:1}"; do
    cur=$(trial "mns${b_mns}_blk${b_blk}_gmem${b_gmem}_recipe${b_recipe}_buck${v}" "$b_mns" "$b_blk" "$b_gmem" "$b_recipe" "$v")
    [ -z "$cur" ] && break
    if improved "$cur" "$prev"; then b_buck="$v"; prev="$cur"; else break; fi
  done
  # axis 5: recipe-cache validate on/off (warm-replay vs cold-correct; perf tie-break, last)
  for v in "${SWEEP_RECIPE_CACHE[@]:1}"; do
    cur=$(trial "mns${b_mns}_blk${b_blk}_gmem${b_gmem}_recipe${v}_buck${b_buck}" "$b_mns" "$b_blk" "$b_gmem" "$v" "$b_buck")
    [ -z "$cur" ] && break
    if improved "$cur" "$prev"; then b_recipe="$v"; prev="$cur"; else break; fi
  done

  # write best.json (the aggregator's per-config winner)
  if [ -n "$best_json" ] && [ -f "$best_json" ]; then
    cp "$best_json" "$rundir/best.json"
    echo "  WINNER: $best_tag -> concurrent $(jget "$rundir/best.json" throughput_tok_s) tok/s (c=$conc) | single-stream $(jget "$rundir/best.json" single_stream_tok_s) tok/s"
  else
    echo "  NO WINNER (all knob points failed/OOM at $ctxtok x $conc)"
    echo '{}' > "$rundir/best.json"
  fi
}

for line in "${CONFIGS[@]}"; do
  IFS='^' read -r id label rt kvdtype kvgate msa regimes <<<"$line"
  want "$id" || continue
  echo; echo "############################################################################"
  echo "# CONFIG $id  —  $label"
  echo "############################################################################"
  # gate check (loud skip on unmet gate — never serve a wrong precision / fake number)
  if [ "$rt" = "fp8inc" ]; then
    if ! gate_fp8inc_ckpt; then echo "SKIP $id: fp8-normal(INC) needs the bf16 ckpt ($BF16_CKPT_HOST) AND the QUANT_CONFIG scales ($QUANT_CONFIG_HOST)."; continue; fi
  else
    if ! gate_int4_ckpt; then echo "SKIP $id: int4 checkpoint missing ($INT4_CKPT_HOST)."; continue; fi
  fi
  if [ -n "$kvgate" ] && ! gate_turboquant; then echo "SKIP $id: TurboQuant KV kernel/plugin (hpu-kvq-ext) not on-card-validated."; continue; fi
  if [ "$msa" = "1" ] && ! gate_msa; then echo "SKIP $id: MSA plugin (hpu-msa-ext) + serve USE_MSA hook absent."; continue; fi

  for reg in $regimes; do
    ctxlabel="${reg%%:*}"; conc="${reg##*:}"; ctxtok=$(ctx_tokens "$ctxlabel")
    rundir="$RUNS/${id}__${ctxlabel}_${conc}c"; mkdir -p "$rundir"
    # compose once (seed knobs) to capture the served checkpoint path for the meta audit.
    compose_env "$rt" "$kvdtype" "$kvgate" "$msa" "$ctxtok" "${SWEEP_MAX_NUM_SEQS[0]}" "${SWEEP_BLOCK_SIZE[0]}" "${SWEEP_GMEM[0]}" "off" "exp"
    # meta.json for the aggregator (ctx + target conc + single-stream point + composition audit)
    "$PY" - "$rundir/meta.json" "$ctxtok" "$conc" "$ctxlabel" "$rt" "$kvdtype" "$kvgate" "$msa" "$SINGLE_STREAM_C" "$MPATH_OVR" <<'PYEOF' 2>/dev/null || true
import json,sys
p,ctxtok,conc,ctxlabel,rt,kvdtype,kvgate,msa,ssc,mpath=sys.argv[1:11]
json.dump(dict(ctx=ctxlabel,ctx_tokens=int(ctxtok),target_conc=int(conc),
               weight=rt,model_path=mpath,kv_cache_dtype=kvdtype,kv_gate=kvgate,msa=int(msa),
               single_stream_conc=int(ssc)),open(p,"w"),indent=2)
PYEOF
    echo "== regime $ctxlabel ($ctxtok tok) @ c=$conc -> $rundir =="
    optimize_config "$rundir" "$rt" "$kvdtype" "$kvgate" "$msa" "$SERVED_NAME" "$ctxtok" "$conc"
  done
done

# ---- emit the single deliverable table ----
echo; echo "============================================================"
echo "Aggregating the matrix into one markdown table -> $REPORT"
"$PY" "$BENCH/aggregate_matrix.py" --runs "$RUNS" --out "$REPORT"
echo "DONE."
