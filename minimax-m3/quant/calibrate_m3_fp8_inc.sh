#!/usr/bin/env bash
# ==================================================================================================
# calibrate_m3_fp8_inc.sh
#
# FP8 (E4M3) **native-W8A8** calibration + serve path for the MiniMax-M3 TEXT TOWER experts,
# using Intel Neural Compressor (INC) on the Gaudi-2 cards. This is the FP8 ANALOG of
# quant/quantize_m3_int4.py — the experts pivot int4 -> fp8:
#
#   * NATIVE MME: fp8 is the Gaudi-2 first-class low-precision matmul (~2x bf16); int4 has NO
#     native matmul (it dequants to bf16/fp8). Optimization Guide §1 ranks FP8(INC) the #1 lever;
#     the deep-search (§"On INT4/INT6") found int4 is a *fit-only fallback*, never a throughput play.
#   * SINGLE-QUANT <1%: experts are quantized ONCE (bf16 -> fp8 e4m3), not int4-then-redequant.
#     Contrast hpu-moe-ext/fp8_routed_moe.py which DOUBLE-quantizes a stored-int4 checkpoint
#     (int4 -> bf16 -> fp8) and documents "a few-% relative error". Native INC fp8 avoids that.
#   * KEEPS MTP: this is a calibration of the SAME bf16 checkpoint (no arch surgery), so the
#     num_mtp_modules=7 nextn layers and the get_spec_layer skip in minimax_m3.py are untouched.
#
# This is NOT the AutoRound/hpu-moe-ext path. It is the STOCK vllm-gaudi INC two-pass flow:
# vllm-hpu-extension/calibration/calibrate_model.sh (in the image at
# /opt/vllm-gaudi/calibration/) which auto-detects the device (step-0 -> g2 here) and runs:
#     MEASURE  (mode=MEASURE,  observer=maxabs)              -> maxabs_measure_g2.json + stats
#     QUANTIZE (mode=QUANTIZE, scale_method=maxabs_hw)       -> maxabs_quant_g2.json
# QUANT_CONFIG is the master switch (its "mode" picks MEASURE vs QUANTIZE). Serving then points
# QUANT_CONFIG at the QUANTIZE json with `--quantization inc --kv-cache-dtype fp8_inc`.
#
# WHAT GETS QUANTIZED vs KEPT BF16 — see the dry-validate (`--dry-validate`) for the exact, config-
# grounded module lists. Summary (default = EXPERTS-ONLY, the faithful int4 mirror):
#   FP8 (W8A8): the 128 ROUTED experts (block_sparse_moe.experts, FusedMoE w1/w2/w3), layers 3..59.
#   KEEP BF16 : lm_head, embed_tokens, ALL norms (RMSNorm — INC never quantizes non-Linear anyway),
#               the router gate (block_sparse_moe.gate) + e_score_correction_bias, AND — the M3-
#               specific load-bearing constraint — the MSA INDEX PROJECTIONS
#               (self_attn.index_{q,k}_proj + index_{q,k}_norm). Selection precision must NOT
#               degrade: the block indexer's top-16 choice is computed from index_q/k; fp8 there
#               would perturb which 16 blocks each token attends to (silent at 4-16k, corrupts
#               selection at very long ctx — exactly the MSA_SPEC §7 long-ctx failure mode the
#               quant-quality-eval flags). Index path STAYS bf16 in BOTH variants.
#   Also bf16 in experts-only (mirror int4): main attention (qkv/o_proj), shared expert, dense MLPs.
#
# A BROADER variant (FP8_SCOPE=broad) additionally quantizes main attention + dense MLPs + shared
# expert (all genuine fp8 MME wins, unlike int4) while STILL protecting index_*/gate/norms/heads.
# Default is experts-only so the quant-quality-eval A-matrix (Δ_quant isolated to the experts)
# holds 1:1. Pick `broad` only after experts-only passes and you want the extra attention/MLP win.
#
# ------------------------------------------------------------------------------------------------
# CARD-FREE here, card-GATED for the RUN:
#   ./calibrate_m3_fp8_inc.sh --dry-validate   # config + safetensors-HEADER only; loads NO weights,
#                                              # needs no card/torch/INC. Mirrors quantize_m3_int4.py.
#   ./calibrate_m3_fp8_inc.sh --emit-config    # write the QUANT_CONFIG json templates to OUT_FP8_DIR.
#   ./calibrate_m3_fp8_inc.sh --build-calib    # build the domain/long-ctx calibration .pkl (card-free;
#                                              # needs pandas+datasets in a scratch env — NOT the
#                                              # serving image).
#   ./calibrate_m3_fp8_inc.sh --run            # the REAL 2-pass MEASURE->QUANTIZE on g2. HARD-GATED:
#                                              # refuses unless (download complete) AND (transformers
#                                              # with models/minimax_m3_vl/) AND (a FREE Gaudi card,
#                                              # NOT vllm-active). Wraps the image's calibrate_model.sh.
#   ./calibrate_m3_fp8_inc.sh --serve-cmd      # print the serve env line + flags (does not launch).
#
# DO NOT touch vllm-active or its cards. The --run path asserts a free card and refuses otherwise;
# card churn can wedge the scheduler (memory: gaudi-kernel-affinity-deadlock) — take ONLY a verifiably
# idle card. Authoring/dry-validate/emit-config/build-calib never touch a card at all.
# ==================================================================================================
set -euo pipefail

# --------------------------------------------------------------------------------------------------
# Static config — the contract. Edit here, nowhere else. Mirrors quantize_m3_int4.py's constants.
# --------------------------------------------------------------------------------------------------
SRC_DIR="${SRC_DIR:-/mnt/data2/minimax-m3-bf16}"          # the bf16 checkpoint (calibration input)
# The fp8 path does NOT emit a re-packed checkpoint: INC fp8 serves the ORIGINAL bf16 weights and
# applies the measured scales at load via QUANT_CONFIG. So the "output" is the SCALES dir, and the
# model you serve is still SRC_DIR (copied to /mnt/fastmodels for the fast NVMe + serve mount).
SERVE_MODEL_DIR="${SERVE_MODEL_DIR:-/mnt/fastmodels/MiniMax-M3-bf16}"   # where serve.sh mounts it
OUT_FP8_DIR="${OUT_FP8_DIR:-/mnt/data1/quant/minimax-m3-fp8-inc}"        # INC measure+quant scales + jsons
MODEL_NAME="${MODEL_NAME:-minimax-m3}"                    # subfolder name calibrate_model.sh uses
DEVICE_TYPE="${DEVICE_TYPE:-g2}"                          # informational; the image auto-detects (step-0)

# Calibration knobs — domain/long-ctx per notes/reviews/quant-quality-eval.md §3 (NOT pile-10k@2048).
CALIB_PKL="${CALIB_PKL:-$OUT_FP8_DIR/m3-domain-longctx-calib.pkl}"   # DataFrame[question,system_prompt]
SAMPLE_LEN="${SAMPLE_LEN:-8192}"     # >= 8192 so the long-ctx / sparse-path activation regime is exercised
NSAMPLES="${NSAMPLES:-256}"          # >= 256 so all 128 experts (top-4) are likely activated
BATCH_SIZE="${BATCH_SIZE:-4}"        # measure batch (small; long seqlen dominates HBM)
TP_SIZE="${TP_SIZE:-8}"              # M3 is a big MoE (>200B active-class) -> TP=8 + EP for calibration

# FP8 scope: experts-only (default, the int4 mirror) | broad (also attn/dense/shared)
FP8_SCOPE="${FP8_SCOPE:-experts}"

# The image's stock calibration toolkit (read-only; we wrap it, never edit it).
CALIB_DIR_IN_IMAGE="/opt/vllm-gaudi/calibration"
IMAGE="${IMAGE:-vllm-gaudi:next}"

# --------------------------------------------------------------------------------------------------
# QUANT_CONFIG json templates. These are the master switch INC reads. The selection logic
# (verified from the image's neural_compressor/.../fp8_quant/_core/utils.py:should_quantize):
#   * a module is quantized iff its TYPE is supported (Linear / FusedMoE) AND its name does NOT
#     match ANY blocklist regex. `is_re_match` uses re.search (substring/REGEX over the full module
#     path). "Allowlist by names is not yet implemented" (quant_config.py) -> we select by an EMPTY
#     allowlist (= allow all supported types) and CARVE OUT keep-bf16 via blocklist regexes. This is
#     exactly the mixtral/deepseek recipe pattern create_quant_config bakes in.
#   * scale_method=maxabs_hw -> the Gaudi pow2 HW grid (the method to validate vs pure maxabs;
#     NEVER unit_scale — Mixtral PPL 11.34->96.65). scale_format=CONST speeds fp8 warmup.
#   * MEASURE: quantize_weight=false, just collect maxabs activation/weight stats into dump_stats_path.
#   * QUANTIZE: reads those stats from the SAME dump_stats_path and bakes per-tensor fp8 scales.
# The g2-scale requirement: maxabs_hw measured ON g2 yields the ±240 E4M3 (fnuz, bias 8) grid. NEVER
# reuse a g3 measurement (±448, bias 7) -> clip. The image's step-0 auto-detects g2 and writes the
# _g2 suffix; running on these cards IS what makes them g2 scales. (For an OCP-fp8 source checkpoint
# you would instead set VLLM_HPU_CONVERT_TO_FP8UZ=1; our source is bf16 so INC measures natively.)
# --------------------------------------------------------------------------------------------------

# Keep-bf16 blocklist regexes, by scope. (Norms are RMSNorm; INC won't quantize non-Linear, but we
# list them anyway — explicit + harmless, belt-and-suspenders.)
read -r -d '' BLOCK_EXPERTS_ONLY <<'JSON' || true
        "lm_head",
        "embed_tokens",
        "self_attn",
        "block_sparse_moe\\.gate\\b",
        "e_score_correction_bias",
        "shared_experts",
        "\\.mlp\\.",
        "layernorm",
        "model\\.norm"
JSON

read -r -d '' BLOCK_BROAD <<'JSON' || true
        "lm_head",
        "embed_tokens",
        "self_attn\\.index_",
        "block_sparse_moe\\.gate\\b",
        "e_score_correction_bias",
        "layernorm",
        "model\\.norm"
JSON

emit_config() {
    mkdir -p "$OUT_FP8_DIR/$MODEL_NAME/$DEVICE_TYPE"
    local block
    if [ "$FP8_SCOPE" = "broad" ]; then block="$BLOCK_BROAD"; else block="$BLOCK_EXPERTS_ONLY"; fi
    local stats="$OUT_FP8_DIR/$MODEL_NAME/$DEVICE_TYPE/inc_output"

    # PASS 1 — MEASURE. observer=maxabs, no weight quant; just record ranges to dump_stats_path.
    cat > "$OUT_FP8_DIR/$MODEL_NAME/maxabs_measure_${DEVICE_TYPE}.json" <<JSON
{
    "method": "HOOKS",
    "mode": "MEASURE",
    "observer": "maxabs",
    "allowlist": { "types": [], "names": [] },
    "blocklist": {
      "types": [],
      "names": [
$block
      ]
    },
    "quantize_weight": false,
    "dump_stats_path": "$stats",
    "calibration_sample_interval": 1
}
JSON

    # PASS 2 — QUANTIZE. scale_method=maxabs_hw (g2 pow2 grid), CONST scale_format (warmup speed).
    # SAME blocklist + SAME dump_stats_path so the QUANTIZE pass reads PASS-1's measured stats.
    cat > "$OUT_FP8_DIR/$MODEL_NAME/maxabs_quant_${DEVICE_TYPE}.json" <<JSON
{
    "method": "HOOKS",
    "mode": "QUANTIZE",
    "observer": "maxabs",
    "scale_method": "maxabs_hw",
    "scale_format": "CONST",
    "allowlist": { "types": [], "names": [] },
    "blocklist": {
      "types": [],
      "names": [
$block
      ]
    },
    "dump_stats_path": "$stats"
}
JSON
    echo "Wrote (scope=$FP8_SCOPE):"
    echo "  MEASURE : $OUT_FP8_DIR/$MODEL_NAME/maxabs_measure_${DEVICE_TYPE}.json"
    echo "  QUANTIZE: $OUT_FP8_DIR/$MODEL_NAME/maxabs_quant_${DEVICE_TYPE}.json"
    echo
    echo "Serve later with:  export QUANT_CONFIG=$OUT_FP8_DIR/$MODEL_NAME/maxabs_quant_${DEVICE_TYPE}.json"
}

# --------------------------------------------------------------------------------------------------
# Domain / long-context calibration set builder (card-free; needs pandas+datasets in a SCRATCH env).
# Produces a pandas pickle of a DataFrame with `question` + `system_prompt` columns — the exact shape
# the image's step-1-prepare-calibration-dataset.py consumes. Per quant-quality-eval §3: code +
# agentic + long documents, so all 128 experts fire and the long-ctx activation regime is hit.
# Swap the sources freely; keep it representative (code / tool-use / long-ctx) and license-clean.
# --------------------------------------------------------------------------------------------------
build_calib() {
    mkdir -p "$(dirname "$CALIB_PKL")"
    python3 - "$CALIB_PKL" "$NSAMPLES" <<'PY'
import sys
out, n = sys.argv[1], int(sys.argv[2])
try:
    import pandas as pd
    from datasets import load_dataset
except Exception as e:
    sys.stderr.write(
        "build-calib needs pandas+datasets in a SCRATCH env (NOT the serving image):\n"
        "  python3 -m venv /tmp/m3calib && . /tmp/m3calib/bin/activate && pip install pandas datasets\n"
        f"import failed: {e!r}\n")
    sys.exit(3)

SYS_CODE   = "You are an expert software engineer. Reason about the code carefully and answer precisely."
SYS_AGENT  = "You are a tool-using agent. Plan, call tools with valid JSON arguments, and use long context."
SYS_LONGCTX= "You are a careful analyst. Read the entire long document and answer using only its content."

rows = []
# share the budget across the three domains so routing coverage is broad (all 128 experts).
per = max(1, n // 3)

# 1) CODE — brittle to quant; sharpest int4/fp8 probe (the-stack-smol is a small, license-filtered code set).
try:
    ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
    for i, ex in zip(range(per), ds):
        code = ex.get("content") or ex.get("text") or ""
        rows.append({"system_prompt": SYS_CODE,
                     "question": "Explain what this code does, then suggest one improvement:\n\n" + code[:24000]})
except Exception as e:
    sys.stderr.write(f"[warn] code source unavailable ({e!r}); skipping code rows\n")

# 2) LONG-CONTEXT — long documents so SAMPLE_LEN>=8192 exercises the sparse-path activation regime.
try:
    ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train", streaming=True)
    for i, ex in zip(range(per), ds):
        txt = ex.get("text") or ""
        rows.append({"system_prompt": SYS_LONGCTX,
                     "question": "Summarize the key points of the following long document:\n\n" + txt[:48000]})
except Exception as e:
    sys.stderr.write(f"[warn] long-ctx source unavailable ({e!r}); skipping long-ctx rows\n")

# 3) AGENTIC / tool-call — M3's stated strength; routing-sensitive. glaive function-calling transcripts.
try:
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
    for i, ex in zip(range(per), ds):
        chat = ex.get("chat") or ex.get("system") or str(ex)
        rows.append({"system_prompt": SYS_AGENT,
                     "question": "Continue this tool-using dialogue with a correct next step:\n\n" + str(chat)[:24000]})
except Exception as e:
    sys.stderr.write(f"[warn] agentic source unavailable ({e!r}); skipping agentic rows\n")

if not rows:
    sys.stderr.write("FATAL: no calibration rows built (all sources failed). Point CALIB_PKL at your own\n"
                     "DataFrame[question,system_prompt].pkl, or fix network/datasets access.\n")
    sys.exit(4)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).head(n).reset_index(drop=True)
df.to_pickle(out)
print(f"wrote {len(df)} calibration rows -> {out}  (code/agentic/long-ctx mix; seqlen capped at step-1)")
PY
    echo "NOTE: pass this to --run via CALIB_PKL=$CALIB_PKL ; the image's step-1 tokenizes it at"
    echo "      --max-model-length $SAMPLE_LEN (>=8192) and --max-dataset-samples $NSAMPLES (>=256)."
}

# --------------------------------------------------------------------------------------------------
# DRY-VALIDATE — config + safetensors-HEADER only. Loads NO weights, needs no card/torch/INC.
# Resolves the fp8-quantize-vs-keep-bf16 module lists against the REAL config.json + shard headers
# and asserts the contract (experts fp8; index_*/gate/heads/norms bf16). Mirrors the int4 dry-run.
# Delegates to the python helper colocated with this script.
# --------------------------------------------------------------------------------------------------
dry_validate() {
    SRC_DIR="$SRC_DIR" FP8_SCOPE="$FP8_SCOPE" \
      python3 "$(dirname "$(readlink -f "$0")")/dry_validate_m3_fp8_inc.py"
}

# --------------------------------------------------------------------------------------------------
# REAL RUN — the 2-pass MEASURE->QUANTIZE on g2, by wrapping the image's calibrate_model.sh.
# HARD-GATED. Takes a FREE card only. NEVER touches vllm-active.
# --------------------------------------------------------------------------------------------------
assert_can_run() {
    echo "Checking --run gates (download / transformers-m3 / free-card) ..." >&2
    local errs=0
    # 1) download complete. set -e-safe counting via bash globs (no `ls | wc` pipe that pipefail
    #    would abort on when a glob matches nothing).
    local shards=0 idx=0 total="" f
    shopt -s nullglob
    for f in "$SRC_DIR"/*.safetensors; do shards=$((shards+1)); total="$f"; done
    for f in "$SRC_DIR"/*.index.json; do idx=$((idx+1)); done
    shopt -u nullglob
    if [ -n "$total" ]; then
        total="$(printf '%s\n' "$total" | sed -n 's/.*-of-\([0-9]\+\)\.safetensors/\1/p')"
    fi
    if [ -z "$total" ] || [ "$shards" -lt "${total:-99999}" ] || [ "$idx" -lt 1 ]; then
        echo "  * download INCOMPLETE: $shards/${total:-?} shards, index=$idx at $SRC_DIR" >&2; errs=1
    fi
    # 2) transformers with minimax_m3_vl modeling (same blocker as the int4 recipe)
    if ! python3 -c "import transformers,os,sys;b=os.path.dirname(transformers.__file__);sys.exit(0 if os.path.isdir(b+'/models/minimax_m3_vl') else 1)" 2>/dev/null; then
        echo "  * transformers lacks models/minimax_m3_vl/ — vLLM cannot build M3 for calibration." >&2
        echo "    Install a transformers build that ships minimax_m3_vl (see quantize_m3_int4.py REQUIREMENTS #2)." >&2
        errs=1
    fi
    # 3) a FREE card — and vllm-active must NOT be the thing we are about to disturb.
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx vllm-active; then
        echo "  * vllm-active is RUNNING. Calibration needs the cards; free it FIRST (stop the serve" >&2
        echo "    deliberately) or run on a verifiably idle card set. Refusing to churn cards." >&2
        errs=1
    fi
    if [ "$errs" -ne 0 ]; then
        echo "REFUSING TO RUN — unmet gates (above)." >&2; exit 2
    fi
}

real_run() {
    assert_can_run
    [ -f "$CALIB_PKL" ] || { echo "no calibration .pkl at $CALIB_PKL — run --build-calib first (or set CALIB_PKL)"; exit 2; }
    emit_config   # ensure the jsons exist with the chosen scope before the 2-pass run
    echo "Launching INC 2-pass MEASURE->QUANTIZE on $DEVICE_TYPE (TP=$TP_SIZE, EP) via the image's calibrate_model.sh."
    echo "Model name folder=$MODEL_NAME ; scales/jsons -> $OUT_FP8_DIR ; calib=$CALIB_PKL"
    # We pass our pre-built domain/long-ctx pkl as the dataset (.pkl => step-1 is skipped; step-2
    # consumes it directly). -u enables expert-parallel measurement unification for the big MoE.
    # NOTE: this MUST run inside the serving image (INC + step-* scripts live there), on free cards.
    # It does NOT use serve.sh; it is the offline calibration container. Mount SRC_DIR + OUT_FP8_DIR.
    local DEV=""; for d in /dev/accel/accel* /dev/infiniband/uverbs* ; do DEV="$DEV --device=$d"; done
    docker run --rm $DEV --cap-add=sys_nice --ipc=host --net=host --ulimit memlock=-1:-1 \
      -e HABANA_VISIBLE_DEVICES=all \
      -e PT_HPU_WEIGHT_SHARING=0 \
      -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
      -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
      -e VLLM_RPC_TIMEOUT=100000000 \
      -v "$SRC_DIR:$SRC_DIR:ro" \
      -v "$OUT_FP8_DIR:$OUT_FP8_DIR" \
      -v "$CALIB_PKL:$CALIB_PKL:ro" \
      "$IMAGE" \
      bash -lc "cd $CALIB_DIR_IN_IMAGE && \
        ./calibrate_model.sh -m '$SRC_DIR' -d '$CALIB_PKL' -o '$OUT_FP8_DIR' \
          -b $BATCH_SIZE -t $TP_SIZE -l $NSAMPLES -u"
    echo
    echo "DONE. The QUANTIZE config to serve with: $OUT_FP8_DIR/$MODEL_NAME/maxabs_quant_${DEVICE_TYPE}.json"
}

# --------------------------------------------------------------------------------------------------
# SERVE COMMAND — print the model env line + vLLM flags (mirrors AI/models/*.env + serve.sh).
# --------------------------------------------------------------------------------------------------
serve_cmd() {
    local qc="$OUT_FP8_DIR/$MODEL_NAME/maxabs_quant_${DEVICE_TYPE}.json"
    cat <<TXT
# ---- AI/models/minimax-m3-fp8.env  (launch: ./serve.sh minimax-m3-fp8) -----------------------
# Serve M3 with NATIVE fp8 experts (INC) + fp8 KV. QUANT_CONFIG = the QUANTIZE json (master switch
# now in QUANTIZE mode). The model served is the ORIGINAL bf16 checkpoint — INC applies the measured
# fp8 scales at load (no re-packed checkpoint). hpu-moe-ext is INERT here (it only patches int4 MoE).
MODEL_PATH=MiniMax-M3-bf16                 # = $SERVE_MODEL_DIR mounted at /models
SERVED_NAME=minimax-m3
IMAGE=$IMAGE
# FP8/INC => LAZY mode (PT_HPU_LAZY_MODE=1): INC is NOT supported under torch.compile (Guide §II.3).
# PT_HPU_WEIGHT_SHARING=0 frees the full-precision weights post-fp8 (avoids OOM on the big MoE).
# PT_HPU_ENABLE_LAZY_COLLECTIVES + --enable-expert-parallel = the big-MoE TP=8 EP path. Raise the
# engine iteration timeout (long fp8 compiles trip the default -> engine crash on first request).
MODEL_ENV="-e PT_HPU_LAZY_MODE=1 -e QUANT_CONFIG=$qc -e VLLM_DISABLE_MARK_SCALES_AS_CONST=true -e PT_HPU_WEIGHT_SHARING=0 -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 -e VLLM_RPC_TIMEOUT=100000000 -e VLLM_HPU_LOG_STEP_CPU_FALLBACKS=true"
VLLM_ARGS="--quantization inc --kv-cache-dtype fp8_inc --tensor-parallel-size 8 --enable-expert-parallel --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.9 --block-size 128 --async-scheduling --no-enable-prefix-caching --trust-remote-code"
# ----------------------------------------------------------------------------------------------
# Equivalent raw QUANT_CONFIG export if launching by hand:
#   export QUANT_CONFIG=$qc
TXT
}

# --------------------------------------------------------------------------------------------------
usage() {
    sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "usage: $0 {--dry-validate|--emit-config|--build-calib|--run|--serve-cmd}"
    echo "  env knobs: FP8_SCOPE=experts|broad  SRC_DIR=  OUT_FP8_DIR=  SAMPLE_LEN=  NSAMPLES=  TP_SIZE="
}

case "${1:-}" in
    --dry-validate) dry_validate ;;
    --emit-config)  emit_config ;;
    --build-calib)  build_calib ;;
    --run)          real_run ;;
    --serve-cmd)    serve_cmd ;;
    -h|--help|"")   usage ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
esac
