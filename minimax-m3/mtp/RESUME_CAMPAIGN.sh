#!/usr/bin/env bash
# M3 MTP single-stream campaign — RESUME runbook (NOT auto-run; commands are reviewed per step).
# Goal: M3 single-stream 5.8-7.6 -> ~25-30 tok/s via a depth-1 draft head (~1.70x).
# STATUS at pause (2026-06-15): CPU pipeline 9/9 green; dump infra debugged (4 fixes: device-glob
# controlD, V1-multiproc worker-side tee in minimax_m3._m3_mtp_dump_tee, jsonl corpus, truncate_prompt_tokens).
# Small dump was loading (validated load+warmup+tee path) when cards were yielded to another window.
# All fixes are ON DISK. Resume = run the steps below in order on IDLE cards (one container at a time).
set -euo pipefail
cd $HOME/AI
DEV=""; for d in /dev/accel/accel* /dev/infiniband/uverbs*; do [ -e "$d" ] && DEV="$DEV --device=$d"; done
MOUNTS="-v /mnt/fastmodels:/models:ro -v /mnt/data1/recipe_cache:/recipe_cache \
 -v $HOME/AI/hpu-moe-ext:/opt/hpu-moe-ext -v $HOME/AI/hpu-m3-ext:/opt/hpu-m3-ext \
 -v $HOME/AI/minimax-m3/vllm:/opt/m3-vllm -v $HOME/AI/minimax-m3:/opt/m3-root"
COMMON="$DEV --cap-add=sys_nice --cap-add=SYS_PTRACE --ipc=host --net=host --ulimit memlock=-1:-1 \
 -e HABANA_VISIBLE_DEVICES=all -e PYTHONDONTWRITEBYTECODE=1 \
 -e M3_VLLM_DIR=/opt/m3-vllm -e M3_ROOT_DIR=/opt/m3-root -e VLLM_WORKER_MULTIPROC_METHOD=fork \
 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 -e VLLM_RPC_TIMEOUT=100000000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600"

# --- STEP 1: SMALL DUMP (validate shards produced; ~5-7 min) -------------------------------------
# Expect: 6 shards in mtp/dumps/, each {hidden:[S,6144]bf16, target_ids:[S], positions:[S]}.
small_dump() {
  docker rm -f m3-mtp-dump 2>/dev/null || true; rm -f minimax-m3/mtp/dumps/shard_*.pt
  docker run -d --name m3-mtp-dump $COMMON \
    -e PT_HPU_RECIPE_CACHE_CONFIG="/recipe_cache/minimax-m3-int4,false,8192" \
    -e HPU_MOE_EXT_CKPT="/models/MiniMax-M3-int4-experts-AutoRound" \
    -e M3_EXPERT_RUNTIME=int4 -e M3_SWIGLUOAI=1 \
    -e M3_MTP_DUMP=1 -e M3_MTP_DUMP_DIR=/opt/m3-root/mtp/dumps -e M3_MTP_TP=8 -e M3_MTP_MAXLEN=4096 \
    $MOUNTS vllm-gaudi:next bash -lc \
    "pip install --no-deps -e /opt/hpu-moe-ext -e /opt/hpu-m3-ext -q 2>&1|tail -1; exec python /opt/m3-root/mtp/dump_m3_hidden.py --offline --prompt-limit 6 --corpus /opt/m3-root/mtp/calib/calib.jsonl"
}

# --- STEP 2: FULL DUMP (all 320 prompts, maxlen 8192 for longer seqs; ~30 min) -------------------
full_dump() {
  docker rm -f m3-mtp-dump 2>/dev/null || true; rm -f minimax-m3/mtp/dumps/shard_*.pt
  docker run -d --name m3-mtp-dump $COMMON \
    -e PT_HPU_RECIPE_CACHE_CONFIG="/recipe_cache/minimax-m3-int4,false,8192" \
    -e HPU_MOE_EXT_CKPT="/models/MiniMax-M3-int4-experts-AutoRound" \
    -e M3_EXPERT_RUNTIME=int4 -e M3_SWIGLUOAI=1 \
    -e M3_MTP_DUMP=1 -e M3_MTP_DUMP_DIR=/opt/m3-root/mtp/dumps -e M3_MTP_TP=8 -e M3_MTP_MAXLEN=8192 \
    $MOUNTS vllm-gaudi:next bash -lc \
    "pip install --no-deps -e /opt/hpu-moe-ext -e /opt/hpu-m3-ext -q 2>&1|tail -1; exec python /opt/m3-root/mtp/dump_m3_hidden.py --offline --corpus /opt/m3-root/mtp/calib/calib.jsonl"
}

# --- STEP 3: TRAIN the lean draft head on HPU (1 card; ~0.5-2 hr) --------------------------------
# ~300-400M trainable (frontend + lean block + final norm); lm_head/embed frozen from bf16 ckpt.
train() {
  docker rm -f m3-mtp-train 2>/dev/null || true
  docker run -d --name m3-mtp-train $COMMON -e HABANA_VISIBLE_MODULES=0 \
    -v /mnt/data2/minimax-m3-bf16:/bf16:ro \
    $MOUNTS vllm-gaudi:next bash -lc \
    "pip install --no-deps -e /opt/hpu-moe-ext -e /opt/hpu-m3-ext -q 2>&1|tail -1; exec python /opt/m3-root/mtp/train_m3_mtp.py \
      --dumps /opt/m3-root/mtp/dumps --bf16-dir /bf16 --mlp-dim 8192 --epochs 3 --bs 8 --lr 6e-4 \
      --device hpu --out /opt/m3-root/mtp/draft_ckpt"
}

# --- STEP 4: ACCEPT-RATE GATE (CPU, card-free; >=60% => GO to serve) -----------------------------
oracle() {
  $HOME/.venvs/m3calib/bin/python minimax-m3/mtp/m3_mtp_oracle.py \
    --draft minimax-m3/mtp/draft_ckpt --dump "$(ls minimax-m3/mtp/dumps/shard_*.pt | tail -1)"
}

# --- STEP 5: SERVE the MTP head + MEASURE single-stream (THE number) -----------------------------
# PREREQ (do first, card-free): wire m3_mtp_method_patch.install() into hpu_m3_ext.register(), and
# CONFIRM the spec-config arch binding (minimax-m3.env "(2b) MTP NEEDS CONFIRMATION") resolves
# MiniMaxM3MTP. Then serve mirroring minimax-m3-int4.env + drop --async-scheduling +
#   --speculative-config '{"model":"<draft_ckpt>","num_speculative_tokens":1}'
# and bench single-stream (mtp/bench) vs the int4 5.8-7.6 baseline. Report acceptance length + tok/s.
echo "Run steps in order: small_dump -> (verify shards) -> full_dump -> train -> oracle(>=60%) -> serve+measure"
