#!/usr/bin/env bash
set -euo pipefail

# ---- project layout (derived from the script location) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export DATA_DIR="${PROJECT_ROOT}/data"
export CACHE_DIR="${PROJECT_ROOT}/cache"
export OUTPUT_DIR="${PROJECT_ROOT}/outputs"

mkdir -p "${DATA_DIR}" "${CACHE_DIR}" "${OUTPUT_DIR}"

# ---- choose GPUs ----
# Use all GPUs by default; override with e.g. CUDA_VISIBLE_DEVICES=0 for single GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"

# ---- model path (local) ----
# Pick one:
# 1) Official:
# export MODEL_PATH="/model/llm/Meta-Llama-3.1-8B-Instruct"
# 2) Unsloth FP16/BF16 (recommended if available locally):
# export MODEL_PATH="/model/HuggingFace/unsloth/Meta-Llama-3.1-8B-Instruct"
export MODEL_PATH="${MODEL_PATH:-/model/llm/Meta-Llama-3.1-8B-Instruct}"

# ---- data files ----
export XGUARD_PARQUET="${XGUARD_PARQUET:-${DATA_DIR}/xguard-train.parquet}"
export ORBENCH_PARQUET="${ORBENCH_PARQUET:-${DATA_DIR}/orbench.parquet}"

# ---- tokenization / sampling ----
export MAX_LENGTH="${MAX_LENGTH:-4096}"                 # 4096 is a sane default
export TARGET_TOTAL_TOKENS="${TARGET_TOTAL_TOKENS:-2000000}"
export TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-512}"
export TOKENIZE_NUM_PROC="${TOKENIZE_NUM_PROC:-4}"
export SEED="${SEED:-42}"

# ---- training hyperparams ----
export BF16="${BF16:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export LR="${LR:-2e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"

# ---- LoRA hyperparams ----
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

echo "[launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "[launch] MODEL_PATH=${MODEL_PATH}"
echo "[launch] MAX_LENGTH=${MAX_LENGTH} TARGET_TOTAL_TOKENS=${TARGET_TOTAL_TOKENS}"

# Run uv from the repository root so the optional finetune dependency group is visible.
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
cd "${REPO_ROOT}"

# Use Python to count GPUs (respects CUDA_VISIBLE_DEVICES)
N_GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "1")
if [ "${N_GPUS:-1}" -gt 1 ]; then
  echo "[launch] Detected ${N_GPUS} GPUs, using accelerate launch"
  uv run accelerate launch --num_processes "${N_GPUS}" "${PROJECT_ROOT}/scripts/train.py"
else
  uv run python "${PROJECT_ROOT}/scripts/train.py"
fi
