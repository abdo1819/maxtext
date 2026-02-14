#!/bin/bash

# --- 1. PROJECT PATHS ---
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAXTEXT_PKG_DIR="${PROJECT_ROOT}/src/MaxText"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# --- 2. MODEL CONFIGURATION ---
export MODEL_NAME="qwen3-asr-1.7b"
export BASE_OUTPUT_DIR="gs://arabic-asr-dataset/checkpoints"
export CHECKPOINT_PATH="${BASE_OUTPUT_DIR}/${MODEL_NAME}"

# --- 3. HUGGINGFACE CONFIG ---
export HF_HOME='/mnt/data/hf_cache'
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
if [ -z "$HF_TOKEN" ]; then
    export HF_TOKEN=""
fi

# --- 4. COMMON ARGS ---
export TOKENIZER_PATH="${PROJECT_ROOT}/src/maxtext/assets/tokenizers/qwen3-tokenizer"
export CONFIG_FILE="${PROJECT_ROOT}/src/maxtext/configs/base.yml"

export TPU_NAME="tpu-v4-32-node"
