#!/bin/bash

# --- 1. PROJECT PATHS ---
export PROJECT_ROOT="/home/abdo1819/maxtext"
export MAXTEXT_PKG_DIR="${PROJECT_ROOT}/src/MaxText"
# Add MaxText to python path so we can run modules
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# --- 2. MODEL CONFIGURATION ---
export MODEL_NAME="qwen3-omni-30b-a3b"
export BASE_OUTPUT_DIR="gs://arabic-asr-dataset/checkpoints"
# The specific path for this model's checkpoint (fp32 original)
export CHECKPOINT_PATH_FP32="${BASE_OUTPUT_DIR}/${MODEL_NAME}-thinking"
# BF16 checkpoint path (smaller, faster loading)
export CHECKPOINT_PATH="${BASE_OUTPUT_DIR}/${MODEL_NAME}-thinking-bf16"

# --- 3. HUGGINGFACE CONFIG ---
export HF_HOME='/mnt/data/hf_cache'
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# Ideally, set HF_TOKEN in your ~/.bashrc, but for this script:
if [ -z "$HF_TOKEN" ]; then
    export HF_TOKEN=""
fi

# --- 4. COMMON ARGS ---
# Paths we use often
export TOKENIZER_PATH="${PROJECT_ROOT}/src/maxtext/assets/tokenizers/qwen3-tokenizer/tokenizer.json"
export CONFIG_FILE="${MAXTEXT_PKG_DIR}/configs/models/qwen3-omni-30b-a3b.yml"