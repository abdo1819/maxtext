#!/bin/bash
set -e

# Single-host inference script for Qwen3-Omni-30B-A3B.
# Use this on a single-host TPU (e.g. v4-8, 4 chips) where JAX distributed
# is not needed. Does NOT work on multi-host pods (v4-32) — use
# inference_multihost.sh via multihost_runner.py for that.
#
# Usage:
#   bash qwen_speech_exp/inference/inference_singlehost.sh
#   bash qwen_speech_exp/inference/inference_singlehost.sh --mode audio --audio /path/to/audio.wav --prompt "Transcribe this"
#   bash qwen_speech_exp/inference/inference_singlehost.sh --prompt "Explain transformers"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../env_vars.sh"

# Defaults
MODE="text"
PROMPT="What is machine learning?"
IMAGE_PATH=""
AUDIO_PATH=""
MAX_PREFILL=272
MAX_TARGET=300

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       MODE="$2"; shift 2 ;;
        --prompt)     PROMPT="$2"; shift 2 ;;
        --image)      IMAGE_PATH="$2"; shift 2 ;;
        --audio)      AUDIO_PATH="$2"; shift 2 ;;
        --max-prefill) MAX_PREFILL="$2"; shift 2 ;;
        --max-target)  MAX_TARGET="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Mode-specific settings
USE_MULTIMODAL="false"
USE_AUDIO="false"
EXTRA_ARGS=""

case $MODE in
    text)
        MAX_PREFILL=${MAX_PREFILL:-272}
        ;;
    image)
        USE_MULTIMODAL="true"
        MAX_PREFILL=${MAX_PREFILL:-512}
        EXTRA_ARGS="image_path=${IMAGE_PATH}"
        ;;
    audio)
        USE_MULTIMODAL="true"
        USE_AUDIO="true"
        MAX_PREFILL=${MAX_PREFILL:-1024}
        EXTRA_ARGS="audio_path=${AUDIO_PATH}"
        ;;
    *)
        echo "Unknown mode: $MODE (choose text, image, or audio)"
        exit 1
        ;;
esac

# Single-host inference (4 chips on one host)
# Key differences from inference_multihost.sh:
#   - skip_jax_distributed_system=true (no multi-host coordination)
#   - ici_expert_parallelism=4 (single host, 4 chips)
python3 -m maxtext.decode "${PROJECT_ROOT}/src/maxtext/configs/base.yml" \
    model_name="${MODEL_NAME}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    tokenizer_type=huggingface \
    load_parameters_path="${CHECKPOINT_PATH}/0/items" \
    max_prefill_predict_length=${MAX_PREFILL} \
    max_target_length=${MAX_TARGET} \
    per_device_batch_size=1 \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=4 \
    ici_tensor_parallelism=1 \
    scan_layers=false \
    weight_dtype=bfloat16 \
    quantize_kvcache=true \
    kv_quant_dtype=int8 \
    attention=dot_product \
    use_multimodal=${USE_MULTIMODAL} \
    use_audio=${USE_AUDIO} \
    prompt="${PROMPT}" \
    skip_jax_distributed_system=true \
    ${EXTRA_ARGS}
