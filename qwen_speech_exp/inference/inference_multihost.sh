#!/bin/bash
set -e

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

# Multi-host inference on v4-32 (16 chips) with bf16 checkpoint.
# With bf16, model fits on 4 chips (one host), so we split:
#   - ici_expert_parallelism=4  (experts on 4 chips, less all-to-all)
#   - ici_data_parallelism=4    (4 independent data-parallel groups)
# This reduces expert routing communication from 16-chip to 4-chip groups.
export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE'

python3 -m maxtext.decode "${PROJECT_ROOT}/src/maxtext/configs/base.yml" \
    model_name="${MODEL_NAME}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    tokenizer_type=huggingface \
    load_parameters_path="${CHECKPOINT_PATH}/0/items" \
    max_prefill_predict_length=${MAX_PREFILL} \
    max_target_length=${MAX_TARGET} \
    per_device_batch_size=1 \
    ici_fsdp_parallelism=1 \
    ici_data_parallelism=4 \
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
    skip_jax_distributed_system=false \
    autoregressive_decode_assert="" \
    ${EXTRA_ARGS}
