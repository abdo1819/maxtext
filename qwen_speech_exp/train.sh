#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env_vars.sh"

# Mount GCS data via gcsfuse
bash "${PROJECT_ROOT}/tools/setup/setup_gcsfuse.sh" \
  DATASET_GCS_BUCKET=arabic-asr-dataset \
  MOUNT_PATH=/tmp/gcsfuse

# Network optimization for multi-host TPU
bash "${PROJECT_ROOT}/preflight.sh" 2>/dev/null || true

# XLA flags for v4 TPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"

python3 -m MaxText.train "${CONFIG_FILE}" \
    model_name="${MODEL_NAME}" \
    run_name="qwen3-omni-finetune-run1" \
    base_output_directory="${BASE_OUTPUT_DIR}" \
    load_parameters_path="${CHECKPOINT_PATH}/0/items" \
    dataset_type=grain \
    grain_file_type=arrayrecord \
    grain_train_files="/tmp/gcsfuse/grain_data_arrayrecord/train/*.array_record" \
    grain_eval_files="/tmp/gcsfuse/grain_data_arrayrecord/validation/*.array_record" \
    grain_worker_count=2 \
    tokenizer_path="${TOKENIZER_PATH}" \
    tokenizer_type=huggingface \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=16 \
    ici_tensor_parallelism=1 \
    per_device_batch_size=1 \
    max_target_length=2048 \
    steps=1000 \
    learning_rate=1e-5 \
    remat_policy=full \
    scan_layers=true \
    weight_dtype=bfloat16 \
    attention=dot_product \
    enable_checkpointing=true \
    checkpoint_period=200 \
    eval_interval=100 \
    use_multimodal=true \
    use_audio=true
