#!/bin/bash
# SFT training on distillation dataset with CoT.
# Based on train.sh but uses the distillation_sft ArrayRecord files
# and increased max_target_length to accommodate CoT + transcription.
#
# Run from jumpbox (maxtext-vm) via multihost_runner.py:
#   python3 tools/orchestration/multihost_runner.py \
#       --TPU_PREFIX=qr-v4-32 --PROJECT=arabic-asr-level2thinkg \
#       --ZONE=us-central2-b --INTERNAL_IP=true \
#       --COMMAND="cd ~/maxtext && source ~/venv-maxtext/bin/activate && bash qwen_speech_exp/train_sft_distillation.sh" \
#       --USE_EXISTING_FOLDER=true --RUN_NAME=sft-distillation
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

python3 -m MaxText.train "${PROJECT_ROOT}/src/maxtext/configs/base.yml" \
    model_name="${MODEL_NAME}" \
    run_name="qwen3-omni-sft-distillation-run1" \
    base_output_directory="${BASE_OUTPUT_DIR}" \
    load_parameters_path="${CHECKPOINT_PATH}/0/items" \
    dataset_type=grain \
    grain_file_type=arrayrecord \
    grain_train_files="/tmp/gcsfuse/grain_data_arrayrecord/distillation_sft/*.array_record" \
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
    use_audio=true \
    use_sft=true \
    sft_train_on_completion_only=true
