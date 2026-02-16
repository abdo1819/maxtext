#!/bin/bash
# Launch GRPO training on TPU v4-32 (4 hosts × 4 chips = 16 chips).
#
# Training and inference share a SINGLE mesh with all 16 devices.
# This avoids TPU ICI "unexpected peer in launch group" errors that
# arise when two separate meshes exist on the same slice.
# Execution is serialized: generate completions, then train, then repeat.
#
# Two config files are passed to the trainer via parse_custom_args:
#   1st YAML + args → training config  (16 chips: fsdp=4 × expert=4)
#   2nd YAML + args → inference config  (same 16 chips, same ICI)
#
# Usage (from jumpbox):
#   bash qwen_speech_exp/training/run_grpo.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../env_vars.sh"

GRPO_CONFIG="${PROJECT_ROOT}/src/maxtext/configs/grpo_audio_qwen3_omni.yml"

python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="cd ~/maxtext && source ~/venv-maxtext/bin/activate && \
git pull && \
bash tools/setup/setup_gcsfuse.sh DATASET_GCS_BUCKET=arabic-asr-dataset MOUNT_PATH=/tmp/gcsfuse && \
bash preflight.sh 2>/dev/null || true && \
export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE' && \
export PYTHONHASHSEED=0 && \
python3 -u -m MaxText.experimental.rl.grpo_trainer \
    src/maxtext/configs/grpo_audio_qwen3_omni.yml \
    ici_expert_parallelism=4 \
    ici_fsdp_parallelism=4 \
    grain_worker_count=0 \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${CHECKPOINT_PATH}/0/items \
    base_output_directory=gs://arabic-asr-dataset/grpo_training \
    grain_train_files=/tmp/gcsfuse/grain_data_arrayrecord/train/*.array_record \
    src/maxtext/configs/grpo_audio_qwen3_omni.yml \
    ici_expert_parallelism=4 \
    ici_fsdp_parallelism=4 \
    per_device_batch_size=1 \
    max_prefill_predict_length=128 \
    max_target_length=384 \
    grain_worker_count=0 \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${CHECKPOINT_PATH}/0/items \
    grain_train_files=/tmp/gcsfuse/grain_data_arrayrecord/train/*.array_record" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=maxtext
