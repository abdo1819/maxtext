#!/bin/bash
# Launch GRPO training on TPU v4-32 (4 hosts × 4 chips = 16 chips).
#
# The GRPO trainer uses all 16 chips in a single multi-host JAX process:
#   - 8 chips  → inference/rollout engine (generates completions)
#   - 8 chips  → training (policy + reference model + gradients)
#
# NOTE: 8/8 split required because MoE activation_batch axis maps to
# (data, fsdp, expert) and that product must divide 16 (expert groups).
# 12 doesn't divide 16 but 8 does (expert=4 × fsdp=2 = 8).
#
# Two config files are passed to the trainer via parse_custom_args:
#   1st YAML + args → training config  (8 chips: fsdp=2 × expert=4)
#   2nd YAML + args → inference config (8 chips: fsdp=2 × expert=4)
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
python3 -m MaxText.experimental.rl.grpo_trainer \
    src/maxtext/configs/grpo_audio_qwen3_omni.yml \
    ici_expert_parallelism=4 \
    ici_fsdp_parallelism=2 \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${CHECKPOINT_PATH}/0/items \
    base_output_directory=gs://arabic-asr-dataset/grpo_training \
    grain_train_files=/tmp/gcsfuse/grain_data_arrayrecord/train/*.array_record \
    src/maxtext/configs/grpo_audio_qwen3_omni.yml \
    ici_expert_parallelism=4 \
    ici_fsdp_parallelism=2 \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${CHECKPOINT_PATH}/0/items \
    grain_train_files=/tmp/gcsfuse/grain_data_arrayrecord/train/*.array_record" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=maxtext
