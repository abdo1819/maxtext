#!/bin/bash
# Launch batch inference on TPU v4-32 via multihost_runner.py.
# Run this from the jumpbox (maxtext-vm).
#
# Usage:
#   ./qwen_speech_exp/run_batch_inference.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env_vars.sh"

python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=tpu-v4-32-node \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="cd ~/maxtext && source ~/venv-maxtext/bin/activate && \
git pull && \
bash tools/setup/setup_gcsfuse.sh DATASET_GCS_BUCKET=arabic-asr-dataset MOUNT_PATH=/tmp/gcsfuse && \
bash preflight.sh 2>/dev/null || true && \
mkdir -p /tmp/distillation && \
gsutil cp gs://arabic-asr-dataset/distillation/inference_results.jsonl /tmp/distillation/inference_results.jsonl 2>/dev/null || true && \
export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE' && \
python3 qwen_speech_exp/batch_inference.py \
    src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    tokenizer_type=huggingface \
    load_parameters_path=${CHECKPOINT_PATH}/0/items \
    max_prefill_predict_length=256 \
    max_target_length=512 \
    per_device_batch_size=4 \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=16 \
    ici_tensor_parallelism=1 \
    scan_layers=false \
    weight_dtype=bfloat16 \
    quantize_kvcache=false \
    attention=dot_product \
    use_multimodal=true \
    use_audio=true \
    skip_jax_distributed_system=false" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=maxtext
