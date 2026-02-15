#!/bin/bash
# Launch 4x parallel single-host batch inference on TPU v4-32.
#
# Each of the 4 TPU workers runs an independent JAX process using its local
# 4 chips (ici_expert_parallelism=4, skip_jax_distributed_system=true).
# Workers auto-detect their shard index from TPU metadata (WORKER_ID).
#
# This gives ~4x throughput vs the multi-host setup (run_batch_inference.sh)
# since 4 prefill streams run in parallel instead of 1.
#
# Usage (from jumpbox):
#   bash qwen_speech_exp/inference/run_batch_inference_sharded.sh
#
# After all shards complete, merge results:
#   bash qwen_speech_exp/inference/merge_shard_results.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../env_vars.sh"

python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="cd ~/maxtext && source ~/venv-maxtext/bin/activate && \
git pull && \
bash tools/setup/setup_gcsfuse.sh DATASET_GCS_BUCKET=arabic-asr-dataset MOUNT_PATH=/tmp/gcsfuse && \
bash preflight.sh 2>/dev/null || true && \
mkdir -p /tmp/distillation && \
sudo rm -f /tmp/tpu-env && \
curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' > /tmp/tpu-env && \
WORKER_ID=\$(grep '^WORKER_ID' /tmp/tpu-env | cut -d \"'\" -f 2) && \
echo \"Worker ID: \$WORKER_ID\" && \
gsutil cp gs://arabic-asr-dataset/distillation/inference_results_shard\${WORKER_ID}.jsonl /tmp/distillation/inference_results_shard\${WORKER_ID}.jsonl 2>/dev/null || true && \
gsutil cp gs://arabic-asr-dataset/distillation/inference_results.jsonl /tmp/distillation/inference_results.jsonl 2>/dev/null || true && \
export SHARD_INDEX=\$WORKER_ID && \
export NUM_SHARDS=4 && \
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 && \
export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE' && \
python3 qwen_speech_exp/inference/batch_inference.py \
    src/maxtext/configs/base.yml \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    tokenizer_type=huggingface \
    load_parameters_path=${CHECKPOINT_PATH}/0/items \
    max_prefill_predict_length=256 \
    max_target_length=512 \
    per_device_batch_size=4 \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=4 \
    ici_tensor_parallelism=1 \
    scan_layers=false \
    weight_dtype=bfloat16 \
    quantize_kvcache=true \
    kv_quant_dtype=int8 \
    attention=dot_product \
    use_multimodal=true \
    use_audio=true \
    skip_jax_distributed_system=true" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=maxtext
