#!/bin/bash
set -e

# Convert the fp32 MaxText checkpoint to bf16 to halve storage and loading memory.
# Run this on a TPU worker (has ~340GB host RAM) or any machine with enough memory.
#
# From jumpbox (run on a single TPU worker):
#   gcloud compute tpus tpu-vm ssh qr-v4-32 --zone=us-central2-b --worker=0 \
#     --project=arabic-asr-level2thinkg --internal-ip \
#     --command="cd ~/maxtext && source ~/venv-maxtext/bin/activate && bash qwen_speech_exp/checkpoint/convert_ckpt_bf16.sh"

source "$(dirname "$0")/../env_vars.sh"

SOURCE_CKPT="${CHECKPOINT_PATH}"
DEST_CKPT="${CHECKPOINT_PATH}-bf16"

echo "=================================================="
echo "Converting checkpoint from fp32 to bf16"
echo "Source: ${SOURCE_CKPT}"
echo "Dest:   ${DEST_CKPT}"
echo "=================================================="

python3 qwen_speech_exp/checkpoint/convert_ckpt_to_bf16.py \
    --source "${SOURCE_CKPT}" \
    --dest "${DEST_CKPT}"

echo ""
echo "=================================================="
echo "Conversion complete!"
echo "Update your training/inference scripts to use:"
echo "  load_parameters_path=${DEST_CKPT}/0/items"
echo "=================================================="
