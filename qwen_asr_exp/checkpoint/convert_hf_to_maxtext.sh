#!/bin/bash
set -e

# Source the variables
source "$(dirname "$0")/../env_vars.sh"

echo "=================================================="
echo "Phase 1: Checking Conversion Status"
echo "Model: $MODEL_NAME"
echo "Target: $CHECKPOINT_PATH"
echo "=================================================="

# Check if checkpoint already exists in GCS to avoid re-running
if gsutil ls "${CHECKPOINT_PATH}/0/items" &> /dev/null; then
    echo "Checkpoint already found at: ${CHECKPOINT_PATH}"
    echo "Skipping conversion step..."
else
    echo "Checkpoint not found. Starting conversion..."

    python "${MAXTEXT_PKG_DIR}/utils/ckpt_conversion/to_maxtext.py" \
        "${CONFIG_FILE}" \
        model_name="$MODEL_NAME" \
        base_output_directory="${CHECKPOINT_PATH}" \
        hf_access_token="$HF_TOKEN" \
        hardware=cpu \
        skip_jax_distributed_system=True \
        scan_layers=False

    echo "Conversion completed successfully!"
fi
