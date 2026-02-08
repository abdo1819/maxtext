#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Mount GCS bucket
bash "${PROJECT_ROOT}/tools/setup/setup_gcsfuse.sh" \
  DATASET_GCS_BUCKET=arabic-asr-dataset \
  MOUNT_PATH=/tmp/gcsfuse

# Convert each split
for SPLIT in train validation test; do
  INPUT_DIR="/tmp/gcsfuse/grain_data/${SPLIT}"
  OUTPUT_DIR="/tmp/gcsfuse/grain_data_arrayrecord/${SPLIT}"

  if [ ! -d "${INPUT_DIR}" ]; then
    echo "=== Skipping ${SPLIT} (directory not found: ${INPUT_DIR}) ==="
    continue
  fi

  echo "=== Converting ${SPLIT} ==="
  python3 "${SCRIPT_DIR}/convert_tfrecord_to_arrayrecord.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --file_pattern "*.tfrecord"
done

echo "All splits converted!"
