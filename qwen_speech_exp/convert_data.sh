#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GCS_BUCKET="gs://arabic-asr-dataset"
LOCAL_OUTPUT="/tmp/arrayrecord_output"

# Mount GCS bucket (read-only for input)
bash "${PROJECT_ROOT}/tools/setup/setup_gcsfuse.sh" \
  DATASET_GCS_BUCKET=arabic-asr-dataset \
  MOUNT_PATH=/tmp/gcsfuse

# Convert each split: read from gcsfuse, write locally, then upload to GCS
for SPLIT in train validation test; do
  INPUT_DIR="/tmp/gcsfuse/grain_data/${SPLIT}"
  LOCAL_OUT="${LOCAL_OUTPUT}/${SPLIT}"
  GCS_OUT="${GCS_BUCKET}/grain_data_arrayrecord/${SPLIT}/"

  if [ ! -d "${INPUT_DIR}" ]; then
    echo "=== Skipping ${SPLIT} (directory not found: ${INPUT_DIR}) ==="
    continue
  fi

  echo "=== Converting ${SPLIT} ==="
  mkdir -p "${LOCAL_OUT}"

  python3 "${SCRIPT_DIR}/convert_tfrecord_to_arrayrecord.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${LOCAL_OUT}" \
    --file_pattern "*.tfrecord"

  echo "=== Uploading ${SPLIT} to GCS ==="
  gsutil -m cp "${LOCAL_OUT}"/*.array_record "${GCS_OUT}"

  echo "=== Cleaning up local ${SPLIT} files ==="
  rm -rf "${LOCAL_OUT}"
done

echo "All splits converted and uploaded!"
echo "Output: ${GCS_BUCKET}/grain_data_arrayrecord/"
