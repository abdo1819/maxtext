#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GCS_BUCKET="gs://arabic-asr-dataset"
LOCAL_OUTPUT="/tmp/arrayrecord_output"
TRACKER_FILE="${LOCAL_OUTPUT}/upload_tracker.json"

# Mount GCS bucket (read-only for input)
bash "${PROJECT_ROOT}/tools/setup/setup_gcsfuse.sh" \
  DATASET_GCS_BUCKET=arabic-asr-dataset \
  MOUNT_PATH=/tmp/gcsfuse

# Initialize tracker JSON — rebuild from GCS if missing
mkdir -p "${LOCAL_OUTPUT}"
if [ ! -f "${TRACKER_FILE}" ]; then
  echo "Tracker file not found, rebuilding from GCS..."
  python3 -c "
import json, subprocess

tracker = {}
for split in ['train', 'validation', 'test']:
    result = subprocess.run(
        ['gsutil', 'ls', f'${GCS_BUCKET}/grain_data_arrayrecord/{split}/'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        files = [line.split('/')[-1] for line in result.stdout.strip().split('\n') if line.strip()]
        tracked = [f for f in files if f.endswith('.array_record')]
        if tracked:
            tracker[split] = tracked
            print(f'  {split}: {len(tracked)} files already in GCS')
    else:
        print(f'  {split}: no existing files in GCS')

with open('${TRACKER_FILE}', 'w') as f:
    json.dump(tracker, f, indent=2)
print(f'Tracker rebuilt with {sum(len(v) for v in tracker.values())} total files.')
"
fi

# Helper: check if a file is already tracked as uploaded
is_uploaded() {
  local split="$1" basename="$2"
  python3 -c "
import json, sys
with open('${TRACKER_FILE}') as f:
    tracker = json.load(f)
uploaded = tracker.get('$split', [])
sys.exit(0 if '$basename' in uploaded else 1)
"
}

# Helper: mark a file as uploaded in tracker
mark_uploaded() {
  local split="$1" basename="$2"
  python3 -c "
import json
with open('${TRACKER_FILE}') as f:
    tracker = json.load(f)
tracker.setdefault('$split', []).append('$basename')
with open('${TRACKER_FILE}', 'w') as f:
    json.dump(tracker, f, indent=2)
"
}

# Helper: upload a single file, track it, delete local copy
upload_and_track() {
  local split="$1" local_file="$2" gcs_dir="$3"
  local basename
  basename="$(basename "${local_file}")"

  gsutil cp "${local_file}" "${gcs_dir}${basename}"
  mark_uploaded "${split}" "${basename}"
  rm -f "${local_file}"
  echo "  Uploaded + cleaned: ${basename}"
}

# Convert each split: one file at a time, upload immediately
for SPLIT in train validation test; do
  INPUT_DIR="/tmp/gcsfuse/grain_data/${SPLIT}"
  LOCAL_OUT="${LOCAL_OUTPUT}/${SPLIT}"
  GCS_OUT="${GCS_BUCKET}/grain_data_arrayrecord/${SPLIT}/"

  if [ ! -d "${INPUT_DIR}" ]; then
    echo "=== Skipping ${SPLIT} (directory not found: ${INPUT_DIR}) ==="
    continue
  fi

  echo "=== Processing ${SPLIT} ==="
  mkdir -p "${LOCAL_OUT}"

  # Phase 1: Upload any leftover local files from a previous interrupted run
  EXISTING_FILES=("${LOCAL_OUT}"/*.array_record)
  if [ -f "${EXISTING_FILES[0]}" ]; then
    echo "--- Found leftover local files, uploading first ---"
    for LOCAL_FILE in "${EXISTING_FILES[@]}"; do
      BASENAME="$(basename "${LOCAL_FILE}")"
      if is_uploaded "${SPLIT}" "${BASENAME}"; then
        echo "  Already tracked: ${BASENAME}, deleting local copy"
        rm -f "${LOCAL_FILE}"
      else
        upload_and_track "${SPLIT}" "${LOCAL_FILE}" "${GCS_OUT}"
      fi
    done
    echo "--- Leftover files handled ---"
  fi

  # Phase 2: Convert each TFRecord file individually, upload immediately
  TFRECORD_FILES=($(ls "${INPUT_DIR}"/*.tfrecord 2>/dev/null | sort))
  TOTAL=${#TFRECORD_FILES[@]}
  echo "Found ${TOTAL} TFRecord files"

  for i in "${!TFRECORD_FILES[@]}"; do
    INPUT_FILE="${TFRECORD_FILES[$i]}"
    BASENAME="$(basename "${INPUT_FILE}" .tfrecord)"
    AR_FILE="${LOCAL_OUT}/${BASENAME}.array_record"
    IDX=$((i + 1))

    # Skip if already uploaded
    if is_uploaded "${SPLIT}" "${BASENAME}.array_record"; then
      echo "[${IDX}/${TOTAL}] ${BASENAME}: already uploaded, skipping"
      continue
    fi

    # Convert
    python3 "${SCRIPT_DIR}/convert_tfrecord_to_arrayrecord.py" \
      --input_dir "${INPUT_DIR}" \
      --output_dir "${LOCAL_OUT}" \
      --file_pattern "$(basename "${INPUT_FILE}")"

    # Upload, track, delete
    if [ -f "${AR_FILE}" ]; then
      echo "[${IDX}/${TOTAL}] Uploading ${BASENAME}.array_record ..."
      upload_and_track "${SPLIT}" "${AR_FILE}" "${GCS_OUT}"
    else
      echo "[${IDX}/${TOTAL}] WARNING: expected ${AR_FILE} not found after conversion"
    fi
  done

  echo "=== ${SPLIT} complete ==="
done

echo ""
echo "All splits converted and uploaded!"
echo "Output: ${GCS_BUCKET}/grain_data_arrayrecord/"
echo "Tracker: ${TRACKER_FILE}"
