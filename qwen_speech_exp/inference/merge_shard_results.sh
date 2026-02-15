#!/bin/bash
# Merge sharded inference results into a single file.
#
# Downloads inference_results_shard{0..3}.jsonl from GCS, concatenates them,
# deduplicates by sample_id, and uploads the merged inference_results.jsonl.
#
# Usage (from jumpbox or any machine with gsutil):
#   bash qwen_speech_exp/inference/merge_shard_results.sh
set -e

GCS_DIR="gs://arabic-asr-dataset/distillation"
LOCAL_DIR="/tmp/distillation/merge"
NUM_SHARDS=4

mkdir -p "$LOCAL_DIR"

# Download all shard files
echo "Downloading shard files..."
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_FILE="inference_results_shard${i}.jsonl"
  echo "  Downloading ${SHARD_FILE}..."
  gsutil cp "${GCS_DIR}/${SHARD_FILE}" "${LOCAL_DIR}/${SHARD_FILE}" 2>/dev/null || \
    echo "  WARNING: ${SHARD_FILE} not found, skipping"
done

# Also download the old unsharded file if it exists (contains pre-sharding results)
echo "  Downloading old unsharded file (if exists)..."
gsutil cp "${GCS_DIR}/inference_results.jsonl" "${LOCAL_DIR}/inference_results_old.jsonl" 2>/dev/null || true

# Concatenate all files and deduplicate by sample_id
echo "Merging and deduplicating..."
MERGED_FILE="${LOCAL_DIR}/inference_results_merged.jsonl"
python3 -c "
import json
import glob
import os

local_dir = '${LOCAL_DIR}'
seen = {}

# Load old unsharded results first (lower priority)
old_file = os.path.join(local_dir, 'inference_results_old.jsonl')
if os.path.exists(old_file):
    with open(old_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    seen[record['sample_id']] = record
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f'  Old unsharded: {len(seen)} records')

# Load shard files (higher priority, overwrite old results)
for i in range(${NUM_SHARDS}):
    shard_file = os.path.join(local_dir, f'inference_results_shard{i}.jsonl')
    count = 0
    if os.path.exists(shard_file):
        with open(shard_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        seen[record['sample_id']] = record
                        count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
    print(f'  Shard {i}: {count} records')

# Write merged output
with open('${MERGED_FILE}', 'w') as f:
    for record in seen.values():
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f'  Total unique records: {len(seen)}')
"

# Upload merged file
echo "Uploading merged file to GCS..."
gsutil cp "$MERGED_FILE" "${GCS_DIR}/inference_results.jsonl"
echo "Done. Merged file: ${GCS_DIR}/inference_results.jsonl"

# Show summary
TOTAL=$(wc -l < "$MERGED_FILE")
echo "Total records in merged file: $TOTAL"
