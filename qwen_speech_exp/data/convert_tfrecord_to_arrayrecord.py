"""Convert TFRecord files to ArrayRecord format for MaxText grain pipeline.

Usage (on TPU VM after gcsfuse mount):
  python3 convert_tfrecord_to_arrayrecord.py \
    --input_dir /tmp/gcsfuse/grain_data/train \
    --output_dir /tmp/gcsfuse/grain_data_arrayrecord/train \
    --file_pattern "*.tfrecord"

Run for each split: train, validation, test.
"""

import argparse
import glob
import os

from array_record.python.array_record_module import ArrayRecordWriter
import tensorflow as tf


def convert_single_file(input_path, output_path):
    """Convert a single TFRecord file to ArrayRecord.

    Records are copied verbatim -- both formats store serialized
    tf.train.Example protobufs, only the container differs.
    """
    writer = ArrayRecordWriter(output_path, "group_size:1")
    count = 0
    for raw_record in tf.compat.v1.io.tf_record_iterator(input_path):
        writer.write(raw_record)
        count += 1
    writer.close()
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert TFRecord files to ArrayRecord format"
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing TFRecord files")
    parser.add_argument("--output_dir", required=True, help="Output directory for ArrayRecord files")
    parser.add_argument("--file_pattern", default="*.tfrecord", help="Glob pattern for input files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.file_pattern)))
    print(f"Found {len(input_files)} TFRecord files in {args.input_dir}")

    if not input_files:
        print("No files found. Check --input_dir and --file_pattern.")
        return

    total = 0
    for i, input_path in enumerate(input_files):
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(args.output_dir, f"{basename}.array_record")
        count = convert_single_file(input_path, output_path)
        total += count
        print(f"[{i+1}/{len(input_files)}] {basename}: {count} records")

    print(f"Done. Total records converted: {total}")


if __name__ == "__main__":
    main()
