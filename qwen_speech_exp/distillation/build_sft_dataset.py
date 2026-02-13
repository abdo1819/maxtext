"""Build SFT dataset with Chain-of-Thought reasoning in ArrayRecord format.

Reads cot_results.jsonl from Stage 2, joins with original audio from ArrayRecord
files, and writes new ArrayRecord files with the text field formatted as:
    <think>
    {cot}
    </think>

    {transcription}

This plugs directly into the existing audio SFT pipeline since BuildAudioSFTSequence
tokenizes the text field and the Qwen3-Omni tokenizer handles <think>/<think> as
special tokens (IDs 151667/151668).

Runs on the jumpbox (maxtext-vm).

Usage:
    python3 qwen_speech_exp/distillation/build_sft_dataset.py [--input INPUT] [--output_dir OUTPUT_DIR]
        [--records_per_file RECORDS_PER_FILE]
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from array_record.python.array_record_module import ArrayRecordWriter

GCSFUSE_BASE = "/tmp/gcsfuse/grain_data_arrayrecord"
DEFAULT_INPUT = "/tmp/gcsfuse/distillation/cot_results.jsonl"
DEFAULT_OUTPUT_DIR = "/tmp/gcsfuse/grain_data_arrayrecord/distillation_sft"
SAMPLE_RATE = 16000


def parse_full_record(raw_record):
  """Parse all 7 fields from a serialized tf.train.Example."""
  parsed = tf.io.parse_single_example(
      raw_record,
      {
          "audio": tf.io.VarLenFeature(tf.float32),
          "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
          "sample_rate": tf.io.FixedLenFeature([], tf.int64, default_value=SAMPLE_RATE),
          "audio_len": tf.io.FixedLenFeature([], tf.int64, default_value=0),
          "duration": tf.io.VarLenFeature(tf.float32),
          "dataset_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
          "sample_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
      },
  )
  audio = tf.sparse.to_dense(parsed["audio"]).numpy()
  audio_len = int(parsed["audio_len"].numpy())
  if audio_len > 0:
    audio = audio[:audio_len]
  else:
    audio_len = len(audio)

  duration_sparse = tf.sparse.to_dense(parsed["duration"]).numpy()
  duration = float(duration_sparse[0]) if len(duration_sparse) > 0 else len(audio) / SAMPLE_RATE

  return {
      "audio": audio,
      "audio_len": audio_len,
      "sample_rate": int(parsed["sample_rate"].numpy()),
      "duration": duration,
      "text": parsed["text"].numpy().decode("utf-8"),
      "dataset_name": parsed["dataset_name"].numpy().decode("utf-8"),
      "sample_id": parsed["sample_id"].numpy().decode("utf-8"),
  }


def build_sft_text(cot, original_text):
  """Format the SFT text field with CoT in Qwen3-Omni <think> format.

  Output format:
      <think>
      {cot}
      </think>

      {transcription}
  """
  return f"<think>\n{cot}\n</think>\n\n{original_text}"


def make_tf_example(audio, audio_len, sample_rate, duration, text, dataset_name, sample_id):
  """Create a tf.train.Example with the full 7-field schema."""
  feature = {
      "audio": tf.train.Feature(float_list=tf.train.FloatList(value=audio.tolist())),
      "audio_len": tf.train.Feature(int64_list=tf.train.Int64List(value=[audio_len])),
      "sample_rate": tf.train.Feature(int64_list=tf.train.Int64List(value=[sample_rate])),
      "duration": tf.train.Feature(float_list=tf.train.FloatList(value=[duration])),
      "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode("utf-8")])),
      "dataset_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[dataset_name.encode("utf-8")])),
      "sample_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample_id.encode("utf-8")])),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
  parser = argparse.ArgumentParser(description="Build SFT dataset with CoT from distillation results")
  parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to cot_results.jsonl")
  parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for ArrayRecord files")
  parser.add_argument("--records_per_file", type=int, default=500, help="Max records per output ArrayRecord file")
  args = parser.parse_args()

  # Load CoT results
  if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

  cot_records = []
  with open(args.input, "r") as f:
    for line in f:
      line = line.strip()
      if line:
        cot_records.append(json.loads(line))
  print(f"Loaded {len(cot_records)} CoT results from {args.input}")

  # Filter out records with empty CoT
  cot_records = [r for r in cot_records if r.get("cot", "").strip()]
  print(f"After filtering empty CoT: {len(cot_records)} records")

  if not cot_records:
    print("No records to process. Exiting.")
    return

  os.makedirs(args.output_dir, exist_ok=True)

  # Group by audio_file for efficient reading
  from collections import defaultdict

  by_file = defaultdict(list)
  for rec in cot_records:
    by_file[rec["audio_file"]].append(rec)

  # Process and write ArrayRecord files
  import grain.python as grain

  total_written = 0
  file_idx = 0
  writer = None
  records_in_current_file = 0

  def open_new_writer():
    nonlocal writer, file_idx, records_in_current_file
    if writer is not None:
      writer.close()
    out_path = os.path.join(args.output_dir, f"distillation_sft_{file_idx:04d}.array_record")
    writer = ArrayRecordWriter(out_path, "group_size:1")
    records_in_current_file = 0
    file_idx += 1
    return out_path

  current_file = open_new_writer()

  for audio_file, records_for_file in by_file.items():
    fpath = os.path.join(GCSFUSE_BASE, audio_file)
    if not os.path.exists(fpath):
      print(f"Warning: ArrayRecord file not found: {fpath}, skipping {len(records_for_file)} records")
      continue

    ds = grain.ArrayRecordDataSource([fpath])

    for cot_rec in records_for_file:
      record_idx = cot_rec["record_idx"]

      # Read original record
      try:
        raw_record = ds[record_idx]
      except (IndexError, Exception) as e:
        print(f"Warning: cannot read record {record_idx} from {audio_file}: {e}")
        continue

      original = parse_full_record(raw_record)

      # Verify sample_id matches
      if original["sample_id"] != cot_rec["sample_id"]:
        print(
            f"Warning: sample_id mismatch at {audio_file}[{record_idx}]: "
            f"expected {cot_rec['sample_id']}, got {original['sample_id']}"
        )
        continue

      # Build SFT text with CoT
      sft_text = build_sft_text(cot_rec["cot"], cot_rec["original_text"])

      # Create new tf.train.Example
      example = make_tf_example(
          audio=original["audio"],
          audio_len=original["audio_len"],
          sample_rate=original["sample_rate"],
          duration=original["duration"],
          text=sft_text,
          dataset_name=original["dataset_name"],
          sample_id=original["sample_id"],
      )

      # Write to ArrayRecord
      writer.write(example.SerializeToString())
      total_written += 1
      records_in_current_file += 1

      # Rotate file if needed
      if records_in_current_file >= args.records_per_file:
        current_file = open_new_writer()

    del ds

  # Close final writer
  if writer is not None:
    writer.close()

  print(f"SFT dataset build complete.")
  print(f"  Total records written: {total_written}")
  print(f"  Output files: {file_idx} ArrayRecord files in {args.output_dir}")

  # Print a sample for verification
  if total_written > 0:
    sample_file = os.path.join(args.output_dir, "distillation_sft_0000.array_record")
    if os.path.exists(sample_file):
      ds = grain.ArrayRecordDataSource([sample_file])
      sample = parse_full_record(ds[0])
      del ds
      print(f"\nSample record:")
      print(f"  sample_id: {sample['sample_id']}")
      print(f"  audio_len: {sample['audio_len']}")
      print(f"  text (first 200 chars): {sample['text'][:200]}")


if __name__ == "__main__":
  main()
