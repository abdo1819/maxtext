"""Batch inference for audio transcription using Qwen3-Omni on TPU v4-32.

Runs across all 4 TPU hosts cooperatively (model-parallel). The model is sharded
across all 16 chips via MaxEngine. All hosts participate in every inference step.
Only host 0 writes output.

Usage (via multihost_runner.py from jumpbox):
    python3 qwen_speech_exp/batch_inference.py

Output: gs://arabic-asr-dataset/distillation/inference_results.jsonl
"""

import os
import json
import glob

import numpy as np
import jax
import tensorflow as tf

from absl import app

from MaxText import maxengine
from MaxText import pyconfig
from maxtext.multimodal.processor_qwen3_omni import (
    pre_process_audio_qwen3_omni,
    QWEN3_OMNI_AUDIO_START_TOKEN,
    QWEN3_OMNI_AUDIO_END_TOKEN,
    QWEN3_OMNI_AUDIO_TOKEN,
    SAMPLE_RATE,
)
from maxtext.utils import max_logging

# --- Configuration ---
GCSFUSE_BASE = "/tmp/gcsfuse/grain_data_arrayrecord"
OUTPUT_DIR = "/tmp/distillation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "inference_results.jsonl")
GCS_OUTPUT = "gs://arabic-asr-dataset/distillation/inference_results.jsonl"
NUM_SAMPLES = 10
SEED = 42
N_WINDOW = 50  # n_window_for_audio from model config


def parse_record(raw_record):
  """Parse a serialized tf.train.Example from ArrayRecord into a dict with all 7 fields."""
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

  duration_sparse = tf.sparse.to_dense(parsed["duration"]).numpy()
  duration = float(duration_sparse[0]) if len(duration_sparse) > 0 else len(audio) / SAMPLE_RATE

  return {
      "audio": audio,
      "text": parsed["text"].numpy().decode("utf-8"),
      "sample_rate": int(parsed["sample_rate"].numpy()),
      "audio_len": len(audio),
      "duration": duration,
      "dataset_name": parsed["dataset_name"].numpy().decode("utf-8"),
      "sample_id": parsed["sample_id"].numpy().decode("utf-8"),
  }


def select_samples(data_dir, num_samples, seed):
  """Select a deterministic subset of samples from ArrayRecord files.

  Returns list of (file_path, record_idx) tuples.
  """
  import grain.python as grain

  ar_files = sorted(glob.glob(os.path.join(data_dir, "*.array_record")))
  if not ar_files:
    raise FileNotFoundError(f"No ArrayRecord files found in {data_dir}")

  # Build index of (file, record_idx) for all records
  all_indices = []
  for fpath in ar_files:
    ds = grain.ArrayRecordDataSource([fpath])
    n = len(ds)
    for idx in range(n):
      all_indices.append((fpath, idx))
    del ds

  max_logging.log(f"Total records across {len(ar_files)} files: {len(all_indices)}")

  # Deterministic shuffle and select
  rng = np.random.RandomState(seed)
  selected = rng.choice(len(all_indices), size=min(num_samples, len(all_indices)), replace=False)
  selected.sort()  # Sort for sequential file access
  return [all_indices[i] for i in selected]


def load_completed_ids(output_file):
  """Load already-processed sample_ids from partial output for resume."""
  completed = set()
  if os.path.exists(output_file):
    with open(output_file, "r") as f:
      for line in f:
        line = line.strip()
        if line:
          try:
            record = json.loads(line)
            completed.add(record["sample_id"])
          except (json.JSONDecodeError, KeyError):
            continue
  return completed


def build_prompt_tokens(hf_tokenizer, num_audio_tokens):
  """Build tokenized prompt with audio placeholders for transcription.

  Returns (token_array, true_length) as int32 numpy array.
  """
  # Encode prefix: system message + user prefix
  prefix_text = (
      "<|im_start|>system\n"
      "You are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n"
  )
  suffix_text = (
      "\nTranscribe this Arabic audio accurately.<|im_end|>\n"
      "<|im_start|>assistant\n"
  )

  prefix_ids = hf_tokenizer.encode(prefix_text, add_special_tokens=False)
  suffix_ids = hf_tokenizer.encode(suffix_text, add_special_tokens=False)

  # Build audio token block: <audio_start> <audio_pad>*N <audio_eos>
  audio_block = (
      [QWEN3_OMNI_AUDIO_START_TOKEN]
      + [QWEN3_OMNI_AUDIO_TOKEN] * num_audio_tokens
      + [QWEN3_OMNI_AUDIO_END_TOKEN]
  )

  full_tokens = np.array(prefix_ids + audio_block + suffix_ids, dtype=np.int32)
  return full_tokens


def pad_audio_features(audio_features, chunk_size):
  """Pad mel spectrogram frames to be divisible by chunk_size.

  Args:
    audio_features: (1, mel_bins, frames) mel spectrogram
    chunk_size: n_window * 2 (must be divisible)

  Returns:
    Padded audio features with frames divisible by chunk_size.
  """
  frames = audio_features.shape[2]
  remainder = frames % chunk_size
  if remainder != 0:
    pad_len = chunk_size - remainder
    audio_features = np.pad(
        audio_features, ((0, 0), (0, 0), (0, pad_len)), mode="constant", constant_values=0.0
    )
  return audio_features


def main(argv):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # Build config from command line argv
  # Filter out non-MaxText args
  config_argv = [a for a in argv if not a.startswith("--")]
  config = pyconfig.initialize(config_argv)

  max_logging.log(f"Process {jax.process_index()} of {jax.process_count()} initialized")
  is_host0 = jax.process_index() == 0

  # --- Build engine and load model (all hosts) ---
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load = jax.random.split(rng)
  params = engine.load_params(rng_load)

  # Get tokenizer
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  hf_tokenizer = tokenizer_model.tokenizer  # Underlying HuggingFace tokenizer
  eos_id = tokenizer_model.eos_id

  max_prefill_length = config.max_prefill_predict_length
  max_target_length = config.max_target_length
  chunk_size = N_WINDOW * 2  # 100

  max_logging.log(
      f"Engine ready. max_prefill={max_prefill_length}, max_target={max_target_length}, "
      f"devices={jax.device_count()}, local_devices={jax.local_device_count()}"
  )

  # --- Select samples (all hosts, deterministic) ---
  data_dir = os.path.join(GCSFUSE_BASE, "train")
  sample_list = select_samples(data_dir, NUM_SAMPLES, SEED)
  max_logging.log(f"Selected {len(sample_list)} samples for inference")

  # --- Check resume (host 0 only, but all hosts skip the same samples) ---
  completed_ids = set()
  if is_host0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    completed_ids = load_completed_ids(OUTPUT_FILE)
    if completed_ids:
      max_logging.log(f"Resuming: {len(completed_ids)} samples already completed")

  # Broadcast completed count to all hosts for sync
  completed_count = len(completed_ids)
  # All hosts need to know which samples to skip, but since we process sequentially
  # and all hosts must stay in sync, we'll load completed_ids on all hosts
  if not is_host0:
    completed_ids = load_completed_ids(OUTPUT_FILE)

  # Open output file for appending (host 0 only)
  output_f = None
  if is_host0:
    output_f = open(OUTPUT_FILE, "a")

  # --- Initialize decode state once (all hosts) ---
  rng, rng_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_decode)

  # --- Process samples ---
  import grain.python as grain

  # Cache open data sources to avoid reopening files
  ds_cache = {}
  num_completed = len(completed_ids)
  num_errors = 0

  try:
    for sample_idx, (fpath, record_idx) in enumerate(sample_list):
      # Load the record (all hosts)
      if fpath not in ds_cache:
        ds_cache[fpath] = grain.ArrayRecordDataSource([fpath])
      ds = ds_cache[fpath]
      raw_record = ds[record_idx]
      record = parse_record(raw_record)

      # Skip if already completed
      if record["sample_id"] in completed_ids:
        continue

      # Process audio
      audio = record["audio"].astype(np.float32)
      if len(audio) == 0:
        max_logging.log(f"Skipping empty audio: {record['sample_id']}")
        continue

      # Get mel spectrogram and pad to chunk-aligned length
      audio_features, audio_mask = pre_process_audio_qwen3_omni(audio)
      audio_features = pad_audio_features(audio_features, chunk_size)

      # Compute number of audio tokens
      num_mel_frames = audio_features.shape[2]
      num_audio_tokens = num_mel_frames // chunk_size

      if num_audio_tokens == 0:
        max_logging.log(f"Skipping too-short audio: {record['sample_id']}")
        continue

      # Build prompt tokens
      prompt_tokens = build_prompt_tokens(hf_tokenizer, num_audio_tokens)
      true_length = len(prompt_tokens)

      if true_length > max_prefill_length:
        max_logging.log(
            f"Skipping too-long prompt ({true_length} > {max_prefill_length}): {record['sample_id']}"
        )
        continue

      # Pad to max_prefill_length
      padded_tokens = np.zeros(max_prefill_length, dtype=np.int32)
      padded_tokens[:true_length] = prompt_tokens

      # --- Run inference (all hosts participate) ---
      try:
        rng, rng_prefill = jax.random.split(rng)
        prefill_result, first_token = engine.prefill(
            params=params,
            padded_tokens=padded_tokens,
            audio_values=audio_features,
            true_length=true_length,
            rng=rng_prefill,
            slot=0,
        )

        # Insert prefill into decode state (reuses pre-allocated KV cache)
        decode_state = engine.insert(prefill_result, decode_state, slot=0)

        # Collect first token
        first_token_id = first_token.get_result_at_slot(0).tokens.item()
        generated_tokens = [first_token_id]

        # Generate loop
        for step in range(max_prefill_length, max_target_length):
          rng, rng_gen = jax.random.split(rng)
          decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_gen)
          token_id = sampled_tokens.get_result_at_slot(0).tokens.item()
          generated_tokens.append(token_id)
          if token_id == eos_id:
            break

        # Decode output text (host 0)
        if is_host0:
          output_text = hf_tokenizer.decode(generated_tokens, skip_special_tokens=True)

          # Write result
          result = {
              "sample_id": record["sample_id"],
              "dataset_name": record["dataset_name"],
              "audio_file": os.path.relpath(fpath, GCSFUSE_BASE),
              "record_idx": record_idx,
              "original_text": record["text"],
              "model_transcription": output_text,
              "duration": record["duration"],
          }
          output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
          output_f.flush()

          num_completed += 1
          max_logging.log(
              f"[{num_completed}/{len(sample_list)}] sample_id={record['sample_id']} "
              f"duration={record['duration']:.1f}s "
              f"gen_tokens={len(generated_tokens)}"
          )

      except Exception as e:
        num_errors += 1
        if is_host0:
          max_logging.log(f"Error processing {record['sample_id']}: {e}")
        if num_errors > 50:
          raise RuntimeError(f"Too many errors ({num_errors}), aborting") from e

  finally:
    if output_f is not None:
      output_f.close()
    for ds in ds_cache.values():
      del ds

  if is_host0:
    max_logging.log(
        f"Batch inference complete. {num_completed} samples processed, {num_errors} errors. "
        f"Output: {OUTPUT_FILE}"
    )
    # Upload results to GCS
    import subprocess

    result = subprocess.run(["gsutil", "cp", OUTPUT_FILE, GCS_OUTPUT], capture_output=True, text=True)
    if result.returncode == 0:
      max_logging.log(f"Uploaded results to {GCS_OUTPUT}")
    else:
      max_logging.log(f"Failed to upload to GCS: {result.stderr}")


if __name__ == "__main__":
  app.run(main)
