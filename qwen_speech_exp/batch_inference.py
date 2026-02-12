"""Batch inference for audio transcription using Qwen3-Omni on TPU v4-32.

Runs across all 4 TPU hosts cooperatively (model-parallel). The model is sharded
across all 16 chips via MaxEngine. All hosts participate in every inference step.
Only host 0 writes output.

Uses continuous batching: all batch slots are filled with samples, one generate
step advances all slots simultaneously, and finished slots are immediately
refilled with new samples from the queue.

Usage (via multihost_runner.py from jumpbox):
    python3 qwen_speech_exp/batch_inference.py

Output: gs://arabic-asr-dataset/distillation/inference_results.jsonl
"""

import collections
import os
import json
import glob
import subprocess
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import jax
import tensorflow as tf

from absl import app

from MaxText import maxengine
from MaxText import pyconfig
from maxtext.multimodal.processor_qwen3_omni import (
    pre_process_audio_qwen3_omni,
    get_rope_index,
    _get_feat_extract_output_lengths,
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
NUM_SAMPLES = 5000
SEED = 42
N_WINDOW = 50  # n_window_for_audio from model config
MAX_GEN_STEPS = 256  # max_target(512) - max_prefill(256)


@dataclass
class SlotState:
  """Tracks the state of a single decode slot in continuous batching."""
  slot_id: int
  sample_idx: int
  record: dict
  fpath: str
  record_idx: int
  generated_tokens: list = field(default_factory=list)
  active: bool = True


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


def prepare_sample(record, hf_tokenizer, max_prefill_length, chunk_size, config):
  """Prepare a single sample for prefill: audio features, tokens, positions.

  Returns None if the sample should be skipped, otherwise returns a dict with
  all inputs needed for engine.prefill().
  """
  audio = record["audio"].astype(np.float32)
  if len(audio) == 0:
    max_logging.log(f"Skipping empty audio: {record['sample_id']}")
    return None

  audio_features, audio_mask = pre_process_audio_qwen3_omni(audio)
  audio_features = pad_audio_features(audio_features, chunk_size)

  padded_mel_frames = audio_features.shape[2]
  num_audio_tokens = int(_get_feat_extract_output_lengths(np.array(padded_mel_frames)).item())

  if num_audio_tokens == 0:
    max_logging.log(f"Skipping too-short audio: {record['sample_id']}")
    return None

  prompt_tokens = build_prompt_tokens(hf_tokenizer, num_audio_tokens)
  true_length = len(prompt_tokens)

  if true_length > max_prefill_length:
    max_logging.log(
        f"Skipping too-long prompt ({true_length} > {max_prefill_length}): {record['sample_id']}"
    )
    return None

  padded_tokens = np.zeros(max_prefill_length, dtype=np.int32)
  padded_tokens[:true_length] = prompt_tokens

  # Compute MRoPE 3D position IDs for audio tokens
  tokens_2d = padded_tokens[np.newaxis, :]
  attention_mask = np.zeros_like(tokens_2d)
  attention_mask[0, :true_length] = 1
  position_ids, mrope_position_deltas = get_rope_index(
      input_ids=tokens_2d,
      image_grid_thw=None,
      video_grid_thw=None,
      attention_mask=attention_mask,
      use_audio_in_video=False,
      audio_lengths=np.array([padded_mel_frames]),
      spatial_merge_size=config.spatial_merge_size_for_vit,
      position_id_per_seconds=config.position_id_per_seconds,
  )
  position_ids = position_ids.astype(np.int32)
  mrope_position_deltas = mrope_position_deltas.astype(np.int32)

  return {
      "padded_tokens": padded_tokens,
      "position_ids": position_ids,
      "mrope_position_deltas": mrope_position_deltas,
      "audio_features": audio_features,
      "audio_mask": audio_mask,
      "true_length": true_length,
      "padded_mel_frames": padded_mel_frames,
      "num_audio_tokens": num_audio_tokens,
  }


def main(argv):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # Build config from command line argv
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
  total_batch = config.per_device_batch_size * jax.device_count()

  max_logging.log(
      f"Engine ready. max_prefill={max_prefill_length}, max_target={max_target_length}, "
      f"per_device_batch={config.per_device_batch_size}, devices={jax.device_count()}, "
      f"total_batch_slots={total_batch}"
  )

  # --- Select samples (all hosts, deterministic) ---
  data_dir = os.path.join(GCSFUSE_BASE, "train")
  sample_list = select_samples(data_dir, NUM_SAMPLES, SEED)
  max_logging.log(f"Selected {len(sample_list)} samples for inference")

  # --- Check resume (all hosts load completed_ids for sync) ---
  completed_ids = set()
  if is_host0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
  completed_ids = load_completed_ids(OUTPUT_FILE)
  if completed_ids and is_host0:
    max_logging.log(f"Resuming: {len(completed_ids)} samples already completed")

  # --- Build sample queue (all hosts, deterministic, skip completed) ---
  import grain.python as grain

  ds_cache = {}
  sample_queue = collections.deque()

  for sample_idx, (fpath, record_idx) in enumerate(sample_list):
    if fpath not in ds_cache:
      ds_cache[fpath] = grain.ArrayRecordDataSource([fpath])
    ds = ds_cache[fpath]
    raw_record = ds[record_idx]
    record = parse_record(raw_record)

    if record["sample_id"] in completed_ids:
      continue

    sample_queue.append((sample_idx, fpath, record_idx, record))

  max_logging.log(f"Sample queue: {len(sample_queue)} samples to process (skipped {len(completed_ids)} completed)")

  if len(sample_queue) == 0:
    max_logging.log("All samples already completed. Nothing to do.")
    for ds in ds_cache.values():
      del ds
    return

  # Open output file for appending (host 0 only)
  output_f = None
  if is_host0:
    output_f = open(OUTPUT_FILE, "a")

  # --- Initialize decode state (all hosts) ---
  rng, rng_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_decode)
  max_logging.log(f"Decode state initialized with {total_batch} slots")

  # --- Slot management ---
  slots = [None] * total_batch  # SlotState or None
  num_completed = len(completed_ids)
  num_completed_this_run = 0
  num_errors = 0
  inference_start_time = time.time()
  last_gcs_upload_time = time.time()
  GCS_UPLOAD_INTERVAL = 300

  def fill_slot(slot_id, sample_info):
    """Prefill one sample into a slot. All hosts must call this together."""
    nonlocal decode_state, rng, slots
    idx, fpath, rec_idx, record = sample_info

    inputs = prepare_sample(record, hf_tokenizer, max_prefill_length, chunk_size, config)
    if inputs is None:
      # Sample is invalid, mark slot as empty and return False
      slots[slot_id] = None
      return False

    max_logging.log(
        f"Prefilling slot {slot_id}: sample {idx}, mel_frames={inputs['padded_mel_frames']}, "
        f"audio_tokens={inputs['num_audio_tokens']}, true_length={inputs['true_length']}"
    )

    rng, rng_prefill = jax.random.split(rng)
    prefill_result, first_token = engine.prefill(
        params=params,
        padded_tokens=inputs["padded_tokens"],
        positions=inputs["position_ids"],
        mrope_deltas=inputs["mrope_position_deltas"],
        audio_values=inputs["audio_features"],
        audio_masks=inputs["audio_mask"],
        true_length=inputs["true_length"],
        rng=rng_prefill,
        slot=slot_id,
    )

    decode_state = engine.insert(prefill_result, decode_state, slot=slot_id)

    # Prefill returns batch-size-1; always use slot 0 to extract
    first_tok = first_token.get_result_at_slot(0).tokens.item()

    slots[slot_id] = SlotState(
        slot_id=slot_id,
        sample_idx=idx,
        record=record,
        fpath=fpath,
        record_idx=rec_idx,
        generated_tokens=[first_tok],
        active=True,
    )
    return True

  try:
    # --- Fill initial slots ---
    max_logging.log(f"Filling initial {min(total_batch, len(sample_queue))} slots...")
    for slot_id in range(total_batch):
      while sample_queue:
        sample_info = sample_queue.popleft()
        try:
          if fill_slot(slot_id, sample_info):
            break
        except Exception as e:
          num_errors += 1
          max_logging.log(
              f"Error prefilling slot {slot_id} sample {sample_info[0]}: {e}"
          )
          traceback.print_exc()
          if num_errors > 50:
            raise RuntimeError(f"Too many errors ({num_errors}), aborting") from e
      # If sample_queue is empty and slot wasn't filled, slot stays None

    active_count = sum(1 for s in slots if s is not None and s.active)
    max_logging.log(f"Initial fill complete. {active_count} active slots out of {total_batch}")

    # --- Generate loop ---
    max_gen_upper_bound = MAX_GEN_STEPS * 2  # generous upper bound
    for step in range(max_gen_upper_bound):
      active_slots = [s for s in slots if s is not None and s.active]
      if not active_slots:
        break

      # One generate step — processes ALL slots simultaneously
      rng, rng_gen = jax.random.split(rng)
      decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_gen)

      # Check each active slot for new tokens
      finished_slots = []
      for s in slots:
        if s is None or not s.active:
          continue
        token_id = sampled_tokens.get_result_at_slot(s.slot_id).tokens.item()
        s.generated_tokens.append(token_id)
        if token_id == eos_id or len(s.generated_tokens) >= MAX_GEN_STEPS:
          s.active = False
          finished_slots.append(s)

      # Save results for finished slots (host 0 only)
      for s in finished_slots:
        if is_host0:
          output_text = hf_tokenizer.decode(s.generated_tokens, skip_special_tokens=True)
          result = {
              "sample_id": s.record["sample_id"],
              "dataset_name": s.record["dataset_name"],
              "audio_file": os.path.relpath(s.fpath, GCSFUSE_BASE),
              "record_idx": s.record_idx,
              "original_text": s.record["text"],
              "model_transcription": output_text,
              "duration": s.record["duration"],
          }
          output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
          output_f.flush()

          num_completed += 1
          num_completed_this_run += 1
          elapsed = time.time() - inference_start_time
          avg_time = elapsed / num_completed_this_run
          remaining_samples = len(sample_queue) + sum(
              1 for ss in slots if ss is not None and ss.active
          )
          throughput = num_completed_this_run / elapsed
          max_logging.log(
              f"[{num_completed}/{len(sample_list)}] slot={s.slot_id} "
              f"sample_id={s.record['sample_id']} "
              f"duration={s.record['duration']:.1f}s "
              f"gen_tokens={len(s.generated_tokens)} "
              f"avg={avg_time:.1f}s/sample throughput={throughput:.2f}samples/s "
              f"remaining={remaining_samples} queue={len(sample_queue)}"
          )

      # Refill finished slots with new samples (all hosts)
      for s in finished_slots:
        filled = False
        while sample_queue and not filled:
          sample_info = sample_queue.popleft()
          try:
            filled = fill_slot(s.slot_id, sample_info)
          except Exception as e:
            num_errors += 1
            max_logging.log(
                f"Error refilling slot {s.slot_id} sample {sample_info[0]}: {e}"
            )
            traceback.print_exc()
            if num_errors > 50:
              raise RuntimeError(f"Too many errors ({num_errors}), aborting") from e
        if not filled:
          slots[s.slot_id] = None  # No more samples for this slot

      # Periodic GCS upload (host 0)
      if is_host0 and time.time() - last_gcs_upload_time > GCS_UPLOAD_INTERVAL:
        subprocess.run(
            ["gsutil", "cp", OUTPUT_FILE, GCS_OUTPUT],
            capture_output=True, text=True,
        )
        last_gcs_upload_time = time.time()
        max_logging.log(f"Periodic upload to GCS ({num_completed} results)")

      # Log batch status periodically
      if is_host0 and step % 50 == 0:
        active_count = sum(1 for s in slots if s is not None and s.active)
        max_logging.log(
            f"Step {step}: {active_count} active slots, "
            f"{len(sample_queue)} in queue, {num_completed_this_run} completed this run"
        )

  finally:
    if output_f is not None:
      output_f.close()
    for ds in ds_cache.values():
      del ds

  if is_host0:
    elapsed = time.time() - inference_start_time
    max_logging.log(
        f"Batch inference complete. {num_completed} total ({num_completed_this_run} this run), "
        f"{num_errors} errors. Elapsed: {elapsed:.0f}s. Output: {OUTPUT_FILE}"
    )
    # Final upload to GCS
    result = subprocess.run(["gsutil", "cp", OUTPUT_FILE, GCS_OUTPUT], capture_output=True, text=True)
    if result.returncode == 0:
      max_logging.log(f"Uploaded results to {GCS_OUTPUT}")
    else:
      max_logging.log(f"Failed to upload to GCS: {result.stderr}")


if __name__ == "__main__":
  app.run(main)
