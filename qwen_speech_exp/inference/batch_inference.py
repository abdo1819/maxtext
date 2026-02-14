"""Batch inference with bucketed audio padding and continuous batching.

Uses per_device_batch_size=4 (64 total slots) with audio mel-frame bucketing
to minimize JIT recompilation. Audio is padded to fixed bucket sizes so that
prefill compiles once per bucket (~8 times) instead of once per unique audio
length (~200+ times).

Slots are filled incrementally: a small initial batch (8 slots) is filled
first to start generation quickly, then remaining slots are filled between
generate steps. This avoids waiting ~45 minutes to fill all 64 slots before
any output is produced.

All 4 TPU hosts participate in every inference step (model-parallel).
Only host 0 writes output.

Usage (via multihost_runner.py from jumpbox):
    python3 qwen_speech_exp/inference/batch_inference.py <config.yml> [key=value ...]

Output: gs://arabic-asr-dataset/distillation/inference_results.jsonl
"""

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
GCS_UPLOAD_INTERVAL = 300  # Upload to GCS every 5 minutes

# Mel-frame bucket boundaries (multiples of 100, matching audio encoder chunk_size).
# Each bucket triggers one JIT compilation for prefill.
# Bucket 1600 → 208 audio tokens → ~248 prompt tokens (fits in max_prefill=256).
AUDIO_BUCKETS = [200, 400, 600, 800, 1000, 1200, 1400, 1600]


@dataclass
class SlotState:
  """Tracks the state of one decode slot in the KV cache."""
  slot_idx: int
  active: bool = False
  sample_id: str = ""
  dataset_name: str = ""
  audio_file: str = ""
  record_idx: int = -1
  original_text: str = ""
  duration: float = 0.0
  generated_tokens: list = field(default_factory=list)
  sample_idx: int = -1


def get_audio_bucket(mel_frames):
  """Find the smallest bucket that fits mel_frames."""
  for b in AUDIO_BUCKETS:
    if mel_frames <= b:
      return b
  # Too long for any bucket — round up to next 100 (will likely be skipped by prompt length check)
  return ((mel_frames + 99) // 100) * 100


def pad_audio_to_bucket(audio_features, audio_mask, bucket_frames):
  """Pad mel spectrogram and mask to exact bucket frame count.

  Both must have the same time dimension for consistent JIT shapes.
  """
  current = audio_features.shape[2]
  if current < bucket_frames:
    pad = bucket_frames - current
    audio_features = np.pad(audio_features, ((0, 0), (0, 0), (0, pad)), mode="constant")
    audio_mask = np.pad(audio_mask, ((0, 0), (0, pad)), mode="constant")
  return audio_features, audio_mask


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
  prefix_text = (
      "<|im_start|>user\n"
  )
  suffix_text = (
      "\nTranscribe the Arabic audio into text.<|im_end|>\n"
      "<|im_start|>assistant\n"
  )

  prefix_ids = hf_tokenizer.encode(prefix_text, add_special_tokens=False)
  suffix_ids = hf_tokenizer.encode(suffix_text, add_special_tokens=False)

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

  config_argv = [a for a in argv if not a.startswith("--")]
  config = pyconfig.initialize(config_argv)

  max_logging.log(f"Process {jax.process_index()} of {jax.process_count()} initialized")
  is_host0 = jax.process_index() == 0

  # --- Build engine and load model (all hosts) ---
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load = jax.random.split(rng)
  params = engine.load_params(rng_load)

  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  hf_tokenizer = tokenizer_model.tokenizer
  eos_id = tokenizer_model.eos_id

  max_prefill_length = config.max_prefill_predict_length
  max_target_length = config.max_target_length
  max_generate_steps = max_target_length - max_prefill_length
  chunk_size = N_WINDOW * 2  # 100
  total_slots = config.per_device_batch_size * jax.device_count()

  max_logging.log(
      f"Engine ready. max_prefill={max_prefill_length}, max_target={max_target_length}, "
      f"max_generate_steps={max_generate_steps}, total_slots={total_slots}, "
      f"devices={jax.device_count()}, buckets={AUDIO_BUCKETS}"
  )

  # --- Select samples (all hosts, deterministic) ---
  data_dir = os.path.join(GCSFUSE_BASE, "train")
  sample_list = select_samples(data_dir, NUM_SAMPLES, SEED)
  max_logging.log(f"Selected {len(sample_list)} samples for inference")

  # --- Resume (all hosts load completed_ids for sync) ---
  completed_ids = load_completed_ids(OUTPUT_FILE)
  if is_host0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if completed_ids:
      max_logging.log(f"Resuming: {len(completed_ids)} samples already completed")

  output_f = open(OUTPUT_FILE, "a") if is_host0 else None

  # --- Initialize decode state and slots ---
  rng, rng_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_decode)
  slots = [SlotState(slot_idx=i) for i in range(total_slots)]

  # --- Sample iterator (lazy loading) ---
  import grain.python as grain
  ds_cache = {}

  def sample_iterator():
    for sample_idx, (fpath, record_idx) in enumerate(sample_list):
      if fpath not in ds_cache:
        ds_cache[fpath] = grain.ArrayRecordDataSource([fpath])
      raw_record = ds_cache[fpath][record_idx]
      record = parse_record(raw_record)
      if record["sample_id"] in completed_ids:
        continue
      yield sample_idx, fpath, record_idx, record

  sample_iter = iter(sample_iterator())

  # --- Tracking ---
  num_completed = len(completed_ids)
  num_errors = 0
  num_skipped = 0
  inference_start = time.time()
  last_gcs_upload = time.time()
  compiled_buckets = set()
  samples_exhausted = False

  def try_fill_slot(slot):
    """Try to fill an empty slot with the next available sample.

    Skips samples with empty/too-short/too-long audio.
    Returns True if slot was filled, False if no more samples.
    """
    nonlocal decode_state, rng, num_skipped, num_errors, samples_exhausted

    while True:
      try:
        sample_idx, fpath, record_idx, record = next(sample_iter)
      except StopIteration:
        samples_exhausted = True
        return False

      audio = record["audio"].astype(np.float32)
      if len(audio) == 0:
        max_logging.log(f"Skip empty audio: {record['sample_id']}")
        num_skipped += 1
        continue

      # --- Audio processing with bucketing ---
      audio_features, audio_mask = pre_process_audio_qwen3_omni(audio)
      audio_features = pad_audio_features(audio_features, chunk_size)
      mel_frames = audio_features.shape[2]
      bucket = get_audio_bucket(mel_frames)

      # Pad both audio_features and audio_mask to bucket size for consistent JIT shapes
      audio_features, audio_mask = pad_audio_to_bucket(audio_features, audio_mask, bucket)

      # Compute audio token count from BUCKETED frame count
      num_audio_tokens = int(_get_feat_extract_output_lengths(np.array(bucket)).item())
      if num_audio_tokens == 0:
        max_logging.log(f"Skip zero tokens: {record['sample_id']}")
        num_skipped += 1
        continue

      # Build prompt with bucketed token count
      prompt_tokens = build_prompt_tokens(hf_tokenizer, num_audio_tokens)
      true_length = len(prompt_tokens)
      if true_length > max_prefill_length:
        max_logging.log(
            f"Skip too long ({true_length}>{max_prefill_length}): {record['sample_id']} "
            f"bucket={bucket} audio_tokens={num_audio_tokens}"
        )
        num_skipped += 1
        continue

      # Pad tokens to max_prefill_length (fixed shape)
      padded_tokens = np.zeros(max_prefill_length, dtype=np.int32)
      padded_tokens[:true_length] = prompt_tokens

      # MRoPE 3D position IDs
      tokens_2d = padded_tokens[np.newaxis, :]
      attention_mask = np.zeros_like(tokens_2d)
      attention_mask[0, :true_length] = 1
      position_ids, mrope_deltas = get_rope_index(
          input_ids=tokens_2d,
          image_grid_thw=None,
          video_grid_thw=None,
          attention_mask=attention_mask,
          use_audio_in_video=False,
          audio_lengths=np.array([bucket]),
          spatial_merge_size=config.spatial_merge_size_for_vit,
          position_id_per_seconds=config.position_id_per_seconds,
      )
      position_ids = position_ids.astype(np.int32)
      mrope_deltas = mrope_deltas.astype(np.int32)

      # --- Prefill + Insert ---
      is_new_bucket = bucket not in compiled_buckets
      if is_new_bucket:
        max_logging.log(
            f"Bucket {bucket} mel frames: first use, JIT compiling prefill... "
            f"({len(compiled_buckets)}/{len(AUDIO_BUCKETS)} buckets compiled)"
        )
        compiled_buckets.add(bucket)

      try:
        t0 = time.time()
        rng, rng_prefill = jax.random.split(rng)
        prefill_result, first_token = engine.prefill(
            params=params,
            padded_tokens=padded_tokens,
            positions=position_ids,
            mrope_deltas=mrope_deltas,
            audio_values=audio_features,
            audio_masks=audio_mask,
            true_length=true_length,
            rng=rng_prefill,
            slot=slot.slot_idx,
        )
        decode_state = engine.insert(prefill_result, decode_state, slot=slot.slot_idx)
        dt = time.time() - t0

        # Prefill always returns batch-size-1 result at index 0
        first_token_id = first_token.get_result_at_slot(0).tokens.item()

        # Populate slot state
        slot.active = True
        slot.sample_id = record["sample_id"]
        slot.dataset_name = record["dataset_name"]
        slot.audio_file = os.path.relpath(fpath, GCSFUSE_BASE)
        slot.record_idx = record_idx
        slot.original_text = record["text"]
        slot.duration = record["duration"]
        slot.generated_tokens = [first_token_id]
        slot.sample_idx = sample_idx

        max_logging.log(
            f"Slot {slot.slot_idx}: filled in {dt:.1f}s, bucket={bucket}, "
            f"mel={mel_frames}, tokens={num_audio_tokens}, sample={sample_idx}"
            + (" (JIT)" if is_new_bucket else "")
        )
        return True

      except Exception as e:
        num_errors += 1
        max_logging.log(
            f"Prefill error slot {slot.slot_idx}, sample {record['sample_id']}: {e}"
        )
        traceback.print_exc()
        if num_errors > 50:
          raise RuntimeError(f"Too many errors ({num_errors}), aborting") from e
        continue  # Try next sample

  def finish_slot(slot):
    """Write result for a completed slot and deactivate it."""
    nonlocal num_completed, last_gcs_upload

    if is_host0 and output_f:
      output_text = hf_tokenizer.decode(slot.generated_tokens, skip_special_tokens=True)
      result = {
          "sample_id": slot.sample_id,
          "dataset_name": slot.dataset_name,
          "audio_file": slot.audio_file,
          "record_idx": slot.record_idx,
          "original_text": slot.original_text,
          "model_transcription": output_text,
          "duration": slot.duration,
      }
      output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
      output_f.flush()
      num_completed += 1

      done_this_run = num_completed - len(completed_ids)
      elapsed = time.time() - inference_start
      avg = elapsed / done_this_run if done_this_run > 0 else 0
      remaining = len(sample_list) - num_completed - num_skipped
      eta_min = (avg * remaining) / 60 if avg > 0 else 0
      max_logging.log(
          f"[{num_completed}/{len(sample_list)}] {slot.sample_id} "
          f"dur={slot.duration:.1f}s tokens={len(slot.generated_tokens)} "
          f"avg={avg:.1f}s/sample eta={eta_min:.0f}min"
      )

      # Periodic GCS upload
      if time.time() - last_gcs_upload > GCS_UPLOAD_INTERVAL:
        subprocess.run(["gsutil", "cp", OUTPUT_FILE, GCS_OUTPUT], capture_output=True)
        last_gcs_upload = time.time()
        max_logging.log(f"GCS upload ({num_completed} results)")

    # Reset slot
    slot.active = False
    slot.generated_tokens = []

  # === Main continuous batching loop ===
  # Fill a small initial batch, then interleave generation with filling.
  # Each prefill takes ~40s (full model forward pass across 16 chips),
  # so filling all 64 slots upfront would waste ~45 minutes before
  # any generation starts. Instead, fill INITIAL_FILL slots, start
  # generating, and fill remaining empty slots between generate steps.
  INITIAL_FILL = min(8, total_slots)
  try:
    # Phase 1: Fill initial batch of slots
    filled_count = 0
    for slot in slots:
      if filled_count >= INITIAL_FILL:
        break
      if not try_fill_slot(slot):
        break
      filled_count += 1

    max_logging.log(f"Initial fill: {filled_count}/{total_slots} slots filled, starting generation")

    # Phase 2: Interleaved generate + fill loop
    while True:
      active_slots = [s for s in slots if s.active]
      if not active_slots and samples_exhausted:
        break
      if not active_slots:
        # No active slots but samples remain — fill one and retry
        for slot in slots:
          if not slot.active and try_fill_slot(slot):
            break
        continue

      active_count = len(active_slots)
      empty_count = sum(1 for s in slots if not s.active)
      max_logging.log(f"Generate phase: {active_count} active, {empty_count} empty slots")

      for step in range(max_generate_steps):
        rng, rng_gen = jax.random.split(rng)
        decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_gen)

        # Check each active slot for tokens
        newly_done = []
        for slot in slots:
          if not slot.active:
            continue
          token_id = sampled_tokens.get_result_at_slot(slot.slot_idx).tokens.item()
          slot.generated_tokens.append(token_id)

          if token_id == eos_id or len(slot.generated_tokens) >= max_generate_steps:
            newly_done.append(slot)

        # Finish completed slots and immediately try to refill them
        for slot in newly_done:
          finish_slot(slot)
          if not samples_exhausted:
            try_fill_slot(slot)

        # Note: proactive ramp-up filling was removed because each prefill
        # takes ~40s, blocking generation for all active slots. Slots are
        # only filled when they naturally complete (newly_done above).

        # If no slots are active, break the generate loop
        if not any(s.active for s in slots):
          break

  except Exception as e:
    max_logging.log(f"Fatal error: {e}")
    traceback.print_exc()
    raise

  finally:
    if output_f:
      output_f.close()
    for ds in ds_cache.values():
      del ds

  if is_host0:
    elapsed = time.time() - inference_start
    max_logging.log(
        f"Batch inference complete. {num_completed} done, {num_errors} errors, "
        f"{num_skipped} skipped in {elapsed/60:.1f} min. Output: {OUTPUT_FILE}"
    )
    result = subprocess.run(
        ["gsutil", "cp", OUTPUT_FILE, GCS_OUTPUT], capture_output=True, text=True
    )
    if result.returncode == 0:
      max_logging.log(f"Uploaded to {GCS_OUTPUT}")
    else:
      max_logging.log(f"GCS upload failed: {result.stderr}")


if __name__ == "__main__":
  app.run(main)
