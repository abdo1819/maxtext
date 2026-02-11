"""Generate Chain-of-Thought reasoning using Gemini API.

Reads inference_results.jsonl from Stage 1, sends each sample's audio + model
transcription + correct transcription to Gemini to generate CoT reasoning
about confusing/unclear audio parts.

Runs on the jumpbox (maxtext-vm) — no TPU needed, just internet access.

Usage:
    python3 qwen_speech_exp/generate_cot.py [--input INPUT] [--output OUTPUT]
        [--rpm RPM] [--max_retries MAX_RETRIES] [--model MODEL]
"""

import argparse
import io
import json
import os
import time
import wave

import numpy as np
import tensorflow as tf

GCSFUSE_BASE = "/tmp/gcsfuse/grain_data_arrayrecord"
DEFAULT_INPUT = "/tmp/gcsfuse/distillation/inference_results.jsonl"
DEFAULT_OUTPUT = "/tmp/gcsfuse/distillation/cot_results.jsonl"
SAMPLE_RATE = 16000

PROMPT_TEMPLATE = """You are an expert Arabic speech analysis assistant. You are given:
1. An audio recording
2. A model-generated transcription: "{model_transcription}"
3. The correct transcription: "{original_text}"

Listen to the audio carefully. Generate a detailed chain-of-thought analysis about:
- Which parts of the audio are unclear or confusing
- Why the model may have made specific errors
- Phonetic similarities that could cause confusion
- Background noise, accent, or pronunciation challenges
- How to correctly identify the unclear segments

Write your analysis in Arabic. Be concise but thorough."""


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


def audio_to_wav_bytes(audio_samples, sample_rate=SAMPLE_RATE):
  """Convert float32 audio samples to WAV bytes in-memory."""
  # Normalize to int16 range
  audio_int16 = np.clip(audio_samples * 32767, -32768, 32767).astype(np.int16)
  buf = io.BytesIO()
  with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(sample_rate)
    wf.writeframes(audio_int16.tobytes())
  return buf.getvalue()


def load_audio_from_arrayrecord(audio_file, record_idx):
  """Load raw audio samples from an ArrayRecord file."""
  from array_record.python.array_record_module import ArrayRecordDataSource

  fpath = os.path.join(GCSFUSE_BASE, audio_file)
  ds = ArrayRecordDataSource(fpath)
  raw_record = ds[record_idx]
  del ds

  parsed = tf.io.parse_single_example(
      raw_record,
      {
          "audio": tf.io.VarLenFeature(tf.float32),
          "audio_len": tf.io.FixedLenFeature([], tf.int64, default_value=0),
      },
  )
  audio = tf.sparse.to_dense(parsed["audio"]).numpy()
  audio_len = int(parsed["audio_len"].numpy())
  if audio_len > 0:
    audio = audio[:audio_len]
  return audio.astype(np.float32)


def call_gemini_with_retry(client, model_name, audio_bytes, prompt, max_retries=5):
  """Call Gemini API with exponential backoff retry."""
  from google import genai
  from google.genai import types

  for attempt in range(max_retries):
    try:
      response = client.models.generate_content(
          model=model_name,
          contents=[
              types.Content(
                  role="user",
                  parts=[
                      types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                      types.Part.from_text(text=prompt),
                  ],
              )
          ],
      )
      if response.text:
        return response.text.strip()
      return ""
    except Exception as e:
      wait_time = min(2**attempt * 2, 60)
      print(f"  Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
      if attempt < max_retries - 1:
        print(f"  Retrying in {wait_time}s...")
        time.sleep(wait_time)
      else:
        raise


def main():
  parser = argparse.ArgumentParser(description="Generate CoT reasoning with Gemini API")
  parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to inference_results.jsonl")
  parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output cot_results.jsonl")
  parser.add_argument("--rpm", type=int, default=10, help="Max requests per minute")
  parser.add_argument("--max_retries", type=int, default=5, help="Max retries per API call")
  parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
  args = parser.parse_args()

  # Initialize Gemini client
  from google import genai

  api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
  client = genai.Client(api_key=api_key)

  # Load inference results
  if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

  samples = []
  with open(args.input, "r") as f:
    for line in f:
      line = line.strip()
      if line:
        samples.append(json.loads(line))
  print(f"Loaded {len(samples)} inference results from {args.input}")

  # Check resume
  completed_ids = load_completed_ids(args.output)
  if completed_ids:
    print(f"Resuming: {len(completed_ids)} samples already completed")
  remaining = [s for s in samples if s["sample_id"] not in completed_ids]
  print(f"Remaining: {len(remaining)} samples to process")

  if not remaining:
    print("All samples already processed. Nothing to do.")
    return

  # Rate limiting
  min_interval = 60.0 / args.rpm  # seconds between requests

  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  num_processed = len(completed_ids)
  num_errors = 0

  with open(args.output, "a") as out_f:
    for i, sample in enumerate(remaining):
      start_time = time.time()
      sample_id = sample["sample_id"]

      try:
        # Load audio from ArrayRecord
        audio = load_audio_from_arrayrecord(sample["audio_file"], sample["record_idx"])
        wav_bytes = audio_to_wav_bytes(audio)

        # Build prompt
        prompt = PROMPT_TEMPLATE.format(
            model_transcription=sample["model_transcription"],
            original_text=sample["original_text"],
        )

        # Call Gemini
        cot = call_gemini_with_retry(client, args.model, wav_bytes, prompt, args.max_retries)

        # Write result
        result = {
            "sample_id": sample_id,
            "dataset_name": sample["dataset_name"],
            "audio_file": sample["audio_file"],
            "record_idx": sample["record_idx"],
            "original_text": sample["original_text"],
            "model_transcription": sample["model_transcription"],
            "cot": cot,
            "duration": sample["duration"],
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

        num_processed += 1
        print(
            f"[{num_processed}/{len(samples)}] sample_id={sample_id} "
            f"cot_len={len(cot)} chars"
        )

      except Exception as e:
        num_errors += 1
        print(f"Error processing {sample_id}: {e}")
        if num_errors > 100:
          raise RuntimeError(f"Too many errors ({num_errors}), aborting") from e

      # Rate limiting
      elapsed = time.time() - start_time
      if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

  print(f"CoT generation complete. {num_processed} processed, {num_errors} errors.")
  print(f"Output: {args.output}")


if __name__ == "__main__":
  main()
