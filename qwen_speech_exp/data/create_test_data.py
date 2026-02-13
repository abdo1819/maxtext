"""Generate a small synthetic ArrayRecord dataset for testing the audio GRPO pipeline.

Creates 30 short audio samples (sine waves at various frequencies, 1-3 seconds
each at 16kHz) paired with ground truth transcription strings. Writes them
as ArrayRecord files in the tf.train.Example format expected by
ParseAudioFeatures in _audio_data_processing.py:

  - 'audio': FloatList (raw audio samples)
  - 'text': BytesList (text transcription)
  - 'sample_rate': Int64List (audio sample rate)
  - 'audio_len': Int64List (number of audio samples)

Usage:
  python3 qwen_speech_exp/data/create_test_data.py --output_dir /tmp/test_grpo_data
"""

import argparse
import os
import numpy as np
import tensorflow as tf

try:
  from array_record.python.array_record_module import ArrayRecordWriter
except ImportError:
  print("array_record not installed. Trying grain...")
  ArrayRecordWriter = None


SAMPLE_RATE = 16000
NUM_SAMPLES = 30

# Simple ground truth transcriptions for testing
GROUND_TRUTHS = [
    "hello world",
    "this is a test",
    "speech recognition",
    "the quick brown fox",
    "jumps over the lazy dog",
    "good morning everyone",
    "how are you today",
    "artificial intelligence",
    "machine learning is fun",
    "deep neural networks",
    "audio processing test",
    "transcription sample",
    "testing one two three",
    "welcome to the demo",
    "natural language processing",
    "automatic speech recognition",
    "reinforcement learning",
    "group relative policy",
    "chain of thought reasoning",
    "this is sample twenty",
    "hello again world",
    "another test sentence",
    "computing on tpu chips",
    "jax and flax framework",
    "training with grpo",
    "audio features extracted",
    "mel spectrogram input",
    "reward function testing",
    "final test example",
    "end of test data",
]


def generate_audio(duration_sec, frequency_hz=440.0):
  """Generate a sine wave audio sample."""
  t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
  # Add some harmonics and noise for variety
  audio = 0.5 * np.sin(2 * np.pi * frequency_hz * t)
  audio += 0.2 * np.sin(2 * np.pi * frequency_hz * 2 * t)
  audio += 0.05 * np.random.randn(len(t))
  audio = audio.astype(np.float32)
  # Normalize
  audio = audio / (np.abs(audio).max() + 1e-8)
  return audio


def make_tf_example(audio, text, sample_rate):
  """Create a tf.train.Example with audio, text, sample_rate, audio_len."""
  feature = {
      "audio": tf.train.Feature(float_list=tf.train.FloatList(value=audio.tolist())),
      "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode("utf-8")])),
      "sample_rate": tf.train.Feature(int64_list=tf.train.Int64List(value=[sample_rate])),
      "audio_len": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(audio)])),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
  parser = argparse.ArgumentParser(description="Generate test ArrayRecord data for audio GRPO")
  parser.add_argument("--output_dir", type=str, default="/tmp/test_grpo_data", help="Output directory for ArrayRecord files")
  parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, help="Number of samples to generate")
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  output_path = os.path.join(args.output_dir, "test_audio_grpo.arrayrecord")

  rng = np.random.RandomState(42)

  if ArrayRecordWriter is not None:
    writer = ArrayRecordWriter(output_path, "group_size:1")
  else:
    # Fallback: use TFRecord format
    output_path = os.path.join(args.output_dir, "test_audio_grpo.tfrecord")
    writer = tf.io.TFRecordWriter(output_path)

  for i in range(args.num_samples):
    duration = rng.uniform(1.0, 3.0)
    frequency = rng.uniform(200.0, 800.0)
    audio = generate_audio(duration, frequency)
    text = GROUND_TRUTHS[i % len(GROUND_TRUTHS)]

    example = make_tf_example(audio, text, SAMPLE_RATE)
    serialized = example.SerializeToString()

    if ArrayRecordWriter is not None:
      writer.write(serialized)
    else:
      writer.write(serialized)

  if ArrayRecordWriter is not None:
    writer.close()
  else:
    writer.close()

  print(f"Wrote {args.num_samples} examples to {output_path}")
  print(f"File size: {os.path.getsize(output_path)} bytes")


if __name__ == "__main__":
  main()
