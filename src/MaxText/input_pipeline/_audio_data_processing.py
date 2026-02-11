# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio SFT data pipeline using Grain for speech fine-tuning.

Reads ArrayRecord files containing speech data (audio waveform + text transcript),
converts audio to mel spectrograms, tokenizes text, and produces training batches
compatible with the Qwen3-Omni model.

Each ArrayRecord example must be a serialized tf.train.Example with:
  - 'audio': FloatList (raw audio samples)
  - 'text': BytesList (text transcription)
  - 'sample_rate': Int64List (audio sample rate)
  - 'audio_len': Int64List (number of audio samples)
"""

import functools

import jax
import numpy as np
import grain.python as grain
from grain.experimental import pick_performance_config
import tensorflow as tf

from MaxText import multihost_dataloading
from MaxText import tokenizer
from MaxText.input_pipeline import _input_pipeline_utils
from MaxText.input_pipeline._grain_data_processing import get_datasets
from maxtext.multimodal.processor_qwen3_omni import (
    pre_process_audio_qwen3_omni,
    add_extra_tokens_for_qwen3_omni,
    _get_feat_extract_output_lengths,
    QWEN3_OMNI_AUDIO_START_TOKEN,
    QWEN3_OMNI_AUDIO_END_TOKEN,
    QWEN3_OMNI_AUDIO_TOKEN,
    SAMPLE_RATE,
)
from maxtext.utils import max_logging


class ParseAudioFeatures(grain.MapTransform):
  """Parse audio features from serialized tf.train.Example."""

  def map(self, raw_example):
    parsed = tf.io.parse_single_example(
        raw_example,
        {
            "audio": tf.io.VarLenFeature(tf.float32),
            "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "sample_rate": tf.io.FixedLenFeature([], tf.int64, default_value=SAMPLE_RATE),
            "audio_len": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        },
    )
    audio = tf.sparse.to_dense(parsed["audio"]).numpy()
    text = parsed["text"].numpy().decode("utf-8")
    sample_rate = int(parsed["sample_rate"].numpy())
    audio_len = int(parsed["audio_len"].numpy())
    if audio_len == 0:
      audio_len = len(audio)
    return {
        "audio": audio[:audio_len],
        "text": text,
        "sample_rate": sample_rate,
    }


class ResampleAudio(grain.MapTransform):
  """Resample audio to target sample rate if needed."""

  def __init__(self, target_sample_rate=SAMPLE_RATE):
    self.target_sample_rate = target_sample_rate

  def map(self, example):
    audio = example["audio"]
    sr = example["sample_rate"]
    if sr != self.target_sample_rate and sr > 0:
      # Simple linear interpolation resampling
      ratio = self.target_sample_rate / sr
      new_len = int(len(audio) * ratio)
      indices = np.linspace(0, len(audio) - 1, new_len)
      audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    example["audio"] = audio
    example["sample_rate"] = self.target_sample_rate
    return example


class AudioToMelSpectrogram(grain.MapTransform):
  """Convert raw audio waveform to log-mel spectrogram features."""

  def __init__(self, max_audio_length=None):
    self.max_audio_length = max_audio_length

  def map(self, example):
    audio = example["audio"]
    if self.max_audio_length is not None:
      audio = audio[: self.max_audio_length]
    # pre_process_audio_qwen3_omni returns (features, mask)
    # features shape: (1, num_mel_bins, audio_length_frames)
    audio_features, audio_mask = pre_process_audio_qwen3_omni(audio)
    example["audio_features"] = audio_features[0]  # Remove batch dim -> (num_mel_bins, frames)
    example["audio_mask"] = audio_mask[0]  # (frames,)
    # Compute number of audio tokens after encoder downsampling
    # The audio encoder uses windowed convolution that produces
    # audio_length // (n_window * 2) output tokens
    return example


class BuildAudioSFTSequence(grain.MapTransform):
  """Build input/target token sequences for audio SFT.

  Constructs sequences like:
    <|audio_start|> <|audio_pad|>*N <|audio_eos|> <text tokens> <eos>

  The audio placeholder tokens get replaced by audio encoder embeddings
  during the forward pass.
  """

  def __init__(self, tokenizer_model, max_target_length, n_window):
    self.tokenizer_model = tokenizer_model
    self.max_target_length = max_target_length
    self.n_window = n_window

  def map(self, example):
    text = example["text"]
    audio_features = example["audio_features"]  # (num_mel_bins, frames)

    # Pad mel frames to be divisible by chunk_size (n_window * 2) for the audio encoder
    chunk_size = self.n_window * 2
    num_frames = audio_features.shape[1]
    remainder = num_frames % chunk_size
    if remainder != 0:
      pad_len = chunk_size - remainder
      audio_features = np.pad(audio_features, ((0, 0), (0, pad_len)), constant_values=0.0)
      example["audio_features"] = audio_features

    # Compute number of audio tokens using the encoder's output length formula.
    # Each chunk of 100 mel frames goes through 3 stride-2 convolutions producing
    # 13 tokens per chunk, so total = (padded_frames // 100) * 13 (plus remainder).
    padded_frames = audio_features.shape[1]
    num_audio_tokens = int(_get_feat_extract_output_lengths(np.array(padded_frames)).item())

    # Tokenize the text transcript
    text_tokens = self.tokenizer_model.encode(text)
    if isinstance(text_tokens, tuple):
      text_tokens = text_tokens[0]
    if hasattr(text_tokens, "numpy"):
      text_tokens = text_tokens.numpy()
    text_tokens = np.array(text_tokens, dtype=np.int32).flatten()

    # Build sequence: <audio_start> <audio_pad>*N <audio_eos> <text> <eos>
    audio_start = np.array([QWEN3_OMNI_AUDIO_START_TOKEN], dtype=np.int32)
    audio_pads = np.full(num_audio_tokens, QWEN3_OMNI_AUDIO_TOKEN, dtype=np.int32)
    audio_end = np.array([QWEN3_OMNI_AUDIO_END_TOKEN], dtype=np.int32)

    # Get EOS token
    eos_id = getattr(self.tokenizer_model, "eos_id", None)
    if eos_id is not None:
      eos_token = np.array([eos_id], dtype=np.int32)
    else:
      eos_token = np.array([], dtype=np.int32)

    full_sequence = np.concatenate([audio_start, audio_pads, audio_end, text_tokens, eos_token])

    # Truncate to max_target_length
    if len(full_sequence) > self.max_target_length:
      full_sequence = full_sequence[: self.max_target_length]

    # For causal LM: inputs = targets (shifted during training by ShiftData)
    example["inputs"] = full_sequence
    example["targets"] = full_sequence
    example["num_audio_tokens"] = num_audio_tokens

    return example


class PadAudioSFTToMaxLength(grain.MapTransform):
  """Pad sequences and audio features to max length."""

  def __init__(self, max_target_length, pad_id, num_mel_bins, max_audio_frames):
    self.max_target_length = max_target_length
    self.pad_id = pad_id
    self.num_mel_bins = num_mel_bins
    self.max_audio_frames = max_audio_frames

  def map(self, example):
    inputs = example["inputs"]
    targets = example["targets"]
    seq_len = len(inputs)

    # Pad token sequences
    pad_len = self.max_target_length - seq_len
    if pad_len > 0:
      inputs = np.pad(inputs, (0, pad_len), constant_values=self.pad_id)
      targets = np.pad(targets, (0, pad_len), constant_values=self.pad_id)
    else:
      inputs = inputs[: self.max_target_length]
      targets = targets[: self.max_target_length]

    # Create segmentation (1 for real tokens, 0 for padding)
    segmentation = np.zeros(self.max_target_length, dtype=np.int32)
    segmentation[:seq_len] = 1

    # Create position ids (sequential)
    positions = np.arange(self.max_target_length, dtype=np.int32)

    # Pad audio features to fixed shape
    audio_features = example["audio_features"]  # (num_mel_bins, frames)
    cur_frames = audio_features.shape[1]
    if cur_frames < self.max_audio_frames:
      pad_width = ((0, 0), (0, self.max_audio_frames - cur_frames))
      audio_features = np.pad(audio_features, pad_width, constant_values=0.0)
    else:
      audio_features = audio_features[:, : self.max_audio_frames]

    return {
        "inputs": inputs.astype(np.int32),
        "targets": targets.astype(np.int32),
        "inputs_position": positions,
        "targets_position": positions.copy(),
        "inputs_segmentation": segmentation,
        "targets_segmentation": segmentation.copy(),
        "audios": audio_features.astype(np.float32),
    }


class BatchAudioSFT(grain.MapTransform):
  """Reshape the audios field after batching to (batch, num_mel_bins, frames)."""

  def map(self, example):
    # After grain.Batch, audios is already (batch, num_mel_bins, frames)
    # Just ensure proper dtype
    if "audios" in example:
      example["audios"] = np.asarray(example["audios"], dtype=np.float32)
    return example


def audio_sft_preprocessing_pipeline(
    dataset,
    config,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Grain pipeline for audio SFT training with Qwen3-Omni.

  Args:
    dataset: Grain IterDataset or MapDataset from get_datasets()
    config: Model/training configuration
    grain_worker_count: Number of grain workers
    grain_per_worker_buffer_size: Buffer size per worker

  Returns:
    Grain dataset ready for multihost dataloading
  """
  # Parse audio features from serialized examples
  dataset = dataset.map(ParseAudioFeatures())

  # Resample to target rate
  dataset = dataset.map(ResampleAudio(target_sample_rate=SAMPLE_RATE))

  # Compute max audio length based on config
  # max_sample_len_for_audio is in samples; convert to spectrogram-safe limit
  max_audio_samples = config.max_sample_len_for_audio * SAMPLE_RATE
  dataset = dataset.map(AudioToMelSpectrogram(max_audio_length=max_audio_samples))

  # Build tokenizer
  tokenizer_model = tokenizer.build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      config.add_bos,
      config.add_eos,
      config.hf_access_token,
      config.dataset_type,
  )
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = 0

  # Build input/target sequences
  dataset = dataset.map(
      BuildAudioSFTSequence(
          tokenizer_model=tokenizer_model,
          max_target_length=config.max_target_length,
          n_window=config.n_window_for_audio,
      )
  )

  # Compute max audio frames for padding
  # n_window_for_audio * 2 samples per token, and we need a fixed audio shape
  # Use the training n_window (not inference) to compute max frames
  chunk_size = config.n_window_for_audio * 2
  # Estimate max frames: max_sample_len_for_audio * SAMPLE_RATE / HOP_LENGTH
  # But for safety, use a reasonable upper bound
  max_audio_frames = chunk_size * (config.max_sample_len_for_audio // config.n_window_for_audio + 1)

  # Pad to max length
  dataset = dataset.map(
      PadAudioSFTToMaxLength(
          max_target_length=config.max_target_length,
          pad_id=pad_id,
          num_mel_bins=config.num_mel_bins_for_audio,
          max_audio_frames=max_audio_frames,
      )
  )

  # Batch
  batch_size = config.global_batch_size_to_load // jax.process_count()
  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  # Shift inputs for teacher-forced training
  dataset = dataset.map(
      _input_pipeline_utils.ShiftData(
          ignored_ids=[pad_id],
          axis=1,
      )
  )

  # Multiprocessing prefetch
  multiprocessing_options = (
      pick_performance_config(
          ds=dataset,
          ram_budget_mb=config.grain_ram_budget_mb,
          max_workers=None,
          max_buffer_size=None,
      ).multiprocessing_options
      if grain_worker_count == -1
      else grain.MultiprocessingOptions(
          num_workers=grain_worker_count,
          per_worker_buffer_size=grain_per_worker_buffer_size,
      )
  )
  dataset = dataset.mp_prefetch(multiprocessing_options)
  return dataset


def make_grain_audio_train_iterator(config, global_mesh, process_indices):
  """Create audio training data iterator using grain pipeline."""
  assert (
      config.global_batch_size_to_load % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."

  train_ds = get_datasets(
      config.grain_train_files,
      config.grain_file_type,
      shuffle=config.enable_data_shuffling,
      shuffle_seed=config.data_shuffle_seed,
      num_epoch=config.num_epoch,
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
      grain_worker_count=config.grain_worker_count,
      grain_num_threads=config.grain_num_threads,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
  )

  train_dataloader = audio_sft_preprocessing_pipeline(
      train_ds,
      config,
      grain_worker_count=config.grain_worker_count,
      grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
  )
  return multihost_dataloading.MultiHostDataLoadIterator(
      train_dataloader,
      global_mesh,
      config.generate_padding_batch_train,
  )


def make_grain_audio_eval_iterator(config, global_mesh, process_indices):
  """Create audio eval data iterator using grain pipeline."""
  assert (
      config.global_batch_size_to_load_eval % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."

  eval_ds = get_datasets(
      config.grain_eval_files,
      config.grain_file_type,
      shuffle=False,
      shuffle_seed=config.data_shuffle_seed,
      num_epoch=1,
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
      grain_worker_count=config.grain_worker_count_eval,
      grain_num_threads=config.grain_num_threads_eval,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size_eval,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
  )

  eval_dataloader = audio_sft_preprocessing_pipeline(
      eval_ds,
      config,
      grain_worker_count=config.grain_worker_count_eval,
      grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
  )
  return multihost_dataloading.MultiHostDataLoadIterator(
      eval_dataloader, global_mesh, config.generate_padding_batch_eval
  )
