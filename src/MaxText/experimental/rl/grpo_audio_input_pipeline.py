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

"""Audio input pipeline for GRPO training with Qwen3-Omni.

Reads ArrayRecord files containing audio + text pairs, produces batches with:
  - Prompt-only token sequences ([AUDIO_START][AUDIO_PAD]*N[AUDIO_EOS])
  - Mel spectrogram features for the audio encoder
  - Ground truth text for reward computation

Reuses audio processing stages from _audio_data_processing.py.
"""

import functools

import jax
import numpy as np
import grain.python as grain
import tensorflow as tf

from MaxText import tokenizer
from MaxText.input_pipeline._audio_data_processing import (
    ParseAudioFeatures,
    ResampleAudio,
    AudioToMelSpectrogram,
)
from MaxText.input_pipeline import input_pipeline_interface
from MaxText.input_pipeline._grain_data_processing import get_datasets
from maxtext.multimodal.processor_qwen3_omni import (
    _get_feat_extract_output_lengths,
    QWEN3_OMNI_AUDIO_START_TOKEN,
    QWEN3_OMNI_AUDIO_END_TOKEN,
    QWEN3_OMNI_AUDIO_TOKEN,
    SAMPLE_RATE,
)
from maxtext.utils import max_logging


class BuildAudioGRPOPrompt(grain.MapTransform):
  """Build prompt-only token sequence for GRPO: <audio_start> <audio_pad>*N <audio_eos>.

  Unlike SFT which includes the target text in the sequence, GRPO only builds
  the audio prompt — the model generates the text completion during rollout.
  The ground truth text is kept separately for reward computation.
  """

  def __init__(self, max_prefill_length, n_window):
    self.max_prefill_length = max_prefill_length
    self.n_window = n_window

  def map(self, example):
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
    # Each chunk of 100 mel frames produces 13 tokens via 3 stride-2 convolutions.
    padded_frames = audio_features.shape[1]
    num_audio_tokens = int(_get_feat_extract_output_lengths(np.array(padded_frames)).item())

    # Build prompt: <audio_start> <audio_pad>*N <audio_eos>
    audio_start = np.array([QWEN3_OMNI_AUDIO_START_TOKEN], dtype=np.int32)
    audio_pads = np.full(num_audio_tokens, QWEN3_OMNI_AUDIO_TOKEN, dtype=np.int32)
    audio_end = np.array([QWEN3_OMNI_AUDIO_END_TOKEN], dtype=np.int32)

    prompt_tokens = np.concatenate([audio_start, audio_pads, audio_end])

    # Truncate if needed (shouldn't happen for reasonable audio lengths)
    if len(prompt_tokens) > self.max_prefill_length:
      prompt_tokens = prompt_tokens[: self.max_prefill_length]

    true_length = len(prompt_tokens)

    example["prompt_tokens"] = prompt_tokens
    example["true_length"] = true_length
    example["num_audio_tokens"] = num_audio_tokens
    return example


class PadAudioGRPOToMaxLength(grain.MapTransform):
  """Pad prompt tokens and audio features to fixed sizes for batching."""

  def __init__(self, max_prefill_length, pad_id, num_mel_bins, max_audio_frames):
    self.max_prefill_length = max_prefill_length
    self.pad_id = pad_id
    self.num_mel_bins = num_mel_bins
    self.max_audio_frames = max_audio_frames

  def map(self, example):
    prompt_tokens = example["prompt_tokens"]
    true_length = example["true_length"]

    # Pad prompt tokens
    pad_len = self.max_prefill_length - len(prompt_tokens)
    if pad_len > 0:
      prompt_tokens = np.pad(prompt_tokens, (0, pad_len), constant_values=self.pad_id)
    else:
      prompt_tokens = prompt_tokens[: self.max_prefill_length]

    # Pad audio features to fixed shape
    audio_features = example["audio_features"]  # (num_mel_bins, frames)
    cur_frames = audio_features.shape[1]
    if cur_frames < self.max_audio_frames:
      pad_width = ((0, 0), (0, self.max_audio_frames - cur_frames))
      audio_features = np.pad(audio_features, pad_width, constant_values=0.0)
    else:
      audio_features = audio_features[:, : self.max_audio_frames]

    return {
        "ar": prompt_tokens.astype(np.int32),
        "ar_true_length": np.array([true_length], dtype=np.int32),
        "audios": audio_features.astype(np.float32),
        "ground_truth_text": example["text"],
    }


class CollectGroundTruth(grain.MapTransform):
  """After batching, extract ground_truth_text as a list of strings."""

  def map(self, example):
    # ground_truth_text after batching is a numpy array of byte strings
    if "ground_truth_text" in example:
      gt = example["ground_truth_text"]
      if hasattr(gt, "tolist"):
        gt = gt.tolist()
      if isinstance(gt, list) and len(gt) > 0 and isinstance(gt[0], bytes):
        gt = [t.decode("utf-8") for t in gt]
      example["ground_truth_text"] = gt
    return example


def audio_grpo_preprocessing_pipeline(dataset, config):
  """Grain pipeline for audio GRPO training with Qwen3-Omni.

  Args:
    dataset: Grain dataset from get_datasets()
    config: Model/training configuration

  Returns:
    Grain dataset producing batches with keys: ar, ar_true_length, audios, ground_truth_text
  """
  # Parse audio features from serialized examples
  dataset = dataset.map(ParseAudioFeatures())

  # Resample to target rate
  dataset = dataset.map(ResampleAudio(target_sample_rate=SAMPLE_RATE))

  # Convert to mel spectrogram
  max_audio_samples = config.max_sample_len_for_audio * SAMPLE_RATE
  dataset = dataset.map(AudioToMelSpectrogram(max_audio_length=max_audio_samples))

  # Build prompt-only token sequence (no target text in tokens)
  max_prefill_length = config.max_prefill_predict_length
  dataset = dataset.map(
      BuildAudioGRPOPrompt(
          max_prefill_length=max_prefill_length,
          n_window=config.n_window_for_audio,
      )
  )

  # Build tokenizer to get pad_id
  tokenizer_model = tokenizer.build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      config.add_bos,
      config.add_eos,
      config.hf_access_token,
      config.dataset_type,
  )
  pad_id = tokenizer_model.pad_id if tokenizer_model.pad_id is not None else 0

  # Compute max audio frames for padding
  chunk_size = config.n_window_for_audio * 2
  max_audio_frames = chunk_size * (config.max_sample_len_for_audio // config.n_window_for_audio + 1)

  # Pad to fixed sizes
  dataset = dataset.map(
      PadAudioGRPOToMaxLength(
          max_prefill_length=max_prefill_length,
          pad_id=pad_id,
          num_mel_bins=config.num_mel_bins_for_audio,
          max_audio_frames=max_audio_frames,
      )
  )

  # Batch
  batch_size = config.global_batch_size_to_load // jax.process_count()
  dataset = dataset.batch(batch_size)

  # Collect ground truth strings after batching
  dataset = dataset.map(CollectGroundTruth())

  return dataset


def create_audio_data_iterator(config, mesh):
  """Create audio data iterator for GRPO training.

  Args:
    config: Training configuration with grain_train_files, audio params, etc.
    mesh: JAX device mesh.

  Returns:
    An iterator yielding batches of audio GRPO data.
  """
  process_indices = input_pipeline_interface.get_process_loading_real_data(
      config.data_sharding,
      config.global_batch_size_to_load,
      config.global_batch_size_to_train_on,
      config.max_target_length,
      mesh,
  )

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

  train_dataloader = audio_grpo_preprocessing_pipeline(train_ds, config)

  multiprocessing_options = grain.MultiprocessingOptions(
      num_workers=config.grain_worker_count,
      per_worker_buffer_size=config.grain_per_worker_buffer_size,
  )
  train_dataloader = train_dataloader.mp_prefetch(multiprocessing_options)

  max_logging.log("Audio GRPO data pipeline created successfully.")
  return iter(train_dataloader)
