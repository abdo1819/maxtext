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

"""Audio GRPO training using TunixMaxTextAudioAdapter + AudioOfflineRollout.

This module provides a custom audio GRPO training loop that:
1. Uses TunixMaxTextAudioAdapter for the training forward pass (NNX model)
2. Uses AudioOfflineRollout (wrapping OfflineEngine) for generation
3. Computes ASR rewards (WER/CER) on the generated completions
4. Implements the GRPO loss and gradient update

Since Tunix's GrpoLearner and RLCluster are built around vLLM rollouts
which cannot handle audio, this module implements a custom training loop
that handles audio rollout + Tunix-compatible model forward passes.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAudioAdapter
from MaxText.rl.audio_rollout import AudioOfflineRollout
from MaxText.rl import rewards_asr
from MaxText.experimental.rl import grpo_audio_input_pipeline
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils


def create_audio_adapter(config, mesh, devices=None):
  """Create a TunixMaxTextAudioAdapter for the given config.

  Args:
    config: MaxText configuration.
    mesh: JAX device mesh.
    devices: Optional devices for model creation.

  Returns:
    TunixMaxTextAudioAdapter wrapping the NNX Transformer.
  """
  model, mesh = model_creation_utils.create_nnx_model(
      config, mesh=mesh, devices=devices
  )
  adapter = TunixMaxTextAudioAdapter(
      base_model=model,
      use_standalone_mappings=True,
      use_no_op_mappings=False,
  )
  return adapter, mesh


def compute_log_probs_nnx(adapter, input_tokens, positions, segmentation, encoder_audios=None):
  """Compute per-token log-probs using the NNX adapter.

  Args:
    adapter: TunixMaxTextAudioAdapter instance.
    input_tokens: [B, L] token IDs.
    positions: [B, L] position indices.
    segmentation: [B, L] segment IDs.
    encoder_audios: Optional [B, mel_bins, frames] audio features.

  Returns:
    Per-token log-probs [B, L-1].
  """
  adapter.set_audio(encoder_audios)
  logits, _ = adapter(
      input_tokens=input_tokens,
      positions=positions,
      cache=None,
      attention_mask=None,
      decoder_segment_ids=segmentation,
  )
  # Compute log-probs from logits
  targets = input_tokens[:, 1:]
  log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
  token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
  return token_log_probs


def audio_grpo_train_step(
    actor_adapter,
    reference_adapter,
    rollout_engine,
    tokenizer_model,
    batch,
    config,
    rng,
):
  """Perform a single audio GRPO training step.

  Args:
    actor_adapter: TunixMaxTextAudioAdapter for the policy model.
    reference_adapter: TunixMaxTextAudioAdapter for the reference model.
    rollout_engine: AudioOfflineRollout for generation.
    tokenizer_model: Tokenizer for decoding completions.
    batch: Data batch with keys: ar, ar_true_length, audios, ground_truth_text.
    config: Training configuration.
    rng: JAX PRNG key.

  Returns:
    Tuple of (loss, metrics_dict).
  """
  prompt_tokens = batch["ar"]
  prompt_lengths = batch["ar_true_length"]
  audio_features = batch["audios"]
  ground_truth = batch["ground_truth_text"]

  # 1. Generate completions (G per prompt)
  num_gen = config.num_generations
  batch_size = prompt_tokens.shape[0]

  # Repeat prompts and audio for G generations
  expanded_prompts = np.repeat(prompt_tokens, num_gen, axis=0)
  expanded_lengths = np.repeat(prompt_lengths, num_gen, axis=0)
  expanded_audio = np.repeat(audio_features, num_gen, axis=0)
  expanded_gt = [t for t in ground_truth for _ in range(num_gen)]

  results = rollout_engine.generate(
      prompt_tokens=expanded_prompts,
      prompt_lengths=expanded_lengths,
      audio_features=expanded_audio,
  )

  # 2. Compute rewards
  completion_texts = []
  for i, r in enumerate(results):
    prompt_len = int(expanded_lengths[i][0]) if hasattr(expanded_lengths[i], "__len__") else int(expanded_lengths[i])
    comp_tokens = r.token_ids[prompt_len:]
    comp_tokens = comp_tokens[comp_tokens != 0]
    text = tokenizer_model.decode(comp_tokens, skip_special_tokens=True)
    completion_texts.append(text)

  wer_rewards = rewards_asr.asr_wer_reward(
      prompts=[""] * len(completion_texts),
      completions=completion_texts,
      answer=expanded_gt,
      tmvp_config=config,
  )
  format_rewards = rewards_asr.asr_format_reward(
      prompts=[""] * len(completion_texts),
      completions=completion_texts,
      tmvp_config=config,
  )
  rewards = jnp.array(
      [w + f for w, f in zip(wer_rewards, format_rewards)], dtype=jnp.float32
  )

  # 3. Build full sequences (prompt + completion) for log-prob computation
  max_target_length = config.max_target_length
  full_sequences = []
  completion_masks = []
  for i, r in enumerate(results):
    seq = r.token_ids[:max_target_length]
    if len(seq) < max_target_length:
      seq = np.pad(seq, (0, max_target_length - len(seq)), constant_values=0)
    full_sequences.append(seq)
    prompt_len = int(expanded_lengths[i][0]) if hasattr(expanded_lengths[i], "__len__") else int(expanded_lengths[i])
    mask = np.zeros(max_target_length, dtype=np.int32)
    mask[prompt_len : len(r.token_ids)] = 1
    completion_masks.append(mask)

  full_sequences = jnp.array(full_sequences)
  completion_masks = jnp.array(completion_masks)
  positions = jnp.broadcast_to(
      jnp.arange(max_target_length), full_sequences.shape
  )
  segmentation = (full_sequences != 0).astype(jnp.int32)

  # 4. Compute log-probs for policy and reference
  policy_logps = compute_log_probs_nnx(
      actor_adapter, full_sequences, positions, segmentation,
      encoder_audios=jnp.array(expanded_audio),
  )
  ref_logps = compute_log_probs_nnx(
      reference_adapter, full_sequences, positions, segmentation,
      encoder_audios=jnp.array(expanded_audio),
  )

  # 5. GRPO loss computation
  valid_mask = completion_masks[:, 1:]  # shifted by 1 for log-prob alignment
  valid_mask = valid_mask[:, : policy_logps.shape[1]]

  # Group rewards and compute advantages
  rewards_grouped = rewards.reshape(-1, num_gen)
  group_mean = jnp.mean(rewards_grouped, axis=1)
  group_std = jnp.std(rewards_grouped, axis=1)
  advantages = (rewards - jnp.repeat(group_mean, num_gen)) / (jnp.repeat(group_std, num_gen) + 1e-8)
  advantages_exp = advantages[:, None]

  # Clipped surrogate loss
  old_logps = jax.lax.stop_gradient(policy_logps)
  ratio = jnp.exp(policy_logps - old_logps)
  clipped_ratio = jnp.clip(ratio, 1 - config.grpo_epsilon, 1 + config.grpo_epsilon)
  loss_tokens = -jnp.minimum(ratio * advantages_exp, clipped_ratio * advantages_exp)

  # KL penalty
  if config.grpo_beta != 0.0:
    kl = jnp.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1
    kl = kl * valid_mask
    loss_tokens = loss_tokens + config.grpo_beta * kl

  # Average over valid completion tokens
  loss_per_example = jnp.sum(loss_tokens * valid_mask, axis=1) / jnp.clip(jnp.sum(valid_mask, axis=1), min=1)
  loss = jnp.mean(loss_per_example)

  metrics = {
      "loss": float(loss),
      "avg_reward": float(jnp.mean(rewards)),
      "avg_reward_std": float(jnp.mean(group_std)),
      "avg_advantage": float(jnp.mean(advantages)),
      "avg_wer_reward": float(np.mean(wer_rewards)),
  }
  if config.grpo_beta != 0.0:
    metrics["avg_kl"] = float(jnp.mean(kl * valid_mask))

  return loss, metrics
