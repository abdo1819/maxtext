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

"""Audio rollout engine for GRPO using MaxText's OfflineEngine.

Replaces vLLM for rollout generation when audio inputs are needed,
since vLLM's MaxTextForCausalLM adapter does not support audio.
"""

import numpy as np

from maxtext.inference.offline_engine import InputData, OfflineEngine
from maxtext.utils import max_logging


class AudioOfflineRollout:
  """Custom rollout engine using MaxText OfflineEngine for audio GRPO.

  Wraps OfflineEngine to generate text completions from audio prompts.
  MaxEngine.prefill() already supports audio_values and audio_masks kwargs;
  this rollout ensures the audio features are plumbed through InputData
  to the prefill call.
  """

  def __init__(self, config, mesh):
    """Initialize the rollout engine.

    Args:
      config: MaxText inference configuration.
      mesh: JAX device mesh for inference.
    """
    self.engine = OfflineEngine(config=config, mesh=mesh)
    self.config = config
    max_logging.log("AudioOfflineRollout initialized with OfflineEngine.")

  def generate(
      self,
      prompt_tokens,
      prompt_lengths,
      audio_features=None,
      audio_masks=None,
  ):
    """Generate text completions from audio prompts.

    Args:
      prompt_tokens: Token sequences [B, L] with audio placeholder tokens.
      prompt_lengths: Actual prompt lengths [B] or [B, 1].
      audio_features: Mel spectrograms [B, mel_bins, frames], or None.
      audio_masks: Audio validity masks [B, frames], or None.

    Returns:
      List of CompletionOutput objects with token_ids and logprobs.
    """
    input_data = []
    batch_size = prompt_tokens.shape[0]
    for i in range(batch_size):
      length = prompt_lengths[i]
      if hasattr(length, "__len__"):
        length = int(length[0])
      else:
        length = int(length)

      audio_feat = audio_features[i] if audio_features is not None else None
      audio_mask = audio_masks[i] if audio_masks is not None else None

      input_data.append(
          InputData(
              id=f"audio_input_{i}",
              tokens=np.array(prompt_tokens[i]),
              true_length=length,
              audio_features=audio_feat,
              audio_mask=audio_mask,
          )
      )

    results = self.engine.batch_inference(input_data)
    return results

  def update_params(self, params):
    """Update the engine's model parameters (for actor weight sync).

    Args:
      params: New model parameters to load into the engine.
    """
    self.engine.update_params(params)
