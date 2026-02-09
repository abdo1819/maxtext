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

"""ASR reward functions for audio GRPO training.

Provides WER/CER-based rewards for speech-to-text chain-of-thought training.
Follows the same signature pattern as reward functions in utils_rl.py:
  (prompts, completions, answer, tmvp_config, **kwargs) -> list[float]
"""

import re


def _edit_distance(ref_tokens, hyp_tokens):
  """Compute Levenshtein edit distance between two token sequences."""
  n = len(ref_tokens)
  m = len(hyp_tokens)
  dp = [[0] * (m + 1) for _ in range(n + 1)]
  for i in range(n + 1):
    dp[i][0] = i
  for j in range(m + 1):
    dp[0][j] = j
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      if ref_tokens[i - 1] == hyp_tokens[j - 1]:
        dp[i][j] = dp[i - 1][j - 1]
      else:
        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
  return dp[n][m]


def compute_wer(hypothesis, reference):
  """Compute Word Error Rate between hypothesis and reference strings.

  Args:
    hypothesis: Predicted transcription string.
    reference: Ground truth transcription string.

  Returns:
    WER as a float in [0, inf). Returns 1.0 if reference is empty and
    hypothesis is non-empty, 0.0 if both are empty.
  """
  ref_words = reference.strip().split()
  hyp_words = hypothesis.strip().split()
  if len(ref_words) == 0:
    return 0.0 if len(hyp_words) == 0 else 1.0
  distance = _edit_distance(ref_words, hyp_words)
  return distance / len(ref_words)


def compute_cer(hypothesis, reference):
  """Compute Character Error Rate between hypothesis and reference strings.

  Args:
    hypothesis: Predicted transcription string.
    reference: Ground truth transcription string.

  Returns:
    CER as a float in [0, inf). Returns 1.0 if reference is empty and
    hypothesis is non-empty, 0.0 if both are empty.
  """
  ref_chars = list(reference.strip())
  hyp_chars = list(hypothesis.strip())
  if len(ref_chars) == 0:
    return 0.0 if len(hyp_chars) == 0 else 1.0
  distance = _edit_distance(ref_chars, hyp_chars)
  return distance / len(ref_chars)


def extract_transcription(completion, tmvp_config):
  """Extract transcription text from between solution tokens.

  Looks for text between solution_start_token and solution_end_token
  (e.g., <answer> and </answer>).

  Args:
    completion: The model's completion string.
    tmvp_config: Config with solution_start_token and solution_end_token.

  Returns:
    The extracted transcription string, or None if format doesn't match.
  """
  pattern = re.compile(
      rf"{re.escape(tmvp_config.solution_start_token)}\s*(.*?)\s*{re.escape(tmvp_config.solution_end_token)}",
      flags=re.DOTALL,
  )
  match = pattern.search(completion)
  if match:
    return match.group(1).strip()
  return None


def asr_format_reward(prompts, completions, tmvp_config, **kwargs):
  """Reward for correct chain-of-thought format.

  Checks for <reasoning>...</reasoning><answer>...</answer> structure.

  Args:
    prompts: List of prompt strings (unused but required by interface).
    completions: List of model completion strings.
    tmvp_config: Config with reasoning/solution tokens and reward values.
    **kwargs: Additional keyword arguments (ignored).

  Returns:
    List of float reward scores.
  """
  scores = []
  format_regex = re.compile(
      (
          r"^[\s]{0,}"
          rf"{re.escape(tmvp_config.reasoning_start_token)}.+?{re.escape(tmvp_config.reasoning_end_token)}.*?"
          rf"{re.escape(tmvp_config.solution_start_token)}.+?{re.escape(tmvp_config.solution_end_token)}"
          r"[\s]{0,}$"
      ),
      flags=re.MULTILINE | re.DOTALL,
  )
  for completion in completions:
    if format_regex.search(completion) is not None:
      scores.append(tmvp_config.reward_exact_format_match)
    else:
      # Partial credit: check individual tags
      score = 0.0
      for tag in [
          tmvp_config.reasoning_start_token,
          tmvp_config.reasoning_end_token,
          tmvp_config.solution_start_token,
          tmvp_config.solution_end_token,
      ]:
        if tag in completion:
          score += 0.5
        else:
          score -= 0.5
      scores.append(score)
  return scores


def asr_wer_reward(prompts, completions, answer, tmvp_config, **kwargs):
  """WER-based reward for ASR chain-of-thought.

  Extracts transcription from <answer> tags and computes WER against
  the ground truth. Reward is structured as:
    - format_bonus for correct format
    - -WER (capped at -2.0) for transcription quality
    - exact_match_bonus for perfect transcription

  Args:
    prompts: List of prompt strings (unused but required by interface).
    completions: List of model completion strings.
    answer: List of ground truth transcription strings.
    tmvp_config: Config with reward parameters.
    **kwargs: Additional keyword arguments (ignored).

  Returns:
    List of float reward scores.
  """
  scores = []
  reward_cap = getattr(tmvp_config, "asr_reward_cap", 2.0)
  exact_match_bonus = getattr(tmvp_config, "asr_exact_match_bonus", 3.0)
  format_bonus = getattr(tmvp_config, "asr_format_bonus", 1.0)
  no_answer_penalty = getattr(tmvp_config, "asr_no_answer_penalty", -2.0)

  for completion, gt_text in zip(completions, answer):
    reward = 0.0

    # Check format
    transcription = extract_transcription(completion, tmvp_config)

    if transcription is None:
      # No valid answer tags found
      reward += no_answer_penalty
    else:
      reward += format_bonus
      wer = compute_wer(transcription, gt_text)
      reward += max(-reward_cap, -wer)
      if wer == 0.0:
        reward += exact_match_bonus

    scores.append(reward)
  return scores


def asr_cer_reward(prompts, completions, answer, tmvp_config, **kwargs):
  """CER-based reward for ASR chain-of-thought.

  Same structure as asr_wer_reward but uses Character Error Rate.

  Args:
    prompts: List of prompt strings (unused but required by interface).
    completions: List of model completion strings.
    answer: List of ground truth transcription strings.
    tmvp_config: Config with reward parameters.
    **kwargs: Additional keyword arguments (ignored).

  Returns:
    List of float reward scores.
  """
  scores = []
  reward_cap = getattr(tmvp_config, "asr_reward_cap", 2.0)
  exact_match_bonus = getattr(tmvp_config, "asr_exact_match_bonus", 3.0)
  format_bonus = getattr(tmvp_config, "asr_format_bonus", 1.0)
  no_answer_penalty = getattr(tmvp_config, "asr_no_answer_penalty", -2.0)

  for completion, gt_text in zip(completions, answer):
    reward = 0.0
    transcription = extract_transcription(completion, tmvp_config)

    if transcription is None:
      reward += no_answer_penalty
    else:
      reward += format_bonus
      cer = compute_cer(transcription, gt_text)
      reward += max(-reward_cap, -cer)
      if cer == 0.0:
        reward += exact_match_bonus

    scores.append(reward)
  return scores
