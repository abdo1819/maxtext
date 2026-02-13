# Audio GRPO for Qwen3-Omni-30B-A3B — Status & Test Plan

## What's Done

### Implementation (All 9 plan items complete)

**Phase 1 — Shared Infrastructure**

1. **Extended `InputData`** (`src/maxtext/inference/offline_engine.py:66-74`)
   Added `audio_features` and `audio_mask` optional fields to the inference data class.

2. **Plumbed audio through prefill path** (`src/MaxText/prefill_packing.py`, `src/maxtext/inference/offline_engine.py`)
   `PrefillProcessor._process()`, `PrefillHelper._jitted_single_prefill()`, `PrefillHelper.process()`, and `_run_continuous_batching()` all pass audio through to `MaxEngine.prefill(audio_values=..., audio_masks=...)`.

3. **ASR reward functions** (`src/MaxText/rl/rewards_asr.py`)
   - `compute_wer()` / `compute_cer()` — edit-distance-based metrics
   - `extract_transcription()` — extracts text from `<answer>...</answer>` tags
   - `asr_wer_reward()` / `asr_cer_reward()` — reward = format_bonus + max(-cap, -WER/CER) + exact_match_bonus
   - `asr_format_reward()` — reward for correct `<reasoning>...</reasoning><answer>...</answer>` format

4. **Audio GRPO data pipeline** (`src/MaxText/experimental/rl/grpo_audio_input_pipeline.py`)
   Grain pipeline: ParseAudioFeatures -> ResampleAudio -> AudioToMelSpectrogram -> BuildAudioGRPOPrompt -> PadAudioGRPOToMaxLength -> Batch -> CollectGroundTruth.
   Output keys: `ar`, `ar_true_length`, `audios`, `ground_truth_text`.

**Phase 2 — Standalone Experimental GRPO**

5. **`compute_log_probs()` + audio** (`src/MaxText/experimental/rl/grpo_utils.py:51`)
   Added `encoder_audios` param, passed to `model.apply()`.

6. **`generate_offline_completions()` + audio** (`src/MaxText/experimental/rl/grpo_utils.py:119`)
   Repeats audio G times per prompt, creates `InputData` with `audio_features`, passes to `OfflineEngine.batch_inference()`.

7. **`grpo_loss_fn()` + pre-computed rewards + audio** (`src/MaxText/experimental/rl/grpo_trainer.py:168`)
   Extracts `encoder_audios` from batch, passes to both policy and reference `compute_log_probs` calls. Uses pre-computed rewards from `data["rewards"]` when available.

8. **`generate_completions()` + `train_loop()` + audio** (`src/MaxText/experimental/rl/grpo_trainer.py`)
   - `generate_completions()`: decodes completion tokens to text, computes WER + format rewards, stores in batch.
   - `setup_train_loop()`: uses `grpo_audio_input_pipeline` when `config.use_audio`.

9. **Config file** (`src/MaxText/configs/grpo_audio_qwen3_omni.yml`)
   Full config with model, audio, GRPO, reward, data, inference, and training settings.

### Bug Fixes Applied

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | CRITICAL | `offline_engine.py:937` | `pad_data()` now carries `audio_features`/`audio_mask` through to padded `InputData` |
| 2 | CRITICAL | `types.py` | Added missing Pydantic fields: `inference_rollouts`, `inference_devices_per_replica`, `inference_replicas`, `num_generations`, `grpo_beta`, `grpo_epsilon`, ASR reward fields |
| 3 | CRITICAL | `types.py:923` | Changed `train_data_columns` type from `list[str]` to `str \| list[str]` (GRPO uses string access) |
| 4 | CRITICAL | `grpo_audio_qwen3_omni.yml` | Added missing keys: `train_data_columns`, `decode_sampling_strategy`, `return_log_prob`, `decode_sampling_temperature`, `async_checkpointing`, `add_bos`, `add_eos` |
| 5 | HIGH | `grpo_trainer.py:383-501` | Fixed `LossAux` access from dict-style `aux["field"]` to attribute-style `aux.field` (8 occurrences) |
| 6 | LOW | `grpo_audio_input_pipeline.py` | Added missing `from MaxText.input_pipeline import input_pipeline_interface` import |

---

## Test Plan — Small Random Qwen Model on v4-8

### Goal

Validate the full audio GRPO pipeline end-to-end without needing the real 30B checkpoint. Use a tiny Qwen3-Omni model with random weights and reduced layers on a single v4-8 TPU (1 host, 4 chips).

### Step 1: Create tiny model config

Create `src/MaxText/configs/models/qwen3-omni-test.yml` with minimal dimensions:

```yaml
model_name: "qwen3-omni-test"
decoder_block: "qwen3_omni"
# Drastically reduced model size
num_decoder_layers: 2          # original: 48
num_query_heads: 4             # original: 32
num_kv_heads: 2                # original: 8
head_dim: 64                   # original: 128
mlp_dim: 512                   # original: ~5120
emb_dim: 256                   # original: ~3584
vocab_size: 152064             # keep original tokenizer vocab
# Audio encoder — keep structure, reduce size
use_multimodal: true
use_audio: true
# MoE — disable for test (use dense)
num_experts: 0
```

Key: keep `vocab_size` and audio token IDs matching the real tokenizer so the data pipeline works unchanged. The audio encoder structure must also match (mel bins, etc.) — only reduce the text decoder.

### Step 2: Create small test ArrayRecord dataset

Create a script `qwen_speech_exp/data/create_test_data.py` that:
- Generates 20-50 short audio samples (synthetic sine waves or noise, 1-3 seconds each at 16kHz)
- Pairs each with a ground truth transcription string
- Writes to ArrayRecord format matching the expected schema (`audio`, `text` fields)
- Outputs to `gs://arabic-asr-dataset/test_grpo_data/` or a local path

### Step 3: Create test GRPO config

Create `src/MaxText/configs/grpo_audio_qwen3_omni_test.yml`:

```yaml
base_config: "base.yml"
model_name: "qwen3-omni-test"

use_multimodal: true
use_audio: true
freeze_audio_encoder_params: true
freeze_vision_encoder_params: true

use_grpo: true
num_generations: 2             # reduced from 4
grpo_beta: 0.04
grpo_epsilon: 0.2

reasoning_start_token: '<reasoning>'
reasoning_end_token: '</reasoning>'
solution_start_token: '<answer>'
solution_end_token: '</answer>'

reward_exact_format_match: 3.0
asr_reward_cap: 2.0
asr_exact_match_bonus: 3.0
asr_format_bonus: 1.0
asr_no_answer_penalty: -2.0

max_prefill_predict_length: 128
max_target_length: 256

dataset_type: grain
grain_file_type: arrayrecord
train_data_columns: 'ar'
grain_train_files: "/path/to/test_grpo_data/"

tokenizer_type: "huggingface"
tokenizer_path: "Qwen/Qwen3-Omni-30B-A3B-Instruct"

decode_sampling_strategy: "weighted"
decode_sampling_temperature: 0.9
return_log_prob: true
inference_rollouts: 1
inference_devices_per_replica: 4
inference_replicas: 1

weight_dtype: 'bfloat16'
attention: 'dot_product'
per_device_batch_size: 2
learning_rate: 1.0e-5
enable_dropout: false
steps: 10
async_checkpointing: false

add_bos: false
add_eos: false
```

### Step 4: Test stages (run on v4-8)

**4a. Config loading** — verify config parses without errors:
```bash
python3 -c "
from MaxText import pyconfig
config = pyconfig.initialize(['configs/grpo_audio_qwen3_omni_test.yml'])
print('use_audio:', config.use_audio)
print('num_generations:', config.num_generations)
print('train_data_columns:', config.train_data_columns)
"
```

**4b. Data pipeline** — verify audio batches are produced:
```bash
python3 -c "
from MaxText import pyconfig
from MaxText.experimental.rl import grpo_audio_input_pipeline
import jax
config = pyconfig.initialize(['configs/grpo_audio_qwen3_omni_test.yml'])
mesh = jax.sharding.Mesh(jax.devices(), ('data',))
it = grpo_audio_input_pipeline.create_audio_data_iterator(config, mesh)
batch = next(it)
print('ar shape:', batch['ar'].shape)
print('audios shape:', batch['audios'].shape)
print('ground_truth_text:', batch['ground_truth_text'][:2])
"
```

**4c. Reward functions** — unit test:
```bash
python3 -c "
from MaxText.rl import rewards_asr
# exact match
r = rewards_asr.compute_wer('hello world', 'hello world')
assert r == 0.0, f'Expected 0.0 got {r}'
# partial match
r = rewards_asr.compute_wer('hello world', 'hello')
assert r > 0.0
# format reward
r = rewards_asr.asr_format_reward(
    prompts=[''], completions=['<reasoning>thinking</reasoning><answer>text</answer>'], tmvp_config=None
)
assert r[0] > 0, f'Expected positive format reward, got {r[0]}'
print('All reward tests passed')
"
```

**4d. Inference with audio** — verify model generates tokens given audio input:
```bash
python3 -c "
# Test that OfflineEngine produces completions with audio input
from maxtext.inference.offline_engine import OfflineEngine, InputData
from MaxText import pyconfig
import numpy as np, jax
config = pyconfig.initialize(['configs/grpo_audio_qwen3_omni_test.yml'])
# ... create engine, build a dummy InputData with audio_features, run inference
"
```

**4e. Full GRPO loop** — run 5-10 training steps:
```bash
python3 -m MaxText.experimental.rl.grpo_trainer \
    configs/grpo_audio_qwen3_omni_test.yml \
    configs/grpo_audio_qwen3_omni_test.yml \
    steps=5
```

### Step 5: What to verify

- [ ] Config loads without Pydantic validation errors
- [ ] Data pipeline produces batches with correct shapes
- [ ] Audio features survive through `pad_data` (not silently dropped)
- [ ] `OfflineEngine` generates non-empty completions when given audio
- [ ] WER/CER rewards are computed and are finite
- [ ] `grpo_loss_fn` computes loss without NaN
- [ ] Loss decreases (or at least changes) over 5-10 steps
- [ ] Audio encoder gradients are zero when `freeze_audio_encoder_params=true`
- [ ] No OOM on v4-8 with the tiny model
