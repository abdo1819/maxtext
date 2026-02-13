# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Project Focus

Fine-tuning **Qwen3-Omni-30B-A3B** (speech/multimodal) on a **TPU v4-32** (4 hosts x 4 chips = 16 chips) in `us-central2-b`. The experiment code lives in `qwen_speech_exp/`. See `qwen_speech_exp/SETUP_GUIDE.md` for the full setup and deployment workflow.

### Infrastructure

- **Jumpbox** (`maxtext-jumpbox`, `n2-standard-2`): orchestrates TPU workers via `multihost_runner.py`
- **TPU workers** (`qr-v4-32`): 4 hosts, code at `~/maxtext` (git clone), venv at `~/venv-maxtext` (Python 3.12)
- **GCS bucket**: `gs://arabic-asr-dataset` — holds dataset (`/grain_data_arrayrecord/`) and checkpoints (`/checkpoints/`)
- **GCP project**: `arabic-asr-level2thinkg`

### Key Experiment Files

All under `qwen_speech_exp/`, organized into subdirectories:
- `env_vars.sh` — shared paths and model config (sourced by all scripts)
- `setup/setup_tpu_worker.sh` — installs Python 3.12, PyTorch XLA, JAX on workers
- `training/train.sh` — fine-tuning launch (`python3 -m MaxText.train`)
- `inference/inference_multihost.sh` — multi-host inference (`python3 -m maxtext.decode`)
- `inference/run_batch_inference.sh` / `inference/batch_inference.py` — batch inference
- `checkpoint/convert_hf_to_maxtext.sh` — HuggingFace-to-MaxText checkpoint conversion
- `data/convert_data.sh` / `data/convert_tfrecord_to_arrayrecord.py` — data conversion
- `distillation/generate_cot.py` / `distillation/build_sft_dataset.py` — distillation pipeline

### Dataset (ArrayRecord, converted from TFRecord)

Location: `gs://arabic-asr-dataset/grain_data_arrayrecord/{train,validation,test}/`

**Splits and record counts:**

| Split      | Files | Records  | Size      |
|------------|-------|----------|-----------|
| train      | 55    | 520,296  | 169.71 GiB |
| validation | 12    | 71,673   | 23.09 GiB  |
| test       | 12    | 71,608   | 23.32 GiB  |
| **Total**  | **79**| **663,577** | **216.12 GiB** |

**Datasets included (7 sources):**
- `MohamedRashad/SADA22` — 19 train / 3 val / 3 test shards
- `MohamedRashad/mgb2_arabic` — 29 train / 4 val / 4 test shards
- `MohamedRashad/arabic_english_code_switching` — 1 train / 1 val / 1 test shard
- `fixie_ai/common_voice_17_0` — 3 train / 2 val / 2 test shards
- `google/fleurs` — 1 train / 1 val / 1 test shard
- `halabi2016/arabic_speech_corpus` — 1 train / 0 val / 1 test shard
- `UBC_NLP/Casablanca` — 1 train / 1 val / 0 test shard

**Record schema** (each record is a serialized `tf.train.Example`):

| Feature        | Type        | Description                                      |
|----------------|-------------|--------------------------------------------------|
| `audio`        | float_list  | Raw waveform samples (variable length)           |
| `audio_len`    | int64_list  | Number of audio samples (scalar)                 |
| `sample_rate`  | int64_list  | Always 16000 Hz (scalar)                         |
| `duration`     | float_list  | Duration in seconds (scalar)                     |
| `text`         | bytes_list  | Transcript text, UTF-8 encoded                   |
| `dataset_name` | bytes_list  | Source dataset identifier                        |
| `sample_id`    | bytes_list  | Unique sample identifier                         |

### Deploying Changes to TPU Workers

Workers clone the repo via git. After pushing changes:
```bash
python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b --INTERNAL_IP=true \
    --COMMAND="cd ~/maxtext && git pull" \
    --USE_EXISTING_FOLDER=true --RUN_NAME=<RUN_NAME>
```

All `--COMMAND` values must prefix with `cd ~/maxtext && source ~/venv-maxtext/bin/activate &&`.

## MaxText Architecture

MaxText is Google's high-performance JAX/Flax LLM library for TPU/GPU training and inference.

### Dual Package Layout

Two parallel packages under `src/` — both are actively used and cross-import:
- `src/MaxText/` — core: model layers, training loop, config system, optimizer, sharding, JetStream engine
- `src/maxtext/` — utilities: decoding, checkpointing, multimodal, post-training (SFT/RL/DPO), kernels

### Configuration System (3-layer)

1. **YAML** (OmegaConf): `configs/base.yml` -> `configs/models/<model>.yml` -> CLI overrides -> env vars (`M_` prefix)
2. **Pydantic**: `configs/types.py` (`MaxTextConfig`) validates the merged dict
3. **Runtime**: `pyconfig.py` `initialize(argv)` loads YAML, merges, validates, returns read-only `HyperParameters`

Important: `pyconfig.py:_prepare_for_pydantic()` only includes keys explicitly in YAML/CLI — not Pydantic defaults. Code accessing `pydantic_kwargs` before `MaxTextConfig` instantiation must use `.get()` or the key must be in the config file.

### Entry Points

- Training: `python3 -m MaxText.train <config.yml> [key=value ...]`
- Inference: `python3 -m maxtext.decode <config.yml> [key=value ...]`
- Checkpoint conversion: `python3 src/MaxText/utils/ckpt_conversion/to_maxtext.py <config.yml> [key=value ...]`
- SFT: `python3 -m MaxText.sft_trainer <config.yml>`
- Multi-host orchestration: `python3 tools/orchestration/multihost_runner.py`

### Model Dispatch

Models are selected via `decoder_block` config key (maps to `DecoderBlockType` enum). Each model has custom layers in `src/MaxText/layers/<model>.py` that plug into the shared `Decoder -> DecoderLayer -> Transformer` pipeline.

### Sharding

Two modes via `shard_mode` config: `AUTO` (sharding constraints/hints) or `EXPLICIT` (direct reshard). Logical axis rules map abstract names (`embed`, `heads`, `mlp`) to physical mesh dimensions (`data`, `fsdp`, `tensor`).

## Common Commands

### Lint and Format

```bash
# Format (Google's pyink, 2-space indent, 122-char line)
pyink src/MaxText --pyink-indentation=2 --line-length=122

# Lint
pylint --disable=R0401,R0917,W0201,W0613 src/MaxText

# Run pre-commit on staged files
pre-commit run --all-files
```

### Tests

```bash
# All unit tests
python3 -m pytest tests/

# Single test file
python3 -m pytest tests/unit/<test_file>.py

# By marker
python3 -m pytest -m cpu_only tests/
python3 -m pytest -m tpu_only tests/
```

Test files use `*_test.py` / `*_tests.py` naming. Markers: `tpu_only`, `gpu_only`, `cpu_only`, `decoupled`, `integration_test`.

### Build

```bash
pip install -e .                    # editable install
pip install -e ".[tpu]"             # with TPU deps
DECOUPLE_GCLOUD=TRUE pip install .  # without GCP dependencies
```

Requires Python >= 3.12. Build backend is Hatchling.
