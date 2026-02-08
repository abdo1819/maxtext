# Qwen3-Omni Speech Fine-tuning Setup Guide

## Architecture

```
┌──────────────┐       SSH (internal IP)       ┌─────────────────────┐
│   Jumpbox    │──────────────────────────────▶│   TPU VM qr-v4-32   │
│ n2-standard-2│                               │  4 hosts × 4 chips  │
│ us-central2-b│       gcsfuse                 │  = 16 TPU v4 chips  │
│              │◀─────────────────────────────▶│                     │
└──────┬───────┘                               └─────────────────────┘
       │                                                │
       │              ┌──────────────────┐              │
       └─────────────▶│   GCS Bucket     │◀─────────────┘
                      │ arabic-asr-dataset│
                      │  /grain_data/     │
                      │  /checkpoints/    │
                      └──────────────────┘
```

---

## Step 1: Create the Jumpbox VM

```bash
gcloud compute instances create maxtext-jumpbox \
    --project=arabic-asr-level2thinkg \
    --zone=us-central2-b \
    --machine-type=n2-standard-2 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=250GB \
    --scopes=cloud-platform
```

SSH into it:

```bash
gcloud compute ssh maxtext-jumpbox \
    --project=arabic-asr-level2thinkg \
    --zone=us-central2-b
```

Ubuntu 22.04 on GCE auto-expands the root partition to the full disk size (243GB usable).
Verify with `df -h /` after SSH.

---

## Step 2: Set Up the Jumpbox

First, get the code onto the jumpbox. From the jumpbox:

```bash
# Clone the repo
git clone https://github.com/abdo1819/maxtext.git ~/maxtext
```

Or copy from your local machine instead (run from local):

```bash
gcloud compute scp --recurse /home/abdo1819/work/master/maxtext \
    maxtext-jumpbox:~/ \
    --project=arabic-asr-level2thinkg --zone=us-central2-b
```

Then run the automated setup script:

```bash
cd ~/maxtext
bash qwen_speech_exp/setup_jumpbox.sh
```

This script installs system packages (git, python3-venv), adds Google's apt
repo and installs gcsfuse, creates a Python venv with tensorflow and
array-record, sets up gcloud SSH keys for TPU access, and mounts
`gs://arabic-asr-dataset` at `/tmp/gcsfuse`.

---

## Step 3: Convert TFRecord to ArrayRecord (on Jumpbox)

Run the conversion directly on the jumpbox. This reads from GCS via gcsfuse,
converts locally, then uploads to GCS via gsutil:

```bash
cd ~/maxtext
source ~/venv/bin/activate

bash qwen_speech_exp/convert_data.sh
```

The script uploads each file incrementally and tracks progress in
`/tmp/arrayrecord_output/upload_tracker.json`. If interrupted, re-run
and it will resume from where it left off.

Verify the output:

```bash
gsutil ls gs://arabic-asr-dataset/grain_data_arrayrecord/train/ | head -5
```

---

## Step 4: Set Up TPU Workers (from Jumpbox)

This copies the code to all 4 TPU workers, installs Python 3.12, PyTorch XLA,
and JAX + MaxText dependencies.

**Important:** Use `$HOME/maxtext` (not `~/maxtext`) for `--SCRIPT_DIR` since
Python does not expand `~`.

```bash
cd ~/maxtext

python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="bash qwen_speech_exp/setup_tpu_worker.sh" \
    --SCRIPT_DIR=$HOME/maxtext
```

The `setup_tpu_worker.sh` script installs Python 3.12 from deadsnakes PPA,
creates a venv at `~/venv-maxtext`, installs PyTorch + XLA for TPU, then runs
the standard MaxText `setup.sh`.

**Save the `RUN_NAME`** printed in the output (e.g. `2026-02-08-19-30-00`).
All subsequent commands use `--USE_EXISTING_FOLDER=true --RUN_NAME=<RUN_NAME>`
to avoid re-copying the code.

---

## Step 5: Mount GCS on TPU Workers

Each TPU worker needs gcsfuse to access the dataset:

```bash
python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="bash tools/setup/setup_gcsfuse.sh DATASET_GCS_BUCKET=arabic-asr-dataset MOUNT_PATH=/tmp/gcsfuse" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=<RUN_NAME>
```

---

## Step 6: Verify TPU Setup

```bash
python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="python3 -c 'import jax; print(f\"Host {jax.process_index()}: {jax.device_count()} devices\")'" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=<RUN_NAME>
```

Expected: each worker reports 4 TPU v4 devices.

---

## Step 7: Run Training (from Jumpbox)

```bash
cd ~/maxtext

python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="cd qwen_speech_exp && bash train.sh" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=<RUN_NAME>
```

Monitor logs on the jumpbox at `/tmp/<RUN_NAME>/output_slice_*.txt`.

---

## Step 8: Run Inference (from Jumpbox)

### Text mode

```bash
python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="cd qwen_speech_exp && bash inference_multihost.sh --mode text --prompt 'What is machine learning?'" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=<RUN_NAME>
```

### Audio mode

```bash
python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="cd qwen_speech_exp && bash inference_multihost.sh --mode audio --prompt 'Transcribe this audio' --audio /tmp/gcsfuse/test_audio.wav" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=<RUN_NAME>
```

---

## Step 9: Cleanup

```bash
# Delete the jumpbox when done
gcloud compute instances delete maxtext-jumpbox \
    --project=arabic-asr-level2thinkg \
    --zone=us-central2-b

# Delete TPU (if using queued resources)
gcloud alpha compute tpus queued-resources delete qr-v4-32 \
    --project=arabic-asr-level2thinkg \
    --zone=us-central2-b \
    --force --async
```

---

## Quick Reference

All commands run from the jumpbox inside `~/maxtext`.

| Task | Command |
|------|---------|
| Setup TPU workers | `python3 tools/orchestration/multihost_runner.py --TPU_PREFIX=qr-v4-32 --PROJECT=arabic-asr-level2thinkg --ZONE=us-central2-b --INTERNAL_IP=true --COMMAND="bash tools/setup/setup.sh MODE=stable" --SCRIPT_DIR=$HOME/maxtext` |
| Run cmd on all workers | `python3 tools/orchestration/multihost_runner.py --TPU_PREFIX=qr-v4-32 --PROJECT=arabic-asr-level2thinkg --ZONE=us-central2-b --INTERNAL_IP=true --COMMAND="CMD" --USE_EXISTING_FOLDER=true --RUN_NAME=<RUN_NAME>` |
| SSH into worker N | `gcloud compute tpus tpu-vm ssh qr-v4-32 --worker=N --project=arabic-asr-level2thinkg --zone=us-central2-b --internal-ip` |
| Convert data | `bash qwen_speech_exp/convert_data.sh` |
| Train | See Step 7 |
| Inference | See Step 8 |
| Check TPU devices | See Step 6 |

---

## File Overview

```
qwen_speech_exp/
├── SETUP_GUIDE.md                      # This file
├── env_vars.sh                         # Environment variables (paths, model name)
├── setup_jumpbox.sh                    # Jumpbox bootstrap script
├── convert.sh                          # HF -> MaxText checkpoint conversion
├── convert_data.sh                     # TFRecord -> ArrayRecord data conversion
├── convert_tfrecord_to_arrayrecord.py  # Python converter script
├── setup_tpu_worker.sh                 # TPU worker setup (Python 3.12 + deps)
├── train.sh                            # Training launch script (v4-32, 16 chips)
├── inference.sh                        # Single-host inference (4 chips)
└── inference_multihost.sh              # Multi-host inference (16 chips)
```
