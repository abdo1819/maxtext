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

This script installs system packages (git, python3-venv, gcsfuse), creates a
Python venv with tensorflow and array-record, sets up gcloud SSH keys for TPU
access, and mounts `gs://arabic-asr-dataset` at `/tmp/gcsfuse`.

---

## Step 3: Mount GCS Data on Jumpbox

```bash
# Create mount point
sudo mkdir -p /tmp/gcsfuse

# Mount the bucket
gcsfuse --implicit-dirs arabic-asr-dataset /tmp/gcsfuse

# Verify data is accessible
ls /tmp/gcsfuse/grain_data/train/ | head -5
```

---

## Step 4: Convert TFRecord to ArrayRecord (on Jumpbox)

Run the conversion directly on the jumpbox. This reads from GCS via gcsfuse
and writes back to GCS via gcsfuse:

```bash
cd ~/maxtext
source ~/venv/bin/activate

# Convert each split
for SPLIT in train validation test; do
    echo "=== Converting ${SPLIT} ==="
    python3 qwen_speech_exp/convert_tfrecord_to_arrayrecord.py \
        --input_dir /tmp/gcsfuse/grain_data/${SPLIT} \
        --output_dir /tmp/gcsfuse/grain_data_arrayrecord/${SPLIT} \
        --file_pattern "*.tfrecord"
done
```

Verify the output:

```bash
ls /tmp/gcsfuse/grain_data_arrayrecord/train/ | head -5
# or via gsutil:
gsutil ls gs://arabic-asr-dataset/grain_data_arrayrecord/train/ | head -5
```

---

## Step 5: Set Up TPU Workers (from Jumpbox)

Since the jumpbox is in the same project/zone, use `--internal-ip` for faster
SSH. These commands are run **from the jumpbox**.

### Option A: Using multihost_runner.py

```bash
cd ~/maxtext

# Install deps on all TPU workers (copies the repo + runs setup.sh)
python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="bash tools/setup/setup.sh MODE=stable" \
    --SCRIPT_DIR=~/maxtext
```

Save the `RUN_NAME` printed in the output (e.g. `2026-02-08-19-30-00`).

### Option B: Using tpu_ssh.sh directly

```bash
cd ~/maxtext

# Run setup on all 4 workers in parallel
bash qwen_speech_exp/tpu_ssh.sh --all \
    --command "cd ~/maxtext && bash tools/setup/setup.sh MODE=stable"
```

Note: For Option B you need to first copy the code to each worker. See
Step 5b below.

### Step 5b: Copy Code to TPU Workers (for tpu_ssh.sh approach)

```bash
# Tar the repo
cd ~ && tar czf maxtext.tar.gz maxtext/

# Copy to all workers
for w in 0 1 2 3; do
    gcloud compute tpus tpu-vm scp maxtext.tar.gz qr-v4-32:~/ \
        --worker=$w \
        --project=arabic-asr-level2thinkg \
        --zone=us-central2-b \
        --internal-ip &
done
wait

# Untar on all workers
bash qwen_speech_exp/tpu_ssh.sh --all \
    --command "cd ~ && tar xzf maxtext.tar.gz"
```

---

## Step 6: Mount GCS on TPU Workers

Each TPU worker needs gcsfuse to access the dataset:

```bash
bash qwen_speech_exp/tpu_ssh.sh --all \
    --command "cd ~/maxtext && bash tools/setup/setup_gcsfuse.sh DATASET_GCS_BUCKET=arabic-asr-dataset MOUNT_PATH=/tmp/gcsfuse"
```

---

## Step 7: Verify TPU Setup

```bash
# Check JAX sees the TPU devices on each worker
bash qwen_speech_exp/tpu_ssh.sh --all \
    --command "cd ~/maxtext && source ~/venv/bin/activate 2>/dev/null; python3 -c 'import jax; print(f\"Host {jax.process_index()}: {jax.device_count()} devices\")'"
```

Expected: each worker reports 4 TPU v4 devices.

---

## Step 8: Run Training (from Jumpbox)

### Option A: multihost_runner.py

```bash
cd ~/maxtext

python3 tools/orchestration/multihost_runner.py \
    --TPU_PREFIX=qr-v4-32 \
    --PROJECT=arabic-asr-level2thinkg \
    --ZONE=us-central2-b \
    --INTERNAL_IP=true \
    --COMMAND="cd qwen_speech_exp && bash train.sh" \
    --USE_EXISTING_FOLDER=true \
    --RUN_NAME=<RUN_NAME_FROM_STEP5>
```

### Option B: tpu_ssh.sh

```bash
bash qwen_speech_exp/tpu_ssh.sh --all \
    --command "cd ~/maxtext/qwen_speech_exp && bash train.sh"
```

Monitor logs: multihost_runner stores logs at `/tmp/<RUN_NAME>/output_slice_*.txt`
on the jumpbox.

---

## Step 9: Run Inference (from Jumpbox)

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
# First copy audio file to all workers (or use a GCS path)
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

## Step 10: Cleanup

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

## Quick Reference: Commands from Jumpbox

| Task | Command |
|------|---------|
| SSH into worker 0 | `bash qwen_speech_exp/tpu_ssh.sh` |
| SSH into worker N | `bash qwen_speech_exp/tpu_ssh.sh --worker N` |
| Run cmd on 1 worker | `bash qwen_speech_exp/tpu_ssh.sh --command "CMD"` |
| Run cmd on ALL workers | `bash qwen_speech_exp/tpu_ssh.sh --all --command "CMD"` |
| Copy file to worker | `bash qwen_speech_exp/tpu_ssh.sh --scp local remote` |
| Convert data | `python3 qwen_speech_exp/convert_tfrecord_to_arrayrecord.py ...` |
| Train (multihost) | `python3 tools/orchestration/multihost_runner.py --INTERNAL_IP=true ...` |
| Check TPU devices | `bash qwen_speech_exp/tpu_ssh.sh --command "python3 -c 'import jax; print(jax.devices())'"` |

---

## File Overview

```
qwen_speech_exp/
├── SETUP_GUIDE.md                      # This file
├── env_vars.sh                         # Environment variables (paths, model name)
├── convert.sh                          # HF -> MaxText checkpoint conversion
├── convert_data.sh                     # TFRecord -> ArrayRecord data conversion
├── convert_tfrecord_to_arrayrecord.py  # Python converter script
├── train.sh                            # Training launch script (v4-32, 16 chips)
├── inference.sh                        # Single-host inference (4 chips)
├── inference_multihost.sh              # Multi-host inference (16 chips)
└── tpu_ssh.sh                          # Helper to SSH/SCP to TPU workers
```
