#!/bin/bash
# Bootstrap script for the jumpbox VM.
# Run this after SSH-ing into the jumpbox for the first time.
#
# Usage:
#   # From your local machine, create jumpbox and SSH in:
#   gcloud compute instances create maxtext-jumpbox \
#       --project=arabic-asr-level2thinkg \
#       --zone=us-central2-b \
#       --machine-type=n2-standard-2 \
#       --image-family=ubuntu-2204-lts \
#       --image-project=ubuntu-os-cloud \
#       --boot-disk-size=100GB \
#       --scopes=cloud-platform
#
#   gcloud compute ssh maxtext-jumpbox \
#       --project=arabic-asr-level2thinkg \
#       --zone=us-central2-b
#
#   # Then on the jumpbox:
#   bash setup_jumpbox.sh

set -e

echo "=== 1/5 Installing system packages ==="
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv

# Install gcsfuse from Google's repo
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" \
    | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo tee /usr/share/keyrings/cloud.google.asc > /dev/null
sudo apt-get update
sudo apt-get install -y gcsfuse

echo "=== 2/5 Setting up Python venv ==="
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install tensorflow array-record

echo "=== 3/5 Setting up gcloud SSH keys ==="
if [ ! -f ~/.ssh/google_compute_engine.pub ]; then
    ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -N "" -q
fi
gcloud compute os-login ssh-keys add --key-file ~/.ssh/google_compute_engine.pub || true

echo "=== 4/5 Mounting GCS bucket ==="
sudo mkdir -p /tmp/gcsfuse
sudo chown "$(whoami)" /tmp/gcsfuse
gcsfuse --implicit-dirs arabic-asr-dataset /tmp/gcsfuse
echo "Mounted gs://arabic-asr-dataset at /tmp/gcsfuse"

echo "=== 5/5 Verifying setup ==="
echo "GCS data:"
ls /tmp/gcsfuse/grain_data/ 2>/dev/null || echo "  (grain_data/ not found)"
echo ""
echo "Python:"
python3 --version
python3 -c "import tensorflow; print(f'  TensorFlow {tensorflow.__version__}')"
python3 -c "from array_record.python.array_record_module import ArrayRecordWriter; print('  array-record OK')"

echo ""
echo "============================================"
echo "Jumpbox setup complete!"
echo ""
echo "Next steps:"
echo "  1. Clone or copy maxtext to ~/maxtext"
echo "  2. Run data conversion:"
echo "     cd ~/maxtext"
echo "     source ~/venv/bin/activate"
echo "     bash qwen_speech_exp/data/convert_data.sh"
echo ""
echo "  3. Set up TPU workers:"
echo "     python3 tools/orchestration/multihost_runner.py \\"
echo "       --TPU_PREFIX=qr-v4-32 \\"
echo "       --PROJECT=arabic-asr-level2thinkg \\"
echo "       --ZONE=us-central2-b \\"
echo "       --INTERNAL_IP=true \\"
echo "       --COMMAND='bash tools/setup/setup.sh MODE=stable' \\"
echo "       --SCRIPT_DIR=~/maxtext"
echo "============================================"
