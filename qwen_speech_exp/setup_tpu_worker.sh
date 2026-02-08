#!/bin/bash
# Non-interactive setup script for TPU workers.
# Run via multihost_runner.py to set up all 4 workers in parallel.
set -e

echo "=== 1/5 Installing Python 3.12 ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev libpython3.12

echo "=== 2/5 Creating Python 3.12 venv ==="
python3.12 -m venv ~/venv-maxtext
source ~/venv-maxtext/bin/activate

echo "=== 3/5 Installing PyTorch + XLA for TPU ==="
pip install --upgrade pip
pip install torch==2.9.0
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

echo "=== 4/5 Running MaxText setup ==="
bash tools/setup/setup.sh MODE=stable

echo "=== 5/5 Verifying ==="
python3 --version
python3 -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.device_count()}')"

echo "Setup complete!"
