#!/bin/bash
# Helper script to SSH into the qr-v4-32 TPU VM and run commands.
#
# Usage:
#   # Interactive shell on worker 0:
#   bash tpu_ssh.sh
#
#   # Run a command on worker 0:
#   bash tpu_ssh.sh --command "python3 -c 'import jax; print(jax.devices())'"
#
#   # Run a command on a specific worker:
#   bash tpu_ssh.sh --worker 2 --command "hostname"
#
#   # Run a command on ALL workers in parallel:
#   bash tpu_ssh.sh --all --command "bash tools/setup/setup.sh MODE=stable"
#
#   # SCP a file to worker 0:
#   bash tpu_ssh.sh --scp local_file.py remote_dest_path
#
#   # Run the data conversion:
#   bash tpu_ssh.sh --command "cd /path/to/maxtext && bash qwen_speech_exp/convert_data.sh"

set -e

# --- TPU VM Configuration ---
TPU_NAME="qr-v4-32"
PROJECT="arabic-asr-level2thinkg"
ZONE="us-central2-b"
WORKER=0
NUM_WORKERS=4  # v4-32 has 4 hosts
RUN_ALL=false
DO_SCP=false
USE_INTERNAL_IP=true  # true when running from jumpbox in same project/zone
COMMAND=""

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --worker)     WORKER="$2"; shift 2 ;;
        --command)    COMMAND="$2"; shift 2 ;;
        --all)        RUN_ALL=true; shift ;;
        --scp)        DO_SCP=true; SCP_SRC="$2"; SCP_DST="$3"; shift 3 ;;
        --external-ip) USE_INTERNAL_IP=false; shift ;;
        --tpu)        TPU_NAME="$2"; shift 2 ;;
        --project)    PROJECT="$2"; shift 2 ;;
        --zone)       ZONE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash tpu_ssh.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --command CMD    Run CMD on the TPU VM (non-interactive)"
            echo "  --worker N       Target worker N (default: 0)"
            echo "  --all            Run command on ALL workers in parallel"
            echo "  --scp SRC DST    Copy local file SRC to remote DST"
            echo "  --external-ip    Use external IP (default: internal IP for jumpbox)"
            echo "  --tpu NAME       TPU VM name (default: qr-v4-32)"
            echo "  --project PROJ   GCP project (default: arabic-asr-level2thinkg)"
            echo "  --zone ZONE      GCP zone (default: us-central2-b)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

COMMON_ARGS="--project=${PROJECT} --zone=${ZONE} --strict-host-key-checking=no"
if [ "$USE_INTERNAL_IP" = true ]; then
    COMMON_ARGS="${COMMON_ARGS} --internal-ip"
fi

# --- SCP mode ---
if [ "$DO_SCP" = true ]; then
    echo ">>> Copying ${SCP_SRC} to ${TPU_NAME}:${SCP_DST} (worker ${WORKER})"
    gcloud compute tpus tpu-vm scp \
        --worker="${WORKER}" \
        ${COMMON_ARGS} \
        "${SCP_SRC}" "${TPU_NAME}:${SCP_DST}"
    echo "Done."
    exit 0
fi

# --- Run on ALL workers ---
if [ "$RUN_ALL" = true ]; then
    if [ -z "$COMMAND" ]; then
        echo "Error: --all requires --command"
        exit 1
    fi
    echo ">>> Running on ALL ${NUM_WORKERS} workers in parallel..."
    PIDS=()
    for w in $(seq 0 $((NUM_WORKERS - 1))); do
        echo "  [worker ${w}] Starting..."
        gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
            --worker="${w}" \
            ${COMMON_ARGS} \
            --command="${COMMAND}" &
        PIDS+=($!)
    done

    # Wait for all and collect exit codes
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "  [worker ${i}] FAILED"
            FAILED=$((FAILED + 1))
        else
            echo "  [worker ${i}] Done"
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo ">>> ${FAILED}/${NUM_WORKERS} workers failed."
        exit 1
    fi
    echo ">>> All ${NUM_WORKERS} workers completed successfully."
    exit 0
fi

# --- Single worker: command or interactive ---
if [ -n "$COMMAND" ]; then
    echo ">>> Running on worker ${WORKER}: ${COMMAND}"
    gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
        --worker="${WORKER}" \
        ${COMMON_ARGS} \
        --command="${COMMAND}"
else
    echo ">>> Opening interactive SSH to ${TPU_NAME} worker ${WORKER}..."
    gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
        --worker="${WORKER}" \
        ${COMMON_ARGS}
fi
