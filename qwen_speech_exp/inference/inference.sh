#!/bin/bash
set -e

# Source the variables
source "$(dirname "$0")/../env_vars.sh"

# --- Usage ---
usage() {
    echo "Usage: $0 --mode <text|image|audio> [options]"
    echo ""
    echo "Modes:"
    echo "  text   - Text-only inference"
    echo "  image  - Image + text inference"
    echo "  audio  - Audio + text inference"
    echo ""
    echo "Options:"
    echo "  --prompt <text>       The prompt/question (required)"
    echo "  --image <path>        Path to image file (required for image mode)"
    echo "  --audio <path>        Path to audio file (required for audio mode)"
    echo "  --max-prefill <int>   Max prefill length (default: 272 for text, 512 for multimodal)"
    echo "  --max-target <int>    Max target length (default: 300)"
    echo ""
    echo "Examples:"
    echo "  $0 --mode text --prompt 'What is machine learning?'"
    echo "  $0 --mode image --prompt 'Describe this image' --image /path/to/image.jpg"
    echo "  $0 --mode audio --prompt 'Transcribe this audio' --audio /path/to/audio.wav"
    exit 1
}

# --- Default values ---
MODE=""
PROMPT=""
IMAGE_PATH=""
AUDIO_PATH=""
MAX_PREFILL=""
MAX_TARGET="300"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --audio)
            AUDIO_PATH="$2"
            shift 2
            ;;
        --max-prefill)
            MAX_PREFILL="$2"
            shift 2
            ;;
        --max-target)
            MAX_TARGET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# --- Validate arguments ---
if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required"
    usage
fi

if [[ -z "$PROMPT" ]]; then
    echo "Error: --prompt is required"
    usage
fi

# --- Configure based on mode ---
USE_MULTIMODAL="false"
USE_AUDIO="false"
EXTRA_ARGS=""

case $MODE in
    text)
        USE_MULTIMODAL="false"
        USE_AUDIO="false"
        MAX_PREFILL="${MAX_PREFILL:-272}"
        ;;
    image)
        if [[ -z "$IMAGE_PATH" ]]; then
            echo "Error: --image is required for image mode"
            usage
        fi
        if [[ ! -f "$IMAGE_PATH" ]]; then
            echo "Error: Image file not found: $IMAGE_PATH"
            exit 1
        fi
        USE_MULTIMODAL="true"
        USE_AUDIO="false"
        MAX_PREFILL="${MAX_PREFILL:-512}"
        EXTRA_ARGS="image_path=\"$IMAGE_PATH\""
        ;;
    audio)
        if [[ -z "$AUDIO_PATH" ]]; then
            echo "Error: --audio is required for audio mode"
            usage
        fi
        if [[ ! -f "$AUDIO_PATH" ]]; then
            echo "Error: Audio file not found: $AUDIO_PATH"
            exit 1
        fi
        USE_MULTIMODAL="true"
        USE_AUDIO="true"
        MAX_PREFILL="${MAX_PREFILL:-1024}"
        EXTRA_ARGS="audio_path=\"$AUDIO_PATH\""
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Must be text, image, or audio."
        usage
        ;;
esac

echo "=================================================="
echo "Running Inference (Decoding)"
echo "=================================================="
echo "Mode:        $MODE"
echo "Prompt:      $PROMPT"
echo "Multimodal:  $USE_MULTIMODAL"
echo "Audio:       $USE_AUDIO"
[[ -n "$IMAGE_PATH" ]] && echo "Image:       $IMAGE_PATH"
[[ -n "$AUDIO_PATH" ]] && echo "Audio file:  $AUDIO_PATH"
echo "Max prefill: $MAX_PREFILL"
echo "Max target:  $MAX_TARGET"
echo "=================================================="

# Generate a unique run name so logs don't overwrite each other
RUN_NAME="infer_${MODE}_$(date +%Y%m%d_%H%M%S)"

# Build the command
CMD="python -m maxtext.decode \
    \"$CONFIG_FILE\" \
    model_name=\"$MODEL_NAME\" \
    tokenizer_path=\"$TOKENIZER_PATH\" \
    load_parameters_path=\"${CHECKPOINT_PATH}/0/items\" \
    per_device_batch_size=1 \
    run_name=\"$RUN_NAME\" \
    max_prefill_predict_length=$MAX_PREFILL \
    max_target_length=$MAX_TARGET \
    steps=1 \
    hardware=tpu \
    async_checkpointing=false \
    scan_layers=false \
    skip_jax_distributed_system=true \
    weight_dtype=bfloat16 \
    checkpoint_storage_concurrent_gb=8 \
    ici_fsdp_parallelism=1 \
    ici_expert_parallelism=4 \
    ici_tensor_parallelism=1 \
    quantize_kvcache=true \
    kv_quant_dtype=int8 \
    use_multimodal=$USE_MULTIMODAL \
    use_audio=$USE_AUDIO \
    prompt='$PROMPT' \
    attention='dot_product'"

# Add extra args if present
if [[ -n "$EXTRA_ARGS" ]]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Execute
eval $CMD