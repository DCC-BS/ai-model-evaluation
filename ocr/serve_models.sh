#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# serve_models.sh — Start / stop / check vLLM docker containers for OCR models
#
# Usage:
#   ./serve_models.sh start <model_key|all>   Start one or all model containers
#   ./serve_models.sh stop  <model_key|all>   Stop and remove containers
#   ./serve_models.sh status                  Show running OCR containers
#
# Model keys: olmocr  glm-ocr  paddleocr  deepseek-ocr  lighton-ocr  dots-ocr
#             lighton-ocr-3  nemotron-parse  paddleocr-1.6
# =============================================================================

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
OCR_GPU="${OCR_GPU:-all}"   # docker --gpus value: "all" or a device index such as 1
VLLM_MEMORY="${VLLM_MEMORY:-}"  # --gpu-memory-utilization for every model; empty keeps each model's own
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"  # seconds to wait for the model to become healthy

# ---------------------------------------------------------------------------
# Model definitions
#   Each model is described by a set of variables keyed by a short name.
# ---------------------------------------------------------------------------

declare -A MODEL_HF_NAME=(
    [olmocr]="allenai/olmOCR-2-7B-1025-FP8"
    [glm-ocr]="zai-org/GLM-OCR"
    [paddleocr]="PaddlePaddle/PaddleOCR-VL-1.5"
    [deepseek-ocr]="deepseek-ai/DeepSeek-OCR-2"
    [lighton-ocr]="lightonai/LightOnOCR-2-1B"
    [dots-ocr]="rednote-hilab/dots.ocr"
    [lighton-ocr-3]="lightonai/LightOnOCR-3-4B"
    [nemotron-parse]="nvidia/Nemotron-3.5-Parse"
    [paddleocr-1.6]="PaddlePaddle/PaddleOCR-VL-1.6"
)

declare -A MODEL_CONTAINER=(
    [olmocr]="ocr-olmocr"
    [glm-ocr]="ocr-glm"
    [paddleocr]="ocr-paddleocr"
    [deepseek-ocr]="ocr-deepseek"
    [lighton-ocr]="ocr-lighton"
    [dots-ocr]="ocr-dotsocr"
    [lighton-ocr-3]="ocr-lighton3"
    [nemotron-parse]="ocr-nemotron-parse"
    [paddleocr-1.6]="ocr-paddleocr16"
)

declare -A MODEL_PORT=(
    [olmocr]=8001
    [glm-ocr]=8002
    [paddleocr]=8003
    [deepseek-ocr]=8004
    [lighton-ocr]=8005
    [dots-ocr]=8006
    [lighton-ocr-3]=8007
    [nemotron-parse]=8008
    [paddleocr-1.6]=8009
)

# Use nightly for models that need latest transformers / custom architectures
declare -A MODEL_IMAGE=(
    [olmocr]="vllm/vllm-openai:v0.32.0"
    [glm-ocr]="vllm/vllm-openai:v0.32.0"
    [paddleocr]="vllm/vllm-openai:v0.32.0"
    [deepseek-ocr]="vllm/vllm-openai:v0.32.0"
    [lighton-ocr]="vllm/vllm-openai:v0.32.0"
    [dots-ocr]="vllm/vllm-openai:v0.32.0"
    # Tested on v0.31.0
    [lighton-ocr-3]="vllm/vllm-openai:v0.31.0"
    [paddleocr-1.6]="vllm/vllm-openai:v0.31.0"
    # The Nemotron plugin pins vllm==0.26.0 (supported range 0.25-0.32)
    [nemotron-parse]="vllm/vllm-openai:v0.26.0"
)

# Optional setup commands run inside the container before vLLM starts.
# When set, the entrypoint is overridden to: bash -c "<setup> && exec vllm serve ..."
declare -A MODEL_SETUP=(
    [glm-ocr]="uv pip install --system --upgrade transformers"
    # The vLLM plugin ships inside the model repo: download it at the pinned revision and install it
    [nemotron-parse]="python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('nvidia/Nemotron-3.5-Parse', revision='e7ea79242a87ea0ad3919b672bfedb429a683206', local_dir='/root/.cache/huggingface/nemotron-3.5-parse')\" && uv pip install --system '/root/.cache/huggingface/nemotron-3.5-parse[vllm]'"
)

# Model-specific vLLM flags (space-separated)
declare -A MODEL_FLAGS=(
    [olmocr]="--trust-remote-code --max-model-len 8192"
    [glm-ocr]="--trust-remote-code"
    [paddleocr]="--trust-remote-code --no-enable-prefix-caching --mm-processor-cache-gb 0"
    [deepseek-ocr]="--trust-remote-code"
    [lighton-ocr]="--trust-remote-code --limit-mm-per-prompt {\"image\":1} --no-enable-prefix-caching --mm-processor-cache-gb 0"
    [dots-ocr]="--trust-remote-code --chat-template-content-format string --gpu-memory-utilization 0.95"
    [lighton-ocr-3]="--trust-remote-code --max-model-len 16384 --limit-mm-per-prompt {\"image\":1} --default-chat-template-kwargs {\"enable_thinking\":false}"
    # Flags from the model card minus --attention-config flash_attn_version 3, which needs a Hopper GPU
    # (it starts through the setup branch, where bash -c parses the flags again: hence the quoted JSON)
    [nemotron-parse]="--trust-remote-code --revision e7ea79242a87ea0ad3919b672bfedb429a683206 --dtype bfloat16 --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 16384 --no-enable-prefix-caching --limit-mm-per-prompt '{\"image\":1}' --chat-template /root/.cache/huggingface/nemotron-3.5-parse/chat_template.jinja"
    [paddleocr-1.6]="--trust-remote-code --no-enable-prefix-caching --mm-processor-cache-gb 0"
)

ALL_MODELS=( olmocr glm-ocr paddleocr deepseek-ocr lighton-ocr dots-ocr lighton-ocr-3 nemotron-parse paddleocr-1.6 )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[serve_models] $*"; }
err()  { echo "[serve_models] ERROR: $*" >&2; }

usage() {
    cat <<'EOF'
Usage:
  ./serve_models.sh start   <model_key|all>
  ./serve_models.sh stop    <model_key|all>
  ./serve_models.sh status

Model keys:
  olmocr        allenai/olmOCR-2-7B-1025-FP8       (port 8001)
  glm-ocr       zai-org/GLM-OCR                    (port 8002)
  paddleocr     PaddlePaddle/PaddleOCR-VL-1.5      (port 8003)
  deepseek-ocr  deepseek-ai/DeepSeek-OCR-2         (port 8004)
  lighton-ocr   lightonai/LightOnOCR-2-1B          (port 8005)
  dots-ocr      rednote-hilab/dots.ocr              (port 8006)
  lighton-ocr-3 lightonai/LightOnOCR-3-4B          (port 8007)
  nemotron-parse nvidia/Nemotron-3.5-Parse         (port 8008)
  paddleocr-1.6 PaddlePaddle/PaddleOCR-VL-1.6      (port 8009)
EOF
    exit 1
}

resolve_models() {
    local key="$1"
    if [[ "$key" == "all" ]]; then
        echo "${ALL_MODELS[@]}"
    elif [[ -n "${MODEL_HF_NAME[$key]+x}" ]]; then
        echo "$key"
    else
        err "Unknown model key: $key"
        err "Valid keys: ${ALL_MODELS[*]} all"
        exit 1
    fi
}

wait_for_health() {
    local port="$1"
    local container="$2"
    local elapsed=0

    log "Waiting for $container to become healthy on port $port ..."
    while (( elapsed < HEALTH_TIMEOUT )); do
        # Check if the container is still running
        if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            err "Container $container exited unexpectedly. Check logs:"
            err "  docker logs $container"
            return 1
        fi

        # Try the /health or /v1/models endpoint
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            log "$container is ready (took ${elapsed}s)"
            return 0
        fi

        sleep 5
        elapsed=$(( elapsed + 5 ))
    done

    err "$container did not become healthy within ${HEALTH_TIMEOUT}s"
    err "Check logs: docker logs $container"
    return 1
}

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

start_model() {
    local key="$1"
    local hf_name="${MODEL_HF_NAME[$key]}"
    local container="${MODEL_CONTAINER[$key]}"
    local port="${MODEL_PORT[$key]}"
    local image="${MODEL_IMAGE[$key]}"
    local flags="${MODEL_FLAGS[$key]}"
    local setup="${MODEL_SETUP[$key]:-}"
    [[ -n "$VLLM_MEMORY" ]] && flags="$flags --gpu-memory-utilization $VLLM_MEMORY"
    local gpus="$OCR_GPU"
    [[ "$gpus" == all ]] || gpus="device=$gpus"

    # Stop existing container if running
    if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
        log "Removing existing container $container ..."
        docker rm -f "$container" > /dev/null 2>&1 || true
    fi

    log "Starting $hf_name as $container on port $port ..."
    log "  Image:  $image"
    log "  Flags:  $flags"
    [[ -n "$setup" ]] && log "  Setup:  $setup"

    # shellcheck disable=SC2086
    if [[ -n "$setup" ]]; then
        # Override entrypoint to run setup commands before vLLM starts
        docker run -d \
            --name "$container" \
            --gpus "$gpus" \
            --ipc=host \
            -p "${port}:8000" \
            -v "${HF_CACHE}:/root/.cache/huggingface" \
            -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
            -e "HF_TOKEN=${HF_TOKEN:-}" \
            -e "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu" \
            --entrypoint /bin/bash \
            "$image" \
            -c "$setup && exec vllm serve --model $hf_name --served-model-name $hf_name --port 8000 $flags"
    else
        docker run -d \
            --name "$container" \
            --gpus "$gpus" \
            --ipc=host \
            -p "${port}:8000" \
            -v "${HF_CACHE}:/root/.cache/huggingface" \
            -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
            -e "HF_TOKEN=${HF_TOKEN:-}" \
            -e "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu" \
            "$image" \
            --model "$hf_name" \
            --served-model-name "$hf_name" \
            --port 8000 \
            $flags
    fi

    wait_for_health "$port" "$container"
}

stop_model() {
    local key="$1"
    local container="${MODEL_CONTAINER[$key]}"

    if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
        log "Stopping $container ..."
        docker rm -f "$container" > /dev/null 2>&1
        log "$container removed."
    else
        log "$container is not running."
    fi
}

show_status() {
    log "Running OCR containers:"
    echo ""
    printf "  %-16s %-14s %-6s %s\n" "MODEL KEY" "CONTAINER" "PORT" "STATUS"
    printf "  %-16s %-14s %-6s %s\n" "---------" "---------" "----" "------"
    for key in "${ALL_MODELS[@]}"; do
        local container="${MODEL_CONTAINER[$key]}"
        local port="${MODEL_PORT[$key]}"
        local status
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
            status="running"
        elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
            status="stopped"
        else
            status="not created"
        fi
        printf "  %-16s %-14s %-6s %s\n" "$key" "$container" "$port" "$status"
    done
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
    usage
fi

ACTION="$1"
TARGET="${2:-}"

case "$ACTION" in
    start)
        [[ -z "$TARGET" ]] && usage
        for m in $(resolve_models "$TARGET"); do
            start_model "$m"
        done
        ;;
    stop)
        [[ -z "$TARGET" ]] && usage
        for m in $(resolve_models "$TARGET"); do
            stop_model "$m"
        done
        ;;
    status)
        show_status
        ;;
    *)
        err "Unknown action: $ACTION"
        usage
        ;;
esac