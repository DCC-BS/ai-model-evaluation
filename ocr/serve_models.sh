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
# =============================================================================

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
HEALTH_TIMEOUT=300  # seconds to wait for the model to become healthy

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
)

declare -A MODEL_CONTAINER=(
    [olmocr]="ocr-olmocr"
    [glm-ocr]="ocr-glm"
    [paddleocr]="ocr-paddleocr"
    [deepseek-ocr]="ocr-deepseek"
    [lighton-ocr]="ocr-lighton"
    [dots-ocr]="ocr-dotsocr"
)

declare -A MODEL_PORT=(
    [olmocr]=8001
    [glm-ocr]=8002
    [paddleocr]=8003
    [deepseek-ocr]=8004
    [lighton-ocr]=8005
    [dots-ocr]=8006
)

# Use nightly for models that need latest transformers / custom architectures
declare -A MODEL_IMAGE=(
    [olmocr]="vllm/vllm-openai:v0.32.0"
    [glm-ocr]="vllm/vllm-openai:v0.32.0"
    [paddleocr]="vllm/vllm-openai:v0.32.0"
    [deepseek-ocr]="vllm/vllm-openai:v0.32.0"
    [lighton-ocr]="vllm/vllm-openai:v0.32.0"
    [dots-ocr]="vllm/vllm-openai:v0.32.0"
)

# Optional setup commands run inside the container before vLLM starts.
# When set, the entrypoint is overridden to: bash -c "<setup> && exec vllm serve ..."
declare -A MODEL_SETUP=(
    [glm-ocr]="uv pip install --system --upgrade transformers"
)

# Model-specific vLLM flags (space-separated)
declare -A MODEL_FLAGS=(
    [olmocr]="--trust-remote-code --max-model-len 8192"
    [glm-ocr]="--trust-remote-code"
    [paddleocr]="--trust-remote-code --no-enable-prefix-caching --mm-processor-cache-gb 0"
    [deepseek-ocr]="--trust-remote-code"
    [lighton-ocr]="--trust-remote-code --limit-mm-per-prompt {\"image\":1} --no-enable-prefix-caching --mm-processor-cache-gb 0"
    [dots-ocr]="--trust-remote-code --chat-template-content-format string --gpu-memory-utilization 0.95"
)

ALL_MODELS=( olmocr glm-ocr paddleocr deepseek-ocr lighton-ocr dots-ocr )

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
            --gpus all \
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
            --gpus all \
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