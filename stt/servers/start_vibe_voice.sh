#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIBEVOICE_DIR="$SCRIPT_DIR/VibeVoice"
CONTAINER_NAME="vibevoice-vllm"

# Clone or update the VibeVoice repo
if [ -d "$VIBEVOICE_DIR/.git" ]; then
  echo "Updating VibeVoice repo..."
  # git -C "$VIBEVOICE_DIR" pull
else
  echo "Cloning VibeVoice repo..."
  git clone https://github.com/microsoft/VibeVoice.git "$VIBEVOICE_DIR"
fi

# Remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container '${CONTAINER_NAME}'..."
  docker rm -f "$CONTAINER_NAME"
fi

docker run -d --gpus all --name "$CONTAINER_NAME" \
  --ipc=host \
  -p 8000:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$VIBEVOICE_DIR":/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py --tp 2 --gpu-memory-utilization 0.92 --max-model-len 65536 --max-num-seqs 1"

echo "Container '${CONTAINER_NAME}' started. Logs: docker logs -f ${CONTAINER_NAME}"
