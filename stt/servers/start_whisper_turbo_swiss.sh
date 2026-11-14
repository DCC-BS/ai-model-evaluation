#!/usr/bin/env bash
set -euo pipefail

# Flurin17/whisper-large-v3-turbo-swiss-german on vLLM's native /v1/audio/transcriptions route.
# vLLM splits audio longer than 30 s at low-energy points before decoding.

CONTAINER_NAME="whisper-turbo-swiss-vllm"
MODEL="Flurin17/whisper-large-v3-turbo-swiss-german"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  vllm/vllm-openai:v0.32.0 \
  "$MODEL" --gpu-memory-utilization 0.5
