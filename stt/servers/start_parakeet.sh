#!/usr/bin/env bash
set -euo pipefail

# parakeet-tdt-0.6b-v3 is NeMo-only (no native vLLM transcription support), so we
# serve it through the OpenAI-compatible NeMo shim (servers/nemo_asr_server.py).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="parakeet-nemo"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$SCRIPT_DIR":/srv \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e "NEMO_MODEL=nvidia/parakeet-tdt-0.6b-v3" \
  --entrypoint /bin/bash \
  nvcr.io/nvidia/nemo:25.09 \
  -c 'pip install --no-cache-dir fastapi uvicorn python-multipart && python /srv/nemo_asr_server.py'
