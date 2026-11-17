#!/usr/bin/env bash
set -euo pipefail

# FluidVoice-Labs/fluid-speech-pro is Parakeet TDT 0.6B v3 plus one extra encoder
# block, loaded with the repository's fluid_speech.py (NEMO_LOADER_FILE). Like
# Parakeet it is NeMo-only (no native vLLM transcription support), so we
# serve it through the OpenAI-compatible NeMo shim (servers/nemo_asr_server.py).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="fluid-speech-pro-nemo"
# GPUs for docker --gpus, e.g. GPUS=device=1 to stay off a busy GPU.
GPUS="${GPUS:-all}"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --gpus "$GPUS" \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$SCRIPT_DIR":/srv \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e "NEMO_MODEL=FluidVoice-Labs/fluid-speech-pro" \
  -e "NEMO_LOADER_FILE=fluid_speech.py" \
  --entrypoint /bin/bash \
  nvcr.io/nvidia/nemo:25.09 \
  -c 'pip install --no-cache-dir fastapi uvicorn python-multipart && python /srv/nemo_asr_server.py'
