#!/usr/bin/env bash
set -euo pipefail

# Flurin17/cohere-transcribe-swiss-german is saved in Transformers' own
# CohereAsr format, which vLLM's Cohere loader cannot read (it expects the
# base checkpoint's tensor names). We serve it with Transformers through the
# OpenAI-compatible shim servers/cohere_asr_server.py, inside the vLLM image
# for its torch/transformers stack.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="cohere-swiss-hf"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$SCRIPT_DIR":/srv:ro \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e "HF_MODEL=Flurin17/cohere-transcribe-swiss-german" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.32.0 \
  -c '
    # The feature extractor needs librosa; keep pip off the image'"'"'s numpy/torch.
    pip freeze | grep -i -E "^(torch|numpy|transformers)==" > /tmp/constraints.txt
    pip install --no-cache-dir -c /tmp/constraints.txt librosa==0.11.0
    exec python3 /srv/cohere_asr_server.py
  '
