#!/usr/bin/env bash
set -euo pipefail

docker rm -f granite-vllm 2>/dev/null || true
docker run -d --rm \
  --name granite-vllm \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.32.0 \
  -c 'uv pip install --system "vllm[audio]==0.23.0" && vllm serve ibm-granite/granite-4.0-1b-speech --max-model-len 2048'
