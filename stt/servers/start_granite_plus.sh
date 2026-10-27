#!/usr/bin/env bash
set -euo pipefail

# granite-speech-4.1-2b-plus requires vLLM v0.23.0 or later (per the model card).

docker rm -f granite-plus-vllm 2>/dev/null || true
docker run -d --rm \
  --name granite-plus-vllm \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.32.0 \
  -c 'uv pip install --system "vllm[audio]==0.23.0" && vllm serve ibm-granite/granite-speech-4.1-2b-plus --max-model-len 4096'
