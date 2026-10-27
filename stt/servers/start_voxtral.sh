#!/usr/bin/env bash
set -euo pipefail

docker rm -f voxtral-vllm 2>/dev/null || true
docker run -d --rm \
  --name voxtral-vllm \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e "LD_LIBRARY_PATH=/lib/x86_64-linux-gnu" \
  --entrypoint bash \
  vllm/vllm-openai:v0.32.0 \
  -c 'pip install --no-cache-dir soundfile && exec vllm serve \
    RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic \
    --tokenizer_mode mistral \
    --tokenizer mistralai/Voxtral-Small-24B-2507 \
    --tensor_parallel_size 2'

