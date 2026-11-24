#!/usr/bin/env bash
set -euo pipefail

# Reson8/Rumble-2-Flash: a Qwen3-ASR-0.6B fine-tune, served natively by vLLM.
# Its options (language, digits, filler, verbatim) go through the chat
# template, so evaluate it with the `qwen3_asr_chat` pipeline.
# Whole files need room: an 8.5-minute file is about 6,700 audio tokens
# (13 per second). vLLM sizes the encoder cache from --max-num-batched-tokens
# (default 2048, which fails longer files with 400), so it is set to 8192.

CONTAINER_NAME="rumble-2-flash-vllm"
MODEL="Reson8/Rumble-2-Flash"
# GPUs for docker --gpus, e.g. GPUS=device=1 to stay off a busy GPU.
GPUS="${GPUS:-all}"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --gpus "$GPUS" \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e VLLM_MAX_AUDIO_DECODE_DURATION_S=900 \
  vllm/vllm-openai:v0.32.0 \
  "$MODEL" --max-model-len 16384 --gpu-memory-utilization 0.5 \
  --max-num-batched-tokens 8192
