#!/usr/bin/env bash
set -euo pipefail

# Flurin17/nemotron-3.5-transcribe-swiss-german: Nemotron 3.5 Transcribe
# fine-tuned on Swiss German, same checkpoint format as the base model, served
# by vLLM 0.32.0 with the nemo_speechlm plugin. Chat completions only, so
# evaluate it with the `nemotron_transcribe` pipeline.
#
# Only BF16 weights are published (67 GB), which do not fit two 24 GB GPUs.
# vLLM quantizes them to FP8 while loading (--quantization fp8). The model
# card reports 17.47% ASGDTS WER for FP8 against 17.40% for BF16.
# The model was trained on clips of at most 30 s, so the pipeline cuts the
# audio (`--arg chunk_s=30`); context and audio length are sized for that.
# Speculative decoding (MTP) is left out: the card did not test it with
# `vllm serve`. Other settings follow start_nemotron_transcribe.sh.

CONTAINER_NAME="nemotron-transcribe-swiss-vllm"
MODEL="Flurin17/nemotron-3.5-transcribe-swiss-german"
# NeMo Speech main as of 2026-11-03; pinned for reproducibility.
NEMO_COMMIT="0fc5d7ad29b5f23601703e2550de33688a1733d0"

mkdir -p ~/.cache/flashinfer
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e "MODEL=$MODEL" \
  -e "NEMO_COMMIT=$NEMO_COMMIT" \
  -e VLLM_PLUGINS=nemo_speechlm \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.32.0 \
  -c '
    set -e
    pip freeze | grep -i -E "^(torch|torchaudio|torchvision|transformers|numpy|triton|flashinfer[-_a-z]*)==" > /tmp/constraints.txt
    echo vllm==0.32.0 >> /tmp/constraints.txt
    pip install --no-cache-dir -c /tmp/constraints.txt \
      "nemo-toolkit[asr] @ https://github.com/NVIDIA-NeMo/Speech/archive/${NEMO_COMMIT}.tar.gz" \
      "peft>=0.18.1" av==19.0.2
    exec vllm serve "$MODEL" \
      --served-model-name "$MODEL" \
      --trust-remote-code --dtype bfloat16 --quantization fp8 \
      --tensor-parallel-size 2 --kv-cache-memory 1G \
      --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 16 \
      --enable-chunked-prefill --no-enable-prefix-caching \
      --mamba-cache-mode none \
      --mm-processor-cache-gb 0 --no-async-scheduling \
      --limit-mm-per-prompt "{\"audio\":{\"count\":1,\"length\":480000}}" \
      --reasoning-parser nemotron_v3
  '
