#!/usr/bin/env bash
set -euo pipefail

# NVIDIA-Nemotron-3.5-Transcribe-30B-A4B is a speech LLM (NeMo SALM) served by
# vLLM 0.32.0 through the nemo_speechlm plugin from NeMo Speech main. It only
# speaks /v1/chat/completions, so evaluate it with the `nemotron_transcribe`
# pipeline, not `openai_api`.
#
# We serve the FP8 checkpoint: it is the precision NVIDIA lists for Ada GPUs,
# and BF16 (~67 GB) does not fit. FP8 (~35 GB) is split over two 24 GB GPUs
# with tensor parallelism. Context, batch and audio length are cut down from
# the model card's 80 GB profile. The KV cache is capped at 1 GiB per GPU
# (~250k tokens, plenty for 4 x 16k): with --gpu-memory-utilization the
# profiler sizes for short audio, and the audio encoder then runs out of
# memory on several long recordings at once.
# Word timestamps (--middleware) are left out; the eval does not use them.
# --use-replayssm and --mamba-backend flashinfer are left out too: FlashInfer's
# ReplaySSM kernel needs sm_90 (Hopper) and fails to compile on Ada (sm_89).

CONTAINER_NAME="nemotron-transcribe-vllm"
MODEL="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"
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
  -e VLLM_MAX_AUDIO_DECODE_DURATION_S=900 \
  -e VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.32.0 \
  -c '
    set -e
    # Install NeMo Speech without letting pip touch the torch/vLLM stack
    # that ships in the image. The image has no git, so fetch a tarball.
    pip freeze | grep -i -E "^(torch|torchaudio|torchvision|transformers|numpy|triton|flashinfer[-_a-z]*)==" > /tmp/constraints.txt
    echo vllm==0.32.0 >> /tmp/constraints.txt
    pip install --no-cache-dir -c /tmp/constraints.txt \
      "nemo-toolkit[asr] @ https://github.com/NVIDIA-NeMo/Speech/archive/${NEMO_COMMIT}.tar.gz" \
      "peft>=0.18.1" av==19.0.2
    exec vllm serve "$MODEL" \
      --served-model-name "$MODEL" \
      --trust-remote-code --dtype bfloat16 --kv-cache-dtype auto \
      --tensor-parallel-size 2 --kv-cache-memory 1G \
      --max-model-len 16384 --max-num-batched-tokens 4096 --max-num-seqs 4 \
      --enable-chunked-prefill --no-enable-prefix-caching \
      --mamba-cache-mode none \
      --mm-processor-cache-gb 0 --no-async-scheduling \
      --limit-mm-per-prompt "{\"audio\":{\"count\":1,\"length\":14400000}}" \
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":8}" \
      --reasoning-parser nemotron_v3
  '
