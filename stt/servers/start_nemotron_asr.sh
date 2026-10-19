#!/usr/bin/env bash
set -euo pipefail

# nemotron-3.5-asr-streaming-0.6b is NeMo-only (no native vLLM transcription
# support). We serve it offline through the OpenAI-compatible NeMo shim
# (servers/nemo_asr_server.py); transcribe() runs the whole file at once.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="nemotron-asr-nemo"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$SCRIPT_DIR":/srv \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
  -e "NEMO_MODEL=nvidia/nemotron-3.5-asr-streaming-0.6b" \
  --entrypoint /bin/bash \
  nvcr.io/nvidia/nemo:25.09 \
  -c '
    # nemo:25.09 ships NeMo 2.5.3, which predates the EncDecRNNTBPEModelWithPrompt
    # class (module nemo.collections.asr.models.rnnt_bpe_models_prompt) that
    # nemotron-3.5-asr-streaming requires. Update the editable /opt/NeMo checkout
    # to a pinned post-Nov-2025 commit that has it, and add kaldialign (a new
    # context-biasing dep pulled in by that newer source).
    set -e
    cd /opt/NeMo
    git fetch --depth 1 origin 61d1a24404157e5ca41cfbc55948f62d48a4de27
    git checkout FETCH_HEAD
    pip install --no-cache-dir kaldialign fastapi uvicorn python-multipart
    python /srv/nemo_asr_server.py
  '
