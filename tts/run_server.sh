#!/bin/bash
# Launch vLLM-Omni TTS server and Gradio demo. Model is specified once (by task type).
# Run it from the tts directory after the install steps in README.md.
#
# Usage:
#   ./run_server.sh                     # Default: VoiceDesign model
#   ./run_server.sh CustomVoice         # CustomVoice model
#   ./run_server.sh VoiceDesign         # VoiceDesign model
#   ./run_server.sh Base                # Base (voice clone) model

set -e

TASK_TYPE="${1:-VoiceDesign}"

case "$TASK_TYPE" in
    CustomVoice)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ;;
    VoiceDesign)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        ;;
    Base)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ;;
    *)
        echo "Unknown task type: $TASK_TYPE"
        echo "Supported: CustomVoice, VoiceDesign, Base"
        exit 1
        ;;
esac

API_PORT="${API_PORT:-8000}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
HOST="${HOST:-127.0.0.1}"
API_BASE="http://localhost:${API_PORT}"

echo "Model: $MODEL"
echo "TTS API: $API_BASE (port $API_PORT, bound to $HOST)"
echo "Gradio UI: http://localhost:${GRADIO_PORT} (port $GRADIO_PORT)"
echo ""

# Start vLLM-Omni server in background
echo "Starting Qwen3-TTS server..."
uv run --no-sync vllm-omni serve "$MODEL" \
    --stage-configs-path qwen3_tts.yaml \
    --host "$HOST" \
    --port "$API_PORT" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni &

SERVER_PID=$!

# Ensure server is killed when this script exits (including Ctrl+C)
cleanup() {
    echo ""
    echo "Stopping TTS server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

# Wait for API to be ready (try /health or /v1/models)
echo "Waiting for API at $API_BASE ..."
for i in $(seq 1 90); do
    if curl -s -o /dev/null -w "%{http_code}" "$API_BASE/health" 2>/dev/null | grep -qE '200|404'; then
        echo "API ready."
        break
    fi
    if curl -s -o /dev/null -w "%{http_code}" "$API_BASE/v1/models" 2>/dev/null | grep -qE '200|401'; then
        echo "API ready."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process exited unexpectedly."
        exit 1
    fi
    sleep 2
done

# Final check: if server died, exit
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server failed to start."
    exit 1
fi

# Start Gradio demo (same model; API base and port from above)
echo "Starting Gradio demo at http://localhost:${GRADIO_PORT}"
uv run --no-sync python main.py \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --port "$GRADIO_PORT"
