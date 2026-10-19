#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_benchmark.sh — End-to-end OCR benchmark: deploy, evaluate, plot, teardown
#
# Usage:
#   ./run_benchmark.sh <model_key|all>
#   ./run_benchmark.sh glm-ocr --limit 50
#   ./run_benchmark.sh all --limit 100 --concurrency 8
#
# Model keys: olmocr  glm-ocr  paddleocr  deepseek-ocr  lighton-ocr  dots-ocr
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_CSV="${SCRIPT_DIR}/ocr_evaluation_results.csv"
RESULTS_PNG="${SCRIPT_DIR}/ocr_evaluation_results.png"

log() { echo "[benchmark] $*"; }
err() { echo "[benchmark] ERROR: $*" >&2; }

usage() {
    cat <<'EOF'
Usage:
  ./run_benchmark.sh <model_key|all> [eval_ocr options]

Steps performed:
  1. Start model container(s)   (serve_models.sh start)
  2. Run OCR evaluation          (eval_ocr.py)
  3. Plot results                (plot_results.py)
  4. Stop model container(s)     (serve_models.sh stop)

Model keys:
  olmocr        allenai/olmOCR-2-7B-1025-FP8       (port 8001)
  glm-ocr       zai-org/GLM-OCR                    (port 8002)
  paddleocr     PaddlePaddle/PaddleOCR-VL-1.5      (port 8003)
  deepseek-ocr  deepseek-ai/DeepSeek-OCR-2         (port 8004)
  lighton-ocr   lightonai/LightOnOCR-2-1B          (port 8005)
  dots-ocr      rednote-hilab/dots.ocr              (port 8006)

Extra options (forwarded to eval_ocr.py):
  --limit N         Limit dataset samples (for quick testing)
  --concurrency N   Max concurrent API requests (default: 4)
  --timeout SECS    Per-request timeout (default: 120)
  --output FILE     Custom CSV output path

Examples:
  ./run_benchmark.sh glm-ocr
  ./run_benchmark.sh all --limit 50
  ./run_benchmark.sh deepseek-ocr --concurrency 8 --limit 100
EOF
    exit 1
}

# ---------------------------------------------------------------------------
# Parse arguments: first positional arg is the model key, rest forwarded
# ---------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
    usage
fi

MODEL_KEY="$1"
shift
EVAL_EXTRA_ARGS=("$@")

# Validate model key early
VALID_KEYS="olmocr glm-ocr paddleocr deepseek-ocr lighton-ocr dots-ocr all"
if ! echo "$VALID_KEYS" | grep -qw "$MODEL_KEY"; then
    err "Unknown model key: $MODEL_KEY"
    err "Valid keys: $VALID_KEYS"
    exit 1
fi

# ---------------------------------------------------------------------------
# Cleanup trap — always stop containers, even on failure
# ---------------------------------------------------------------------------

cleanup() {
    log "Stopping model container(s) for: $MODEL_KEY ..."
    "${SCRIPT_DIR}/serve_models.sh" stop "$MODEL_KEY" || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 1: Start model container(s)
# ---------------------------------------------------------------------------

log "=== Step 1/4: Starting model container(s) for: $MODEL_KEY ==="
"${SCRIPT_DIR}/serve_models.sh" start "$MODEL_KEY"

# ---------------------------------------------------------------------------
# Step 2: Run evaluation
# ---------------------------------------------------------------------------

log "=== Step 2/4: Running OCR evaluation ==="
eval_cmd=(
    uv run "${SCRIPT_DIR}/eval_ocr.py"
    --model "$MODEL_KEY"
    --output "$RESULTS_CSV"
)
if (( ${#EVAL_EXTRA_ARGS[@]} )); then
    eval_cmd+=("${EVAL_EXTRA_ARGS[@]}")
fi
"${eval_cmd[@]}"

# ---------------------------------------------------------------------------
# Step 3: Plot results
# ---------------------------------------------------------------------------

log "=== Step 3/4: Plotting results ==="
uv run "${SCRIPT_DIR}/plot_results.py" \
    --input "$RESULTS_CSV" \
    --output "$RESULTS_PNG"

# ---------------------------------------------------------------------------
# Step 4: Teardown (handled by EXIT trap, but log it)
# ---------------------------------------------------------------------------

log "=== Step 4/4: Tearing down ==="
# The EXIT trap will call cleanup automatically.
# Remove the trap and call cleanup explicitly so we get cleaner output.
trap - EXIT
cleanup

log "=== Benchmark complete ==="
log "  Results CSV: $RESULTS_CSV"
log "  Results PNG: $RESULTS_PNG"
