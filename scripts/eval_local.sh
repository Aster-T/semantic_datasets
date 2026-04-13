#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-openml}"
MODEL="${2:-Qwen/Qwen3.5-35B-A3B}"
CONCURRENCY="${3:-8}"
LIMIT="${4:-}"
BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:8000/v1}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

MODELS_ENDPOINT="${BASE_URL%/}/models"
START_TS="$(date +%s)"

until curl -fsS "$MODELS_ENDPOINT" >/dev/null 2>&1; do
    NOW_TS="$(date +%s)"
    if [ $((NOW_TS - START_TS)) -ge "$WAIT_SECONDS" ]; then
        echo "Local LLM server is not ready after ${WAIT_SECONDS}s: $MODELS_ENDPOINT" >&2
        exit 1
    fi
    sleep 2
done

CMD=(python3 src/evaluate.py
    --source "$SOURCE"
    --model "$MODEL"
    --local
    --base-url "$BASE_URL"
    --concurrency "$CONCURRENCY"
)

if [ -n "$LIMIT" ]; then
    CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"
