#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-openml}"
MODEL="${2:-Qwen/Qwen3.5-4B}"
CONCURRENCY="${3:-30}"
LIMIT="${4:-}"

CMD=(python src/evaluate.py
    --source "$SOURCE"
    --model "$MODEL"
    --local
    --concurrency "$CONCURRENCY"
)

if [ -n "$LIMIT" ]; then
    CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"
