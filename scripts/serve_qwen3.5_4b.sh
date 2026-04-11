#!/usr/bin/env bash
set -euo pipefail

vllm serve Qwen/Qwen3.5-4B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --performance-mode throughput \
    --enable-prefix-caching \
    -O3 \
    --uvicorn-log-level warning \
    --language-model-only
