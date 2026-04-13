#!/usr/bin/env bash
set -euo pipefail

vllm serve Qwen/Qwen3.5-32B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --performance-mode throughput \
    --enable-prefix-caching \
    -O3 \
    --uvicorn-log-level warning \
    --language-model-only
