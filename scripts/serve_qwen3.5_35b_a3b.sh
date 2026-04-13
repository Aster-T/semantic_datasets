#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

# Conservative defaults for 2 x 48GB Ada 6000.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"

CMD=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --cpu-offload-gb "$CPU_OFFLOAD_GB"
    --performance-mode throughput
    -O3
    --uvicorn-log-level warning
    --language-model-only
)

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    CMD+=(--enable-prefix-caching)
fi

"${CMD[@]}"
