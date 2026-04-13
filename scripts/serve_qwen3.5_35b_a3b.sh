#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

# Conservative defaults for 2 x 48GB RTX 4090.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-12}"
SWAP_SPACE="${SWAP_SPACE:-16}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
ENABLE_THROUGHPUT_MODE="${ENABLE_THROUGHPUT_MODE:-0}"
VLLM_OPT_LEVEL="${VLLM_OPT_LEVEL:-}"
VLLM_HELP="$(vllm serve -h 2>&1 || true)"

supports_flag() {
    local flag="$1"
    [[ "$VLLM_HELP" == *"$flag"* ]]
}

CMD=(
    vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --uvicorn-log-level warning
    --language-model-only
)

if supports_flag "--kv-cache-dtype"; then
    CMD+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

if supports_flag "--cpu-offload-gb"; then
    CMD+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi

if supports_flag "--swap-space"; then
    CMD+=(--swap-space "$SWAP_SPACE")
fi

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    if supports_flag "--enable-prefix-caching"; then
        CMD+=(--enable-prefix-caching)
    fi
fi

if [ "$ENFORCE_EAGER" = "1" ]; then
    if supports_flag "--enforce-eager"; then
        CMD+=(--enforce-eager)
    fi
fi

if [ "$DISABLE_CUSTOM_ALL_REDUCE" = "1" ]; then
    if supports_flag "--disable-custom-all-reduce"; then
        CMD+=(--disable-custom-all-reduce)
    fi
fi

if [ "$ENABLE_THROUGHPUT_MODE" = "1" ]; then
    if supports_flag "--performance-mode"; then
        CMD+=(--performance-mode throughput)
    fi
fi

if [ -n "$VLLM_OPT_LEVEL" ]; then
    CMD+=("$VLLM_OPT_LEVEL")
fi

"${CMD[@]}"
