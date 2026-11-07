#!/usr/bin/env bash
# Serve the quantized SQL model standalone (CPU, mmap; ~1.06 GB RSS).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/llama.cpp/build/bin/llama-server" \
    -m "${CLARION_GGUF:-$ROOT/models/qwen25-coder-1.5b-sql-write-q4_k_m.gguf}" \
    --port "${SQL_MODEL_PORT:-18766}" -c 2048 -t 8 --no-webui \
    -b 512 -ub 256 "$@"
