#!/usr/bin/env bash
# Export the already-trained write adapter merge to a separate Q4_K_M model.
# The read-only baseline GGUF is never overwritten.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/llama.cpp/gguf-py"
BIN="$ROOT/llama.cpp/build/bin"
MERGED="$ROOT/models/merged-sql-write"
F16="$ROOT/models/qwen25-coder-1.5b-sql-write-f16.gguf"
FINAL="$ROOT/models/qwen25-coder-1.5b-sql-write-q4_k_m.gguf"

"$ROOT/.venv/bin/python" "$ROOT/llama.cpp/convert_hf_to_gguf.py" \
  "$MERGED" --outfile "$F16" --outtype f16
"$BIN/llama-quantize" "$F16" "$FINAL" q4_k_m
echo "Built $FINAL"
