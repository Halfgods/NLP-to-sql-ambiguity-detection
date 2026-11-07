#!/usr/bin/env bash
# Convert the merged model to GGUF (F16), quantize to Q4_K_M (~941 MB),
# then clean up intermediates so only the Q4 GGUF + adapter remain.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/llama.cpp/gguf-py"
BIN="$ROOT/llama.cpp/build/bin"

F16="$ROOT/models/qwen25c-1.5b-sql-f16.gguf"
FINAL="$ROOT/models/qwen25-coder-1.5b-sql-q4_k_m.gguf"

echo "[1/3] merged safetensors -> GGUF F16"
"$ROOT/.venv/bin/python" "$ROOT/llama.cpp/convert_hf_to_gguf.py" \
    "$ROOT/models/merged-sql" --outfile "$F16" --outtype f16

echo "[2/3] F16 -> Q4_K_M"
"$BIN/llama-quantize" "$F16" "$FINAL" q4_k_m

echo "[3/3] sizes + cleanup"
ls -l "$F16" "$FINAL"
rm -f "$F16"
rm -rf "$ROOT/models/merged-sql"
echo "done -> $FINAL"
