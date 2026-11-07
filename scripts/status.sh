#!/usr/bin/env bash
# Live status snapshot for the Clarion rebuild. Run anytime: scripts/status.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
line(){ printf '%s\n' "―――――――――――――――――――――― $1 ――――――――――――――――――――――"; }

echo "snapshot $(date '+%H:%M:%S')"

line "GPU"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  no GPU"

line "training process"
ps -o pid,etime,args -C python 2>/dev/null | grep -F "train_qlora" | grep -v grep \
  && echo "  ^ live" || echo "  not running (finished or not started)"

line "training log tail (last 4 steps)"
LOG="${CLARION_TRAIN_LOG:-$ROOT/runs/train.log}"
[ -f "$LOG" ] && tail -c 1200 "$LOG" | tr '\r' '\n' | grep -E "it\]|loss" | tail -4 || echo "  no log yet"

line "artifacts"
[ -f "$ROOT/adapters/qwen25c-1.5b-sql-lora/adapter_model.safetensors" ] \
  && echo "  adapter     : DONE ($(du -h $ROOT/adapters/qwen25c-1.5b-sql-lora/adapter_model.safetensors | cut -f1))" \
  || echo "  adapter     : pending"
[ -f "$ROOT/models/qwen25-coder-1.5b-sql-q4_k_m.gguf" ] \
  && echo "  Q4_K_M GGUF : DONE ($(du -h $ROOT/models/qwen25-coder-1.5b-sql-q4_k_m.gguf | cut -f1))" \
  || echo "  Q4_K_M GGUF : pending"

line "llama-server"
SRV=$(ps aux | grep -F "llama-server" | grep -v grep | awk '{print $2}')
if [ -n "$SRV" ]; then
  RSS=$(ps -o rss= -p "$SRV" | awk '{printf "%.2f GB", $1/1048576}')
  echo "  pid $SRV resident $RSS"
else
  echo "  not started"
fi

line "parallel agents' outputs"
for p in "$ROOT/data/ambiguity_train"/*; do [ -f "$p" ] && echo "  data: $(basename $p) $(wc -l < $p) lines"; done
[ -d "$ROOT/ambiguity/artifacts" ] && ls "$ROOT/ambiguity/artifacts" 2>/dev/null | sed 's/^/  classifier artifact: /'
[ -f "$ROOT/db_introspect.py" ] && echo "  db_introspect.py present"
for e in "$ROOT/evals/heldout"/*; do [ -f "$e" ] && echo "  eval: $(basename $e) $(wc -l < $e) lines"; done
ls "$ROOT/evals"/REPORT*.md 2>/dev/null | sed 's/^/  report: /'

[ -f "$PY" ] || true
