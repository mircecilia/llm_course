#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${LAB06_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
PY="$TASK_ROOT/.venv/bin/python"
CODE="$TASK_ROOT/src/hw6_rsi_localize.py"
export LAB06_ROOT="$TASK_ROOT"
export LAB06_MODEL=${LAB06_MODEL:-$(dirname "$TASK_ROOT")/llm_course_lab05/models/SmolVLM-256M-Instruct}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [ ! -e "$TASK_ROOT/hw6_imgs/gt.json" ]; then
  "$PY" "$CODE" --mode make_data
fi
"$PY" "$CODE" --mode check
CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode model_check > "$TASK_ROOT/logs/implementation_check.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode run --grid 4 --steps 10 --rounds 2 > "$TASK_ROOT/logs/grid4.log" 2>&1 &
p4=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE" --mode run --grid 8 --steps 10 --rounds 2 > "$TASK_ROOT/logs/grid8.log" 2>&1 &
p8=$!
s4=0; s8=0
wait "$p4" || s4=$?
wait "$p8" || s8=$?
printf 'WORKERS_FINISHED grid4=%s grid8=%s\n' "$s4" "$s8"
if [ "$s4" -ne 0 ] || [ "$s8" -ne 0 ]; then exit 1; fi
"$PY" "$TASK_ROOT/src/audit_results.py" --root "$TASK_ROOT"
printf 'LAB06_ALL_COMPLETE\n'
