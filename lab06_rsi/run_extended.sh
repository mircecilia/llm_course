#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${LAB06_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
SEED_STEPS=${1:-60}
RSI_STEPS=${2:-10}
RUN_NAME=${3:-s${SEED_STEPS}_r${RSI_STEPS}}
TOOL_CONTEXT=${4:-reference}
PY="$TASK_ROOT/.venv/bin/python"
CODE="$TASK_ROOT/src/hw6_rsi_localize.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
test -e "$TASK_ROOT/hw6_imgs/gt.json"
test ! -e "$TASK_ROOT/outputs/$RUN_NAME"
mkdir "$TASK_ROOT/logs/$RUN_NAME"
CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --root "$TASK_ROOT" --mode run --grid 4 --seed-steps "$SEED_STEPS" --rsi-steps "$RSI_STEPS" --rounds 2 --run-name "$RUN_NAME" --tool-context "$TOOL_CONTEXT" > "$TASK_ROOT/logs/$RUN_NAME/grid4.log" 2>&1 &
p4=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE" --root "$TASK_ROOT" --mode run --grid 8 --seed-steps "$SEED_STEPS" --rsi-steps "$RSI_STEPS" --rounds 2 --run-name "$RUN_NAME" --tool-context "$TOOL_CONTEXT" > "$TASK_ROOT/logs/$RUN_NAME/grid8.log" 2>&1 &
p8=$!
s4=0; s8=0
wait "$p4" || s4=$?
wait "$p8" || s8=$?
printf 'WORKERS_FINISHED seed=%s rsi=%s grid4=%s grid8=%s\n' "$SEED_STEPS" "$RSI_STEPS" "$s4" "$s8"
if [ "$s4" -ne 0 ] || [ "$s8" -ne 0 ]; then exit 1; fi
"$PY" "$TASK_ROOT/src/audit_results.py" --root "$TASK_ROOT" --run-name "$RUN_NAME"
printf 'EXTENDED_COMPLETE %s\n' "$RUN_NAME"
