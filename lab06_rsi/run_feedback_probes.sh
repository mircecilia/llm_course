#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${LAB06_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
PY="$TASK_ROOT/.venv/bin/python"
CODE="$TASK_ROOT/src/hw6_rsi_localize.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for SOURCE in s60_r30 s150_r10; do
  TAG=feedback_${SOURCE}
  test ! -e "$TASK_ROOT/outputs/$TAG"
  mkdir "$TASK_ROOT/logs/$TAG"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --root "$TASK_ROOT" --mode feedback_probe --grid 4 --source-run "$SOURCE" --run-name "$TAG" --tool-context candidate > "$TASK_ROOT/logs/$TAG/grid4.log" 2>&1 &
  p4=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE" --root "$TASK_ROOT" --mode feedback_probe --grid 8 --source-run "$SOURCE" --run-name "$TAG" --tool-context candidate > "$TASK_ROOT/logs/$TAG/grid8.log" 2>&1 &
  p8=$!
  s4=0; s8=0
  wait "$p4" || s4=$?
  wait "$p8" || s8=$?
  printf 'FEEDBACK_WORKERS_FINISHED source=%s grid4=%s grid8=%s\n' "$SOURCE" "$s4" "$s8"
  if [ "$s4" -ne 0 ] || [ "$s8" -ne 0 ]; then exit 1; fi
done
