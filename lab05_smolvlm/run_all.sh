#!/usr/bin/env bash
# 两卡只并行独立对照，不做多卡数据并行，所有 DPO 从同一 SFT 适配器开始。
set -euo pipefail
export LAB05_ROOT="${LAB05_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$LAB05_ROOT"
export HF_HOME="$LAB05_ROOT/cache/huggingface"
export MPLCONFIGDIR="$LAB05_ROOT/cache/matplotlib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY="$LAB05_ROOT/.venv/bin/python"
CODE="$LAB05_ROOT/src/hw5_dpo.py"
STATUS=logs/status.txt
DPO12_RUN=dpo12
DPO17_RUN=dpo17
if [ "${1:-}" = "--resume-dpo" ]; then
    STATUS=logs/resume_status.txt
    DPO12_RUN=dpo12_gc
    DPO17_RUN=dpo17_gc
fi
trap 'printf "FAILED line=%s\n" "$LINENO" > "$STATUS"' ERR
if [ "${1:-}" != "--resume-dpo" ]; then
printf 'CHECKING\n' > logs/status.txt
CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode make_data > logs/make_data.log 2>&1
CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode check > logs/check.log 2>&1
printf 'BASELINE\n' > logs/status.txt
CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode infer > logs/baseline.log 2>&1
printf 'SFT_60\n' > logs/status.txt
CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode sft --run sft --steps 60 > logs/sft.log 2>&1
CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode eval --run sft > logs/eval_sft.log 2>&1
fi
printf 'DPO_AND_BETA_SCAN\n' > "$STATUS"
(
    CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode dpo --run "$DPO12_RUN" --pairs 12 --steps 40 --beta 0.1 > "logs/$DPO12_RUN.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode eval --run "$DPO12_RUN" > "logs/eval_$DPO12_RUN.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode dpo --run beta005 --pairs 17 --steps 40 --beta 0.05 > logs/beta005.log 2>&1
    CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE" --mode eval --run beta005 > logs/eval_beta005.log 2>&1
) &
task_gpu0=$!
(
    CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE" --mode dpo --run "$DPO17_RUN" --pairs 17 --steps 40 --beta 0.1 > "logs/$DPO17_RUN.log" 2>&1
    CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE" --mode eval --run "$DPO17_RUN" > "logs/eval_$DPO17_RUN.log" 2>&1
    CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE" --mode dpo --run beta05 --pairs 17 --steps 40 --beta 0.5 > logs/beta05.log 2>&1
    CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE" --mode eval --run beta05 > logs/eval_beta05.log 2>&1
) &
task_gpu1=$!
wait "$task_gpu0"
wait "$task_gpu1"
"$PY" "$CODE" --mode summarize > logs/summarize.log 2>&1
printf 'COMPLETE\n' > "$STATUS"
