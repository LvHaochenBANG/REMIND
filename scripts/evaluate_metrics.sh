#!/usr/bin/env bash
# 从 Remind-cr 运行；仅评估指定 checkpoint，不重新训练。
# 用法：bash scripts/evaluate_metrics.sh energy /path/to/model.pt [额外参数]
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "用法：bash scripts/evaluate_metrics.sh energy|camels|pems|pems-7years checkpoint [额外参数]"
    exit 2
fi
dataset="$1"
checkpoint="$2"
shift 2
case "$dataset" in
    energy) data_dir=./data/ENERGY; topk=2; memory=4; robust=6; gpu=0 ;;
    camels) data_dir=./data/CAMELS; topk=22; memory=4; robust=2; gpu=0 ;;
    pems-7years) data_dir=./data/PEMS-7years; topk=12; memory=4; robust=4; gpu=0 ;;
    pems) data_dir=./data/PEMS; topk=12; memory=4; robust=4; gpu=0 ;;
    *) echo "未知数据集：$dataset" >&2; exit 2 ;;
esac
exec "${PYTHON:-python}" main.py --eval_only --checkpoint "$checkpoint" \
    --run_name "${dataset}_metrics" --dataset_dir "${DATA_DIR:-$data_dir}" \
    --device_index "${DEVICE_INDEX:-$gpu}" --topk "$topk" \
    --memory_mixtures "$memory" --robust_mixtures "$robust" \
    --mae_weight 30 --similarity_alignment_weight 0.5 --dispersion_alignment_weight 0.5 \
    --metric_samples 100 --metric_seed 2025 "$@"
