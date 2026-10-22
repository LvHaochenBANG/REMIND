#!/usr/bin/env bash
# 公共运行逻辑：由 train_energy.sh / train_camels.sh / train_pems.sh 加载。
# 各数据集的实验参数请在对应 train_*.sh 顶部修改。

run_dataset() {
    local mode="train"
    local dry_run=false
    local -a extra_args=() command=()

    # 脚本选项可以放在任意位置，其他参数保留原顺序。
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prepare) mode="prepare" ;;
            --dry-run) dry_run=true ;;
            -h|--help)
                printf '用法：bash scripts/train_%s.sh [--prepare] [--dry-run] [Python 参数]\n' "${DATASET,,}"
                printf '默认训练；--prepare 重建并替换检索缓存；--dry-run 只打印命令。\n'
                return 0 ;;
            *) extra_args+=("$1") ;;
        esac
        shift
    done

    # 使用数组保存命令，确保带空格的路径和参数能正确传递。
    if [[ "$mode" == "prepare" ]]; then
        command=(
            "$PYTHON" run_retrieval.py
            --all --force
            --data_dir "$DATA_DIR"
            --topk_sample "$RETRIEVAL_CANDIDATES"
            --topk_node "$TOP_K"
        )
    else
        command=(
            "$PYTHON" main.py --train
            --run_name "${DATASET}_k${TOP_K}_m${MEMORY_MIXTURES}_r${ROBUST_MIXTURES}"
            --dataset_dir "$DATA_DIR"
            --device_index "$DEVICE_INDEX"
            --topk "$TOP_K"
            --memory_mixtures "$MEMORY_MIXTURES"
            --robust_mixtures "$ROBUST_MIXTURES"
            --mae_weight "$MAE_WEIGHT"
            --similarity_alignment_weight "$SIMILARITY_WEIGHT"
            --dispersion_alignment_weight "$DISPERSION_WEIGHT"
            --epochs "$EPOCHS"
            --batch_size "$BATCH_SIZE"
            --learning_rate "$LEARNING_RATE"
        )
    fi
    command+=("${extra_args[@]}")

    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$dry_run" == true ]]; then
        return 0
    fi
    exec "${command[@]}"
}
