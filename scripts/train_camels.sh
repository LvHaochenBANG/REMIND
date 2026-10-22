#!/usr/bin/env bash
# CAMELS 实验：直接修改下面的参数即可调整配置。
# 用法：bash scripts/train_camels.sh [--prepare | --dry-run | --help]
set -euo pipefail

# 数据集与检索参数
DATASET="CAMELS"
DATA_DIR="${DATA_DIR:-./data/CAMELS}"
TOP_K=22                       # 每个节点使用的检索邻居数
RETRIEVAL_CANDIDATES=30        # 样本级检索候选数

# 两个预测分支的高斯混合分量数
MEMORY_MIXTURES=4             # 记忆分支
ROBUST_MIXTURES=2             # 鲁棒分支

# 总损失 = NLL + 相似度系数 × L_sim + 离散度系数 × L_var + MAE系数 × L_mae
MAE_WEIGHT=30
SIMILARITY_WEIGHT=0.5
DISPERSION_WEIGHT=0.5

# 常用训练参数；环境变量可覆盖 GPU 编号和 Python 路径。
DEVICE_INDEX="${DEVICE_INDEX:-0}"
PYTHON="${PYTHON:-python}"
EPOCHS=50
BATCH_SIZE=12
LEARNING_RATE=0.0001

# 从 Remind-cr 目录运行。
source scripts/run_dataset.sh
run_dataset "$@"
