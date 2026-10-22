"""
检索系统主运行脚本

功能：
1. 数据预处理（可选）
2. 执行检索
3. 保存节点级检索结果

智能加载：
- 如果训练集/验证集/测试集已存在，直接加载不重新分割
- 如果FAISS索引已存在，直接加载不重新构建
- 如果节点检索结果已存在，直接加载不重新检索
- 使用 --force 参数可强制重新处理

用法：
    # 完整流程（数据预处理 + 检索）
    python run_retrieval.py --all

    # 只执行数据预处理
    python run_retrieval.py --preprocess

    # 只执行检索（自动加载已存在的结果）
    python run_retrieval.py --retrieve

    # 强制重新检索（忽略已存在的结果）
    python run_retrieval.py --retrieve --force

    # 自定义配置
    python run_retrieval.py --all --topk_sample 30 --batch_size 100 --node_batch_size 25
"""

import argparse
import os
import sys
import numpy as np


def check_prerequisites(data_dir):
    """检查必要的文件是否存在"""

    if not os.path.exists(data_dir):
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return False

    # 检查原始数据文件
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if len(npy_files) == 0:
        print(f"[ERROR] 未找到任何 .npy 数据文件")
        return False


    # 检查预处理数据
    required_files = [
        "train.npz",
        "val.npz",
        "test.npz"
    ]

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    if missing_files:
        print(f"[ERROR] 缺少预处理文件: {missing_files}")
        return False

    # 检查邻接矩阵是否存在
    adj_path = os.path.join(data_dir, "adj.npy")
    if not os.path.exists(adj_path):
        print(f"[ERROR] 邻接矩阵不存在: {adj_path}")
        return False

    return True


def check_existing_retrieval_results(data_dir):
    """检查节点级检索结果是否已存在"""
    required_files = [
        "train_retrieval_results.npz",
        "val_retrieval_results.npz",
        "test_retrieval_results.npz"
    ]

    existing_files = []
    missing_files = []

    for f in required_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            existing_files.append(f)
        else:
            missing_files.append(f)

    return existing_files, missing_files


def load_retrieval_results(data_dir):
    """加载已存在的节点级检索结果"""

    results = {}
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(data_dir, f"{split}_retrieval_results.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            results[split] = {
                'node_history': data['node_history'],
                'node_target': data['node_target'],
                'node_similarities': data['node_similarities']
            }

    return results


def run_preprocess(data_dir):
    """运行数据预处理"""
    print("阶段 1: 数据预处理")

    try:
        # 导入数据预处理模块
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data_processor import process_data

        # 运行数据预处理
        result = process_data(data_dir)

        if result == 0:
            return True
        else:
            print("\n[ERROR] 数据预处理失败")
            return False

    except Exception as e:
        print(f"\n[ERROR] 数据预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_retrieve(data_dir, topk_sample, batch_size, node_batch_size, force=False, topk_node=5):
    """运行检索

    参数:
        data_dir: 数据目录
        topk_sample: 样本级检索候选数
        batch_size: 样本级批次大小
        node_batch_size: 节点级批次大小
        force: 是否强制重新检索（忽略已存在的结果）
    """
    print("阶段 2: 检索系统")

    # 检查节点级检索结果是否已存在
    existing_files, missing_files = check_existing_retrieval_results(data_dir)

    if len(existing_files) == 3 and not force:

        print("[CACHE] 使用已有检索缓存")


        return True


    try:
        # 导入检索系统模块
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from retrieval_system import RetrievalSystem


        # 创建检索系统
        retrieval_system = RetrievalSystem(
            topk_sample=topk_sample,
            topk_node=topk_node,
            batch_size=batch_size,
            node_batch_size=node_batch_size,
            max_workers=1,
            faiss_index_path=os.path.join(data_dir, "faiss_global.index"),
            use_multiprocessing=False,
            enable_batch_processing=True
        )

        # 确定需要检索的数据集
        if force:
            datasets_to_retrieve = ['train', 'val', 'test']
        else:
            datasets_to_retrieve = [f.replace('_retrieval_results.npz', '') for f in missing_files]


        # 运行检索
        retrieval_system.run_full_retrieval_pipeline(
            data_dir=data_dir,
            datasets=datasets_to_retrieve
        )


        return True

    except Exception as e:
        print(f"\n[ERROR] 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检索系统主运行脚本')
    parser.add_argument('--all', action='store_true', help='运行完整流程（数据预处理 + 检索）')
    parser.add_argument('--preprocess', action='store_true', help='只运行数据预处理')
    parser.add_argument('--retrieve', action='store_true', help='只运行检索')
    parser.add_argument('--force', action='store_true', help='强制重新检索（忽略已存在的结果）')
    parser.add_argument('--data_dir', type=str, default='./PEMS', help='数据目录（默认: ./ENERGY）')
    parser.add_argument('--topk_node', type=int, default=5, help='节点级检索邻居数')
    parser.add_argument('--topk_sample', type=int, default=30, help='样本级检索候选数（默认: 30）')
    parser.add_argument('--batch_size', type=int, default=100, help='样本级批次大小（默认: 100）')
    parser.add_argument('--node_batch_size', type=int, default=25, help='节点级批次大小（默认: 25）')

    args = parser.parse_args()
    if args.topk_node < 1 or args.topk_sample < 1:
        parser.error("topk_node and topk_sample must be positive")

    # 检查至少指定了一个操作
    if not (args.all or args.preprocess or args.retrieve):
        parser.print_help()
        print("\n[ERROR] 请指定至少一个操作: --all, --preprocess, 或 --retrieve")
        return 1

    # 打印系统信息


    success = True

    # 运行数据预处理
    if args.all or args.preprocess:
        if not run_preprocess(args.data_dir):
            success = False
            if args.all:
                return 1

    # 运行检索
    if args.all or args.retrieve:
        # 检查预处理数据
        if not check_prerequisites(args.data_dir):
            return 1

        if not run_retrieve(args.data_dir, args.topk_sample, args.batch_size, args.node_batch_size, args.force, topk_node=args.topk_node):
            success = False

    # 最终状态
    if success:
        print("[INFO] 所有操作完成")
    else:
        print("[INFO] 部分操作失败")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
