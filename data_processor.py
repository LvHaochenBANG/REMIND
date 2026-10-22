"""
数据预处理脚本

功能：
1. 加载原始数据
2. 创建滑动窗口
3. 划分训练集/验证集/测试集
4. 标准化数据
5. 保存预处理结果

用法：
    python data_processor.py
"""

import os
import re
import numpy as np
import argparse


def extract_number(filename):
    """提取文件名中的数字，用于排序"""
    numbers = re.findall(r'\d+', os.path.basename(filename))
    return int(numbers[0]) if numbers else 0


def check_existing_preprocessed_data(data_dir):
    """检查预处理数据是否已存在"""
    required_files = [
        "train.npz",
        "val.npz",
        "test.npz",
        "normalization_params.npz"
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


def load_preprocessed_data(data_dir):
    """加载已存在的预处理数据和归一化参数"""

    results = {}
    # 加载数据集
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(data_dir, f"{split}.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            results[split] = {
                'history': data['history'],
                'targets': data['targets']
            }

    # 加载归一化参数
    norm_params_path = os.path.join(data_dir, "normalization_params.npz")
    if os.path.exists(norm_params_path):
        norm_params = np.load(norm_params_path)
        results['normalization'] = {
            'mean': norm_params['mean'],
            'std': norm_params['std']
        }
    else:
        print(f"  [WARNING] normalization_params.npz 不存在")

    return results


def get_npy_file_paths(folder_path):
    """
    获取文件夹下所有npy文件的路径，并区分邻接矩阵和其他数据文件

    参数:
        folder_path (str): 包含npy文件的文件夹路径

    返回:
        tuple: (adj_matrix_path, data_paths)
            adj_matrix_path (str): 邻接矩阵文件的绝对路径，如果未找到则为None
            data_paths (list): 其他npy文件的绝对路径列表
    """
    # 获取文件夹中所有.npy文件
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                all_files.append(os.path.join(root, file))

    # 初始化返回值
    adj_matrix_path = None
    data_paths = []

    # 遍历所有文件，查找邻接矩阵
    for file_path in all_files:
        filename = os.path.basename(file_path).lower()

        # 检查是否为邻接矩阵文件（可以根据实际命名规则调整）
        if filename.startswith('adj') or 'adjacency' in filename:
            adj_matrix_path = os.path.abspath(file_path)
            continue

        # 其他数据文件
        data_paths.append(os.path.abspath(file_path))

    # 如果没有找到邻接矩阵，使用第一个文件作为邻接矩阵（可选）
    if adj_matrix_path is None and data_paths:
        print("警告：未找到明显的邻接矩阵文件，将使用第一个数据文件作为邻接矩阵")
        adj_matrix_path = data_paths.pop(0)

    # 按提取的数字对data_paths排序
    data_paths.sort(key=extract_number)

    return adj_matrix_path, data_paths


class SeriesWindowProcessor:
    """时间序列窗口处理器

    功能：
    1. 加载数据
    2. 创建滑动窗口
    3. 划分数据集
    4. 标准化数据
    """

    def __init__(self, data_paths, adj_matrix_path, history_window=12, future_window=12,
                 train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        # 初始化参数
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.adj_matrix_path = adj_matrix_path
        self.history_window = history_window
        self.future_window = future_window
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 数据相关变量
        self.data_list = []     # 所有原始数据列表
        self.all_samples = None    # 所有样本 (total_samples, history_window, N)
        self.all_targets = None    # 所有目标 (total_samples, future_window, N)
        self.train_history = None; self.val_history = None; self.test_history = None
        self.train_targets = None; self.val_targets = None; self.test_targets = None

        # 标准化参数
        self.mean = None
        self.std = None

        # 总是加载邻接矩阵
        self.load_adjacency_matrix()

        # 获取数据目录路径
        data_dir = os.path.dirname(self.data_paths[0])

        # 检查是否所有预处理文件都存在
        existing_files, missing_files = check_existing_preprocessed_data(data_dir)
        if len(existing_files) == 4:  # train.npz, val.npz, test.npz, normalization_params.npz
            self.load_preprocessed_files(data_dir)
        else:
            self.load_data()
            self.create_sliding_windows()  # 先对每个数据集划窗
            self.split_datasets()           # 再划分数据集
            self.fit_scaler()              # 用训练集计算标准化参数
            self.apply_scaler()            # 对三部分标准化

    def load_preprocessed_files(self, data_dir):
        """加载已存在的预处理文件和归一化参数"""
        # 加载数据集
        train_data = np.load(os.path.join(data_dir, "train.npz"))
        self.train_history = train_data['history']
        self.train_targets = train_data['targets']

        val_data = np.load(os.path.join(data_dir, "val.npz"))
        self.val_history = val_data['history']
        self.val_targets = val_data['targets']

        test_data = np.load(os.path.join(data_dir, "test.npz"))
        self.test_history = test_data['history']
        self.test_targets = test_data['targets']


        # 加载归一化参数
        norm_params = np.load(os.path.join(data_dir, "normalization_params.npz"))
        self.mean = norm_params['mean']
        self.std = norm_params['std']

        # 设置窗口参数
        self.N = self.train_history.shape[2]
        self.history_window = self.train_history.shape[1]
        self.future_window = self.train_targets.shape[1]

    def load_data(self):
        """加载所有数据文件"""
        self.data_list = []
        for path in self.data_paths:
            data = np.load(path)


            # 检查数据是否有 NaN 或 Inf
            if np.any(np.isnan(data)):
                raise ValueError(f"数据 {path} 包含 NaN 值")
            if np.any(np.isinf(data)):
                raise ValueError(f"数据 {path} 包含 Inf 值")

            self.data_list.append(data)

    def load_adjacency_matrix(self):
        """加载邻接矩阵"""
        self.adj_matrix = np.load(self.adj_matrix_path)
        self.N = self.adj_matrix.shape[0]

    def split_datasets(self):
        """按顺序划分合并后的样本"""
        total_samples = len(self.all_samples)
        train_split = int(total_samples * self.train_ratio)
        val_split = int(total_samples * (self.train_ratio + self.val_ratio))

        self.train_history = self.all_samples[:train_split]
        self.val_history = self.all_samples[train_split:val_split]
        self.test_history = self.all_samples[val_split:]

        self.train_targets = self.all_targets[:train_split]
        self.val_targets = self.all_targets[train_split:val_split]
        self.test_targets = self.all_targets[val_split:]


    def fit_scaler(self):
        """计算训练集的均值和标准差"""
        self.mean = self.train_history.mean(axis=(0, 1, 2), keepdims=False)
        self.std = self.train_history.std(axis=(0, 1, 2), keepdims=False) + 1e-6

    def apply_scaler(self):
        """使用训练集参数标准化"""
        self.train_history = (self.train_history - self.mean) / self.std
        self.val_history = (self.val_history - self.mean) / self.std
        self.test_history = (self.test_history - self.mean) / self.std

        # 对 targets 也进行归一化
        self.train_targets = (self.train_targets - self.mean) / self.std
        self.val_targets = (self.val_targets - self.mean) / self.std
        self.test_targets = (self.test_targets - self.mean) / self.std

    def reverse_normalization(self, data):
        """反标准化"""
        return data * self.std + self.mean

    def create_windows(self, data):
        """生成历史窗口与未来窗口对"""
        T, N = data.shape
        window_size = self.history_window + self.future_window
        n_samples = T - window_size + 1

        samples = np.zeros((n_samples, self.history_window, N))
        targets = np.zeros((n_samples, self.future_window, N))

        for i in range(n_samples):
            end_idx = i + window_size
            samples[i] = data[i:i+self.history_window, :]
            targets[i] = data[i+self.history_window:end_idx, :]

        return samples, targets

    def create_sliding_windows(self):
        """对每个数据集分别划窗，然后合并所有样本"""
        all_samples_list = []
        all_targets_list = []

        for idx, data in enumerate(self.data_list):
            samples, targets = self.create_windows(data)
            all_samples_list.append(samples)
            all_targets_list.append(targets)


        # 合并所有样本
        self.all_samples = np.concatenate(all_samples_list, axis=0)
        self.all_targets = np.concatenate(all_targets_list, axis=0)


    def save_processed_data(self, save_dir):
        """保存预处理后的数据到三个npz文件

        参数:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存训练集
        train_path = os.path.join(save_dir, "train.npz")
        np.savez_compressed(
            train_path,
            history=self.train_history,
            targets=self.train_targets
        )

        # 保存验证集
        val_path = os.path.join(save_dir, "val.npz")
        np.savez_compressed(
            val_path,
            history=self.val_history,
            targets=self.val_targets
        )

        # 保存测试集
        test_path = os.path.join(save_dir, "test.npz")
        np.savez_compressed(
            test_path,
            history=self.test_history,
            targets=self.test_targets
        )

        # 保存归一化参数
        norm_params_path = os.path.join(save_dir, "normalization_params.npz")
        np.savez_compressed(
            norm_params_path,
            mean=self.mean,
            std=self.std
        )


def process_data(data_dir, force=False):
    """数据预处理的核心逻辑

    参数:
        data_dir: 数据目录
        force: 是否强制重新预处理

    返回:
        0: 成功, 1: 失败
    """


    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return 1

    # 检查预处理数据是否已存在
    existing_files, missing_files = check_existing_preprocessed_data(data_dir)

    if len(existing_files) == 4 and not force:


        print("数据预处理完成（使用已有结果）")
        return 0


    # 设置数据路径
    adj_matrix_path, data_paths = get_npy_file_paths(data_dir)

    if len(data_paths) == 0:
        print(f"[ERROR] 未找到任何 .npy 数据文件")
        return 1

    # 创建数据处理器
    processor = SeriesWindowProcessor(
        data_paths=data_paths,
        adj_matrix_path=adj_matrix_path,
        history_window=12,
        future_window=12,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )


    # 保存预处理数据
    processor.save_processed_data(data_dir)

    print("数据预处理完成!")

    return 0


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--data_dir', type=str, default='./ENERGY', help='数据目录（默认: ./ENERGY）')
    parser.add_argument('--force', action='store_true', help='强制重新预处理（忽略已存在的结果）')

    args = parser.parse_args()

    return process_data(args.data_dir, args.force)


if __name__ == '__main__':
    import sys
    sys.exit(main())
