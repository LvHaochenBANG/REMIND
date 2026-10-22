"""
检索系统脚本

功能：
1. 加载预处理后的数据
2. 构建样本级FAISS索引
3. 执行两阶段检索（样本级 + 节点级）
4. 保存节点级检索结果

用法：
    python retrieval_system.py
"""

import os
import numpy as np
import faiss
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _node_retrieval_one_sample(args):
    """
    工具函数：为一个样本做所有节点的 node-level 检索，便于多进程
    使用 numpy 计算 L1 距离（避免 FAISS 多进程兼容性问题）

    参数:
        args: 包含以下元素的元组
            - neighbor_data_dict: 预计算的邻居数据字典 {node_idx: (neighbor_feats, neighbor_targets)}
            - query_sample: 查询样本 (N, history_window)，已转换为 float32
            - N: 节点数量
            - topk_node: 要检索的邻居节点数量
            - history_window: 历史窗口长度（用于检索）
            - future_window: 未来窗口长度（用于目标）
    返回:
        (node_history, node_target, node_similarities)
    """
    (neighbor_data_dict, query_sample,
     N, topk_node, history_window, future_window) = args

    node_history = np.zeros((N, topk_node, history_window), dtype=np.float32)
    node_target = np.zeros((N, topk_node, future_window), dtype=np.float32)
    node_similarities = np.zeros((N, topk_node), dtype=np.float32)

    for node_idx in range(N):
        # 使用预计算的邻居数据
        if node_idx not in neighbor_data_dict:
            continue

        neighbor_feats, neighbor_targets = neighbor_data_dict[node_idx]

        # 使用 numpy 计算 L1 距离（向量化）
        # neighbor_feats: (n * num_neighbors, history_window) - 已经 flatten 过了
        # query_vec: (history_window,)
        query_vec = query_sample[node_idx:node_idx+1, :]

        # 计算所有候选向量到查询向量的 L1 距离
        # 广播: (n * num_neighbors, history_window) - (history_window,)
        # 结果: (n * num_neighbors, history_window)
        abs_diff = np.abs(neighbor_feats - query_vec)
        distances = np.sum(abs_diff, axis=1)  # (n * num_neighbors,)

        # 使用 argpartition 找 topk（O(n) 时间复杂度）
        actual_topk = min(topk_node, len(distances))
        topk_indices = np.argpartition(distances, actual_topk)[:actual_topk]
        topk_distances = distances[topk_indices]
        sorted_order = np.argsort(topk_distances)
        topk_indices = topk_indices[sorted_order]
        topk_distances = topk_distances[sorted_order]

        # 存储结果
        node_history[node_idx, :actual_topk] = neighbor_feats[topk_indices]
        node_target[node_idx, :actual_topk] = neighbor_targets[topk_indices]
        node_similarities[node_idx, :actual_topk] = -topk_distances

    return (node_history, node_target, node_similarities)


class BatchRetrievalManager:
    """管理检索的分批处理和磁盘缓存

    功能：
    1. 创建临时文件目录
    2. 保存批次检索结果
    3. 合并批次数据
    4. 清理临时文件
    """

    def __init__(self, data_dir, dataset_name):
        """初始化批次管理器

        参数:
            data_dir: 数据目录
            dataset_name: 数据集名称 (train/val/test)
        """
        import time
        timestamp = int(time.time())
        self.temp_dir = os.path.join(data_dir, f"temp_{dataset_name}_{timestamp}")
        self.dataset_name = dataset_name

        # 创建临时目录
        os.makedirs(self.temp_dir, exist_ok=True)

        # 进度日志
        self.progress_log = os.path.join(self.temp_dir, "progress.log")

        # 批次文件列表
        self.node_batch_files = []


    def save_node_batch(self, batch_idx, node_history, node_target, node_similarities):
        """保存节点级检索结果批次

        参数:
            batch_idx: 批次索引
            node_history: 节点历史 (batch_size, N, topk_node, history_window)
            node_target: 节点目标 (batch_size, N, topk_node, future_window)
            node_similarities: 节点相似性 (batch_size, N, topk_node)
        """
        batch_file = os.path.join(self.temp_dir, f"node_batch_{batch_idx}.npz")
        np.savez_compressed(
            batch_file,
            node_history=node_history,
            node_target=node_target,
            node_similarities=node_similarities
        )
        self.node_batch_files.append(batch_file)

        # 记录进度
        with open(self.progress_log, 'a') as f:
            f.write(f"node_batch_{batch_idx}: {len(node_history)} samples\n")

        # 检查内存使用
        # mem_info = get_memory_info()

    def merge_node_batches(self, num_samples, N, topk_node, history_window, future_window):
        """合并节点级批次为完整结果

        参数:
            num_samples: 总样本数
            N: 节点数量
            topk_node: 节点级检索的邻居数
            history_window: 历史窗口长度
            future_window: 未来窗口长度

        返回:
            node_retrieval_results: 字典包含 node_history, node_target, node_similarities
        """

        # 初始化合并数组
        node_history = np.zeros((num_samples, N, topk_node, history_window), dtype=np.float32)
        node_target = np.zeros((num_samples, N, topk_node, future_window), dtype=np.float32)
        node_similarities = np.zeros((num_samples, N, topk_node), dtype=np.float32)

        # 合并批次
        current_idx = 0
        for batch_file in self.node_batch_files:
            batch_data = np.load(batch_file, allow_pickle=True)
            batch_size = len(batch_data['node_history'])

            node_history[current_idx:current_idx + batch_size] = batch_data['node_history']
            node_target[current_idx:current_idx + batch_size] = batch_data['node_target']
            node_similarities[current_idx:current_idx + batch_size] = batch_data['node_similarities']

            current_idx += batch_size
            batch_data.close()

        # 转换维度: (num_samples, topk_node, history_window/future_window, N)
        node_history = np.transpose(node_history, (0, 2, 3, 1))
        node_target = np.transpose(node_target, (0, 2, 3, 1))

        node_retrieval_results = {
            'node_history': node_history,
            'node_target': node_target,
            'node_similarities': node_similarities
        }


        return node_retrieval_results

    def cleanup(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class RetrievalSystem:
    """两阶段检索系统：样本级检索 + 节点级时序检索"""

    def __init__(self, topk_sample=30, topk_node=5, batch_size=100, node_batch_size=25,
                 max_workers=1, faiss_index_path="faiss_global.index",
                 use_multiprocessing=False, enable_batch_processing=True):
        """
        参数:
            topk_sample: 样本级检索的候选样本数量
            topk_node: 节点级检索的邻居节点数量
            batch_size: 批量检索大小
            node_batch_size: 节点级检索批次大小
            max_workers: 最大进程数
            faiss_index_path: FAISS 索引保存/加载路径
            use_multiprocessing: 是否使用多进程（默认False，避免内存爆炸）
            enable_batch_processing: 是否启用分批处理（默认True）
        """
        self.topk_node = topk_node
        self.topk_sample = topk_sample
        self.batch_size = batch_size
        self.node_batch_size = node_batch_size
        self.faiss_index_path = faiss_index_path
        self.enable_batch_processing = enable_batch_processing

        self.max_workers = 1 if not use_multiprocessing else max_workers
        self.use_multiprocessing = use_multiprocessing


        # 结果存储
        self.retrieval_results = {}
        self.train_history = None
        self.train_targets = None
        self.N = None
        self.history_window = None
        self.future_window = None
        self.exclude_window = None

        # 初始化全局索引
        self.global_index = None

    def load_preprocessed_data(self, data_dir):
        """加载预处理后的数据

        参数:
            data_dir: 数据目录
        """

        # 加载训练集
        train_data = np.load(os.path.join(data_dir, "train.npz"))
        self.train_history = train_data['history']
        self.train_targets = train_data['targets']

        # 加载验证集
        val_data = np.load(os.path.join(data_dir, "val.npz"))
        val_history = val_data['history']
        val_targets = val_data['targets']

        # 加载测试集
        test_data = np.load(os.path.join(data_dir, "test.npz"))
        test_history = test_data['history']
        test_targets = test_data['targets']

        # 加载邻接矩阵（从原路径）
        adj_matrix = np.load(os.path.join(data_dir, "adj.npy"))

        # 设置窗口参数
        self.N = self.train_history.shape[2]
        self.history_window = self.train_history.shape[1]
        self.future_window = self.train_targets.shape[1]
        self.exclude_window = self.history_window + self.future_window - 1


        return {
            'train_history': self.train_history,
            'train_targets': self.train_targets,
            'val_history': val_history,
            'val_targets': val_targets,
            'test_history': test_history,
            'test_targets': test_targets,
            'adj_matrix': adj_matrix
        }

    def build_global_faiss(self, save_to_disk=False):
        """构建全局样本级 FAISS 索引 (L1 距离)，支持保存到磁盘"""
        n_samples = len(self.train_history)


        db_flat = self.train_history.reshape(n_samples, -1).astype('float32')
        self.global_index = faiss.IndexFlat(db_flat.shape[1], faiss.METRIC_L1)
        self.global_index.add(db_flat)


        if save_to_disk and self.faiss_index_path is not None:
            faiss.write_index(self.global_index, self.faiss_index_path)

    def load_global_faiss(self):
        """从磁盘加载FAISS 索引"""
        self.global_index = faiss.read_index(self.faiss_index_path)

    def retrieve_topk_samples_batch(self, history, dataset_name):
        """使用批量 FAISS 检索整个数据集

        参数:
            history: 查询样本历史 (num_queries, history_window, N)
            dataset_name: 数据集名称，用于 mask
        返回:
            all_indices: 每个查询样本 topk 样本索引 (num_queries, topk)
            all_distances: 每个查询样本 topk 距离 (num_queries, topk)
        """
        num_queries = len(history)
        all_indices = np.zeros((num_queries, self.topk_sample), dtype=int)
        all_distances = np.zeros((num_queries, self.topk_sample), dtype=float)

        history_flat = history.reshape(num_queries, -1).astype('float32')

        for start in tqdm(range(0, num_queries, self.batch_size), desc=f"[Sample Retrieval] {dataset_name}", mininterval=30):
            end = min(start + self.batch_size, num_queries)
            batch_queries = history_flat[start:end]
            batch_size = end - start

            if dataset_name == 'train':
                k_search = self.topk_sample + self.exclude_window
                D, I = self.global_index.search(batch_queries, k_search)

                query_indices = np.arange(start, end)[:, None]
                starts = np.maximum(0, query_indices - self.exclude_window)
                ends = np.minimum(len(self.train_history), query_indices + self.exclude_window + 1)

                for i in range(batch_size):
                    valid_mask = ~((I[i] >= starts[i]) & (I[i] < ends[i]))
                    valid_idx = I[i][valid_mask]
                    valid_dist = D[i][valid_mask]
                    cnt = min(len(valid_idx), self.topk_sample)
                    all_indices[start + i, :cnt] = valid_idx[:cnt]
                    all_distances[start + i, :cnt] = valid_dist[:cnt]
            else:
                D, I = self.global_index.search(batch_queries, self.topk_sample)
                all_indices[start:end] = I
                all_distances[start:end] = D

        return all_indices, all_distances

    def node_level_faiss_batch(self, similar_history, similar_target, history, adj,
                              dataset_name, batch_manager, node_batch_size=None):
        """节点级时序分批检索

        参数:
            similar_history: 相似样本历史 (num_queries, topk_sample, history_window, N)
            similar_target: 相似样本目标 (num_queries, topk_sample, future_window, N)
            history: 查询样本历史 (num_queries, history_window, N)
            adj: 邻接矩阵 (N, N)
            dataset_name: 数据集名称
            batch_manager: BatchRetrievalManager 实例
            node_batch_size: 节点级检索批次大小
        返回:
            node_retrieval_results: 字典包含 node_history, node_target, node_similarities
        """
        if node_batch_size is None:
            node_batch_size = self.node_batch_size

        num_samples = len(history)
        history_window = self.history_window
        future_window = self.future_window


        # 转换数据维度（一次性完成）
        node_data_cand = np.transpose(similar_history, (0, 1, 3, 2)).astype(np.float32)
        target_data_cand = np.transpose(similar_target, (0, 1, 3, 2)).astype(np.float32)
        query_data = np.transpose(history, (0, 2, 1)).astype(np.float32)

        # 预计算邻居索引
        all_neighbors = [np.where(adj[i] > 0)[0] for i in range(self.N)]

        # 分批处理节点级检索
        batch_idx = 0
        for batch_start in tqdm(range(0, num_samples, node_batch_size),
                               desc=f"[Node Retrieval] {dataset_name}", mininterval=30):
            batch_end = min(batch_start + node_batch_size, num_samples)
            batch_size = batch_end - batch_start

            # 为当前批次预计算邻居数据字典
            batch_neighbor_dicts = []
            for sample_idx in range(batch_start, batch_end):
                neighbor_dict = {}
                for node_idx, neighbors in enumerate(all_neighbors):
                    if len(neighbors) > 0:
                        neighbor_feats = node_data_cand[sample_idx, :, neighbors, :].reshape(-1, history_window)
                        neighbor_targets = target_data_cand[sample_idx, :, neighbors, :].reshape(-1, future_window)
                        neighbor_dict[node_idx] = (neighbor_feats, neighbor_targets)
                batch_neighbor_dicts.append(neighbor_dict)

            # 准备进程池参数（仅当前批次）
            pool_args = [
                (
                    batch_neighbor_dicts[i - batch_start],
                    query_data[i],
                    self.N,
                    self.topk_node,
                    history_window,
                    future_window,
                ) for i in range(batch_start, batch_end)
            ]

            # 执行节点级检索
            if self.use_multiprocessing and batch_size > 1:
                results = []
                pool = Pool(processes=self.max_workers)
                try:
                    results = list(pool.imap(_node_retrieval_one_sample, pool_args))
                finally:
                    pool.close()
                    pool.join()
            else:
                results = [_node_retrieval_one_sample(args) for args in pool_args]

            # 保存批次到磁盘
            batch_node_history = np.zeros((batch_size, self.N, self.topk_node, history_window), dtype=np.float32)
            batch_node_target = np.zeros((batch_size, self.N, self.topk_node, future_window), dtype=np.float32)
            batch_node_similarities = np.zeros((batch_size, self.N, self.topk_node), dtype=np.float32)

            for i, (nh, nt, ns) in enumerate(results):
                batch_node_history[i] = nh
                batch_node_target[i] = nt
                batch_node_similarities[i] = ns

            batch_manager.save_node_batch(
                batch_idx=batch_idx,
                node_history=batch_node_history,
                node_target=batch_node_target,
                node_similarities=batch_node_similarities
            )

            batch_idx += 1

            # 释放内存
            del batch_neighbor_dicts, results, batch_node_history, batch_node_target, batch_node_similarities


        # 合并批次
        node_retrieval_results = batch_manager.merge_node_batches(
            num_samples, self.N, self.topk_node, history_window, future_window
        )

        return node_retrieval_results

    def retrieve_for_dataset(self, history, targets, dataset_name, adj=None):  # TODO: 开关控制“online”or“offline”，区别是验证集和测试集是否允许随数据流加入检索库，注意样本不要有覆盖。
        """两阶段检索主流程（分批流水线，避免内存爆炸）

        参数:
            history: 查询样本历史 (num_queries, history_window, N)
            targets: 查询样本目标 (num_queries, future_window, N)
            dataset_name: 数据集名称
            adj: 邻接矩阵 (N, N)，如果为None则不进行节点级检索
        返回:
            node_retrieval_results: 节点级检索结果
        """
        history = np.asarray(history, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        num_queries = len(history)


        # 初始化批次管理器（如果启用分批处理）
        batch_manager = None
        if self.enable_batch_processing:
            batch_manager = BatchRetrievalManager(
                os.path.dirname(self.faiss_index_path),
                dataset_name
            )

        # 预计算邻居索引（如果需要节点级检索）
        all_neighbors = None
        if adj is not None:
            all_neighbors = [np.where(adj[i] > 0)[0] for i in range(self.N)]

        # 分批流水线：样本级检索 + 节点级检索 + 保存到磁盘
        batch_idx = 0
        for batch_start in tqdm(range(0, num_queries, self.batch_size),
                               desc=f"[Pipeline] {dataset_name}", mininterval=30):
            batch_end = min(batch_start + self.batch_size, num_queries)
            batch_size = batch_end - batch_start

            # Stage 1: 样本级检索（只查当前批次）
            batch_history = history[batch_start:batch_end]
            batch_history_flat = batch_history.reshape(batch_size, -1).astype('float32')

            if dataset_name == 'train':
                k_search = self.topk_sample + self.exclude_window
                D, I = self.global_index.search(batch_history_flat, k_search)

                # 排除时间相邻的样本
                batch_topk_indices = np.zeros((batch_size, self.topk_sample), dtype=int)
                batch_topk_distances = np.zeros((batch_size, self.topk_sample), dtype=float)

                query_indices = np.arange(batch_start, batch_end)[:, None]
                starts = np.maximum(0, query_indices - self.exclude_window)
                ends = np.minimum(len(self.train_history), query_indices + self.exclude_window + 1)

                for i in range(batch_size):
                    valid_mask = ~((I[i] >= starts[i]) & (I[i] < ends[i]))
                    valid_idx = I[i][valid_mask]
                    valid_dist = D[i][valid_mask]
                    cnt = min(len(valid_idx), self.topk_sample)
                    batch_topk_indices[i, :cnt] = valid_idx[:cnt]
                    batch_topk_distances[i, :cnt] = valid_dist[:cnt]
            else:
                batch_topk_distances, batch_topk_indices = self.global_index.search(
                    batch_history_flat, self.topk_sample
                )

            # 加载当前批次的样本级检索结果
            batch_similar_history = self.train_history[batch_topk_indices]  # (batch_size, topk_sample, history_window, N)
            batch_similar_target = self.train_targets[batch_topk_indices]    # (batch_size, topk_sample, future_window, N)

            # Stage 2: 节点级检索（当前批次）
            batch_node_results = None
            if adj is not None:
                # 转换维度
                batch_node_data_cand = np.transpose(batch_similar_history, (0, 1, 3, 2)).astype(np.float32)
                batch_target_data_cand = np.transpose(batch_similar_target, (0, 1, 3, 2)).astype(np.float32)
                batch_query_data = np.transpose(batch_history, (0, 2, 1)).astype(np.float32)

                # 准备进程池参数
                pool_args = []
                for i in range(batch_size):
                    neighbor_dict = {}
                    for node_idx, neighbors in enumerate(all_neighbors):
                        if len(neighbors) > 0:
                            neighbor_feats = batch_node_data_cand[i, :, neighbors, :].reshape(-1, self.history_window)
                            neighbor_targets = batch_target_data_cand[i, :, neighbors, :].reshape(-1, self.future_window)
                            neighbor_dict[node_idx] = (neighbor_feats, neighbor_targets)
                    pool_args.append((
                        neighbor_dict,
                        batch_query_data[i],
                        self.N,
                        self.topk_node,
                        self.history_window,
                        self.future_window,
                    ))

                # 执行节点级检索
                if self.use_multiprocessing and batch_size > 1:
                    results = []
                    pool = Pool(processes=self.max_workers)
                    try:
                        results = list(pool.imap(_node_retrieval_one_sample, pool_args))
                    finally:
                        pool.close()
                        pool.join()
                else:
                    results = [_node_retrieval_one_sample(args) for args in pool_args]

                # 组装批次节点级结果
                batch_node_history = np.zeros((batch_size, self.N, self.topk_node, self.history_window), dtype=np.float32)
                batch_node_target = np.zeros((batch_size, self.N, self.topk_node, self.future_window), dtype=np.float32)
                batch_node_similarities = np.zeros((batch_size, self.N, self.topk_node), dtype=np.float32)

                for i, (nh, nt, ns) in enumerate(results):
                    batch_node_history[i] = nh
                    batch_node_target[i] = nt
                    batch_node_similarities[i] = ns

                batch_node_results = (batch_node_history, batch_node_target, batch_node_similarities)

                # 释放节点级检索的临时数据
                del batch_node_data_cand, batch_target_data_cand, batch_query_data
                del pool_args, results, batch_node_history, batch_node_target, batch_node_similarities

            # 保存批次到磁盘
            if self.enable_batch_processing and batch_manager and batch_node_results is not None:
                batch_manager.save_node_batch(
                    batch_idx=batch_idx,
                    node_history=batch_node_results[0],
                    node_target=batch_node_results[1],
                    node_similarities=batch_node_results[2]
                )

            batch_idx += 1

            # 释放当前批次的样本级数据（关键！）
            del batch_topk_indices, batch_topk_distances
            del batch_similar_history, batch_similar_target
            if batch_node_results is not None:
                del batch_node_results

        # 合并批次结果
        node_retrieval_results = None
        if self.enable_batch_processing and batch_manager and adj is not None:
            node_retrieval_results = batch_manager.merge_node_batches(
                num_queries, self.N, self.topk_node, self.history_window, self.future_window
            )
            batch_manager.cleanup()

        return node_retrieval_results

    def run_full_retrieval_pipeline(self, data_dir, datasets=['train', 'val', 'test']):
        """完整检索流程

        参数:
            data_dir: 数据目录
            datasets: 要处理的数据集列表

        返回:
            retrieval_results: 检索结果字典
        """
        # 加载预处理数据
        data = self.load_preprocessed_data(data_dir)

        # 检查FAISS索引文件
        if os.path.exists(self.faiss_index_path):
            self.load_global_faiss()
            print(f"[DONE] 样本级FAISS索引加载完成。")
        else:
            print(f"[PROCESS] 样本级FAISS索引文件不存在，正在重建并保存至 {self.faiss_index_path} ...")
            self.build_global_faiss(save_to_disk=True)
            print(f"[DONE] 样本级FAISS索引构建并存盘完毕。")

        # 执行检索（每个数据集处理完立即保存）
        for split in datasets:
            print(f"[PROCESS] 检索 {split} ...")
            history = data[f"{split}_history"]
            targets = data[f"{split}_targets"]
            adj = data.get('adj_matrix', None)

            node_results = self.retrieve_for_dataset(history, targets, split, adj)
            self.retrieval_results[split] = node_results


            # 立即保存当前数据集的结果
            if node_results is not None:
                save_path = os.path.join(data_dir, f"{split}_retrieval_results.npz")

                np.savez_compressed(
                    save_path,
                    node_history=node_results['node_history'],
                    node_target=node_results['node_target'],
                    node_similarities=node_results['node_similarities']
                )

                print(f"[SAVE] {split} 节点级结果已保存至 {save_path}")


        return self.retrieval_results
