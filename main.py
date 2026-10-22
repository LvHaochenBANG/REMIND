import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from easydict import EasyDict as edict
from utils.common_utils import dir_check, exp_softmax
from config import get_params, default_config
from data_processor import SeriesWindowProcessor, get_npy_file_paths
from run_retrieval import check_existing_retrieval_results, load_retrieval_results
from algorithm import REMIND
from algorithm.trainer import (
    fit,
    load_checkpoint,
    run_debug_forward,
    run_single_batch_backward,
    run_smoke_training,
    test_and_save,
)



def as_float_tensor(array):
    array = np.asarray(array, dtype=np.float32)
    return torch.from_numpy(array)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':
    setup_seed(2025)
    torch.set_num_threads(1)
    param = get_params()
    config = default_config(param)

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.forecast_dir):
        os.makedirs(config.forecast_dir)

    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    # Keep reproducible configuration without logger object memory addresses.
    log_config = {key: value for key, value in config.items() if key != "logger"}
    config.logger.write("[CONFIG] " + json.dumps(log_config, default=str, ensure_ascii=False) + "\n", is_terminal=False)
    
    dir_check(config.model_path)

    dir_check(config.forecast_path)
    print(f"[RUN] {config.run_name} | device={config.device} | output={config.output_root}")
    
    
    # ==== 初始化数据处理类 ====
    adj_matrix_path, data_paths = get_npy_file_paths(config.dataset_dir)
    processor = SeriesWindowProcessor(
        data_paths=data_paths,
        adj_matrix_path=adj_matrix_path,
        history_window=config.history_len,
        future_window=config.pred_len,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    train_x = as_float_tensor(processor.train_history)
    train_y = as_float_tensor(processor.train_targets)
    val_x = as_float_tensor(processor.val_history)
    val_y = as_float_tensor(processor.val_targets)
    test_x = as_float_tensor(processor.test_history)
    test_y = as_float_tensor(processor.test_targets)
    config.normalization = edict(mean=np.asarray(processor.mean, dtype=np.float32), std=np.asarray(processor.std, dtype=np.float32))
    
    existing_files, missing_files = check_existing_retrieval_results(config.dataset_dir)
    if len(existing_files) == 3:
        retrieval_results = load_retrieval_results(config.dataset_dir)
    else:
        config.logger.write(f"[ERROR] Missing retrieval result files: {missing_files}. Please run run_retrieval.py first.\n", is_terminal=True)
        exit(1)
        
    # Cache retrievals are sorted by descending node similarity. Slice before
    # transforming similarities so the weights use precisely the selected K.
    if config.topk is not None:
        for split, result in retrieval_results.items():
            available = result["node_history"].shape[1]
            if available < config.topk:
                raise ValueError(
                    f"{split}: requested topk={config.topk}, cache has {available}. "
                    "Run the dataset script with --prepare to rebuild retrievals."
                )
            result["node_history"] = result["node_history"][:, :config.topk]
            result["node_target"] = result["node_target"][:, :config.topk]
            result["node_similarities"] = result["node_similarities"][..., :config.topk]

    # 训练、验证、测试集的检索结果，并放到 config.device 上
    train_retrieval_x   = as_float_tensor(retrieval_results['train']['node_history'])
    train_retrieval_y   = as_float_tensor(retrieval_results['train']['node_target'])
    train_retrieval_smi = as_float_tensor(exp_softmax(retrieval_results['train']['node_similarities'], config.data.tau))

    val_retrieval_x   = as_float_tensor(retrieval_results['val']['node_history'])
    val_retrieval_y   = as_float_tensor(retrieval_results['val']['node_target'])
    val_retrieval_smi = as_float_tensor(exp_softmax(retrieval_results['val']['node_similarities'], config.data.tau))

    test_retrieval_x   = as_float_tensor(retrieval_results['test']['node_history'])
    test_retrieval_y   = as_float_tensor(retrieval_results['test']['node_target'])
    test_retrieval_smi = as_float_tensor(exp_softmax(retrieval_results['test']['node_similarities'], config.data.tau))

    del retrieval_results
    
    train_dataset = TensorDataset(train_x, train_y, train_retrieval_x, train_retrieval_y, train_retrieval_smi)
    val_dataset = TensorDataset(val_x, val_y, val_retrieval_x, val_retrieval_y, val_retrieval_smi)
    test_dataset = TensorDataset(test_x, test_y, test_retrieval_x, test_retrieval_y, test_retrieval_smi)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        pin_memory=True, num_workers=config.num_workers, persistent_workers=config.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        pin_memory=True, num_workers=config.num_workers, persistent_workers=config.num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        pin_memory=True, num_workers=config.num_workers, persistent_workers=config.num_workers > 0
    )
    
    remind_model = REMIND(config.remind, processor.adj_matrix).to(config.device)
    if config.debug_forward:
        sample_train = next(iter(train_loader))
        print("[DEBUG] train_loader sample shapes:")
        for i, item in enumerate(sample_train):
            print(f"  Item {i}: {item.shape}")

        debug_shapes = run_debug_forward(remind_model, train_loader, config.device)
        print("[DEBUG] REMIND forward shapes:")
        for key, value in debug_shapes.items():
            print(f"  {key}: {value}")

        single_batch = run_single_batch_backward(
            remind_model,
            train_loader,
            config.device,
            config.learning_rate,
            norm_mean=config.normalization.mean,
            norm_std=config.normalization.std,
        )
        print("[DEBUG] single batch backward:")
        for key, value in single_batch.items():
            print(f"  {key}: {value}")

    if config.smoke_test:
        smoke_history = run_smoke_training(
            remind_model,
            train_loader,
            val_loader,
            config.device,
            config.learning_rate,
            config.epochs,
            config.smoke_steps,
            config.normalization.mean,
            config.normalization.std,
        )
        print("[SMOKE] training summary:")
        for record in smoke_history:
            print(
                "  epoch={epoch} train_loss={train_loss:.6f} train_nll={train_loss_nll:.6f} "
                "train_sim={train_loss_sim:.6f} train_var={train_loss_var:.6f} train_mae_loss={train_loss_mae:.6f} "
                "train_mae={train_mae:.6f} val_loss={val_loss:.6f} val_nll={val_loss_nll:.6f} "
                "val_sim={val_loss_sim:.6f} val_var={val_loss_var:.6f} val_mae_loss={val_loss_mae:.6f} "
                "val_mae={val_mae:.6f} val_gate={val_gate:.6f}".format(**record)
            )

    if config.train:
        fit_result = fit(remind_model, train_loader, val_loader, config)
        config.logger.write(
            f"[TRAIN DONE] best_epoch={fit_result['best_epoch']} best_val_loss={fit_result['best_val_loss']:.6f}\n",
            is_terminal=True,
        )
        load_checkpoint(remind_model, config.model_path, map_location=config.device)
        test_and_save(remind_model, test_loader, config, split_name="test")

    if config.eval_only:
        config.model_path = config.checkpoint or config.model_path
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Checkpoint not found: {config.model_path}")
        load_checkpoint(remind_model, config.model_path, map_location=config.device)
        test_and_save(remind_model, test_loader, config, split_name="test")
    
