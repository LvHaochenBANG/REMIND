import os
import time
import argparse

import torch
from easydict import EasyDict as edict

from utils.common_utils import Logger, get_workspace


ws = get_workspace()


def get_params():
    parser = argparse.ArgumentParser(description="REMIND configuration")

    parser.add_argument("--run_name", type=str, default="remind", help="Name for this run")
    parser.add_argument("--dataset_dir", type=str, default="./data/PEMS", help="Dataset directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for evaluation")
    parser.add_argument("--metric_samples", type=int, default=100)
    parser.add_argument("--metric_seed", type=int, default=2025)
    parser.add_argument("--topk", type=int, default=None, help="Number of node retrievals to use (default: all cached neighbors)")
    parser.add_argument("--device_index", type=int, default=0, help="CUDA device index")

    parser.add_argument("--history_len", type=int, default=12, help="History window length")
    parser.add_argument("--pred_len", type=int, default=12, help="Prediction horizon length")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--train", action="store_true", help="Run full training")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate a saved checkpoint")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping value")

    parser.add_argument("--debug_forward", action="store_true", help="Run one debug forward pass")
    parser.add_argument("--smoke_test", action="store_true", help="Run a short training smoke test")
    parser.add_argument("--smoke_steps", type=int, default=3, help="Training steps per epoch for smoke test")

    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--memory_mixtures", type=int, default=4, help="Number of memory branch mixtures")
    parser.add_argument("--robust_mixtures", type=int, default=2, help="Number of robust branch mixtures")
    parser.add_argument("--memory_sigma_floor", type=float, default=1e-3, help="Minimum sigma for memory branch")
    parser.add_argument("--robust_sigma_floor", type=float, default=0.1, help="Minimum sigma for robust branch")
    parser.add_argument("--similarity_alignment_weight", type=float, default=0., help="Weight for similarity alignment loss")
    parser.add_argument("--dispersion_alignment_weight", type=float, default=0., help="Weight for dispersion alignment loss")
    parser.add_argument("--mae_weight", type=float, default=30, help="Weight for MAE loss")
    parser.add_argument("--memory_dropout", type=float, default=0.3, help="Dropout strength applied to gate during training")
    parser.add_argument("--retrieval_noise_std", type=float, default=0.00, help="Std of retrieval noise")
    parser.add_argument("--deviation_residual_scale", type=float, default=0.3, help="Scale for deviation residual in robust branch")

    return parser.parse_args()


def default_config(params):
    config = edict()
    config.key = f"{params.run_name}_{time.strftime('%m-%d %H:%M:%S', time.localtime())}_{time.time_ns() % 1000000000:09d}"
    config.output_root = os.path.join(ws, f"output_{config.key}")
    config.model_dir = os.path.join(config.output_root, "model")
    config.log_dir = os.path.join(config.output_root, "log")
    config.forecast_dir = os.path.join(config.output_root, "forecast")

    config.run_name = params.run_name
    config.log_path = os.path.join(config.log_dir, f"{config.run_name}.log")
    config.model_path = os.path.join(config.model_dir, "model.pt")
    config.forecast_path = os.path.join(config.forecast_dir, "forecast.pkl")

    config.checkpoint = params.checkpoint
    config.metric_samples = params.metric_samples
    config.metric_seed = params.metric_seed
    config.topk = params.topk
    if config.topk is not None and config.topk < 1:
        raise ValueError("topk must be positive")
    config.dataset_dir = params.dataset_dir
    config.device = torch.device(
        f"cuda:{params.device_index}" if torch.cuda.is_available() else "cpu"
    )
    config.history_len = params.history_len
    config.pred_len = params.pred_len
    config.batch_size = params.batch_size
    config.learning_rate = params.learning_rate
    config.epochs = params.epochs
    config.debug_forward = params.debug_forward
    config.smoke_test = params.smoke_test
    config.smoke_steps = params.smoke_steps
    config.train = params.train
    config.eval_only = params.eval_only
    config.num_workers = params.num_workers
    config.early_stop_patience = params.early_stop_patience
    config.grad_clip = params.grad_clip

    config.data = edict()
    config.data.name = params.dataset_dir.split("/")[-1]

    dataset_configs = {
        "PEMS08": (1, 170, "flow", "5min", "2016-07-01", None),
        "Seattle": (1, 323, "flow", "1h", "2015-01-01", None),
        "Fresno": (1, 211, "flow", "5min", "2023-04-01", None),
        "PM2.5": (1, 20, "pm25", "1h", "2014-05-01", None),
        "camels-13": (1, 7, "hydrology", "1D", "1980-01-01", None),
        "CAMELS": (1, 7, "hydrology", "1D", "1980-01-01", 0.7),
        "PEMS": (1, 581, "flow", "5min", None, 1.0),
        "PEMS-7years": (1, 581, "flow", "5min", None, 1.0),
        "ENERGY": (1, 103, "wind", "10min", None, 1.7),
    }

    if config.data.name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {config.data.name}")

    (
        config.data.num_features,
        config.data.num_vertices,
        config.data.file_name,
        config.data.freq,
        config.data.start_time,
        config.data.tau,
    ) = dataset_configs[config.data.name]

    config.model_name = "REMIND"
    config.optimizer = "adam"
    config.logger = Logger()

    config.remind = edict(
        hidden_dim=params.hidden_dim,
        dropout=params.dropout,
        pred_len=params.pred_len,
        memory_mixtures=params.memory_mixtures,
        robust_mixtures=params.robust_mixtures,
        memory_sigma_floor=params.memory_sigma_floor,
        robust_sigma_floor=params.robust_sigma_floor,
        similarity_alignment_weight=params.similarity_alignment_weight,
        dispersion_alignment_weight=params.dispersion_alignment_weight,
        mae_weight=params.mae_weight,
        memory_dropout=params.memory_dropout,
        retrieval_noise_std=params.retrieval_noise_std,
        deviation_residual_scale=params.deviation_residual_scale,
        grad_clip=params.grad_clip,
    )

    return config
