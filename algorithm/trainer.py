import pickle

import torch
from utils.metric import Metrics


def move_batch_to_device(batch, device):
    x, y, ret_x, ret_y, ret_s = batch
    return {
        "x": x.to(device, non_blocking=True),
        "y": y.to(device, non_blocking=True),
        "ret_x": ret_x.to(device, non_blocking=True),
        "ret_y": ret_y.to(device, non_blocking=True),
        "ret_s": ret_s.to(device, non_blocking=True),
    }


def denormalize_tensor(tensor, mean, std):
    if mean is None or std is None:
        return tensor

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    return tensor * std + mean


def denormalized_mae(pred, target, mean, std):
    pred = denormalize_tensor(pred, mean, std)
    target = denormalize_tensor(target, mean, std)
    return Metrics.mae(pred, target)


def additional_point_metrics(predictions, mean, std):
    pred = denormalize_tensor(torch.as_tensor(predictions["pred_mean"], dtype=torch.float64), mean, std)
    target = denormalize_tensor(torch.as_tensor(predictions["target"], dtype=torch.float64), mean, std)
    # Aggregate over all samples, horizons and nodes in original units.
    mape = Metrics.mape(pred, target, mask_val=1.0)
    return {
        "rmse": Metrics.rmse(pred, target),
        "mape": mape if torch.any(target >= 1.0).item() else None,
        "mape_unit": "percent",
        "mape_min_target": 1.0,
        "point_metric_aggregation": "global_elements",
    }


@torch.no_grad()
def run_debug_forward(model, data_loader, device):
    model.eval()
    batch = next(iter(data_loader))
    batch = move_batch_to_device(batch, device)
    outputs = model(batch, compute_loss=True)
    return {
        "x": batch["x"].shape,
        "y": batch["y"].shape,
        "ret_x": batch["ret_x"].shape,
        "ret_y": batch["ret_y"].shape,
        "ret_s": batch["ret_s"].shape,
        "pred_mean": outputs["pred_mean"].shape,
        "mu_mem": outputs["mu_mem"].shape,
        "mu_rob": outputs["mu_rob"].shape,
        "gate": outputs["gate"].shape,
        "loss": float(outputs["loss"].item()),
    }


def train_one_epoch(model, data_loader, optimizer, device, norm_mean=None, norm_std=None, max_steps=None):
    model.train()
    total_loss = 0.0
    total_loss_nll = 0.0
    total_loss_sim = 0.0
    total_loss_var = 0.0
    total_loss_mae = 0.0
    total_mae = 0.0
    total_gate = 0.0
    num_steps = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, compute_loss=True)

        optimizer.zero_grad(set_to_none=True)
        outputs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(model.config, "grad_clip", 5.0))
        optimizer.step()

        total_loss += outputs["loss"].item()
        total_loss_nll += outputs["loss_nll"].item()
        total_loss_sim += outputs["loss_sim"].item()
        total_loss_var += outputs["loss_var"].item()
        total_loss_mae += outputs["loss_mae"].item()
        total_mae += denormalized_mae(outputs["pred_mean"].detach(), batch["y"], norm_mean, norm_std)
        total_gate += outputs["gate_raw"].mean().item()
        num_steps += 1

        if max_steps is not None and num_steps >= max_steps:
            break

    if num_steps == 0:
        return {
            "loss": 0.0,
            "loss_nll": 0.0,
            "loss_sim": 0.0,
            "loss_var": 0.0,
            "loss_mae": 0.0,
            "mae": 0.0,
            "gate": 0.0,
        }

    return {
        "loss": total_loss / num_steps,
        "loss_nll": total_loss_nll / num_steps,
        "loss_sim": total_loss_sim / num_steps,
        "loss_var": total_loss_var / num_steps,
        "loss_mae": total_loss_mae / num_steps,
        "mae": total_mae / num_steps,
        "gate": total_gate / num_steps,
    }


@torch.no_grad()
def evaluate(model, data_loader, device, norm_mean=None, norm_std=None, max_steps=None):
    model.eval()
    total_loss = 0.0
    total_loss_nll = 0.0
    total_loss_sim = 0.0
    total_loss_var = 0.0
    total_loss_mae = 0.0
    total_mae = 0.0
    total_gate = 0.0
    num_steps = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, compute_loss=True)

        total_loss += outputs["loss"].item()
        total_loss_nll += outputs["loss_nll"].item()
        total_loss_sim += outputs["loss_sim"].item()
        total_loss_var += outputs["loss_var"].item()
        total_loss_mae += outputs["loss_mae"].item()
        total_mae += denormalized_mae(outputs["pred_mean"], batch["y"], norm_mean, norm_std)
        total_gate += outputs["gate_raw"].mean().item()
        num_steps += 1

        if max_steps is not None and num_steps >= max_steps:
            break

    if num_steps == 0:
        return {
            "loss": 0.0,
            "loss_nll": 0.0,
            "loss_sim": 0.0,
            "loss_var": 0.0,
            "loss_mae": 0.0,
            "mae": 0.0,
            "gate": 0.0,
        }

    return {
        "loss": total_loss / num_steps,
        "loss_nll": total_loss_nll / num_steps,
        "loss_sim": total_loss_sim / num_steps,
        "loss_var": total_loss_var / num_steps,
        "loss_mae": total_loss_mae / num_steps,
        "mae": total_mae / num_steps,
        "gate": total_gate / num_steps,
    }


def run_single_batch_backward(model, data_loader, device, lr, norm_mean=None, norm_std=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch = next(iter(data_loader))
    batch = move_batch_to_device(batch, device)

    outputs = model(batch, compute_loss=True)
    optimizer.zero_grad(set_to_none=True)
    outputs["loss"].backward()
    optimizer.step()

    grad_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += param.grad.detach().pow(2).sum().item()

    return {
        "loss": float(outputs["loss"].item()),
        "mae": denormalized_mae(outputs["pred_mean"].detach(), batch["y"], norm_mean, norm_std),
        "gate_mean": float(outputs["gate_raw"].mean().item()),
        "grad_norm": grad_norm_sq ** 0.5,
    }


def run_smoke_training(model, train_loader, val_loader, device, lr, epochs, max_steps, norm_mean=None, norm_std=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, norm_mean=norm_mean, norm_std=norm_std, max_steps=max_steps
        )
        val_metrics = evaluate(
            model, val_loader, device, norm_mean=norm_mean, norm_std=norm_std, max_steps=max_steps
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_loss_nll": train_metrics["loss_nll"],
                "train_loss_sim": train_metrics["loss_sim"],
                "train_loss_var": train_metrics["loss_var"],
                "train_loss_mae": train_metrics["loss_mae"],
                "train_mae": train_metrics["mae"],
                "train_gate": train_metrics["gate"],
                "val_loss": val_metrics["loss"],
                "val_loss_nll": val_metrics["loss_nll"],
                "val_loss_sim": val_metrics["loss_sim"],
                "val_loss_var": val_metrics["loss_var"],
                "val_loss_mae": val_metrics["loss_mae"],
                "val_mae": val_metrics["mae"],
                "val_gate": val_metrics["gate"],
            }
        )

    return history


@torch.no_grad()
def predict(model, data_loader, device, max_steps=None):
    model.eval()
    preds = []
    trues = []
    gates = []
    num_steps = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, compute_loss=False)
        preds.append(outputs["pred_mean"].cpu())
        trues.append(batch["y"].cpu())
        gates.append(outputs["gate_raw"].cpu())
        num_steps += 1

        if max_steps is not None and num_steps >= max_steps:
            break

    return {
        "pred_mean": torch.cat(preds, dim=0).numpy(),
        "target": torch.cat(trues, dim=0).numpy(),
        "gate": torch.cat(gates, dim=0).numpy(),
    }


def save_checkpoint(model, optimizer, epoch, best_val_loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(model, path, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def fit(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            config.device,
            norm_mean=config.normalization.mean,
            norm_std=config.normalization.std,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            config.device,
            norm_mean=config.normalization.mean,
            norm_std=config.normalization.std,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_loss_nll": train_metrics["loss_nll"],
            "train_loss_sim": train_metrics["loss_sim"],
            "train_loss_var": train_metrics["loss_var"],
            "train_loss_mae": train_metrics["loss_mae"],
            "train_mae": train_metrics["mae"],
            "train_gate": train_metrics["gate"],
            "val_loss": val_metrics["loss"],
            "val_loss_nll": val_metrics["loss_nll"],
            "val_loss_sim": val_metrics["loss_sim"],
            "val_loss_var": val_metrics["loss_var"],
            "val_loss_mae": val_metrics["loss_mae"],
            "val_mae": val_metrics["mae"],
            "val_gate": val_metrics["gate"],
        }
        history.append(record)

        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, config.model_path)
        else:
            patience += 1

        config.logger.write(
            (
                f"[Epoch {epoch:03d}] "
                f"train_loss={record['train_loss']:.6f} "
                f"train_nll={record['train_loss_nll']:.6f} "
                f"train_sim={record['train_loss_sim']:.6f} "
                f"train_var={record['train_loss_var']:.6f} "
                f"train_mae_loss={record['train_loss_mae']:.6f} "
                f"train_mae={record['train_mae']:.6f} train_gate_val={record['train_gate']:.6f} "
                f"val_loss={record['val_loss']:.6f} "
                f"val_nll={record['val_loss_nll']:.6f} "
                f"val_sim={record['val_loss_sim']:.6f} "
                f"val_var={record['val_loss_var']:.6f} "
                f"val_mae_loss={record['val_loss_mae']:.6f} "
                f"val_mae={record['val_mae']:.6f} val_gate={record['val_gate']:.6f}\n"
            ),
            is_terminal=True,
        )

        if patience >= config.early_stop_patience:
            config.logger.write(
                f"[Early Stop] no improvement for {config.early_stop_patience} epochs. Best epoch={best_epoch}\n",
                is_terminal=True,
            )
            break

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def test_and_save(model, data_loader, config, split_name="test"):
    metrics = evaluate(
        model,
        data_loader,
        config.device,
        norm_mean=config.normalization.mean,
        norm_std=config.normalization.std,
    )
    from .probabilistic_metrics import evaluate_probabilistic
    metrics.update(evaluate_probabilistic(
        model, data_loader, config.device,
        config.normalization.mean, config.normalization.std,
        samples=config.metric_samples, seed=config.metric_seed,
    ))
    predictions = predict(model, data_loader, config.device)
    metrics.update(additional_point_metrics(
        predictions, config.normalization.mean, config.normalization.std,
    ))
    metrics["checkpoint"] = config.model_path
    import json
    from pathlib import Path
    Path(config.forecast_dir, "probabilistic_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with open(config.forecast_path, "wb") as file:
        pickle.dump(
            {
                "split": split_name,
                "metrics": metrics,
                "pred_mean": predictions["pred_mean"],
                "target": predictions["target"],
                "gate": predictions["gate"],
            },
            file,
        )

    config.logger.write(
        f"[{split_name.upper()}] loss={metrics['loss']:.6f} nll={metrics['loss_nll']:.6f} sim={metrics['loss_sim']:.6f} var={metrics['loss_var']:.6f} mae_loss={metrics['loss_mae']:.6f} mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} mape_percent={metrics['mape']} crps={metrics['crps']:.6f} es={metrics['es']:.6f} gate_val={metrics['gate']:.6f}\n",
        is_terminal=True,
    )
    return metrics
