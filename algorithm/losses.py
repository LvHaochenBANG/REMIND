import torch

from .distributions import mdn_log_prob


def _binary_alignment_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    target = target.clamp(0.0, 1.0)
    return -(target * torch.log(pred) + (1.0 - target) * torch.log1p(-pred)).mean()


def _normalize_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    x_detached = x.detach()
    x_min = x_detached.amin()
    x_max = x_detached.amax()
    return (x_detached - x_min) / (x_max - x_min + 1e-6)


def compute_remind_loss(
    y: torch.Tensor,
    outputs: dict,
    lambda_gate: float,
    lambda_mono: float,
    lambda_mae: float,
):
    gate_mix = outputs["gate_mix"].clamp(1e-6, 1.0 - 1e-6)
    gate_raw = outputs["gate_raw"].clamp(1e-6, 1.0 - 1e-6)

    log_p_mem = mdn_log_prob(y, outputs["mem_logits"], outputs["mem_mu"], outputs["mem_sigma"])
    log_p_rob = mdn_log_prob(y, outputs["rob_logits"], outputs["rob_mu"], outputs["rob_sigma"])

    log_gate = torch.log(gate_mix)
    log_inv_gate = torch.log1p(-gate_mix)
    stacked = torch.stack([log_gate + log_p_mem, log_inv_gate + log_p_rob], dim=-1)
    log_prob = torch.logsumexp(stacked, dim=-1)
    loss_nll = -log_prob.mean()

    sim_target = outputs["s_bar"].detach()
    var_target = _normalize_to_unit_interval(outputs["c_mem_ref"])

    loss_sim = _binary_alignment_loss(sim_target, gate_raw)
    loss_var = _binary_alignment_loss(var_target, 1.0 - gate_raw)
    loss_mae = torch.abs(outputs["pred_mean"] - y).mean()
    loss = loss_nll + lambda_gate * loss_sim + lambda_mono * loss_var + lambda_mae * loss_mae

    return {
        "loss": loss,
        "loss_nll": loss_nll,
        "loss_sim": loss_sim,
        "loss_var": loss_var,
        "loss_mae": loss_mae,
        # Compatibility keys for the existing trainer/logging code.
        "loss_task": loss_nll,
        "loss_gate": loss_sim,
        "loss_mono": loss_var,
        "log_prob": log_prob,
        "log_p_mem": log_p_mem,
        "log_p_rob": log_p_rob,
        "sim_target": sim_target,
        "var_target": var_target,
    }
