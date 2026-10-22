import math

import torch


LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-6)
    y = y.unsqueeze(1)
    log_density = -0.5 * (((y - mu) / sigma) ** 2 + 2.0 * torch.log(sigma) + LOG_2PI)
    return log_density.sum(dim=2)


def mdn_log_prob(y: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # y: [B, T, N]
    # logits: [B, N, M]
    # mu/sigma: [B, M, T, N]
    component_log_prob = gaussian_log_prob(y, mu, sigma).permute(0, 2, 1)
    log_mix = torch.log_softmax(logits, dim=-1)
    return torch.logsumexp(log_mix + component_log_prob, dim=-1)
