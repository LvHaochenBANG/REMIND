"""Sampled normalized quantile CRPS and spatial Energy Score on original scale.

Metric convention follows zhangjianhao/Remind-v4-a/utils/metric.py.
Sampling follows the actual gated mixture likelihood, not a weighted sum of draws.
"""
import torch


def sample_forecast(outputs, count, generator):
    gate = outputs['gate_raw'].unsqueeze(-1)
    weights = torch.cat((gate * outputs['mem_logits'].softmax(-1),
                         (1 - gate) * outputs['rob_logits'].softmax(-1)), dim=-1)
    mu = torch.cat((outputs['mem_mu'], outputs['rob_mu']), dim=1)
    sigma = torch.cat((outputs['mem_sigma'], outputs['rob_sigma']), dim=1)
    b, _, t, n = mu.shape
    # One component per node/trajectory, shared across the prediction horizon.
    indices = torch.multinomial(weights.reshape(b * n, -1), count,
                                replacement=True, generator=generator)
    indices = indices.reshape(b, n, count).permute(0, 2, 1).unsqueeze(2).expand(-1, -1, t, -1)
    noise = torch.randn((b, count, t, n), device=mu.device, dtype=mu.dtype, generator=generator)
    return mu.gather(1, indices) + sigma.gather(1, indices) * noise


def score_sums(samples, target):
    """Return additive CRPS numerator/denominator and ES sum/count.

    samples [B,S,T,N], target [B,T,N]. ES uses the empirical S² denominator,
    including diagonal pairs, matching the reference implementation.
    """
    qs = torch.arange(0.05, 1.0, 0.05, device=samples.device)
    numerator = target.new_zeros(())
    for q in qs:
        prediction = torch.quantile(samples, q, dim=1)
        numerator += (2 * ((prediction-target) * ((target <= prediction).to(target.dtype)-q)).abs()).sum() / len(qs)
    first = (samples - target.unsqueeze(1)).norm(dim=-1).mean(dim=1)
    # cdist avoids materializing [B,S,S,T,N].
    vectors = samples.permute(0, 2, 1, 3)
    second = torch.cdist(vectors, vectors, compute_mode='donot_use_mm_for_euclid_dist').mean(dim=(-1, -2)) / 2
    return numerator.item(), target.abs().sum().item(), (first-second).sum().item(), first.numel()


@torch.no_grad()
def evaluate_probabilistic(model, loader, device, mean, std, samples=100, seed=2025):
    if samples < 2:
        raise ValueError('metric_samples must be at least 2')
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(std, dtype=torch.float32, device=device)
    totals = [0., 0., 0., 0]
    for values in loader:
        batch = {k: v.to(device) for k, v in zip(('x','y','ret_x','ret_y','ret_s'), values)}
        output = model(batch, compute_loss=False)
        draws = sample_forecast(output, samples, generator) * std + mean
        target = batch['y'] * std + mean
        totals = [a+b for a,b in zip(totals, score_sums(draws, target))]
    if not totals[3]:
        raise ValueError('Empty test loader')
    return {'crps': totals[0]/totals[1] if totals[1] else float('nan'),
            'es': totals[2]/totals[3], 'metric_samples': samples, 'metric_seed': seed,
            'crps_method': 'normalized_quantile_0.05_to_0.95',
            'es_method': 'spatial_l2_empirical_pairs', 'metric_scale': 'original',
            'sampling_method': 'gated_gaussian_mixture'}
