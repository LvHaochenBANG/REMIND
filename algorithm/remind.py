import torch
import torch.nn as nn

from .encoder import SpatioTemporalEncoder
from .losses import compute_remind_loss
from .memory_branch import MemoryBranch
from .robust_branch import RobustBranch
from .trust_gate import TrustGate


class REMIND(nn.Module):
    def __init__(self, config, adj_matrix):
        super().__init__()
        self.config = config

        adj = torch.as_tensor(adj_matrix, dtype=torch.float32)
        self.register_buffer("adj", adj)
        self.register_buffer("adj_norm", self.normalize_adjacency(adj))

        hidden_dim = config.hidden_dim
        num_nodes = adj.size(0)
        self.memory_encoder = SpatioTemporalEncoder(1, hidden_dim, config.dropout, num_nodes=num_nodes)
        self.robust_encoder = SpatioTemporalEncoder(1, hidden_dim, config.dropout, num_nodes=num_nodes)

        self.memory_branch = MemoryBranch(
            encoder=self.memory_encoder,
            hidden_dim=hidden_dim,
            pred_len=config.pred_len,
            n_mixtures=config.memory_mixtures,
            sigma_floor=config.memory_sigma_floor,
        )
        self.robust_branch = RobustBranch(
            encoder=self.robust_encoder,
            hidden_dim=hidden_dim,
            pred_len=config.pred_len,
            n_mixtures=config.robust_mixtures,
            sigma_floor=config.robust_sigma_floor,
            residual_scale=config.deviation_residual_scale,
        )
        self.trust_gate = TrustGate(hidden_dim)

    @staticmethod
    def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        adj = adj + eye
        degree = adj.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(0)

    def _sample_memory_dropout(self, ret_s: torch.Tensor):
        # ret_s: [B, N, K]
        if not self.training or self.config.memory_dropout <= 0:
            # 如果不训练或无丢弃，返回全1掩码，形状 [B, N]
            return torch.ones_like(ret_s[..., 0])

        # 按 [B, N]，按区间采样随机掩码值
        return torch.empty_like(ret_s[..., 0]).uniform_(1.0 - self.config.memory_dropout, 1.0)

    def _apply_retrieval_noise(self, ret_x: torch.Tensor, rho: torch.Tensor):
        # ret_x: [B, K, T_h, N, 1]
        # rho: [B, N]
        if self.training and self.config.retrieval_noise_std > 0 and self.config.memory_dropout > 0:
            # sigma_eff: [B, N]，加噪标准差
            sigma_eff = self.config.retrieval_noise_std * (1.0 - rho) / self.config.memory_dropout
            # 调整形状 [B, 1, 1, N, 1]，以便与ret_x广播
            sigma_eff = sigma_eff.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
            return ret_x + sigma_eff * torch.randn_like(ret_x)
        return ret_x

    def forward(self, batch: dict, compute_loss: bool = True):
        # x: [B, T_h, N] -> [B, T_h, N, 1]
        x = batch["x"].float().unsqueeze(-1)
        # ret_x: [B, K, T_h, N] -> [B, K, T_h, N, 1]
        ret_x = batch["ret_x"].float().unsqueeze(-1)
        # ret_y: [B, K, T_p, N] -> [B, K, T_p, N, 1]
        ret_y = batch["ret_y"].float().unsqueeze(-1)
        # ret_s: [B, N, K]
        ret_s = batch["ret_s"].float()


        # rho: [B, N] 采样记忆掉落mask
        rho = self._sample_memory_dropout(ret_s)
        # ret_x加入检索噪声，返回 [B, K, T_h, N, 1]
        ret_x = self._apply_retrieval_noise(ret_x, rho)

        # 记忆分支输出
        mem_out = self.memory_branch(x, ret_x, ret_y, ret_s, self.adj_norm)
        # 鲁棒分支输出
        rob_out = self.robust_branch(x, ret_x, ret_s, self.adj_norm)
        # 信任门输出
        gate_out = self.trust_gate(
            ret_s,                           # [B, N, K]
            mem_out["c_mem"],                # [B, N]
            mem_out["c_mem_ref"],            # [B, N]
            rob_out["d_node"],               # [B, N, D]
        )

        # gate_raw: [B, N]
        gate_raw = gate_out["gate"]
        gate_mix = gate_raw
        if self.training and self.config.memory_dropout > 0:
            # gate_mix: [B, N]，按rho缩放
            gate_mix = gate_raw * rho

        # pred_mean: [B, T_p, N]，记忆分支和鲁棒分支加权
        pred_mean = gate_raw.unsqueeze(1) * mem_out["mu_mem"] + (1.0 - gate_raw.unsqueeze(1)) * rob_out["mu_rob"]

        outputs = {
            **mem_out,
            **rob_out,
            **gate_out,
            "ret_s_sorted": ret_s,     # [B, N, K]
            "rho": rho,                # [B, N]
            "gate": gate_raw,          # [B, N]
            "gate_raw": gate_raw,      # [B, N]
            "gate_mix": gate_mix,      # [B, N]
            "pred_mean": pred_mean,    # [B, T_p, N]
        }

        if compute_loss and "y" in batch:
            # y: [B, T_p, N]
            y = batch["y"].float()
            outputs.update(
                compute_remind_loss(
                    y=y,
                    outputs=outputs,
                    lambda_gate=self.config.similarity_alignment_weight,
                    lambda_mono=self.config.dispersion_alignment_weight,
                    lambda_mae=self.config.mae_weight,
                )
            )

        return outputs
