import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TemporalConvPool


class RobustBranch(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        pred_len: int,
        n_mixtures: int,
        sigma_floor: float,
        residual_scale: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.pred_len = pred_len
        self.n_mixtures = n_mixtures
        self.sigma_floor = sigma_floor
        self.residual_scale = residual_scale

        self.dev_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_pool = TemporalConvPool(hidden_dim, hidden_dim)
        self.base_pool = TemporalConvPool(hidden_dim, hidden_dim)
        self.sigma_pool = TemporalConvPool(hidden_dim, hidden_dim)
        self.base_head = nn.Linear(hidden_dim, pred_len)
        self.dev_head = nn.Linear(hidden_dim, pred_len)
        self.base_sigma_head = nn.Linear(hidden_dim, pred_len)
        self.inflation_head = nn.Linear(hidden_dim, pred_len)
        self.mix_logits_head = nn.Linear(hidden_dim * 2, n_mixtures)
        self.mix_delta_head = nn.Linear(hidden_dim * 2, n_mixtures * pred_len)
        self.mix_sigma_head = nn.Linear(hidden_dim * 2, n_mixtures * pred_len)

    def forward(self, x: torch.Tensor, ret_x: torch.Tensor, ret_s: torch.Tensor, adj: torch.Tensor):
        # x: [B, T, N, C]
        # ret_x: [B, K, T, N, C]
        proto_weights = ret_s / ret_s.sum(dim=-1, keepdim=True).clamp_min(1e-6)     # [B, N, K]
        proto_weights = proto_weights.permute(0, 2, 1).unsqueeze(2).unsqueeze(-1)    # [B, K, 1, N, 1]
        proto_x = (proto_weights * ret_x).sum(dim=1)                                  # [B, T, N, C]

        # 编码器返回
        # h_x_seq: [B, T, N, D]
        # h_x_node: [B, N, D]
        h_x_seq, h_x_node, _ = self.encoder(x, adj)
        h_p_seq, h_p_node, _ = self.encoder(proto_x, adj)

        # dev_input: [B, T, N, D*4]，按最后一维拼接
        dev_input = torch.cat([h_x_seq, h_p_seq, h_x_seq - h_p_seq, h_x_seq * h_p_seq], dim=-1)  # [B, T, N, 4D]
        d_seq = self.dev_proj(dev_input)  # [B, T, N, D]
        d_node = self.temporal_pool(d_seq)  # [B, N, D]，先时间卷积再平均

        base_context = self.base_pool(h_x_seq)  # [B, N, D]
        sigma_context = self.sigma_pool(h_x_seq)  # [B, N, D]

        # 均值主分支 [B, N, pred_len]
        mu_base = self.base_head(base_context)  # [B, N, pred_len]
        # 差异分支 [B, N, pred_len]
        mu_dev = self.dev_head(d_node)  # [B, N, pred_len]
        # 最终均值 [B, pred_len, N]
        mu_rob = (mu_base + self.residual_scale * mu_dev).permute(0, 2, 1)  # [B, pred_len, N]

        # 主分支sigma [B, N, pred_len]
        base_sigma = F.softplus(self.base_sigma_head(sigma_context))  # [B, N, pred_len]
        inflation = F.softplus(self.inflation_head(d_node))      # [B, N, pred_len]
        sigma_rob = (base_sigma + inflation + self.sigma_floor).permute(0, 2, 1)  # [B, pred_len, N]

        # 混合分支
        # 拼接上下文 [B, N, 2D]
        context = torch.cat([base_context, d_node], dim=-1)  # [B, N, 2D]
        rob_logits = self.mix_logits_head(context)  # [B, N, n_mixtures]
        # delta_mu和mix_sigma [B, N, n_mixtures * pred_len] -> [B, N, n_mixtures, pred_len]
        delta_mu = self.mix_delta_head(context).reshape(x.size(0), x.size(2), self.n_mixtures, self.pred_len)  # [B, N, n_mixtures, pred_len]
        mix_sigma = self.mix_sigma_head(context).reshape(x.size(0), x.size(2), self.n_mixtures, self.pred_len)  # [B, N, n_mixtures, pred_len]

        # rob_mu 计算
        # mu_rob: [B, pred_len, N] -> [B, N, pred_len] (permute(0,2,1))
        # unsqueeze(2): [B, N, 1, pred_len]
        # delta_mu: [B, N, n_mixtures, pred_len]
        rob_mu = mu_rob.permute(0, 2, 1).unsqueeze(2) + delta_mu      # [B, N, n_mixtures, pred_len]  # TODO: 考虑直接在上一步输出均值和方差，不使用这种一个值+多个delta的形式。
        rob_sigma = F.softplus(mix_sigma) + sigma_rob.permute(0, 2, 1).unsqueeze(2) + self.sigma_floor  # [B, N, n_mixtures, pred_len]

        # 变为 [B, n_mixtures, pred_len, N]
        rob_mu = rob_mu.permute(0, 2, 3, 1)        # [B, n_mixtures, pred_len, N]
        rob_sigma = rob_sigma.permute(0, 2, 3, 1)  # [B, n_mixtures, pred_len, N]

        return {
            "proto_x": proto_x,
            "d_seq": d_seq,
            "d_node": d_node,
            "mu_rob": mu_rob,
            "sigma_rob": sigma_rob,
            "rob_logits": rob_logits,
            "rob_mu": rob_mu,
            "rob_sigma": rob_sigma,
            "h_x_node": h_x_node,
            "h_p_node": h_p_node,
            "proto_weights": proto_weights.squeeze(2).squeeze(-1).permute(0, 2, 1),
            "rob_context": context,
        }
