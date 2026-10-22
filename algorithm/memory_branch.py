import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TemporalConvPool


class MemoryBranch(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, pred_len: int, n_mixtures: int, sigma_floor: float):
        super().__init__()
        self.encoder = encoder
        self.pred_len = pred_len
        self.n_mixtures = n_mixtures
        self.sigma_floor = sigma_floor

        self.score_feat_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_pool = TemporalConvPool(hidden_dim, hidden_dim)
        self.sim_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.calib_feat_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.calib_pool = TemporalConvPool(hidden_dim, hidden_dim)
        self.a_node_head = nn.Linear(hidden_dim, 1)
        self.b_node_head = nn.Linear(hidden_dim, 1)
        self.a_time_head = nn.Linear(hidden_dim, pred_len)
        self.b_time_head = nn.Linear(hidden_dim, pred_len)
        self.uncertainty_pool = TemporalConvPool(1, 1)

        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len),
        )
        self.mix_logits_head = nn.Linear(hidden_dim * 2 + 1, n_mixtures)
        self.mix_delta_head = nn.Linear(hidden_dim * 2 + 1, n_mixtures * pred_len)
        self.mix_sigma_head = nn.Linear(hidden_dim * 2 + 1, n_mixtures * pred_len)

    def forward(
        self,
        x: torch.Tensor,
        ret_x: torch.Tensor,
        ret_y: torch.Tensor,
        ret_s: torch.Tensor,
        adj: torch.Tensor,
    ):
        # x: [B, T_h, N, 1]
        # ret_x: [B, K, T_h, N, 1]
        # ret_y: [B, K, T_p, N, 1]
        bsz, topk, _, nodes, _ = ret_x.shape

        h_q_seq, h_q_node, _ = self.encoder(x, adj)                                  # [B, T_h, N, D], [B, N, D]

        ret_x_flat = ret_x.reshape(bsz * topk, ret_x.size(2), nodes, ret_x.size(4))
        h_r_seq, h_r_node, _ = self.encoder(ret_x_flat, adj)
        h_r_seq = h_r_seq.reshape(bsz, topk, ret_x.size(2), nodes, -1)               # [B, K, T_h, N, D]
        h_r_node = h_r_node.reshape(bsz, topk, nodes, -1)                            # [B, K, N, D]

        q_expand_seq = h_q_seq.unsqueeze(1).expand(-1, topk, -1, -1, -1)             # [B, K, T_h, N, D]
        s_feat = ret_s.permute(0, 2, 1).unsqueeze(-1)                                 # [B, K, N, 1]

        score_feat = torch.cat(
            [q_expand_seq, h_r_seq, q_expand_seq - h_r_seq, q_expand_seq * h_r_seq],
            dim=-1,
        )                                                                             # [B, K, T_h, N, 4D]
        score_feat = self.score_feat_proj(score_feat)                                 # [B, K, T_h, N, D]
        score_summary = self.score_pool(score_feat)                                   # [B, K, N, D]
        sim_summary = self.sim_proj(s_feat)                                           # [B, K, N, D]
        scores = self.score_head(torch.cat([score_summary, sim_summary], dim=-1)).squeeze(-1)  # [B, K, N]
        alpha_k = torch.softmax(scores, dim=1)                                        # [B, K, N]
        alpha = alpha_k.permute(0, 2, 1)                                              # [B, N, K]

        calib_feat = torch.cat([q_expand_seq, h_r_seq, q_expand_seq - h_r_seq], dim=-1)   # [B, K, T_h, N, 3D]
        calib_feat = self.calib_feat_proj(calib_feat)                                      # [B, K, T_h, N, D]
        calib_summary = self.calib_pool(calib_feat)                                         # [B, K, N, D]
        calib_global = calib_summary.mean(dim=2)                                           # [B, K, D]

        a_node = 1.0 + 0.1 * torch.tanh(self.a_node_head(calib_summary))                   # [B, K, N, 1]
        b_node = self.b_node_head(calib_summary)                                            # [B, K, N, 1]
        a_time = 1.0 + 0.1 * torch.tanh(self.a_time_head(calib_global)).unsqueeze(2)       # [B, K, 1, T_p]
        b_time = self.b_time_head(calib_global).unsqueeze(2)                                # [B, K, 1, T_p]

        ret_y_node_first = ret_y.permute(0, 1, 3, 2, 4).squeeze(-1)                        # [B, K, N, T_p]
        calibrated_node_first = (a_node * a_time) * ret_y_node_first + (b_node + b_time)   # [B, K, N, T_p]
        calibrated = calibrated_node_first.permute(0, 1, 3, 2)                             # [B, K, T_p, N]

        alpha_for_future = alpha.permute(0, 2, 1).unsqueeze(2)                             # [B, K, 1, N]
        mu_mem = (alpha_for_future * calibrated).sum(dim=1)                                # [B, T_p, N]

        c_mem_seq = calibrated.var(dim=1, unbiased=False).unsqueeze(-1)                    # [B, T_p, N, 1]
        c_mem = self.uncertainty_pool(c_mem_seq).squeeze(-1)                               # [B, N]
        # 时间维度做平均
        c_mem_ref = ret_y_node_first.var(dim=1, unbiased=False).mean(dim=2)                # [B, N]

        weighted_h_r = (
            alpha.unsqueeze(-1) * h_r_node.permute(0, 2, 1, 3)
        ).sum(dim=2)                                                                        # [B, N, D]
        mem_context = torch.cat([h_q_node, weighted_h_r, c_mem.unsqueeze(-1)], dim=-1)     # [B, N, 2D+1]

        sigma_node = F.softplus(self.sigma_head(mem_context)) + self.sigma_floor            # [B, N, T_p]
        sigma_mem = sigma_node.permute(0, 2, 1)                                             # [B, T_p, N]

        mem_logits = self.mix_logits_head(mem_context)                                      # [B, N, M]
        delta_mu = self.mix_delta_head(mem_context).reshape(bsz, nodes, self.n_mixtures, self.pred_len)
        mix_sigma = self.mix_sigma_head(mem_context).reshape(bsz, nodes, self.n_mixtures, self.pred_len)

        mem_mu = mu_mem.permute(0, 2, 1).unsqueeze(2) + delta_mu                            # [B, N, M, T_p]
        mem_sigma = F.softplus(mix_sigma) + sigma_node.unsqueeze(2) + self.sigma_floor      # [B, N, M, T_p]

        mem_mu = mem_mu.permute(0, 2, 3, 1)                                                 # [B, M, T_p, N]
        mem_sigma = mem_sigma.permute(0, 2, 3, 1)                                           # [B, M, T_p, N]

        return {
            "alpha": alpha,
            "calibrated_y": calibrated,
            "mu_mem": mu_mem,
            "sigma_mem": sigma_mem,
            "c_mem": c_mem,
            "c_mem_ref": c_mem_ref,
            "mem_logits": mem_logits,
            "mem_mu": mem_mu,
            "mem_sigma": mem_sigma,
            "h_q_node": h_q_node,
            "h_r_node": h_r_node,
            "mem_context": mem_context,
        }
