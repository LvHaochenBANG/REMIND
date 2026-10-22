import torch
import torch.nn as nn


class TrustGate(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ret_s: torch.Tensor,
        c_mem: torch.Tensor,
        c_mem_ref: torch.Tensor,
        d_node: torch.Tensor,
    ):
        s1 = ret_s[..., 0]
        s_bar = ret_s.mean(dim=-1)
        d_norm = torch.norm(d_node, dim=-1)

        gate_input = torch.cat(
            [
                s_bar.unsqueeze(-1),
                s1.unsqueeze(-1),
                c_mem.unsqueeze(-1),
                d_norm.unsqueeze(-1),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_input)).squeeze(-1)

        return {
            "gate": gate,
            "s_bar": s_bar,
            "s1": s1,
            "d_norm": d_norm,
        }
