import math

import torch
import torch.nn as nn


class TemporalConvPool(nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None, kernel_size: int = 3):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        self.output_dim = output_dim
        self.temporal_conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D] or [B, K, T, N, D]
        if x.dim() == 4:
            bsz, steps, nodes, dim = x.shape
            x_conv = x.permute(0, 2, 3, 1).reshape(bsz * nodes, dim, steps)
            x_conv = self.temporal_conv(x_conv)
            x_conv = x_conv.mean(dim=-1).reshape(bsz, nodes, self.output_dim)
            return x_conv

        if x.dim() == 5:
            bsz, topk, steps, nodes, dim = x.shape
            x_conv = x.permute(0, 1, 3, 4, 2).reshape(bsz * topk * nodes, dim, steps)
            x_conv = self.temporal_conv(x_conv)
            x_conv = x_conv.mean(dim=-1).reshape(bsz, topk, nodes, self.output_dim)
            return x_conv

        raise ValueError(f"Unsupported tensor shape for TemporalConvPool: {tuple(x.shape)}")


def _valid_num_heads(hidden_dim: int, preferred_heads: int) -> int:
    for heads in range(min(preferred_heads, hidden_dim), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0).unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, D]
        steps = x.size(2)
        if steps > self.pe.size(2):
            raise ValueError(f"Temporal length {steps} exceeds positional encoding max_len {self.pe.size(2)}")
        return self.dropout(x + self.pe[:, :, :steps, :])


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, D]
        nodes = x.size(1)
        node_ids = torch.arange(nodes, device=x.device)
        return self.dropout(x + self.embedding(node_ids).unsqueeze(0).unsqueeze(2))




class PureGCNLayer(nn.Module):
    """Pure Graph Convolutional Network layer with learnable node embeddings"""
    def __init__(self, hidden_dim: int, num_nodes: int, dropout: float = 0.1):
        super().__init__()
        self.theta = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Learnable node embeddings
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.node_embedding)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, D], adj: [N, N] (already normalized)
        bsz, nodes, steps, dim = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(bsz * steps, nodes, dim)

        # Add learnable node embeddings to each time step
        node_emb_expanded = self.node_embedding.unsqueeze(0).expand(bsz * steps, -1, -1)  # [B*steps, N, D]
        x_with_emb = x_flat + node_emb_expanded

        # Pure GCN operation: A * (X + node_embedding) * theta
        h = torch.matmul(adj.unsqueeze(0), x_with_emb)  # [B*steps, N, D]
        h = self.theta(h)  # [B*steps, N, D]
        h = self.dropout(h)

        # Reshape back and apply normalization
        h = h.reshape(bsz, steps, nodes, dim).transpose(1, 2)
        return self.norm(h)




class AwareTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        padding = (kernel_size - 1) // 2
        self.query_conv = nn.Conv2d(hidden_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.key_conv = nn.Conv2d(hidden_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, D]
        bsz, nodes, steps, _ = x.shape
        q = self.query_conv(x.permute(0, 3, 1, 2))
        k = self.key_conv(x.permute(0, 3, 1, 2))
        q = q.reshape(bsz, self.num_heads, self.head_dim, nodes, steps).permute(0, 3, 1, 4, 2)
        k = k.reshape(bsz, self.num_heads, self.head_dim, nodes, steps).permute(0, 3, 1, 4, 2)
        v = self.value_proj(x).reshape(bsz, nodes, steps, self.num_heads, self.head_dim).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        h = torch.matmul(attn, v)
        h = h.transpose(2, 3).reshape(bsz, nodes, steps, self.hidden_dim)
        return self.out_proj(h)


class ASTGNNEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_nodes: int, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.temporal_attn = AwareTemporalAttention(hidden_dim, num_heads, kernel_size, dropout)
        # Replace with pure GCN with learnable node embeddings
        self.spatial_gcn = PureGCNLayer(hidden_dim, num_nodes, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # ASTGNN encoder block: trend-aware temporal attention, then pure GCN.
        x = x + self.dropout(self.temporal_attn(self.norm1(x)))
        x = x + self.dropout(self.spatial_gcn(self.norm2(x), adj))
        return x


class SpatioTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_nodes: int | None = None,
        num_layers: int = 1,
        num_heads: int = 4,
        kernel_size: int = 3,
        max_len: int = 512,
    ):
        super().__init__()
        if num_nodes is None:
            raise ValueError("SpatioTemporalEncoder requires num_nodes for ASTGNN spatial positional encoding")
        num_heads = _valid_num_heads(hidden_dim, num_heads)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.temporal_pos = TemporalPositionalEncoding(hidden_dim, dropout, max_len)
        self.spatial_pos = SpatialPositionalEncoding(num_nodes, hidden_dim, dropout)
        self.layers = nn.ModuleList(
            [ASTGNNEncoderLayer(hidden_dim, num_heads, num_nodes, dropout, kernel_size) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.temporal_pool = TemporalConvPool(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x: [B, T, N, C]
        h = self.input_proj(x).permute(0, 2, 1, 3)  # [B, N, T, D]
        h = self.spatial_pos(self.temporal_pos(h))
        for layer in self.layers:
            h = layer(h, adj)
        h = self.norm(h).permute(0, 2, 1, 3)  # [B, T, N, D]
        h_node = self.temporal_pool(h)  # [B, N, D]，先时间卷积再平均
        h_graph = h_node.mean(dim=1)  # [B, D]，沿节点维度取平均
        return h, h_node, h_graph
