"""
models/baselines.py
===================
3 mô hình baseline để so sánh với Zone-Aware AH-GNN.
Tất cả đều dùng interface thống nhất:

    forward(X, Z=None, time_idx=None, A_static=None) → (B, N, T_out)

Models:
  1. LSTMBaseline    — temporal only, không dùng graph
  2. GCNGRUBaseline  — GCN chuẩn (shared weight) + GRU
  3. STGCNBaseline   — Spatio-Temporal GCN (Yu et al., 2018, simplified)

Chạy: python -c "from models.baselines import *; print('OK')"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════
# BASELINE 1: LSTM (Temporal-only, không graph)
# ══════════════════════════════════════════════
class LSTMBaseline(nn.Module):
    """
    Flatten tất cả node features → LSTM → FC → output.

    Mục đích: Chứng minh rằng cấu trúc đồ thị (GNN) là cần thiết.
    Nếu LSTM tốt hơn GNN thì GNN không có giá trị trên bài toán này.
    """

    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.lstm = nn.LSTM(
            input_size=num_nodes * in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_nodes * out_channels)

    def forward(self, X, Z=None, time_idx=None, A_static=None):
        """
        X: (B, N, in_channels) — đã flatten time vào feature dim
        → out: (B, N, out_channels)
        """
        B, N, F_in = X.shape
        # (B, N*F_in) → unsqueeze thành (B, 1, N*F_in) cho LSTM
        x = X.reshape(B, 1, N * F_in)
        h, _ = self.lstm(x)            # (B, 1, hidden_dim)
        out = self.fc(h[:, -1, :])     # (B, N * out_channels)
        return out.view(B, N, self.out_channels)


# ══════════════════════════════════════════════
# Shared helper: Standard GCN layer (weight sharing)
# ══════════════════════════════════════════════
class _StandardGCN(nn.Module):
    """
    GCN với trọng số dùng chung cho tất cả nodes.
    Đây là chuẩn GCN (Kipf & Welling, 2017).

    H' = σ(Â · H · W)    ← W dùng chung — zone-blind
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        H: (B, N, in_dim)
        A: (N, N) hoặc (B, N, N)
        """
        if A.dim() == 2:
            # (N,N) × (B,N,in) → (B,N,in)  via einsum
            H_agg = torch.einsum("nm,bmd->bnd", A, H)
        else:
            H_agg = torch.bmm(A, H)
        return self.norm(F.relu(self.W(H_agg)))


# ══════════════════════════════════════════════
# BASELINE 2: GCN-GRU (Standard GCN + GRU)
# ══════════════════════════════════════════════
class GCNGRUBaseline(nn.Module):
    """
    GCN chuẩn (không node-specific, không adaptive adj) + GRU temporal.

    Mục đích: So sánh với Zone-Aware model — chứng minh rằng:
      (a) Adaptive adjacency là cần thiết (GCN fixed adj vs AH-GNN)
      (b) Node-specific weight là cần thiết (shared W vs ZM-Conv)
    """

    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.gcn1 = _StandardGCN(in_channels, hidden_dim)
        self.gcn2 = _StandardGCN(hidden_dim, hidden_dim)
        self.gru  = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, out_channels)
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

    def forward(self, X, Z=None, time_idx=None, A_static=None):
        """
        X:        (B, N, in_channels)
        A_static: (N, N) OSRM adjacency — cố định, không adaptive
        → out:    (B, N, out_channels)
        """
        B, N, _ = X.shape

        # Nếu không có A_static thì dùng identity
        if A_static is None:
            A = torch.eye(N, device=X.device)
        else:
            A = A_static

        # Spatial convolution (2 layers GCN)
        h = self.gcn1(X, A)   # (B, N, hidden_dim)
        h = self.gcn2(h, A)   # (B, N, hidden_dim)

        # Temporal: GRU per node
        # (B, N, hidden) → (B*N, 1, hidden) → GRU → (B*N, 1, hidden) → (B, N, hidden)
        h_r = h.reshape(B * N, 1, self.hidden_dim)
        h_gru, _ = self.gru(h_r)
        h_out = h_gru[:, -1, :].reshape(B, N, self.hidden_dim)

        return self.fc(h_out)    # (B, N, out_channels)


# ══════════════════════════════════════════════
# BASELINE 3: STGCN (Spatio-Temporal GCN, simplified)
# ══════════════════════════════════════════════
class _TemporalConv(nn.Module):
    """Temporal convolution 1D trên chiều node features."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=kernel_size // 2)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, channels) → conv trên chiều T
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(F.relu(out))


class STGCNBaseline(nn.Module):
    """
    Spatio-Temporal GCN kiểu sandwich: TemporalConv → GCN → TemporalConv.
    Simplified version của Yu et al. (2018).

    Mục đích: Benchmark với kiến trúc ST-GCN state-of-the-art.
    Không có adaptive adjacency và không có zone awareness.
    """

    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.tconv1     = _TemporalConv(hidden_dim)
        self.gcn        = _StandardGCN(hidden_dim, hidden_dim)
        self.tconv2     = _TemporalConv(hidden_dim)
        self.fc         = nn.Linear(hidden_dim, out_channels)

    def forward(self, X, Z=None, time_idx=None, A_static=None):
        """
        X:        (B, N, in_channels)
        A_static: (N, N) OSRM adjacency — cố định
        → out:    (B, N, out_channels)
        """
        B, N, _ = X.shape

        if A_static is None:
            A = torch.eye(N, device=X.device)
        else:
            A = A_static

        # Project input
        h = self.input_proj(X)       # (B, N, hidden_dim)

        # Temporal conv 1 (coi mỗi node feature theo B như một sequence length-1)
        h = h.unsqueeze(1)           # (B, 1, N, hidden) — fake T dim
        h = h.reshape(B * N, 1, -1) # (B*N, T=1, hidden)
        h = self.tconv1(h)           # (B*N, 1, hidden)
        h = h.reshape(B, N, -1)      # (B, N, hidden)

        # Spatial GCN
        h = self.gcn(h, A)           # (B, N, hidden)

        # Temporal conv 2
        h = h.reshape(B * N, 1, -1)
        h = self.tconv2(h)
        h = h[:, -1, :].reshape(B, N, -1)  # (B, N, hidden)

        return self.fc(h)            # (B, N, out_channels)


# ══════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, F_in, T_out = 32, 17, 48, 3   # 48 = T_in(12) * F(4)

    X        = torch.randn(B, N, F_in)
    Z        = torch.randint(0, 2, (N, 8)).float()
    time_idx = torch.randint(0, 4, (B,))
    A        = torch.rand(N, N)
    A        = A / A.sum(dim=-1, keepdim=True)

    models = {
        "LSTMBaseline":    LSTMBaseline(N, F_in, T_out),
        "GCNGRUBaseline":  GCNGRUBaseline(N, F_in, T_out),
        "STGCNBaseline":   STGCNBaseline(N, F_in, T_out),
    }

    print("=" * 50)
    print("  Baseline Model Sanity Check")
    print("=" * 50)
    for name, model in models.items():
        out = model(X, Z, time_idx, A)
        n_params = sum(p.numel() for p in model.parameters())
        assert out.shape == (B, N, T_out), f"Shape mismatch: {out.shape}"
        print(f"  ✅ {name:<20} output={tuple(out.shape)}  params={n_params:,}")
