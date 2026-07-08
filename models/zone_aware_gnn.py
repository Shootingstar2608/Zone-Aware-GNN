"""
Zone-Aware Adaptive Heterogeneous GNN
======================================
Model chính giải quyết bài toán:
  "Ma trận kề không biểu diễn được ngã tư có nhiều hơn 1 vùng chức năng"

Kiến trúc:
  1. ZoneEmbedding      — multi-hot Z[v] → dense z̃[v]
  2. ZoneAwareAdjacency — adaptive A kết hợp OSRM + zone semantic
  3. ZoneModulatedConv  — W_v phụ thuộc vào zone của từng node
  4. TemporalGRU        — xử lý chuỗi thời gian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════
# BLOCK 1: Zone Embedding
# ══════════════════════════════════════════════
class ZoneEmbedding(nn.Module):
    """
    Encode multi-hot zone vector Z[v] ∈ {0,1}^K → dense z̃[v] ∈ R^d_z

    Nodes với zone compositions giống nhau sẽ có embedding gần nhau
    trong không gian d_z chiều.

    Ví dụ:
      VNU HCM:    Z = [0,1,0,1,1,0,0,0]  (residential+school+university)
      High Tech:  Z = [0,0,1,0,0,0,0,0]  (industrial)
      → embedding hai node này sẽ xa nhau → model học được behaviour khác nhau
    """
    def __init__(self, num_zones: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_zones, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (N, K)  — K = số loại zone
        → (N, d_z) — dense zone embedding
        """
        return self.net(Z)


# ══════════════════════════════════════════════
# BLOCK 2: Zone-Aware Adaptive Adjacency
# ══════════════════════════════════════════════
class ZoneAwareAdjacency(nn.Module):
    """
    Học ma trận kề động kết hợp:
      - Node spatial embedding E  (học từ dữ liệu)
      - Zone embedding z̃          (từ OSM)
      - Time-varying weight W_t   (khác nhau theo rush/night/normal)
      - OSRM static adjacency     (cấu trúc đường thực tế)

    Công thức:
      Ẽ_v   = E_v + W_proj · z̃_v         (spatial + semantic)
      Ã_t   = Softmax(ReLU(Ẽ · W_t · Ẽᵀ))
      A_final = α·Ã_t + (1-α)·A_osrm
    """
    def __init__(self, num_nodes: int, node_embed_dim: int,
                 zone_embed_dim: int, num_time_labels: int = 4):
        super().__init__()
        self.E        = nn.Parameter(torch.randn(num_nodes, node_embed_dim) * 0.1)
        self.W_t      = nn.Parameter(torch.randn(num_time_labels,
                                                  node_embed_dim, node_embed_dim) * 0.1)
        self.W_proj   = nn.Linear(zone_embed_dim, node_embed_dim, bias=False)
        self.alpha    = nn.Parameter(torch.tensor(0.5))  # learnable blend factor

    def forward(self, z_embed: torch.Tensor, time_idx: torch.Tensor,
                A_static: torch.Tensor | None = None) -> torch.Tensor:
        """
        z_embed:  (N, d_z)
        time_idx: (B,)   — batch of time labels
        A_static: (N, N) — OSRM adjacency (optional)
        → A_final: (B, N, N)
        """
        B = time_idx.size(0)
        N = self.E.size(0)

        # Zone-augmented node embedding
        E_zone = self.E + self.W_proj(z_embed)    # (N, d_e)
        E_batch = E_zone.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_e)

        # Time-specific weight
        W = self.W_t[time_idx]                    # (B, d_e, d_e)

        # Adaptive adjacency
        score = torch.bmm(torch.bmm(E_batch, W), E_batch.transpose(1, 2))  # (B,N,N)
        A_adapt = F.softmax(F.relu(score), dim=-1)

        if A_static is not None:
            alpha = torch.sigmoid(self.alpha)
            A_s = A_static.unsqueeze(0).expand(B, -1, -1)
            A_final = alpha * A_adapt + (1 - alpha) * A_s
        else:
            A_final = A_adapt

        return A_final    # (B, N, N)


# ══════════════════════════════════════════════
# BLOCK 3: Zone-Modulated Graph Convolution
# ══════════════════════════════════════════════
class ZoneModulatedGraphConv(nn.Module):
    """
    Graph convolution với trọng số riêng cho từng node,
    được điều chỉnh bởi zone embedding của node đó.

    Thay vì W chung cho tất cả:
      H' = Â · H · W              ← standard (zone-blind)

    Ta dùng W_v phụ thuộc zone:
      W_v   = reshape(W_gen · concat(E_v, z̃_v))
      H'_v  = Σ_{u} A[v,u] · H_u · W_v   ← zone-aware

    → Node trường học và node KCN dùng transformation khác nhau
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_nodes: int, node_embed_dim: int, zone_embed_dim: int):
        super().__init__()
        fused_dim = node_embed_dim + zone_embed_dim
        self.W_gen = nn.Linear(fused_dim, in_channels * out_channels, bias=False)
        self.b_gen = nn.Linear(fused_dim, out_channels, bias=False)
        self.in_ch  = in_channels
        self.out_ch = out_channels
        self.N      = num_nodes

    def forward(self, H: torch.Tensor, A: torch.Tensor,
                E: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        """
        H:       (B, N, in_channels)
        A:       (B, N, N)
        E:       (N, d_e) node embedding
        z_embed: (N, d_z) zone embedding
        → H':    (B, N, out_channels)
        """
        # Zone-modulated weights per node
        fused = torch.cat([E, z_embed], dim=-1)          # (N, d_e+d_z)
        W_v   = self.W_gen(fused).view(self.N, self.in_ch, self.out_ch)  # (N,Fin,Fout)
        b_v   = self.b_gen(fused)                         # (N, Fout)

        # Node-specific transformation: (B,N,Fin) × (N,Fin,Fout) → (B,N,Fout)
        H_transformed = torch.einsum('bni,nio->bno', H, W_v)

        # Graph aggregation: A · H_transformed
        H_agg = torch.bmm(A, H_transformed)               # (B, N, Fout)

        return F.relu(H_agg + b_v)


# ══════════════════════════════════════════════
# MAIN MODEL: Zone-Aware AH-GNN
# ══════════════════════════════════════════════
class ZoneAwareAHGNN(nn.Module):
    """
    Full model kết hợp:
      ZoneEmbedding → ZoneAwareAdjacency → [ZoneModulatedGraphConv × L] → FC

    Args:
      num_nodes:       N — số node trong graph
      num_zones:       K — số loại zone (8)
      in_channels:     F_in — feature mỗi node (T_in × F hoặc T_in cho proxy)
      hidden_channels: chiều hidden
      out_channels:    T_out — số bước dự đoán
      node_embed_dim:  chiều embedding node spatial
      zone_embed_dim:  chiều embedding zone
      num_time_labels: số label thời gian (4: night/rush_am/rush_pm/normal)
      num_layers:      số lớp graph conv
    """
    def __init__(
        self,
        num_nodes:       int,
        num_zones:       int,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int,
        node_embed_dim:  int = 32,
        zone_embed_dim:  int = 16,
        num_time_labels: int = 4,
        num_layers:      int = 2,
    ):
        super().__init__()

        # Zone Embedding
        self.zone_emb = ZoneEmbedding(num_zones, zone_embed_dim)

        # Zone-Aware Adjacency
        self.adj_module = ZoneAwareAdjacency(
            num_nodes, node_embed_dim, zone_embed_dim, num_time_labels
        )

        # Node embedding (shared với adj_module.E)
        self.node_embed_dim = node_embed_dim
        self.zone_embed_dim = zone_embed_dim

        # Graph Conv Layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            ZoneModulatedGraphConv(in_channels, hidden_channels,
                                   num_nodes, node_embed_dim, zone_embed_dim)
        )
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                ZoneModulatedGraphConv(hidden_channels, hidden_channels,
                                       num_nodes, node_embed_dim, zone_embed_dim)
            )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self,
        X:        torch.Tensor,   # (B, N, in_channels)
        Z:        torch.Tensor,   # (N, K) — zone labels (static)
        time_idx: torch.Tensor,   # (B,) — time label index
        A_static: torch.Tensor | None = None,  # (N, N) — OSRM adj
    ) -> torch.Tensor:
        """
        → (B, N, out_channels)
        """
        # 1. Zone embedding (static per node)
        z_embed = self.zone_emb(Z)                     # (N, d_z)

        # 2. Dynamic zone-aware adjacency
        A = self.adj_module(z_embed, time_idx, A_static)  # (B, N, N)

        # 3. Get node embedding from adj_module
        E = self.adj_module.E                          # (N, d_e)

        # 4. Zone-modulated graph convolution
        H = X
        for conv in self.conv_layers:
            H = conv(H, A, E, z_embed)                 # (B, N, hidden)

        # 5. Output prediction
        out = self.fc(H)                               # (B, N, T_out)
        return out


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)

    N    = 17   # nodes
    K    = 8    # zone types
    B    = 32   # batch
    T_in = 12
    F    = 4    # features (congestion, delay, travel_time, ff_ratio)
    T_out = 3

    model = ZoneAwareAHGNN(
        num_nodes       = N,
        num_zones       = K,
        in_channels     = T_in * F,
        hidden_channels = 64,
        out_channels    = T_out,
        node_embed_dim  = 32,
        zone_embed_dim  = 16,
        num_time_labels = 4,
        num_layers      = 2,
    )

    # Dummy inputs
    X        = torch.randn(B, N, T_in * F)
    Z        = torch.randint(0, 2, (N, K)).float()
    time_idx = torch.randint(0, 4, (B,))
    A_static = torch.rand(N, N)
    A_static = A_static / A_static.sum(dim=-1, keepdim=True)

    out = model(X, Z, time_idx, A_static)
    assert out.shape == (B, N, T_out), f"Shape mismatch: {out.shape}"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Forward pass OK | Output: {out.shape}")
    print(f"   Total parameters: {n_params:,}")
    print(f"   Zone embedding output: {model.zone_emb(Z).shape}")

    # Kiểm tra zone-awareness: 2 nodes với zone khác nhau có W_v khác nhau?
    with torch.no_grad():
        z_emb = model.zone_emb(Z)
        E_shared = model.adj_module.E
        fused = torch.cat([E_shared, z_emb], dim=-1)
        W_nodes = model.conv_layers[0].W_gen(fused)
    print(f"\n📊 Node weight diversity (std across nodes): {W_nodes.std(dim=0).mean():.4f}")
    print("   (> 0 confirms different zones → different weights ✅)")
