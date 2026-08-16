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

Variants:
  - TimeZoneAwareAHGNN     — dùng nn.Embedding cho time (Tân, tuần 1)
  - SinusoidalZoneAwareAHGNN — dùng SeasonalTimeEncoder (Bảo, tuần 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════
# BLOCK 0 (MỚI): Seasonal Time Encoder
# Bảo — tuần 2
# ══════════════════════════════════════════════
class SeasonalTimeEncoder(nn.Module):
    """
    Encode thời gian liên tục bằng Sinusoidal + MLP thay vì nn.Embedding.

    Lý do cần thiết:
      - nn.Embedding coi 4 label là 4 điểm rời rạc, không có quan hệ
      - SeasonalTimeEncoder mã hoá tính tuần hoàn thực sự của thời gian:
          8h sáng thứ 2 ≈ 8h sáng thứ 3 (cùng giờ, khác ngày)
          7h30 ≈ 8h00 (gần nhau trong ngày)

    Input:
      time_idx: (B,) — 0=night, 1=normal, 2=rush_morning, 3=rush_evening
    Output:
      (B, embed_dim) — continuous time embedding

    Cách mã hoá:
      1. Map label → giờ đại diện: night=2h, normal=11h, rush_m=8h, rush_e=17h
      2. Map label → ngày đại diện: rush_m/e=2 (weekday), night/normal=4 (any)
      3. Sinusoidal encoding theo giờ và ngày
      4. MLP chiếu → embed_dim
    """

    # Giờ đại diện cho mỗi label: night, normal, rush_morning, rush_evening
    HOUR_MAP = torch.tensor([2.0, 11.0, 8.0, 17.0])
    # Ngày đại diện (0=Mon ... 6=Sun): rush=weekday(2), else=mid(4)
    DAY_MAP = torch.tensor([4.0, 4.0, 2.0, 2.0])

    def __init__(self, embed_dim: int, d_model: int = 32):
        """
        embed_dim: chiều output sau MLP
        d_model  : chiều sinusoidal encoding (nên là bội số của 4)
        """
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim

        # MLP chiếu sinusoidal → embed_dim
        # Input = d_model * 2 (sin giờ + sin ngày, mỗi cái d_model/2 sin + d_model/2 cos)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _sinusoidal(self, values: torch.Tensor, max_val: float) -> torch.Tensor:
        """
        Tạo sinusoidal encoding cho 1 chiều thời gian.
        values : (B,) — giá trị thực (giờ hoặc ngày)
        max_val: chu kỳ (24 cho giờ, 7 cho ngày)
        → (B, d_model)
        """
        device = values.device
        d = self.d_model // 2  # số cặp sin/cos
        i = torch.arange(d, device=device).float()

        # Tần số: sin(2π * value/max_val * k) với k = 1,2,...,d
        freq = (2.0 * math.pi / max_val) * (i + 1)  # (d,)
        angles = values.unsqueeze(1) * freq.unsqueeze(0)  # (B, d)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, d_model)

    def forward(self, time_idx: torch.Tensor) -> torch.Tensor:
        """
        time_idx: (B,) long tensor
        → (B, embed_dim)
        """
        device = time_idx.device

        # Map label → giờ và ngày đại diện
        hour = self.HOUR_MAP.to(device)[time_idx]  # (B,)
        day = self.DAY_MAP.to(device)[time_idx]  # (B,)

        # Sinusoidal encoding cho giờ và ngày
        enc_hour = self._sinusoidal(hour, max_val=24.0)  # (B, d_model)
        enc_day = self._sinusoidal(day, max_val=7.0)  # (B, d_model)

        # Concat và project
        enc = torch.cat([enc_hour, enc_day], dim=-1)  # (B, d_model*2)
        return self.mlp(enc)  # (B, embed_dim)


# ══════════════════════════════════════════════
# BLOCK 1: Zone Embedding (Tân — tuần 1)
# ══════════════════════════════════════════════
class TimeZoneEmbedding(nn.Module):
    """
    Encode multi-hot zone vector Z[v] ∈ {0,1}^K → dense z̃[v] ∈ R^d_z
    Dùng nn.Embedding cho time (discrete).
    Dùng cơ chế gating: z̃ = sigmoid(W_gate * t_emb) * MLP(Z)
    để tránh representation collapse.
    """

    def __init__(self, num_zones: int, embed_dim: int, num_time_labels: int):
        super().__init__()
        self.zone_mlp = nn.Sequential(
            nn.Linear(num_zones, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.time_emb = nn.Embedding(num_time_labels, embed_dim)
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, Z: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        """
        Z       : (N, K)
        time_idx: (B,)
        → (B, N, d_z)
        """
        z_static = self.zone_mlp(Z)
        t_emb = self.time_emb(time_idx)
        gate = self.gate_proj(t_emb)
        return gate.unsqueeze(1) * z_static.unsqueeze(0)


# ══════════════════════════════════════════════
# BLOCK 1b (MỚI): Sinusoidal Zone Embedding
# Bảo — tuần 2
# ══════════════════════════════════════════════
class SinusoidalZoneEmbedding(nn.Module):
    """
    Giống TimeZoneEmbedding nhưng dùng SeasonalTimeEncoder
    thay vì nn.Embedding → encode thời gian liên tục hơn.
    Dùng cơ chế gating: z̃ = sigmoid(W_gate * t_enc) * MLP(Z)
    để tránh representation collapse.
    """

    def __init__(
        self,
        num_zones: int,
        embed_dim: int,
        num_time_labels: int = 4,
        d_model: int = 32,
    ):
        super().__init__()
        self.zone_mlp = nn.Sequential(
            nn.Linear(num_zones, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.time_encoder = SeasonalTimeEncoder(embed_dim, d_model)
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, Z: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        """
        Z       : (N, K)
        time_idx: (B,)
        → (B, N, d_z)
        """
        z_static = self.zone_mlp(Z)
        t_enc = self.time_encoder(time_idx)
        gate = self.gate_proj(t_enc)
        return gate.unsqueeze(1) * z_static.unsqueeze(0)


# ══════════════════════════════════════════════
# BLOCK 2: Zone-Aware Adaptive Adjacency
# ══════════════════════════════════════════════
class TimeZoneAwareAdjacency(nn.Module):
    """
    Học ma trận kề động:
      Ẽ_v   = E_v + W_proj · z̃_v
      Ã_t   = Softmax(ReLU(Ẽ · W_t · Ẽᵀ))
      A_final = α·Ã_t + (1-α)·A_osrm
    """

    def __init__(
        self,
        num_nodes: int,
        node_embed_dim: int,
        zone_embed_dim: int,
        num_time_labels: int = 4,
    ):
        super().__init__()
        self.E = nn.Parameter(torch.randn(num_nodes, node_embed_dim) * 0.1)
        self.W_t = nn.Parameter(
            torch.randn(num_time_labels, node_embed_dim, node_embed_dim) * 0.1
        )
        self.W_proj = nn.Linear(zone_embed_dim, node_embed_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, z_embed: torch.Tensor, time_idx: torch.Tensor, A_static=None
    ) -> torch.Tensor:
        """
        z_embed : (B, N, d_z)
        time_idx: (B,)
        → (B, N, N)
        """
        B = time_idx.size(0)
        E_batch = self.E.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_e)
        E_zone = E_batch + self.W_proj(z_embed)  # (B, N, d_e)
        W = self.W_t[time_idx]  # (B, d_e, d_e)
        score = torch.bmm(torch.bmm(E_zone, W), E_zone.transpose(1, 2))
        A_adapt = F.softmax(F.relu(score), dim=-1)

        if A_static is not None:
            alpha = torch.sigmoid(self.alpha)
            A_s = A_static.unsqueeze(0).expand(B, -1, -1)
            return alpha * A_adapt + (1 - alpha) * A_s
        return A_adapt


# ══════════════════════════════════════════════
# BLOCK 3: Zone-Modulated Graph Convolution
# ══════════════════════════════════════════════
class TimeZoneModulatedGraphConv(nn.Module):
    """
    GCN với W_v riêng từng node, phụ thuộc zone embedding.
    W_v = reshape(W_gen · concat(E_v, z̃_v))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        node_embed_dim: int,
        zone_embed_dim: int,
    ):
        super().__init__()
        fused_dim = node_embed_dim + zone_embed_dim
        self.W_gen = nn.Linear(fused_dim, in_channels * out_channels, bias=False)
        self.b_gen = nn.Linear(fused_dim, out_channels, bias=False)
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.N = num_nodes

    def forward(
        self, H: torch.Tensor, A: torch.Tensor, E: torch.Tensor, z_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        H      : (B, N, in_ch)
        A      : (B, N, N)
        E      : (N, d_e)
        z_embed: (B, N, d_z)
        → (B, N, out_ch)
        """
        B = H.size(0)
        E_batch = E.unsqueeze(0).expand(B, -1, -1)
        fused = torch.cat([E_batch, z_embed], dim=-1)  # (B, N, d_e+d_z)
        W_v = self.W_gen(fused).view(B, self.N, self.in_ch, self.out_ch)
        b_v = self.b_gen(fused)  # (B, N, out_ch)
        H_t = torch.einsum("bni,bnio->bno", H, W_v)
        H_agg = torch.bmm(A, H_t)
        return F.relu(H_agg + b_v)


# ══════════════════════════════════════════════
# MODEL 1: TimeZoneAwareAHGNN (Tân — tuần 1)
# ══════════════════════════════════════════════
class TimeZoneAwareAHGNN(nn.Module):
    """Dùng nn.Embedding cho time — variant zone_full_tc"""

    def __init__(
        self,
        num_nodes,
        num_zones,
        in_channels,
        hidden_channels,
        out_channels,
        node_embed_dim=32,
        zone_embed_dim=16,
        num_time_labels=4,
        num_layers=2,
    ):
        super().__init__()
        self.zone_emb = TimeZoneEmbedding(num_zones, zone_embed_dim, num_time_labels)
        self.adj_module = TimeZoneAwareAdjacency(
            num_nodes, node_embed_dim, zone_embed_dim, num_time_labels
        )
        self.conv_layers = nn.ModuleList(
            [
                TimeZoneModulatedGraphConv(
                    in_channels,
                    hidden_channels,
                    num_nodes,
                    node_embed_dim,
                    zone_embed_dim,
                )
            ]
            + [
                TimeZoneModulatedGraphConv(
                    hidden_channels,
                    hidden_channels,
                    num_nodes,
                    node_embed_dim,
                    zone_embed_dim,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, X, Z, time_idx, A_static=None):
        z_embed = self.zone_emb(Z, time_idx)  # (B, N, d_z)
        
        use_zone_adj = getattr(self, "use_zone_adj", True)
        use_zone_weight = getattr(self, "use_zone_weight", True)

        z_embed_adj = z_embed if use_zone_adj else (z_embed * 0.0)
        z_embed_conv = z_embed if use_zone_weight else (z_embed * 0.0)

        A = self.adj_module(z_embed_adj, time_idx, A_static)
        E = self.adj_module.E
        H = X
        for conv in self.conv_layers:
            H = conv(H, A, E, z_embed_conv)
        return self.fc(H)  # (B, N, T_out)


# ══════════════════════════════════════════════
# MODEL 2: SinusoidalZoneAwareAHGNN (Bảo — tuần 2)
# variant: zone_full_sinc
# ══════════════════════════════════════════════
class SinusoidalZoneAwareAHGNN(nn.Module):
    """
    Giống TimeZoneAwareAHGNN nhưng thay TimeZoneEmbedding
    bằng SinusoidalZoneEmbedding — encode thời gian liên tục.

    Kỳ vọng cải thiện:
      - Cosine similarity giữa các zone khác nhau thấp hơn
      - MAPE thấp hơn zone_full_tc
      - Phân biệt tốt hơn giữa 8h thứ 2 và 8h thứ 7
    """

    def __init__(
        self,
        num_nodes,
        num_zones,
        in_channels,
        hidden_channels,
        out_channels,
        node_embed_dim=32,
        zone_embed_dim=16,
        num_time_labels=4,
        num_layers=2,
        d_model=32,
    ):
        super().__init__()
        # SinusoidalZoneEmbedding thay vì TimeZoneEmbedding
        self.zone_emb = SinusoidalZoneEmbedding(
            num_zones, zone_embed_dim, num_time_labels, d_model
        )
        self.adj_module = TimeZoneAwareAdjacency(
            num_nodes, node_embed_dim, zone_embed_dim, num_time_labels
        )
        self.conv_layers = nn.ModuleList(
            [
                TimeZoneModulatedGraphConv(
                    in_channels,
                    hidden_channels,
                    num_nodes,
                    node_embed_dim,
                    zone_embed_dim,
                )
            ]
            + [
                TimeZoneModulatedGraphConv(
                    hidden_channels,
                    hidden_channels,
                    num_nodes,
                    node_embed_dim,
                    zone_embed_dim,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, X, Z, time_idx, A_static=None):
        z_embed = self.zone_emb(Z, time_idx)  # (B, N, d_z)
        
        use_zone_adj = getattr(self, "use_zone_adj", True)
        use_zone_weight = getattr(self, "use_zone_weight", True)

        z_embed_adj = z_embed if use_zone_adj else (z_embed * 0.0)
        z_embed_conv = z_embed if use_zone_weight else (z_embed * 0.0)

        A = self.adj_module(z_embed_adj, time_idx, A_static)
        E = self.adj_module.E
        H = X
        for conv in self.conv_layers:
            H = conv(H, A, E, z_embed_conv)
        return self.fc(H)  # (B, N, T_out)


# ══════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)

    N = 17
    K = 8
    B = 32
    T_in = 12
    F_feat = 4
    T_out = 3

    X = torch.randn(B, N, T_in * F_feat)
    Z = torch.randint(0, 2, (N, K)).float()
    time_idx = torch.randint(0, 4, (B,))
    A_static = torch.rand(N, N)
    A_static = A_static / A_static.sum(dim=-1, keepdim=True)

    print("=" * 55)
    print("  SeasonalTimeEncoder Test")
    print("=" * 55)
    enc = SeasonalTimeEncoder(embed_dim=16, d_model=32)
    out_enc = enc(time_idx)
    print(f"  Input time_idx shape : {time_idx.shape}")
    print(f"  Output encoding shape: {out_enc.shape}")

    # Kiểm tra: rush_morning (2) và night (0) có encoding khác nhau không?
    t_night = torch.tensor([0])
    t_rush = torch.tensor([2])
    e_night = enc(t_night)
    e_rush = enc(t_rush)
    cos_sim = F.cosine_similarity(e_night, e_rush).item()
    print(f"  Cosine(night, rush_morning): {cos_sim:.4f}")
    print(f"  {'✅ Phân biệt tốt' if cos_sim < 0.9 else '⚠️ Quá giống nhau'}")

    print()
    print("=" * 55)
    print("  Model Comparison: TimeZone vs Sinusoidal")
    print("=" * 55)

    cfg = dict(
        num_nodes=N,
        num_zones=K,
        in_channels=T_in * F_feat,
        hidden_channels=64,
        out_channels=T_out,
        node_embed_dim=32,
        zone_embed_dim=16,
        num_time_labels=4,
        num_layers=2,
    )

    models = {
        "zone_full_tc   (Tân)": TimeZoneAwareAHGNN(**cfg),
        "zone_full_sinc (Bảo)": SinusoidalZoneAwareAHGNN(**cfg, d_model=32),
    }

    for name, model in models.items():
        out = model(X, Z, time_idx, A_static)
        n_p = sum(p.numel() for p in model.parameters())
        assert out.shape == (B, N, T_out)

        # Cosine similarity giữa 2 zone khác nhau
        with torch.no_grad():
            Z_test = torch.zeros(N, K)
            Z_test[0, 0] = 1.0  # commercial
            Z_test[1, 1] = 1.0  # residential
            Z_test[1, 4] = 1.0  # + university
            z_e = model.zone_emb(Z_test, time_idx)  # (B, N, d_z)
            cos = F.cosine_similarity(
                z_e[0, 0].unsqueeze(0), z_e[0, 1].unsqueeze(0)
            ).item()

        print(f"  {name}")
        print(f"    Output: {tuple(out.shape)} | Params: {n_p:,}")
        print(
            f"    Zone cosine (commercial vs uni+res): {cos:.4f} "
            f"{'✅' if cos < 0.9 else '⚠️'}"
        )
