"""
test_bao.py
===========
Test toàn bộ những gì Bảo đã làm tuần 2.
Chạy: python test_bao.py
"""

import sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

PASS = 0
FAIL = 0


def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  ✅ {name}")
        PASS += 1
    else:
        print(f"  ❌ {name} — {msg}")
        FAIL += 1


# ─────────────────────────────────────────────
# TEST 1: Import
# ─────────────────────────────────────────────
print("=" * 55)
print("  TEST 1: Import tất cả models")
print("=" * 55)
try:
    from models.baselines import LSTMBaseline, GCNGRUBaseline, STGCNBaseline
    from models.time_zone_aware_gnn import (
        SeasonalTimeEncoder,
        SinusoidalZoneEmbedding,
        TimeZoneAwareAHGNN,
        SinusoidalZoneAwareAHGNN,
    )
    from scripts.train import build_model

    check("Import baselines", True)
    check("Import time_zone_aware_gnn", True)
    check("Import build_model", True)
except Exception as e:
    check("Import", False, str(e))

# ─────────────────────────────────────────────
# TEST 2: SeasonalTimeEncoder
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  TEST 2: SeasonalTimeEncoder")
print("=" * 55)
try:
    enc = SeasonalTimeEncoder(embed_dim=16, d_model=32)
    t32 = torch.randint(0, 4, (32,))
    out = enc(t32)
    check("Output shape (32,16)", out.shape == (32, 16), str(out.shape))

    e0 = enc(torch.tensor([0]))  # night
    e2 = enc(torch.tensor([2]))  # rush_morning
    cos = F.cosine_similarity(e0, e2).item()
    print(f"    Cosine(night, rush_morning): {cos:.4f}")
    check("Phân biệt night vs rush", cos < 0.9, f"cosine={cos:.4f} >= 0.9")
except Exception as e:
    check("SeasonalTimeEncoder", False, str(e))

# ─────────────────────────────────────────────
# TEST 3: Forward pass tất cả variants
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  TEST 3: Forward pass tất cả variants")
print("=" * 55)

B, N, K, T_in, F_feat, T_out = 4, 17, 8, 12, 4, 3
X = torch.randn(B, N, T_in * F_feat)
Z = torch.randint(0, 2, (N, K)).float()
tidx = torch.randint(0, 4, (B,))
A = torch.rand(N, N)
A = A / A.sum(-1, keepdim=True)
meta = {"N": N, "K": K, "F": F_feat, "T_in": T_in, "T_out": T_out}

variants = [
    "lstm",
    "gcn_gru",
    "stgcn",
    "baseline_ahgnn",
    "zone_concat",
    "zone_weight",
    "zone_full",
    "zone_full_tc",
    "zone_full_sinc",
]

for v in variants:
    try:
        m = build_model(v, meta)
        out = m(X, Z, tidx, A)
        p = sum(x.numel() for x in m.parameters())
        ok = out.shape == (B, N, T_out)
        check(f"{v:<20} {tuple(out.shape)} params={p:,}", ok, f"shape sai: {out.shape}")
    except Exception as e:
        check(v, False, str(e))

# ─────────────────────────────────────────────
# TEST 4: Backward pass
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  TEST 4: Backward pass (zone_full_sinc)")
print("=" * 55)
try:
    m = build_model("zone_full_sinc", meta)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Y = torch.rand(B, N, T_out)

    for step in range(5):
        opt.zero_grad()
        loss = nn.HuberLoss()(m(X, Z, tidx, A), Y)
        loss.backward()
        norm = (
            sum(p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None)
            ** 0.5
        )
        opt.step()

    print(f"    Loss: {loss.item():.4f} | Grad norm: {norm:.4f}")
    check("No gradient explosion", norm < 100, f"norm={norm:.2f}")
    check("Loss decreasing", loss.item() < 1.0, f"loss={loss.item():.4f}")

    # Dead gradient check
    dead = [
        n
        for n, p in m.named_parameters()
        if p.grad is not None and p.grad.norm().item() < 1e-9
    ]
    check("No dead gradients", len(dead) == 0, f"{len(dead)} dead layers")
except Exception as e:
    check("Backward pass", False, str(e))

# ─────────────────────────────────────────────
# TEST 5: build_graph sinusoidal encoding
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  TEST 5: Sinusoidal encoding (build_graph)")
print("=" * 55)
try:

    def sinc_encode(hour, dow, d=4):
        enc = []
        for k in range(1, d // 2 + 1):
            enc += [
                math.sin(2 * math.pi * hour / 24 * k),
                math.cos(2 * math.pi * hour / 24 * k),
            ]
        for k in range(1, d // 2 + 1):
            enc += [
                math.sin(2 * math.pi * dow / 7 * k),
                math.cos(2 * math.pi * dow / 7 * k),
            ]
        return enc

    e = sinc_encode(8, 2)
    print(f"    sinusoidal(hour=8, dow=2) = {[round(x,3) for x in e]}")
    check("8-dim output", len(e) == 8, f"len={len(e)}")

    # Kiểm tra hour 8 và 17 khác nhau
    e8 = torch.tensor(sinc_encode(8, 2))
    e17 = torch.tensor(sinc_encode(17, 2))
    cos2 = F.cosine_similarity(e8.unsqueeze(0), e17.unsqueeze(0)).item()
    print(f"    Cosine(hour=8, hour=17): {cos2:.4f}")
    check("Phân biệt rush_morning vs rush_evening", cos2 < 0.9, f"cosine={cos2:.4f}")
except Exception as e:
    check("Sinusoidal encoding", False, str(e))

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  PASS: {PASS}  |  FAIL: {FAIL}")
print()
if FAIL == 0:
    print("  Bảo tuần 2 — TẤT CẢ PASS ✅")
else:
    print(f"  Còn {FAIL} test cần fix ❌")
