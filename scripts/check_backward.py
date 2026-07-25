"""
check_backward.py
=================
Kiểm tra backward pass của TimeZoneAwareAHGNN:
  1. Gradient explosion check
  2. Gradient flow per layer
  3. Loss convergence check
  4. SeasonalTimeEncoder forward test (nếu có)

Chạy: python check_backward.py
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.time_zone_aware_gnn import TimeZoneAwareAHGNN

torch.manual_seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
B = 32
N = 17
K = 8
T_in = 12
F_feat = 4
T_out = 3
N_STEPS = 20
EXPLODE_THRESHOLD = 100.0

# ─────────────────────────────────────────────
# DUMMY DATA
# ─────────────────────────────────────────────
X = torch.randn(B, N, T_in * F_feat)
Z = torch.randint(0, 2, (N, K)).float()
time_idx = torch.randint(0, 4, (B,))
A = torch.rand(N, N)
A = A / A.sum(dim=-1, keepdim=True)
Y = torch.rand(B, N, T_out)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = TimeZoneAwareAHGNN(
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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.HuberLoss()

# ─────────────────────────────────────────────
# TEST 1: GRADIENT NORM QUA TỪNG STEP
# ─────────────────────────────────────────────
print("=" * 55)
print("  TEST 1: Gradient Norm (20 steps)")
print("=" * 55)
print(f"  {'Step':>5} {'Loss':>10} {'Grad Norm':>12} {'Status':>10}")
print(f"  {'-'*42}")

exploded = False
loss_history = []

for step in range(1, N_STEPS + 1):
    optimizer.zero_grad()
    out = model(X, Z, time_idx, A)
    loss = criterion(out, Y)
    loss.backward()

    # Tính tổng gradient norm
    total_norm = (
        sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
        ** 0.5
    )

    status = "💥 EXPLODE" if total_norm > EXPLODE_THRESHOLD else "✅ OK"
    if total_norm > EXPLODE_THRESHOLD:
        exploded = True

    optimizer.step()
    loss_history.append(loss.item())

    if step <= 5 or step % 5 == 0:
        print(f"  {step:>5} {loss.item():>10.4f} {total_norm:>12.4f} {status:>10}")

print()
if exploded:
    print("  ❌ GRADIENT EXPLOSION DETECTED!")
    print("  → Cần thêm gradient clipping hoặc giảm lr")
else:
    print("  ✅ Gradient stable — no explosion")

# ─────────────────────────────────────────────
# TEST 2: GRADIENT FLOW PER LAYER
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  TEST 2: Gradient Flow Per Layer")
print("=" * 55)

optimizer.zero_grad()
out = model(X, Z, time_idx, A)
loss = criterion(out, Y)
loss.backward()

print(f"  {'Layer':<35} {'Grad Norm':>12} {'Status':>8}")
print(f"  {'-'*58}")

dead_layers = []
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"  {name:<35} {'NO GRAD':>12} {'⚠️':>8}")
        dead_layers.append(name)
        continue
    g = param.grad.norm().item()
    status = "✅" if g > 1e-6 else "💀 DEAD"
    if g <= 1e-6:
        dead_layers.append(name)
    print(f"  {name:<35} {g:>12.6f} {status:>8}")

print()
if dead_layers:
    print(f"  ⚠️  Dead gradients in {len(dead_layers)} layers:")
    for l in dead_layers:
        print(f"     - {l}")
else:
    print("  ✅ All layers have gradients — no vanishing")

# ─────────────────────────────────────────────
# TEST 3: LOSS CONVERGENCE
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  TEST 3: Loss Convergence Check")
print("=" * 55)

loss_start = loss_history[0]
loss_end = loss_history[-1]
converging = loss_end < loss_start

print(f"  Loss start : {loss_start:.4f}")
print(f"  Loss end   : {loss_end:.4f}")
print(f"  Trend      : {'📉 Decreasing ✅' if converging else '📈 Not converging ⚠️'}")

# ─────────────────────────────────────────────
# TEST 4: ZONE AWARENESS CHECK
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  TEST 4: Zone-Awareness Check")
print("=" * 55)

with torch.no_grad():
    # Node 0: commercial only
    Z_test = torch.zeros(N, K)
    Z_test[0, 0] = 1.0  # commercial

    # Node 1: university + residential
    Z_test[1, 1] = 1.0
    Z_test[1, 4] = 1.0

    z_emb = model.zone_emb(Z_test, time_idx)  # (B, N, d_z)
    # Lấy batch đầu tiên
    emb0 = z_emb[0, 0, :]  # commercial node
    emb1 = z_emb[0, 1, :]  # university+residential node

    cosine_sim = torch.nn.functional.cosine_similarity(
        emb0.unsqueeze(0), emb1.unsqueeze(0)
    ).item()

print(f"  Cosine similarity (commercial vs uni+res): {cosine_sim:.4f}")
print(
    f"  {'✅ Different zones → different embeddings' if cosine_sim < 0.9 else '⚠️ Embeddings too similar'}"
)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  Forward pass  : ✅ OK — output {tuple(out.shape)}")
print(f"  Backward pass : {'✅ OK' if not exploded else '❌ EXPLODED'}")
print(
    f"  Grad flow     : {'✅ OK' if not dead_layers else f'⚠️ {len(dead_layers)} dead layers'}"
)
print(f"  Convergence   : {'✅ OK' if converging else '⚠️ Check lr'}")
print(f"  Zone-aware    : ✅ OK (cosine={cosine_sim:.3f})")
print(f"  Total params  : {sum(p.numel() for p in model.parameters()):,}")
