"""
train.py
=========
Huấn luyện và đánh giá Zone-Aware AH-GNN.

Chạy:
  python scripts/train.py                  # Chỉ chạy zone_full
  python scripts/train.py --ablation        # 4 ablation variants
  python scripts/train.py --baselines       # 3 baselines (LSTM, GCN-GRU, STGCN)
  python scripts/train.py --all             # Tất cả: ablation + baselines
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add parent dir to path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.zone_aware_gnn import ZoneAwareAHGNN  # fix: bỏ T_out
from models.ah_gnn import AH_GNN
from models.baselines import LSTMBaseline, GCNGRUBaseline, STGCNBaseline
from models.time_zone_aware_gnn import (
    TimeZoneAwareAHGNN,
    SinusoidalZoneAwareAHGNN,
)  # Bảo

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET_PATH = "data/processed/graph_dataset.pt"
META_PATH = "data/processed/meta.json"
OUT_DIR = "data/results"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# ABLATION VARIANTS
# ──────────────────────────────────────────────
ABLATION_VARIANTS = {
    "baseline_ahgnn": (False, False, False),
    "zone_concat": (True, False, False),
    "zone_weight": (True, True, False),
    "zone_full": (True, True, True),
    "zone_full_tc": (True, True, True),  # Tân  — discrete time embedding
    "zone_full_sinc": (True, True, True),  # Bảo  — sinusoidal time encoder
}

BASELINE_NAMES = ["lstm", "gcn_gru", "stgcn"]


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────
def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict:
    """pred, true: (S, N, T_out)"""
    mae = (pred - true).abs().mean().item()
    rmse = ((pred - true) ** 2).mean().sqrt().item()
    mask = true.abs() > 1e-5
    mape = ((pred - true).abs() / (true.abs() + 1e-8))[mask].mean().item() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def compute_zone_stratified_metrics(pred, true, Z, zone_types) -> dict:
    """
    Tính MAE riêng cho từng zone type.
    Đây là metric chính chứng minh zone-awareness hiệu quả.
    """
    results = {}
    Z_np = Z.cpu().numpy()
    for k, zone in enumerate(zone_types):
        node_mask = Z_np[:, k] == 1
        if node_mask.sum() == 0:
            continue
        pred_z = pred[:, node_mask, :]
        true_z = true[:, node_mask, :]
        results[f"MAE_{zone}"] = (pred_z - true_z).abs().mean().item()

    multi_mask = Z_np.sum(axis=1) > 1
    if multi_mask.sum() > 0:
        pred_m = pred[:, multi_mask, :]
        true_m = true[:, multi_mask, :]
        results["MAE_multi_zone"] = (pred_m - true_m).abs().mean().item()

    return results


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, A, Z, device):
    model.train()
    total_loss = 0.0
    for X_b, Y_b, T_b in loader:
        X_b, Y_b, T_b = X_b.to(device), Y_b.to(device), T_b.to(device)
        pred = model(X_b, Z, T_b, A)
        loss = nn.HuberLoss()(pred, Y_b)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, A, Z, device):
    model.eval()
    preds, trues = [], []
    for X_b, Y_b, T_b in loader:
        X_b, T_b = X_b.to(device), T_b.to(device)
        pred = model(X_b, Z, T_b, A)
        preds.append(pred.cpu())
        trues.append(Y_b)
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return preds, trues


# ──────────────────────────────────────────────
# BUILD MODEL
# ──────────────────────────────────────────────
def build_model(
    variant_name, meta, use_zone_emb=True, use_zone_weight=True, use_zone_adj=True
):
    N = meta["N"]
    K = meta["K"]
    F = meta["F"]
    T_in = meta["T_in"]
    T_out = meta["T_out"]
    in_ch = T_in * F

    # ── External baselines ──
    if variant_name == "lstm":
        return LSTMBaseline(N, in_ch, T_out, hidden_dim=128)
    if variant_name == "gcn_gru":
        return GCNGRUBaseline(N, in_ch, T_out, hidden_dim=64)
    if variant_name == "stgcn":
        return STGCNBaseline(N, in_ch, T_out, hidden_dim=64)

    # ── AH-GNN baseline ──
    if variant_name == "baseline_ahgnn":
        return AH_GNN(
            num_nodes=N,
            in_channels=in_ch,
            hidden_channels=64,
            out_channels=T_out,
            embed_dim=32,
            num_time_labels=4,
            num_layers=2,
        )

    # ── Tân: Time-conditioned discrete embedding ──
    if variant_name == "zone_full_tc":
        model = TimeZoneAwareAHGNN(
            num_nodes=N,
            num_zones=K,
            in_channels=in_ch,
            hidden_channels=64,
            out_channels=T_out,
            node_embed_dim=32,
            zone_embed_dim=16,
            num_time_labels=4,
            num_layers=2,
        )
        model.use_zone_weight = use_zone_weight
        model.use_zone_adj = use_zone_adj
        return model

    # ── Bảo: Sinusoidal time encoder ──
    if variant_name == "zone_full_sinc":
        model = SinusoidalZoneAwareAHGNN(
            num_nodes=N,
            num_zones=K,
            in_channels=in_ch,
            hidden_channels=64,
            out_channels=T_out,
            node_embed_dim=32,
            zone_embed_dim=16,
            num_time_labels=4,
            num_layers=2,
            d_model=32,
        )
        model.use_zone_weight = use_zone_weight
        model.use_zone_adj = use_zone_adj
        return model

    # ── Zone-Aware ablation variants (zone_concat, zone_weight, zone_full) ──
    model = ZoneAwareAHGNN(
        num_nodes=N,
        num_zones=K,
        in_channels=in_ch,
        hidden_channels=64,
        out_channels=T_out,
        node_embed_dim=32,
        zone_embed_dim=16,
        num_time_labels=4,
        num_layers=2,
    )
    model.use_zone_weight = use_zone_weight
    model.use_zone_adj = use_zone_adj
    if not use_zone_emb:
        for p in model.zone_emb.parameters():
            p.requires_grad_(False)
            p.data.zero_()
    return model


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_experiment(variant_name, meta, dataset_dict, ablation_cfg):
    set_seed(42)
    use_zone_emb, use_zone_weight, use_zone_adj = ablation_cfg

    print(f"\n{'='*55}")
    print(f"  Variant: {variant_name}")
    print(
        f"  zone_emb={use_zone_emb} | zone_weight={use_zone_weight} | zone_adj={use_zone_adj}"
    )
    print(f"{'='*55}")

    X = dataset_dict["X"]
    Y = dataset_dict["Y"]
    TL = dataset_dict["time_labels"]
    A = dataset_dict["A"].to(DEVICE)
    Z = dataset_dict["Z"].to(DEVICE)

    S = X.size(0)
    n_train = int(S * TRAIN_RATIO)
    n_val = int(S * VAL_RATIO)
    n_test = S - n_train - n_val

    full_ds = TensorDataset(X, Y, TL)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)
    test_loader = DataLoader(test_ds, BATCH_SIZE)

    model = build_model(variant_name, meta, use_zone_emb, use_zone_weight, use_zone_adj)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_mae = float("inf")
    patience_cnt = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, A, Z, DEVICE)
        val_preds, val_trues = evaluate(model, val_loader, A, Z, DEVICE)
        val_metrics = compute_metrics(val_preds, val_trues)
        scheduler.step(val_metrics["MAE"])

        if epoch % 10 == 0:
            print(
                f"  Ep {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_MAE={val_metrics['MAE']:.4f} | val_RMSE={val_metrics['RMSE']:.4f}"
            )

        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_preds, test_trues = evaluate(model, test_loader, A, Z, DEVICE)
    test_metrics = compute_metrics(test_preds, test_trues)
    zone_metrics = compute_zone_stratified_metrics(
        test_preds, test_trues, Z.cpu(), meta["zone_types"]
    )

    print(f"\n  📊 Test Results:")
    print(
        f"     MAE={test_metrics['MAE']:.4f} | RMSE={test_metrics['RMSE']:.4f} | MAPE={test_metrics['MAPE']:.2f}%"
    )
    print(f"  📊 Zone-Stratified MAE:")
    for k, v in zone_metrics.items():
        print(f"     {k}: {v:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    # ==================================================================================================================================
    t_out = meta["T_out"]
    torch.save(best_state, os.path.join(OUT_DIR, f"{variant_name}_T{t_out}_best.pt"))
    # ===================================================================================================================================

    return {
        **test_metrics,
        **zone_metrics,
        "variant": variant_name,
        "n_params": n_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--variant",
        default="zone_full",
        choices=list(ABLATION_VARIANTS.keys()) + BASELINE_NAMES,
    )
    args = parser.parse_args()

    print(f"📂 Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset not found. Run: python scripts/build_graph.py")
        return

    dataset = torch.load(DATASET_PATH, weights_only=False)
    with open(META_PATH) as f:
        meta = json.load(f)

    print(f"  Nodes: {meta['N']} | Zones: {meta['K']} | Features: {meta['F']}")
    print(f"  Samples: {meta['S']} | T_in: {meta['T_in']} | T_out: {meta['T_out']}")
    print(f"  TomTom: {'✅' if meta['has_tomtom'] else '⚠️ using OSRM proxy'}")
    print(f"  Zones:  {'✅' if meta['has_zones']  else '⚠️ using zero vectors'}")
    print(f"  Device: {DEVICE}")

    if args.all:
        run_queue = list(ABLATION_VARIANTS.items()) + [
            (n, (False, False, False)) for n in BASELINE_NAMES
        ]
        save_ablation = True
        save_baselines = True
    elif args.ablation:
        run_queue = list(ABLATION_VARIANTS.items())
        save_ablation = True
        save_baselines = False
    elif args.baselines:
        run_queue = [(n, (False, False, False)) for n in BASELINE_NAMES]
        save_ablation = False
        save_baselines = True
    else:
        vname = args.variant
        vcfg = ABLATION_VARIANTS.get(vname, (False, False, False))
        run_queue = [(vname, vcfg)]
        save_ablation = False
        save_baselines = False

    all_results = []
    for vname, vcfg in run_queue:
        result = run_experiment(vname, meta, dataset, vcfg)
        all_results.append(result)

    import pandas as pd

    os.makedirs(OUT_DIR, exist_ok=True)

    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print("  SUMMARY")
        print(f"{'='*72}")
        print(
            f"  {'Variant':<22} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Multi-Zone MAE':>15}"
        )
        print(f"  {'-'*62}")
        for r in all_results:
            mz = r.get("MAE_multi_zone", float("nan"))
            marker = " ← PROPOSED" if r["variant"] == "zone_full" else ""
            print(
                f"  {r['variant']:<22} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} "
                f"{r['MAPE']:>8.2f} {mz:>15.4f}{marker}"
            )

    df = pd.DataFrame(all_results)
    if save_ablation or save_baselines or args.all:
        if save_ablation and not save_baselines:
            out_path = os.path.join(OUT_DIR, "ablation_results.csv")
        elif save_baselines and not save_ablation:
            out_path = os.path.join(OUT_DIR, "baseline_results.csv")
        else:
            out_path = os.path.join(OUT_DIR, "all_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
