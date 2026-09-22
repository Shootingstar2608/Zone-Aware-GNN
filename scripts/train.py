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
import inspect
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

# Add parent dir to path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.normalizer import ZScoreNormalizer  # Tôn — Z-Score module
from utils.eval_protocol import (
    chronological_split,
    fit_normalizers,
    compute_metrics,
    compute_zone_stratified_metrics,
    evaluate_with_inverse,
    MAPE_EPS,
    PURGE_GAP_DEFAULT,
    TRAIN_RATIO as _TRAIN_RATIO,
    VAL_RATIO as _VAL_RATIO,
)

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

# Re-export từ eval_protocol cho backward compatibility
TRAIN_RATIO = _TRAIN_RATIO
VAL_RATIO = _VAL_RATIO
TEST_RATIO = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Bảo — Cosine Regularization (Hướng A)
LAMBDA_COS = 0.1  # hệ số phạt cosine similarity giữa zone khác nhau

# ══════════════════════════════════════════════════════════════════
# TÔN — CHRONOLOGICAL SPLIT + NORMALIZATION
# chronological_split, compute_metrics, compute_zone_stratified_metrics,
# fit_normalizers, PURGE_GAP_DEFAULT, MAPE_EPS đã chuyển sang
# utils/eval_protocol.py và import ở đầu file.
# ══════════════════════════════════════════════════════════════════

# Đường dẫn lưu stats normalizer để dùng lại lúc inference
NORMALIZER_DIR = "data/processed"


# ──────────────────────────────────────────────
# ABLATION VARIANTS
# ──────────────────────────────────────────────
ABLATION_VARIANTS = {
    "baseline_ahgnn": (False, False, False),
    "zone_concat": (True, False, False),
    "zone_weight": (True, True, False),
    "zone_full": (True, True, True),
    "zone_full_tc": (True, True, True),  # Tôn  — discrete time embedding
    "zone_full_sinc": (True, True, True),  # Bảo  — sinusoidal time encoder
}

BASELINE_NAMES = ["lstm", "gcn_gru", "stgcn"]

# Các variant có zone embedding thực sự → mới áp dụng cosine_reg
ZONE_AWARE_VARIANTS = {
    "zone_concat",
    "zone_weight",
    "zone_full",
    "zone_full_tc",
    "zone_full_sinc",
}


# ──────────────────────────────────────────────
# METRICS — imported from utils.eval_protocol
# compute_metrics() và compute_zone_stratified_metrics() đã chuyển
# sang utils/eval_protocol.py. Import ở đầu file, re-export tự động.
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# COSINE REGULARIZATION (Bảo — Hướng A)
# ──────────────────────────────────────────────
def get_zone_embeddings(
    model: nn.Module, Z: torch.Tensor, time_idx: torch.Tensor = None
):
    """
    Lấy zone embedding (N, d_z) từ model, nếu model có module `zone_emb`.

    Đã xác nhận trực tiếp từ source 2 kiểu chữ ký thật:
      - ZoneEmbedding.forward(Z)                → (N, d_z)     [ZoneAwareAHGNN]
      - TimeZoneEmbedding.forward(Z, time_idx)   → (B, N, d_z)  [TimeZoneAwareAHGNN]
      - SinusoidalZoneEmbedding.forward(Z, time_idx) → (B, N, d_z) [SinusoidalZoneAwareAHGNN]

    Dùng inspect để gọi đúng số tham số, không đoán/try-except mù.
    Trả về None nếu model không có zone_emb (vd: baseline_ahgnn, lstm, gcn_gru, stgcn).
    """
    zone_emb_module = getattr(model, "zone_emb", None)
    if zone_emb_module is None:
        return None

    n_params = len(inspect.signature(zone_emb_module.forward).parameters)

    if n_params >= 2:
        if time_idx is None:
            raise ValueError(
                "zone_emb của model này cần time_idx (TimeZoneEmbedding/"
                "SinusoidalZoneEmbedding) nhưng không được truyền vào."
            )
        z = zone_emb_module(Z, time_idx)  # (B, N, d_z)
    else:
        z = zone_emb_module(Z)  # (N, d_z)

    if z.dim() == 3:
        # (B, N, d_z) -> gộp theo batch (embedding phụ thuộc time_idx của batch)
        z = z.mean(dim=0)
    return z


@torch.no_grad()
def compute_final_cosine_sim(
    model: nn.Module, Z: torch.Tensor, loader, device
) -> float | None:
    """
    Tính cosine similarity (khác zone) trung bình trên TOÀN BỘ test set
    (không phải chỉ 1 batch), để số liệu ổn định khi báo cáo trong paper.
    Với model time-aware (zone_emb phụ thuộc time_idx), gộp theo weighted
    mean qua tất cả batch để phản ánh đúng phân bố thời gian thực tế.
    """
    model.eval()
    total_z = None
    total_n = 0
    for _, _, T_b in loader:
        T_b = T_b.to(device)
        # Dùng time_idx trung bình hoặc dummy
        dummy_t = torch.zeros(1, dtype=torch.long, device=DEVICE)
        z_emb = get_zone_embeddings(model, Z, dummy_t)
        if z_emb is None:
            return None
        bsz = T_b.size(0)
        z_emb_weighted = z_emb * bsz
        total_z = z_emb_weighted if total_z is None else total_z + z_emb_weighted
        total_n += bsz

    if total_z is None or total_n == 0:
        return None

    z_avg = total_z / total_n
    return compute_cosine_reg(z_avg, Z).item()


def compute_cosine_reg(z: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    z: (N, d_z) — zone embedding của từng node
    Z: (N, K)   — multi-hot zone label của từng node

    Loss = mean cosine similarity giữa các cặp node (i, j) có zone KHÁC nhau
    (không chia sẻ chung bất kỳ zone type nào). Phạt cao khi 2 node khác
    zone nhưng embedding lại quá giống nhau.
    """
    N = Z.size(0)
    z_norm = nn.functional.normalize(z, dim=-1, eps=1e-8)
    sim = z_norm @ z_norm.t()  # (N, N)

    Z_f = Z.float()
    shared_zone = (Z_f @ Z_f.t()) > 0  # True nếu 2 node chung ít nhất 1 zone
    eye = torch.eye(N, dtype=torch.bool, device=Z.device)
    diff_mask = (~shared_zone) & (~eye)  # chỉ lấy cặp KHÁC zone hoàn toàn

    if diff_mask.sum() == 0:
        return torch.zeros((), device=z.device)

    return sim[diff_mask].mean()


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, A, Z, device, lambda_cos=LAMBDA_COS):
    model.train()
    total_loss = 0.0
    total_huber = 0.0
    total_cos = 0.0
    for X_b, Y_b, T_b in loader:
        X_b, Y_b, T_b = X_b.to(device), Y_b.to(device), T_b.to(device)
        pred = model(X_b, Z, T_b, A)
        huber_loss = nn.HuberLoss()(pred, Y_b)

        # Bảo — Cosine Regularization: ép zone khác nhau ra xa nhau
        z_emb = get_zone_embeddings(model, Z, T_b)
        if z_emb is not None and lambda_cos > 0:
            cos_reg = compute_cosine_reg(z_emb, Z)
            loss = huber_loss + lambda_cos * cos_reg
        else:
            cos_reg = torch.zeros((), device=device)
            loss = huber_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        total_huber += huber_loss.item()
        total_cos += cos_reg.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "huber": total_huber / n,
        "cos_reg": total_cos / n,
    }


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

    # ── Tôn: Time-conditioned discrete embedding ──
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


# ══════════════════════════════════════════════════════════════════
# TÔN — CHRONOLOGICAL SPLIT VỚI PURGE GAP
# Đã chuyển sang utils/eval_protocol.py, import ở đầu file.
# Pipeline «LEGACY» (random_split) đã bị XÓA khỏi train.py.
# ══════════════════════════════════════════════════════════════════


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiment(variant_name, meta, dataset_dict, ablation_cfg, lambda_cos=LAMBDA_COS):
    set_seed(42)
    use_zone_emb, use_zone_weight, use_zone_adj = ablation_cfg

    # Bảo — chỉ áp dụng cosine_reg cho các variant có zone embedding thật
    effective_lambda_cos = lambda_cos if variant_name in ZONE_AWARE_VARIANTS else 0.0

    print(f"\n{'='*55}")
    print(f"  Variant: {variant_name}")
    print(
        f"  zone_emb={use_zone_emb} | zone_weight={use_zone_weight} | zone_adj={use_zone_adj}"
    )
    print(
        f"  cosine_reg={'ON (λ=' + str(effective_lambda_cos) + ')' if effective_lambda_cos > 0 else 'OFF'}"
    )
    print(f"{'='*55}")

    X = dataset_dict["X"]
    Y = dataset_dict["Y"]
    TL = dataset_dict["time_labels"]
    A = dataset_dict["A"].to(DEVICE)
    Z = dataset_dict["Z"].to(DEVICE)

    S = X.size(0)

    # ── Chronological Split với Purge Gap (Tôn) ─────────────────────────
    purge_gap = meta["T_in"] + meta["T_out"] - 1
    train_idx, val_idx, test_idx = chronological_split(
        S, TRAIN_RATIO, VAL_RATIO, purge_gap=purge_gap
    )
    print(
        f"  [Split] chrono: train={len(train_idx)} | gap={purge_gap}"
        f" | val={len(val_idx)} | gap={purge_gap} | test={len(test_idx)}"
        f" (bỏ {2 * purge_gap} mẫu purge)"
    )

    # ── Z-Score Normalization — fit CHỈ trên train (Tôn) ────────────────
    x_normalizer, y_normalizer, X_norm, Y_norm = fit_normalizers(
        X, Y, train_idx
    )

    print(f"  [Norm-X] {x_normalizer}")
    print(f"  [Norm-Y] {y_normalizer}")

    # Lưu stats để dùng lại khi inference (chỉ lưu với zone_full)
    if variant_name == "zone_full":
        x_normalizer.save(os.path.join(NORMALIZER_DIR, "x_normalizer.pt"))
        y_normalizer.save(os.path.join(NORMALIZER_DIR, "y_normalizer.pt"))

    full_ds = TensorDataset(X_norm, Y_norm, TL)
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, BATCH_SIZE)
    test_loader  = DataLoader(test_ds, BATCH_SIZE)

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
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            A,
            Z,
            DEVICE,
            lambda_cos=effective_lambda_cos,
        )
        val_preds, val_trues = evaluate(model, val_loader, A, Z, DEVICE)
        val_preds_real = y_normalizer.inverse_transform(val_preds)
        val_trues_real = y_normalizer.inverse_transform(val_trues)
        val_metrics = compute_metrics(val_preds_real, val_trues_real)
        scheduler.step(val_metrics["MAE"])

        if epoch % 10 == 0:
            cos_str = (
                f" | cos_reg={train_stats['cos_reg']:.4f}"
                if effective_lambda_cos > 0
                else ""
            )
            print(
                f"  Ep {epoch:3d} | train_loss={train_stats['loss']:.4f} "
                f"(huber={train_stats['huber']:.4f}{cos_str}) | "
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

    # ── Inverse-transform về đơn vị gốc trước khi tính metric (Tôn) ─────
    test_preds_real = y_normalizer.inverse_transform(test_preds)
    test_trues_real = y_normalizer.inverse_transform(test_trues)

    test_metrics = compute_metrics(test_preds_real, test_trues_real)
    zone_metrics = compute_zone_stratified_metrics(
        test_preds_real, test_trues_real, Z.cpu(), meta["zone_types"]
    )

    # Bảo — báo cáo cosine similarity trung bình cuối cùng (để so sánh trong paper)
    final_cos_sim = None
    if effective_lambda_cos > 0:
        model.eval()
        with torch.no_grad():
            # Truyền dummy time_idx=0 để tương thích với TimeZoneEmbedding / SinusoidalZoneEmbedding
            dummy_t = torch.zeros(1, dtype=torch.long, device=DEVICE)
            z_emb = get_zone_embeddings(model, Z, time_idx=dummy_t)
            if z_emb is not None:
                final_cos_sim = compute_cosine_reg(z_emb, Z).item()

    print(f"\n  📊 Test Results (đơn vị gốc congestion_ratio, sau inverse-transform):")
    print(
        f"     MAE={test_metrics['MAE']:.4f} | RMSE={test_metrics['RMSE']:.4f}"
        f" | MAPE={test_metrics['MAPE']:.2f}% | WAPE={test_metrics['WAPE']:.2f}%"
    )
    if final_cos_sim is not None:
        print(f"     Cosine similarity (khác zone, cuối train): {final_cos_sim:.4f}")
    print(f"  📊 Zone-Stratified MAE:")
    for k, v in zone_metrics.items():
        print(f"     {k}: {v:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    # ==================================================================================================================================
    t_out = meta["T_out"]
    torch.save(best_state, os.path.join(OUT_DIR, f"{variant_name}_T{t_out}_best.pt"))
    torch.save(best_state, os.path.join(OUT_DIR, f"{variant_name}_best.pt"))
    # ===================================================================================================================================

    result = {
        **test_metrics,
        **zone_metrics,
        "variant": variant_name,
        "n_params": n_params,
        "lambda_cos": effective_lambda_cos,
    }
    if final_cos_sim is not None:
        result["final_cosine_sim"] = final_cos_sim
    return result


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
    parser.add_argument(
        "--lambda-cos",
        type=float,
        default=LAMBDA_COS,
        help="Hệ số cosine regularization (Bảo). Đặt 0 để tắt hoàn toàn.",
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
    print(f"  Cosine reg λ: {args.lambda_cos}")

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
        result = run_experiment(vname, meta, dataset, vcfg, lambda_cos=args.lambda_cos)
        all_results.append(result)

    import pandas as pd

    os.makedirs(OUT_DIR, exist_ok=True)

    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print("  SUMMARY")
        print(f"{'='*72}")
        print(
            f"  {'Variant':<22} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Multi-Zone MAE':>15} {'CosSim':>8}"
        )
        print(f"  {'-'*70}")
        for r in all_results:
            mz = r.get("MAE_multi_zone", float("nan"))
            cs = r.get("final_cosine_sim", float("nan"))
            marker = " ← PROPOSED" if r["variant"] == "zone_full" else ""
            print(
                f"  {r['variant']:<22} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} "
                f"{r['MAPE']:>8.2f} {mz:>15.4f} {cs:>8.4f}{marker}"
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
