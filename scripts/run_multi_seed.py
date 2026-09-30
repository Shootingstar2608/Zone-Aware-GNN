"""
run_multi_seed.py
=================
Chạy lặp lại quá trình huấn luyện với NHIỀU SEED khác nhau để đo độ ổn định
của từng mô hình, phục vụ báo cáo Mean ± Std và kiểm định ý nghĩa thống kê.

TAI SAO CẦN FILE RIÊNG THAY VÌ SỬa train.py
-------------------------------------------
1. `train.py` đang fix cứng seed 42 ở HAI chỗ (set_seed và generator của
   random_split). Sửa trực tiếp sẽ làm mọi kết quả cũ không tái lập được.
2. `train.py` đang được nhiều người trong nhóm sửa song song (Người A thêm
   zone_full_tc, Người B thêm zone_full_sinc) -> sửa file đó = conflict merge.
3. File này IMPORT LẠI toàn bộ helper từ train.py (build_model, compute_metrics,
   train_one_epoch, evaluate...) nên nếu nhóm đổi kiến trúc/loss/optimizer,
   script này tự động kế thừa, không bị lệch pha.

TACH RỜI HUẤN LUYỆN VA THỐNG KÊ
-------------------------------
File này CHỈ sinh dữ liệu thô: mỗi lần train = 1 dòng trong
`data/results/multiseed_runs.csv`. Việc tính Mean±Std, t-test, p-value nằm ở
`scripts/stat_analysis.py`. Lý do: huấn luyện tốn hàng giờ, còn thống kê chỉ
tốn vài giây — tách ra để chỉnh cách phân tích mà không phải train lại.

BA GIAO THỨC CHIA DỮ LIỆU (--split-modes)
-----------------------------------------

  ╔═════════════════════════════════════════════════════════════════╗
  ║  PIPELINE «LEGACY»  (KÔNG KHUYẾN DÙNG khi báo cáo paper)      ║
  ║  random_fixed / random_paired — chia NGẠU NHIÊN                  ║
  ║  => Có data leakage do sliding window chồng lấp                 ║
  ║  => Chỉ dùng để so sánh lịch sử với kết quả cũ                ║
  ╚═════════════════════════════════════════════════════════════════╝

  random_fixed  : chia ngỪu nhiên, seed chia CỐ ĐỊNH = 42 cho mọi lần chạy.
                  -> Chỉ có khởi tạo mô hình thay đổi. Tái lập đúng giao thức
                  hiện tại của train.py. Test set giống hệt nhau giữa các
                  model => so sánh trực tiếp được, nhưng kết luận chỉ đúng
                  trên MỘT split. Kiểm định phù hợp: Welch t-test (độc lập).

  random_paired : chia ngỪu nhiên, seed chia = seed của lần chạy.
                  -> Mỗi seed cho một split khác nhau, NHƯNG mọi model đều dùng
                  chung split đó ở cùng seed => ghép cặp được theo seed.
                  Kiểm định phù hợp: paired t-test (mạnh hơn vì khử được
                  nhiễu do "split dễ/khó"). Kết luận rộng hơn.

  ╔═════════════════════════════════════════════════════════════════╗
  ║  PIPELINE «CLEAN»  (KHUYẾN DÙNG cho mọi báo cáo chính thức)    ║
  ║  chrono — chia theo THỜI GIAN + purge gap (= T_in + T_out - 1) ║
  ║  => Không có data leakage, đúng giao thức DCRNN/STGCN/WaveNet   ║
  ║  => Đây là con số nên đưa vào paper                            ║
  ╚═════════════════════════════════════════════════════════════════╝

  chrono        : chia theo THỜI GIAN (70%/10%/20%) + purge gap giữa các tập.
                  -> Không có rò rỉ thời gian, không chồng lấp cửa sổ.
                  Đây là giao thức chuẩn của DCRNN / STGCN / Graph WaveNet.

Chạy:
  python scripts/run_multi_seed.py                          # mặc định: 9 model, 5 seed, 2 mode
  python scripts/run_multi_seed.py --models proposed        # chỉ 4 model chính
  python scripts/run_multi_seed.py --split-modes chrono     # pipeline CLEAN — dùng khi báo cáo
  python scripts/run_multi_seed.py --resume                 # chạy tiếp sau khi bị ngắt
  python scripts/run_multi_seed.py --epochs 5 --seeds 0 1   # smoke test nhanh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

# ── Cho phép import package `models` và `scripts` khi chạy từ bất kỳ đâu ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# Tái sử dụng toàn bộ logic gốc — KHÔNG copy-paste lại
from scripts.train import (  # noqa: E402
    ABLATION_VARIANTS,
    BASELINE_NAMES,
    BATCH_SIZE,
    EPOCHS,
    LR,
    PATIENCE,
    build_model,
    evaluate,
    set_seed,
    train_one_epoch,
)
from utils.eval_protocol import (  # noqa: E402
    chronological_split,
    fit_normalizers,
    compute_metrics,
    compute_zone_stratified_metrics,
    get_git_commit_hash,
    TRAIN_RATIO,
    VAL_RATIO,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = os.path.join(ROOT, "data", "processed", "graph_dataset.pt")
META_PATH = os.path.join(ROOT, "data", "processed", "meta.json")
OUT_DIR = os.path.join(ROOT, "data", "results")
RUNS_CSV = os.path.join(OUT_DIR, "multiseed_runs.csv")

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
SPLIT_MODES = ["random_fixed", "random_paired", "chrono"]

# Nhóm model để tiện chọn nhanh
MODEL_GROUPS = {
    "all": list(ABLATION_VARIANTS.keys()) + BASELINE_NAMES,
    "proposed": ["zone_full_tc", "zone_full_sinc", "zone_full", "gcn_gru"],
    "baselines": BASELINE_NAMES,
    "ablation": list(ABLATION_VARIANTS.keys()),
}

# Khoá định danh duy nhất của một lần chạy (dùng cho --resume)
RUN_KEY = ["variant", "seed", "split_mode"]


# ══════════════════════════════════════════════════════════════
# CHIA DỮ LIỆU
# ══════════════════════════════════════════════════════════════
def make_splits(full_ds, S: int, split_mode: str, seed: int, meta: dict):
    """
    Trả về (train_idx, val_idx, test_idx, split_seed_used).

    Trả về INDEX thay vì Subset để caller fit normalizer đúng trước khi
    tạo DataLoader. Điểm mấu chốt: split_seed quyết định kiểm định
    thống kê sau này là PAIRED hay INDEPENDENT.

    NOTE: mode random_fixed / random_paired thuộc pipeline «LEGACY»
          (có data leakage). Chỉ dùng để so sánh lịch sử. Pipeline
          chuẩn để báo cáo paper là mode 'chrono'.
    """
    n_train = int(S * TRAIN_RATIO)
    n_val   = int(S * VAL_RATIO)
    n_test  = S - n_train - n_val

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE «CLEAN» — DÙNG CHO BÁO CÁO PAPER
    # ══════════════════════════════════════════════════════════════════
    if split_mode == "chrono":
        purge_gap = meta["T_in"] + meta["T_out"] - 1
        train_idx, val_idx, test_idx = chronological_split(
            S, TRAIN_RATIO, VAL_RATIO, purge_gap=purge_gap
        )
        return train_idx, val_idx, test_idx, None

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE «LEGACY» — random_fixed / random_paired
    #  CẢNH BÁO: có data leakage do sliding window chồng lấp.
    #  Chỉ dùng để so sánh lịch sử với kết quả cũ. KHÔNG đưa vào paper.
    # ══════════════════════════════════════════════════════════════════
    split_seed = 42 if split_mode == "random_fixed" else seed
    from torch.utils.data import random_split as _random_split
    tmp_ds = TensorDataset(
        torch.arange(S)  # dummy, chỉ cần lấy index
    )
    tmp_train, tmp_val, tmp_test = _random_split(
        tmp_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(split_seed),
    )
    train_idx = list(tmp_train.indices)
    val_idx   = list(tmp_val.indices)
    test_idx  = list(tmp_test.indices)
    return train_idx, val_idx, test_idx, split_seed


# ══════════════════════════════════════════════════════════════
# MỘT LẦN CHẠY
# ══════════════════════════════════════════════════════════════
def run_once(
    variant: str,
    seed: int,
    split_mode: str,
    meta: dict,
    dataset: dict,
    epochs: int,
    save_checkpoint: bool,
    verbose: bool = False,
) -> dict:
    """Huấn luyện 1 model với 1 seed, trả về dict metrics trên test set."""
    t0 = time.time()

    # 1) Seed TOÀN CỤC: chi phối khởi tạo trọng số + thứ tự shuffle + dropout
    set_seed(seed)

    cfg = ABLATION_VARIANTS.get(variant, (False, False, False))
    use_zone_emb, use_zone_weight, use_zone_adj = cfg

    X, Y, TL = dataset["X"], dataset["Y"], dataset["time_labels"]
    A = dataset["A"].to(DEVICE)
    Z = dataset["Z"].to(DEVICE)
    S = X.size(0)

    # ── Split (trả về index để fit normalizer đúng trước) ──────────────────
    full_ds = TensorDataset(X, Y, TL)
    train_idx, val_idx, test_idx, split_seed = make_splits(
        full_ds, S, split_mode, seed, meta
    )

    # ── Z-Score Normalization — fit CHỈ trên train_idx ───────────────────
    x_normalizer, y_normalizer, X_norm, Y_norm = fit_normalizers(
        X, Y, train_idx
    )

    full_ds_norm = TensorDataset(X_norm, Y_norm, TL)
    train_ds = Subset(full_ds_norm, train_idx)
    val_ds   = Subset(full_ds_norm, val_idx)
    test_ds  = Subset(full_ds_norm, test_idx)

    # Generator riêng cho DataLoader: đảm bảo thứ tự batch tái lập được 100%
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, generator=loader_gen)
    val_loader = DataLoader(val_ds, BATCH_SIZE)
    test_loader = DataLoader(test_ds, BATCH_SIZE)

    model = build_model(variant, meta, use_zone_emb, use_zone_weight, use_zone_adj)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_mae = float("inf")
    best_state, patience_cnt, last_epoch = None, 0, 0

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, A, Z, DEVICE)
        val_preds, val_trues = evaluate(model, val_loader, A, Z, DEVICE)
        val_preds_real = y_normalizer.inverse_transform(val_preds)
        val_trues_real = y_normalizer.inverse_transform(val_trues)
        val_metrics = compute_metrics(val_preds_real, val_trues_real)
        scheduler.step(val_metrics["MAE"])

        if verbose and epoch % 20 == 0:
            print(
                f"      ep {epoch:3d} loss={train_loss:.4f} val_MAE={val_metrics['MAE']:.4f}"
            )

        if val_metrics["MAE"] < best_val_mae - 1e-9:
            best_val_mae = val_metrics["MAE"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    test_preds, test_trues = evaluate(model, test_loader, A, Z, DEVICE)

    # ── Inverse-transform về đơn vị gốc trước khi tính metric ────────────
    test_preds_real = y_normalizer.inverse_transform(test_preds)
    test_trues_real = y_normalizer.inverse_transform(test_trues)

    test_metrics = compute_metrics(test_preds_real, test_trues_real)
    zone_metrics = compute_zone_stratified_metrics(
        test_preds_real, test_trues_real, Z.cpu(), meta["zone_types"]
    )

    # Mặc định KHÔNG lưu checkpoint: 9 model × 5 seed × 3 mode = 135 file,
    # và quan trọng hơn là sẽ ghi đè các file `{variant}_best.pt` mà
    # scripts/zone_stratified_analysis.py đang phụ thuộc vào.
    if save_checkpoint:
        ckpt_dir = os.path.join(OUT_DIR, "checkpoints_multiseed")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            best_state, os.path.join(ckpt_dir, f"{variant}_{split_mode}_s{seed}.pt")
        )

    return {
        "variant": variant,
        "seed": seed,
        "split_mode": split_mode,
        "split_seed": split_seed if split_seed is not None else -1,
        **test_metrics,
        **zone_metrics,
        "best_val_MAE": best_val_mae,
        "epochs_run": last_epoch,
        "n_params": n_params,
        "train_time_s": round(time.time() - t0, 1),
        "device": DEVICE,
        "status": "ok",
        "commit": get_git_commit_hash(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# ══════════════════════════════════════════════════════════════
# CSV VERSIONING + GHI TĂNG DẦN
# ══════════════════════════════════════════════════════════════
def find_versioned_csv(base_path: str, resume: bool) -> str:
    """
    Tìm hoặc tạo CSV versioned: multiseed_runs_v{N}.csv
    - resume=True: trả về file v{max} hiện có (hoặc v1 nếu chưa có).
    - resume=False: tạo v{max+1} mới.
    """
    import glob
    import re

    directory = os.path.dirname(base_path)
    basename = os.path.splitext(os.path.basename(base_path))[0]  # "multiseed_runs"
    ext = os.path.splitext(base_path)[1]  # ".csv"

    pattern = os.path.join(directory, f"{basename}_v*{ext}")
    existing = glob.glob(pattern)

    versions = []
    for f in existing:
        m = re.search(rf"_v(\d+){re.escape(ext)}$", f)
        if m:
            versions.append(int(m.group(1)))

    if resume and versions:
        v = max(versions)
    elif versions:
        v = max(versions) + 1
    else:
        v = 1

    return os.path.join(directory, f"{basename}_v{v}{ext}")


def append_row(row: dict, path: str):
    """Ghi ngay sau mỗi lần train xong. Sweep 90 lần train mà crash ở lần thứ
    80 rồi mất sạch thì rất đau — nên ghi từng dòng một."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        # Lần chạy mới đè lên lần cũ nếu trùng khoá
        df = df.drop_duplicates(subset=RUN_KEY, keep="last")
    else:
        df = df_new
    df.to_csv(path, index=False)


def load_done_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if not set(RUN_KEY).issubset(df.columns):
        return set()
    df = df[df.get("status", "ok") == "ok"]
    return {(r.variant, int(r.seed), r.split_mode) for r in df.itertuples()}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Chạy đa seed để đo Mean±Std và phục vụ kiểm định thống kê."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Tên group (all/proposed/baselines/ablation) hoặc liệt kê tên model.",
    )
    p.add_argument(
        "--split-modes",
        nargs="+",
        default=["random_fixed", "random_paired"],
        choices=SPLIT_MODES,
    )
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--out", default=RUNS_CSV)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Bỏ qua các (model, seed, split_mode) đã có trong CSV.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: --epochs 5 --seeds 42 43 --split-modes chrono",
    )
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # --smoke override
    if args.smoke:
        args.epochs = 5
        args.seeds = [42, 43]
        args.split_modes = ["chrono"]
        if args.models == ["all"]:
            args.models = ["proposed"]

    # Giải nghĩa group -> danh sách model
    variants = []
    for m in args.models:
        variants.extend(MODEL_GROUPS.get(m, [m]))
    seen = set()
    variants = [v for v in variants if not (v in seen or seen.add(v))]

    valid = set(ABLATION_VARIANTS) | set(BASELINE_NAMES)
    unknown = [v for v in variants if v not in valid]
    if unknown:
        raise SystemExit(f"❌ Model không tồn tại: {unknown}\n   Hợp lệ: {sorted(valid)}")

    if not os.path.exists(DATASET_PATH):
        raise SystemExit("❌ Chưa có dataset. Chạy: python scripts/build_graph.py")

    dataset = torch.load(DATASET_PATH, weights_only=False)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    # CSV versioned path
    csv_path = find_versioned_csv(args.out, resume=args.resume)

    done = load_done_keys(csv_path) if args.resume else set()

    queue = [
        (v, s, sm)
        for sm in args.split_modes
        for v in variants
        for s in args.seeds
        if (v, int(s), sm) not in done
    ]
    total = len(queue)

    print("=" * 68)
    print("  MULTI-SEED SWEEP")
    print("=" * 68)
    print(f"  Models      : {len(variants)} → {', '.join(variants)}")
    print(f"  Seeds       : {args.seeds}")
    print(f"  Split modes : {args.split_modes}")
    print(f"  Epochs      : {args.epochs} (early-stop patience={PATIENCE})")
    print(f"  Device      : {DEVICE}")
    print(f"  Bỏ qua      : {len(done)} lần chạy đã có (--resume)")
    print(f"  Cần chạy    : {total} lần train")
    print(f"  Ghi vào     : {csv_path}")
    print(f"  Commit      : {get_git_commit_hash()}")
    print("=" * 68)

    t_start = time.time()
    n_fail = 0

    for i, (variant, seed, split_mode) in enumerate(queue, 1):
        elapsed = time.time() - t_start
        eta = (elapsed / (i - 1) * (total - i + 1)) if i > 1 else 0
        print(
            f"\n[{i:3d}/{total}] {variant:<16} seed={seed:<4} mode={split_mode:<14}"
            f" | đã chạy {elapsed/60:5.1f}m | còn ~{eta/60:5.1f}m"
        )
        try:
            row = run_once(
                variant,
                seed,
                split_mode,
                meta,
                dataset,
                args.epochs,
                args.save_checkpoints,
                args.verbose,
            )
            print(
                f"          MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}"
                f"  MAPE={row['MAPE']:.2f}%  ({row['epochs_run']} ep,"
                f" {row['train_time_s']:.0f}s)"
            )
        except Exception as e:  # một lần chạy hỏng không được giết cả sweep
            n_fail += 1
            print(f"          ❌ LỖI: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            row = {
                "variant": variant,
                "seed": seed,
                "split_mode": split_mode,
                "status": f"failed: {type(e).__name__}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        append_row(row, csv_path)

    print("\n" + "=" * 68)
    print(f"  XONG {total - n_fail}/{total} lần chạy trong {(time.time()-t_start)/60:.1f} phút")
    if n_fail:
        print(f"  ⚠️  {n_fail} lần thất bại — xem cột `status` trong CSV")
    print(f"  → Kết quả thô: {csv_path}")
    print("  → Bước tiếp theo: python scripts/stat_analysis.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
