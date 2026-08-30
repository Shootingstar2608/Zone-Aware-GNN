"""
stat_analysis.py
================
Đọc `data/results/multiseed_runs.csv` (do run_multi_seed.py sinh ra) và xuất:

  1. Bảng Mean ± Std của MAE / RMSE / MAPE cho từng model
  2. Kiểm định ý nghĩa thống kê giữa model đề xuất và baseline mạnh nhất
  3. Bảng LaTeX dán thẳng vào paper

Chạy:
  python scripts/stat_analysis.py
  python scripts/stat_analysis.py --baseline gcn_gru --proposed zone_full_tc zone_full_sinc
  python scripts/stat_analysis.py --metrics MAE RMSE MAPE MAE_multi_zone

───────────────────────────────────────────────────────────────────────────
GHI CHÚ VỀ LỰA CHỌN KIỂM ĐỊNH — đọc kỹ trước khi viết vào paper
───────────────────────────────────────────────────────────────────────────

• PAIRED vs WELCH (độc lập). Không phải chọn bừa, mà phụ thuộc giao thức chia
  dữ liệu:

    - split_mode = random_paired: seed s cho model A và seed s cho model B dùng
      CHUNG một tập test. Độ khó của tập test là biến gây nhiễu chung, và
      paired t-test khử được nó (chỉ xét hiệu d_s = A_s − B_s). Đây là kiểm định
      ĐÚNG và MẠNH hơn ở mode này.

    - split_mode = random_fixed: tập test giống hệt nhau ở mọi lần chạy, seed
      chỉ đổi khởi tạo trọng số. "Seed 42 của A" và "seed 42 của B" không chia
      sẻ nguồn nhiễu nào cả (kiến trúc khác nhau, tensor khởi tạo khác nhau)
      → ghép cặp là VÔ NGHĨA. Ở mode này Welch t-test (không giả định phương sai
      bằng nhau) là kiểm định đúng.

  Script tính CẢ HAI ở mọi mode, nhưng cột `recommended` đánh dấu cái nên dùng.

• MỘT PHÍA hay HAI PHÍA. Giả thuyết của bạn có hướng ("model đề xuất TỐT HƠN"),
  nên one-sided về mặt lý thuyết là hợp lệ. Nhưng reviewer thường nghi ngờ
  one-sided vì dễ bị lạm dụng (chọn hướng sau khi thấy dữ liệu). Script báo cáo
  hai phía làm CHÍNH và một phía để tham khảo. Nếu dùng một phía, phải nói rõ
  trong paper là đã đăng ký hướng trước khi chạy thí nghiệm.

• WILCOXON VỚI n = 5 KHÔNG BAO GIỜ ĐẠT p < 0.05 Ở HAI PHÍA.
  Wilcoxon signed-rank chính xác có p hai phía nhỏ nhất = 2 / 2^n. Với n = 5:
  2/32 = 0.0625 > 0.05. Dù model của bạn thắng tuyệt đối cả 5 seed, p vẫn là
  0.0625. Một phía thì nhỏ nhất = 1/32 = 0.03125 (đạt được). Muốn Wilcoxon hai
  phía có ý nghĩa, cần n ≥ 6 (2/64 = 0.031). Cột `p_floor` in ra giới hạn này.
  → Khuyến nghị: chạy 8–10 seed thay vì 5 nếu muốn dùng kiểm định phi tham số.

• ĐA SO SÁNH. 2 model đề xuất × 3 metric = 6 kiểm định. Nếu mỗi cái dùng
  α = 0.05 độc lập thì xác suất có ÍT NHẤT một dương tính giả là
  1 − 0.95^6 ≈ 26%. Hiệu chỉnh Holm–Bonferroni (chặt chẽ hơn Bonferroni nhưng
  vẫn kiểm soát FWER) được áp dụng trong từng họ (split_mode × loại kiểm định).

• p-VALUE KHÔNG NÓI ĐỘ LỚN. Với n nhỏ và phương sai nhỏ, p có thể rất bé dù
  cải thiện chỉ 0.001 MAE. Vì vậy luôn báo cáo kèm Cohen's d (độ lớn hiệu ứng),
  khoảng tin cậy 95% của chênh lệch, và % cải thiện tương đối.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "results")
RUNS_CSV = os.path.join(OUT_DIR, "multiseed_runs.csv")

DEFAULT_METRICS = ["MAE", "RMSE", "MAPE"]
DEFAULT_PROPOSED = ["zone_full_tc", "zone_full_sinc"]
BASELINE_POOL = ["lstm", "gcn_gru", "stgcn"]

# Với mọi metric ở đây, GIÁ TRỊ NHỎ HƠN = TỐT HƠN.
# => improvement = baseline − proposed (dương nghĩa là đề xuất thắng)
LOWER_IS_BETTER = True


# ══════════════════════════════════════════════════════════════
# CÁC HÀM THỐNG KÊ
# ══════════════════════════════════════════════════════════════
def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d_z cho thiết kế ghép cặp = mean(d) / sd(d).
    Quy ước diễn giải: |d| ≈ 0.2 nhỏ, 0.5 vừa, 0.8 lớn."""
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else float("inf")


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g = Cohen's d có hiệu chỉnh chệch cho mẫu nhỏ.
    Với n = 5 mỗi nhóm, Cohen's d thổi phồng hiệu ứng ~10–15%, nên dùng g."""
    n1, n2 = len(a), len(b)
    s_pool = np.sqrt(
        ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    )
    if s_pool < 1e-12:
        return float("inf")
    d = (a.mean() - b.mean()) / s_pool
    J = 1 - 3 / (4 * (n1 + n2) - 9)  # hệ số hiệu chỉnh
    return float(d * J)


def ci95_mean(x: np.ndarray) -> tuple[float, float]:
    """Khoảng tin cậy 95% cho trung bình, dùng phân phối t (n nhỏ)."""
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    se = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return (float(x.mean() - t_crit * se), float(x.mean() + t_crit * se))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """
    Hiệu chỉnh Holm–Bonferroni (step-down).
    Sắp p tăng dần, nhân p_(i) với (m − i), rồi ép đơn điệu không giảm.
    Kiểm soát FWER giống Bonferroni nhưng mạnh hơn (bác bỏ được nhiều hơn).
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)  # ép đơn điệu
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ══════════════════════════════════════════════════════════════
# BẢNG TỔNG HỢP MEAN ± STD
# ══════════════════════════════════════════════════════════════
def build_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (mode, variant), g in df.groupby(["split_mode", "variant"], sort=False):
        row = {"split_mode": mode, "variant": variant, "n_runs": len(g)}
        for m in metrics:
            if m not in g.columns:
                continue
            v = g[m].dropna().to_numpy(dtype=float)
            row[f"{m}_mean"] = v.mean() if len(v) else np.nan
            # ddof=1 = phương sai mẫu (ước lượng không chệch). Dùng ddof=0 với
            # n=5 sẽ báo cáo std thấp hơn thực tế ~11% — sai về mặt thống kê.
            row[f"{m}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
        row["n_params"] = g["n_params"].iloc[0] if "n_params" in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# KIỂM ĐỊNH
# ══════════════════════════════════════════════════════════════
def compare(
    df: pd.DataFrame, mode: str, baseline: str, proposed: str, metric: str
) -> dict | None:
    """So sánh 1 cặp (proposed vs baseline) trên 1 metric, trong 1 split_mode."""
    sub = df[df.split_mode == mode]
    b = sub[sub.variant == baseline][["seed", metric]].dropna()
    p = sub[sub.variant == proposed][["seed", metric]].dropna()
    if len(b) < 2 or len(p) < 2:
        return None

    # Ghép theo seed. Ở random_paired, cùng seed = cùng tập test.
    merged = b.merge(p, on="seed", suffixes=("_base", "_prop")).sort_values("seed")
    x_b = merged[f"{metric}_base"].to_numpy(dtype=float)
    x_p = merged[f"{metric}_prop"].to_numpy(dtype=float)
    n = len(merged)
    if n < 2:
        return None

    diff = x_b - x_p  # dương = model đề xuất tốt hơn (metric càng nhỏ càng tốt)

    # ── Paired t-test ──
    t_pair, p_pair_2 = stats.ttest_rel(x_b, x_p)
    p_pair_1 = stats.ttest_rel(x_b, x_p, alternative="greater").pvalue

    # ── Welch t-test (không giả định phương sai bằng nhau) ──
    t_welch, p_welch_2 = stats.ttest_ind(x_b, x_p, equal_var=False)
    p_welch_1 = stats.ttest_ind(x_b, x_p, equal_var=False, alternative="greater").pvalue

    # ── Wilcoxon signed-rank (phi tham số, không giả định phân phối chuẩn) ──
    try:
        p_wil_2 = stats.wilcoxon(diff, alternative="two-sided").pvalue
        p_wil_1 = stats.wilcoxon(diff, alternative="greater").pvalue
    except ValueError:  # tất cả hiệu bằng 0
        p_wil_2 = p_wil_1 = 1.0

    lo, hi = ci95_mean(diff)

    return {
        "split_mode": mode,
        "metric": metric,
        "baseline": baseline,
        "proposed": proposed,
        "n_seeds": n,
        "baseline_mean": x_b.mean(),
        "baseline_std": x_b.std(ddof=1),
        "proposed_mean": x_p.mean(),
        "proposed_std": x_p.std(ddof=1),
        "improvement_abs": diff.mean(),
        "improvement_pct": 100 * diff.mean() / x_b.mean() if x_b.mean() else np.nan,
        "ci95_low": lo,
        "ci95_high": hi,
        # CI không chứa 0 <=> paired t-test hai phía có p < 0.05. Đây là cách
        # trình bày thuyết phục hơn p-value đơn thuần.
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and lo > 0),
        "wins": int((diff > 0).sum()),  # thắng bao nhiêu / n seed
        "t_paired": t_pair,
        "p_paired_2sided": p_pair_2,
        "p_paired_1sided": p_pair_1,
        "t_welch": t_welch,
        "p_welch_2sided": p_welch_2,
        "p_welch_1sided": p_welch_1,
        "p_wilcoxon_2sided": p_wil_2,
        "p_wilcoxon_1sided": p_wil_1,
        "p_floor_wilcoxon_2sided": 2 / (2**n),  # p nhỏ nhất Wilcoxon đạt được
        "cohens_dz_paired": cohens_d_paired(diff),
        "hedges_g_indep": hedges_g(x_b, x_p),
        # random_paired => cùng test set theo seed => ghép cặp hợp lệ
        "recommended_test": "paired" if mode == "random_paired" else "welch",
    }


def run_tests(
    df: pd.DataFrame, baseline: str, proposed_list: list[str], metrics: list[str]
) -> pd.DataFrame:
    rows = []
    for mode in df.split_mode.unique():
        for prop in proposed_list:
            for m in metrics:
                r = compare(df, mode, baseline, prop, m)
                if r:
                    rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)

    # ── Hiệu chỉnh Holm trong TỪNG HỌ = (split_mode) × (loại kiểm định) ──
    # Họ = tập các kiểm định trả lời cùng một câu hỏi khoa học, ở đây là
    # "model đề xuất có tốt hơn baseline không" trên 2 model × 3 metric.
    for mode, g in out.groupby("split_mode"):
        for col, newcol in [
            ("p_paired_2sided", "p_paired_holm"),
            ("p_welch_2sided", "p_welch_holm"),
        ]:
            adj = holm_bonferroni(g[col].tolist())
            out.loc[g.index, newcol] = adj

    out["p_recommended"] = np.where(
        out.recommended_test == "paired", out.p_paired_2sided, out.p_welch_2sided
    )
    out["p_recommended_holm"] = np.where(
        out.recommended_test == "paired", out.p_paired_holm, out.p_welch_holm
    )
    out["significant_holm"] = out.p_recommended_holm < 0.05
    return out


# ══════════════════════════════════════════════════════════════
# XUẤT LATEX
# ══════════════════════════════════════════════════════════════
def latex_main_table(
    summary: pd.DataFrame, tests: pd.DataFrame, mode: str, metrics: list[str]
) -> str:
    s = summary[summary.split_mode == mode].copy()
    if s.empty:
        return ""
    order = [
        "lstm", "stgcn", "gcn_gru",
        "baseline_ahgnn", "zone_concat", "zone_weight",
        "zone_full", "zone_full_sinc", "zone_full_tc",
    ]
    s["_o"] = s.variant.apply(lambda v: order.index(v) if v in order else 99)
    s = s.sort_values("_o")

    pretty = {
        "lstm": "LSTM", "gcn_gru": "GCN-GRU", "stgcn": "STGCN",
        "baseline_ahgnn": "AH-GNN (no zone)", "zone_concat": "\\; + zone concat",
        "zone_weight": "\\; + zone weight", "zone_full": "Zone-Aware AH-GNN",
        "zone_full_sinc": "\\; + sinusoidal time", "zone_full_tc": "\\; + time-conditioned",
    }
    best = {m: s[f"{m}_mean"].min() for m in metrics if f"{m}_mean" in s}

    sig = {}
    for r in tests[tests.split_mode == mode].itertuples():
        sig[(r.proposed, r.metric)] = stars(r.p_recommended_holm)

    n_seeds = int(s.n_runs.max())
    lines = [
        "% Sinh tự động bởi scripts/stat_analysis.py",
        f"% split_mode = {mode}, n = {n_seeds} seeds",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Kết quả trên tập test, trung bình $\\pm$ độ lệch chuẩn trên "
        f"{n_seeds} seed ngẫu nhiên. Ký hiệu $^{{*}}$/$^{{**}}$/$^{{***}}$ chỉ mức "
        f"$p<0.05$/$0.01$/$0.001$ so với baseline mạnh nhất (GCN-GRU), "
        f"t-test đã hiệu chỉnh Holm.}}",
        f"\\label{{tab:main-results-{mode.replace('_','-')}}}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "Model & " + " & ".join(metrics) + " \\\\",
        "\\midrule",
    ]
    for r in s.itertuples():
        cells = []
        for m in metrics:
            mu = getattr(r, f"{m}_mean", np.nan)
            sd = getattr(r, f"{m}_std", np.nan)
            txt = f"{mu:.4f} $\\pm$ {sd:.4f}"
            if np.isfinite(mu) and abs(mu - best.get(m, np.inf)) < 1e-12:
                txt = f"\\textbf{{{mu:.4f}}} $\\pm$ \\textbf{{{sd:.4f}}}"
            mark = sig.get((r.variant, m), "")
            if mark and mark != "n.s.":
                txt += f"$^{{{mark}}}$"
            cells.append(txt)
        lines.append(f"{pretty.get(r.variant, r.variant)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# BÁO CÁO RA MÀN HÌNH
# ══════════════════════════════════════════════════════════════
def print_report(summary: pd.DataFrame, tests: pd.DataFrame, metrics: list[str]):
    for mode in summary.split_mode.unique():
        s = summary[summary.split_mode == mode].sort_values(f"{metrics[0]}_mean")
        print("\n" + "═" * 78)
        print(f"  SPLIT MODE: {mode}")
        print("═" * 78)
        hdr = f"  {'Model':<18}{'n':>3}  " + "".join(f"{m + ' (mean±std)':>24}" for m in metrics)
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in s.itertuples():
            cells = ""
            for m in metrics:
                mu, sd = getattr(r, f"{m}_mean", np.nan), getattr(r, f"{m}_std", np.nan)
                cells += f"{mu:>14.4f} ± {sd:<8.4f}"
            print(f"  {r.variant:<18}{r.n_runs:>3}  {cells}")

        t = tests[tests.split_mode == mode]
        if t.empty:
            continue
        print(f"\n  KIỂM ĐỊNH Ý NGHĨA (kiểm định khuyến nghị: {t.recommended_test.iloc[0]})")
        print("  " + "─" * 74)
        for r in t.itertuples():
            verdict = "CÓ Ý NGHĨA" if r.significant_holm else "chưa đủ bằng chứng"
            print(f"\n  ▸ {r.proposed}  vs  {r.baseline}   [{r.metric}]")
            print(
                f"      {r.baseline_mean:.4f}±{r.baseline_std:.4f}  →  "
                f"{r.proposed_mean:.4f}±{r.proposed_std:.4f}   "
                f"(giảm {r.improvement_abs:+.4f}, {r.improvement_pct:+.2f}%)"
            )
            print(f"      Thắng {r.wins}/{r.n_seeds} seed")
            print(
                f"      p (paired,  2-sided) = {r.p_paired_2sided:.2e}   "
                f"→ Holm = {r.p_paired_holm:.2e} {stars(r.p_paired_holm)}"
            )
            print(
                f"      p (Welch,   2-sided) = {r.p_welch_2sided:.2e}   "
                f"→ Holm = {r.p_welch_holm:.2e} {stars(r.p_welch_holm)}"
            )
            print(
                f"      p (Wilcoxon 2-sided) = {r.p_wilcoxon_2sided:.4f} "
                f"(sàn lý thuyết với n={r.n_seeds}: {r.p_floor_wilcoxon_2sided:.4f})"
            )
            print(f"      Cohen's d_z = {r.cohens_dz_paired:.2f} | Hedges' g = {r.hedges_g_indep:.2f}")
            print(
                f"      CI 95% của chênh lệch: [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]"
                f"{'  (không chứa 0 ✓)' if r.ci_excludes_zero else '  (chứa 0 ✗)'}"
            )
            print(f"      → KẾT LUẬN (sau Holm): {verdict}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=RUNS_CSV)
    p.add_argument("--baseline", default=None, help="Mặc định: tự chọn baseline mạnh nhất theo MAE.")
    p.add_argument("--proposed", nargs="+", default=DEFAULT_PROPOSED)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--out-prefix", default=os.path.join(OUT_DIR, "multiseed"))
    args = p.parse_args()

    if not os.path.exists(args.runs):
        sys.exit(f"❌ Chưa có {args.runs}. Chạy trước: python scripts/run_multi_seed.py")

    df = pd.read_csv(args.runs)
    if "status" in df.columns:
        n_bad = (df.status != "ok").sum()
        if n_bad:
            print(f"⚠️  Bỏ qua {n_bad} lần chạy thất bại.")
        df = df[df.status == "ok"]

    metrics = [m for m in args.metrics if m in df.columns]
    if not metrics:
        sys.exit(f"❌ Không có metric nào trong {args.metrics} tồn tại trong CSV.")

    # Cảnh báo nếu số seed quá ít so với mức cần cho kiểm định phi tham số
    n_seeds = df.groupby(["split_mode", "variant"]).size().max()
    if n_seeds < 6:
        print(
            f"\n⚠️  n = {n_seeds} seed. Wilcoxon hai phía có p nhỏ nhất = "
            f"{2/2**n_seeds:.4f} — KHÔNG THỂ đạt p<0.05 dù thắng tuyệt đối.\n"
            f"    Cần n ≥ 6 để kiểm định phi tham số dùng được. t-test vẫn OK."
        )

    summary = build_summary(df, metrics)

    # Tự chọn baseline mạnh nhất (MAE trung bình thấp nhất trong nhóm baseline)
    baseline = args.baseline
    if baseline is None:
        cand = summary[summary.variant.isin(BASELINE_POOL)]
        if cand.empty:
            sys.exit("❌ Không tìm thấy baseline nào trong dữ liệu.")
        baseline = (
            cand.groupby("variant")["MAE_mean"].mean().idxmin()
            if "MAE_mean" in cand
            else BASELINE_POOL[1]
        )
        print(f"\nℹ️  Baseline mạnh nhất (tự chọn theo MAE trung bình): {baseline}")

    proposed = [v for v in args.proposed if v in set(df.variant)]
    if not proposed:
        sys.exit(f"❌ Không tìm thấy model đề xuất nào trong {args.proposed}.")

    tests = run_tests(df, baseline, proposed, metrics)
    print_report(summary, tests, metrics)

    os.makedirs(OUT_DIR, exist_ok=True)
    f_sum = f"{args.out_prefix}_summary.csv"
    f_tests = f"{args.out_prefix}_significance.csv"
    f_tex = f"{args.out_prefix}_tables.tex"
    summary.to_csv(f_sum, index=False)
    tests.to_csv(f_tests, index=False)
    with open(f_tex, "w", encoding="utf-8") as fh:
        for mode in summary.split_mode.unique():
            fh.write(latex_main_table(summary, tests, mode, metrics) + "\n\n")

    print("\n" + "═" * 78)
    print(f"  ✅ Bảng tổng hợp : {f_sum}")
    print(f"  ✅ Bảng kiểm định: {f_tests}")
    print(f"  ✅ LaTeX cho paper: {f_tex}")
    print("═" * 78)


if __name__ == "__main__":
    main()
