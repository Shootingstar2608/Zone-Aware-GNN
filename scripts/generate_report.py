"""
scripts/generate_report.py
===========================
Sinh report Markdown + LaTeX từ multiseed_runs CSV.

Mở rộng stat_analysis.py, thêm:
  - Breakdown theo horizon (nếu có T_out column)
  - Breakdown theo Non-IID scenario (alpha)
  - Template report cho 10-seed chrono experiment

Output:
  data/results/report_{split_mode}_{timestamp}.md
  data/results/report_{split_mode}_{timestamp}.tex

Chạy:
  python scripts/generate_report.py
  python scripts/generate_report.py --runs data/results/multiseed_runs_v1.csv
  python scripts/generate_report.py --non-iid data/results/non_iid_eval.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.eval_protocol import get_git_commit_hash

OUT_DIR = os.path.join(ROOT, "data", "results")
DEFAULT_METRICS = ["MAE", "RMSE", "MAPE", "WAPE"]
DEFAULT_PROPOSED = ["zone_full", "zone_full_tc", "zone_full_sinc"]
BASELINE_POOL = ["lstm", "gcn_gru", "stgcn"]


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def ci95_mean(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    se = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return (float(x.mean() - t_crit * se), float(x.mean() + t_crit * se))


def cohens_d_paired(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else float("inf")


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s_pool = np.sqrt(
        ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    )
    if s_pool < 1e-12:
        return float("inf")
    d = (a.mean() - b.mean()) / s_pool
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * J)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ══════════════════════════════════════════════════════════════════════════════
# BUILD SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def build_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["split_mode", "variant"]
    if "split_mode" not in df.columns:
        df = df.copy()
        df["split_mode"] = "chrono"

    for keys, g in df.groupby(group_cols, sort=False):
        mode, variant = keys
        row = {"split_mode": mode, "variant": variant, "n_runs": len(g)}
        for m in metrics:
            if m not in g.columns:
                continue
            v = g[m].dropna().to_numpy(dtype=float)
            row[f"{m}_mean"] = v.mean() if len(v) else np.nan
            row[f"{m}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
            lo, hi = ci95_mean(v) if len(v) > 1 else (np.nan, np.nan)
            row[f"{m}_ci95_lo"] = lo
            row[f"{m}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PAIRWISE TESTS
# ══════════════════════════════════════════════════════════════════════════════
def compare_pair(df, mode, baseline, proposed, metric) -> dict | None:
    sub = df[df.split_mode == mode]
    b = sub[sub.variant == baseline][["seed", metric]].dropna()
    p = sub[sub.variant == proposed][["seed", metric]].dropna()
    if len(b) < 2 or len(p) < 2:
        return None

    merged = b.merge(p, on="seed", suffixes=("_base", "_prop")).sort_values("seed")
    x_b = merged[f"{metric}_base"].to_numpy(dtype=float)
    x_p = merged[f"{metric}_prop"].to_numpy(dtype=float)
    n = len(merged)
    if n < 2:
        return None

    diff = x_b - x_p

    t_pair, p_pair = stats.ttest_rel(x_b, x_p)
    t_welch, p_welch = stats.ttest_ind(x_b, x_p, equal_var=False)
    lo, hi = ci95_mean(diff)

    is_paired = mode in ("random_paired", "chrono")

    return {
        "split_mode": mode,
        "metric": metric,
        "baseline": baseline,
        "proposed": proposed,
        "n_seeds": n,
        "baseline_mean": x_b.mean(),
        "proposed_mean": x_p.mean(),
        "improvement_abs": diff.mean(),
        "improvement_pct": 100 * diff.mean() / x_b.mean() if x_b.mean() else np.nan,
        "ci95_low": lo,
        "ci95_high": hi,
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and lo > 0),
        "p_paired": p_pair,
        "p_welch": p_welch,
        "cohens_dz": cohens_d_paired(diff),
        "hedges_g": hedges_g(x_b, x_p),
        "recommended_test": "paired" if is_paired else "welch",
    }


def run_all_tests(df, baseline, proposed_list, metrics):
    rows = []
    for mode in df.split_mode.unique():
        for prop in proposed_list:
            for m in metrics:
                r = compare_pair(df, mode, baseline, prop, m)
                if r:
                    rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)

    # Holm correction per family
    for mode, g in out.groupby("split_mode"):
        for col, newcol in [("p_paired", "p_paired_holm"), ("p_welch", "p_welch_holm")]:
            adj = holm_bonferroni(g[col].tolist())
            out.loc[g.index, newcol] = adj

    out["p_recommended"] = np.where(
        out.recommended_test == "paired", out.p_paired, out.p_welch
    )
    out["p_recommended_holm"] = np.where(
        out.recommended_test == "paired", out.p_paired_holm, out.p_welch_holm
    )
    out["significant_holm"] = out.p_recommended_holm < 0.05
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_markdown(summary, tests, metrics, df, non_iid_df=None) -> str:
    lines = [
        "# Evaluation Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Commit**: `{get_git_commit_hash()}`  ",
        "",
    ]

    # Config
    if "split_mode" in df.columns:
        modes = df.split_mode.unique().tolist()
        lines.append(f"**Split modes**: {', '.join(modes)}  ")
    if "seed" in df.columns:
        seeds = sorted(df.seed.unique().tolist())
        lines.append(f"**Seeds**: {seeds}  ")
    lines.append("")

    # Summary table per mode
    for mode in summary.split_mode.unique():
        s = summary[summary.split_mode == mode].copy()
        lines.append(f"## Results — {mode}")
        lines.append("")

        # Table header
        hdr = "| Model | n |"
        sep = "|---|---:|"
        for m in metrics:
            if f"{m}_mean" in s.columns:
                hdr += f" {m} |"
                sep += "---:|"
        lines.append(hdr)
        lines.append(sep)

        for _, r in s.iterrows():
            row_str = f"| {r['variant']} | {int(r['n_runs'])} |"
            for m in metrics:
                mu = r.get(f"{m}_mean", np.nan)
                sd = r.get(f"{m}_std", np.nan)
                if np.isfinite(mu) and np.isfinite(sd):
                    row_str += f" {mu:.4f} ± {sd:.4f} |"
                else:
                    row_str += " — |"
            lines.append(row_str)
        lines.append("")

    # Statistical tests
    if not tests.empty:
        lines.append("## Statistical Significance")
        lines.append("")
        for _, r in tests.iterrows():
            sig = "✅ significant" if r.get("significant_holm") else "❌ n.s."
            lines.append(
                f"- **{r['proposed']}** vs {r['baseline']} [{r['metric']}]: "
                f"Δ={r['improvement_abs']:+.4f} ({r['improvement_pct']:+.1f}%), "
                f"p={r['p_recommended_holm']:.2e} {stars(r['p_recommended_holm'])}, "
                f"Cohen's d={r['cohens_dz']:.2f}, "
                f"CI95%=[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] "
                f"→ {sig}"
            )
        lines.append("")

    # Horizon breakdown
    horizon_cols = [c for c in df.columns if c.startswith("MAE_") and c.split("_")[1].isdigit()]
    if horizon_cols:
        lines.append("## Horizon Breakdown (Step-by-step MAE/RMSE)")
        lines.append("")
        steps = sorted(list(set(int(c.split("_")[1]) for c in horizon_cols)))
        
        # Build table header
        hdr = "| Model | " + " | ".join(f"MAE t={t} | RMSE t={t}" for t in steps) + " |"
        sep = "|---|" + "---:|---:|"*len(steps)
        lines.append(hdr)
        lines.append(sep)
        
        # Build table body
        for variant, group in df.groupby("variant"):
            row = f"| {variant} |"
            for t in steps:
                mae_m = group[f"MAE_{t}"].mean()
                rmse_m = group[f"RMSE_{t}"].mean()
                row += f" {mae_m:.4f} | {rmse_m:.4f} |"
            lines.append(row)
        lines.append("")

    # Non-IID breakdown
    if non_iid_df is not None and not non_iid_df.empty:
        lines.append("## Non-IID Scenario Breakdown")
        lines.append("")
        if "alpha" in non_iid_df.columns:
            pivot = non_iid_df.groupby(["variant", "alpha"])["MAE"].mean().unstack("alpha")
            lines.append(pivot.to_markdown())
            lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLES
# ══════════════════════════════════════════════════════════════════════════════
def tex_esc(s):
    return str(s).replace("_", "\\_")

def generate_latex(summary, tests, metrics, df, non_iid_df=None) -> str:
    lines = []
    for mode in summary.split_mode.unique():
        s = summary[summary.split_mode == mode].copy()
        n_seeds = int(s.n_runs.max()) if len(s) else 0

        sig = {}
        t_mode = tests[tests.split_mode == mode] if not tests.empty else pd.DataFrame()
        for _, r in t_mode.iterrows():
            sig[(r["proposed"], r["metric"])] = stars(r.get("p_recommended_holm", 1.0))

        best = {m: s[f"{m}_mean"].min() for m in metrics if f"{m}_mean" in s}

        lines.extend([
            f"% split_mode = {mode}, n = {n_seeds} seeds",
            "\\begin{table}[H]",
            "\\centering",
            f"\\caption{{Results (mean $\\pm$ std over {n_seeds} seeds, "
            f"Holm-corrected significance)}}",
            f"\\label{{tab:results-{mode}}}",
            "\\begin{tabular}{l" + "c" * len(metrics) + "}",
            "\\toprule",
            "Model & " + " & ".join(metrics) + " \\\\",
            "\\midrule",
        ])
        for _, r in s.iterrows():
            cells = []
            for m in metrics:
                mu = r.get(f"{m}_mean", np.nan)
                sd = r.get(f"{m}_std", np.nan)
                txt = f"{mu:.4f} $\\pm$ {sd:.4f}"
                if np.isfinite(mu) and abs(mu - best.get(m, np.inf)) < 1e-12:
                    txt = f"\\textbf{{{mu:.4f}}} $\\pm$ {sd:.4f}"
                mark = sig.get((r["variant"], m), "")
                if mark and mark != "n.s.":
                    txt += f"$^{{{mark}}}$"
                cells.append(txt)
            lines.append(f"{tex_esc(r['variant'])} & " + " & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    # 2. Horizon Breakdown Table
    horizon_cols = [c for c in df.columns if c.startswith("MAE_") and c.split("_")[1].isdigit()]
    if horizon_cols:
        steps = sorted(list(set(int(c.split("_")[1]) for c in horizon_cols)))
        chunk_size = 6
        for i in range(0, len(steps), chunk_size):
            chunk_steps = steps[i:i+chunk_size]
            t_start = chunk_steps[0]
            t_end = chunk_steps[-1]

            lines.extend([
                "",
                "\\begin{table}[H]",
                "\\centering",
                f"\\caption{{Horizon Breakdown (t={t_start} to t={t_end})}}",
                f"\\label{{tab:horizon-breakdown-{t_start}-{t_end}}}",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{l" + "cc" * len(chunk_steps) + "}",
                "\\toprule",
                "Model & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{t={t}}}" for t in chunk_steps) + " \\\\",
                "\\cmidrule(lr){2-" + str(2 * len(chunk_steps) + 1) + "}",
                " & " + " & ".join(["MAE & RMSE"] * len(chunk_steps)) + " \\\\",
                "\\midrule",
            ])
            for variant, group in df.groupby("variant"):
                row_cells = []
                for t in chunk_steps:
                    row_cells.append(f"{group[f'MAE_{t}'].mean():.4f}")
                    row_cells.append(f"{group[f'RMSE_{t}'].mean():.4f}")
                lines.append(f"{tex_esc(variant)} & " + " & ".join(row_cells) + " \\\\")
            lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Generate evaluation report")
    p.add_argument("--runs", default=None,
                    help="Path to multiseed_runs CSV. Auto-detect latest _v{N}.csv if not set.")
    p.add_argument("--non-iid", default=None,
                    help="Path to non_iid_eval.csv for scenario breakdown.")
    p.add_argument("--baseline", default=None)
    p.add_argument("--proposed", nargs="+", default=DEFAULT_PROPOSED)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = p.parse_args()

    # Auto-detect latest versioned CSV
    if args.runs is None:
        import glob
        import re
        pattern = os.path.join(OUT_DIR, "multiseed_runs_v*.csv")
        candidates = glob.glob(pattern)
        if candidates:
            # Sort by version number
            def _ver(f):
                m = re.search(r"_v(\d+)\.csv$", f)
                return int(m.group(1)) if m else 0
            args.runs = max(candidates, key=_ver)
        else:
            fallback = os.path.join(OUT_DIR, "multiseed_runs.csv")
            if os.path.exists(fallback):
                args.runs = fallback
            else:
                sys.exit("❌ No multiseed CSV found. Run scripts/run_multi_seed.py first.")

    print(f"📂 Loading: {args.runs}")
    df = pd.read_csv(args.runs)
    if "status" in df.columns:
        df = df[df.status == "ok"]
    if "split_mode" not in df.columns:
        df["split_mode"] = "chrono"

    metrics = [m for m in args.metrics if m in df.columns]
    if not metrics:
        sys.exit(f"❌ No metrics found in {args.metrics}")

    summary = build_summary(df, metrics)

    # Auto-select strongest baseline
    baseline = args.baseline
    if baseline is None:
        cand = summary[summary.variant.isin(BASELINE_POOL)]
        if not cand.empty and "MAE_mean" in cand.columns:
            baseline = cand.groupby("variant")["MAE_mean"].mean().idxmin()
        elif BASELINE_POOL:
            baseline = BASELINE_POOL[1]
        else:
            baseline = "gcn_gru"
    print(f"ℹ️  Baseline: {baseline}")

    proposed = [v for v in args.proposed if v in set(df.variant)]
    tests = run_all_tests(df, baseline, proposed, metrics) if proposed else pd.DataFrame()

    # Non-IID
    non_iid_df = None
    if args.non_iid and os.path.exists(args.non_iid):
        non_iid_df = pd.read_csv(args.non_iid)

    # Generate outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    mode_tag = "_".join(df.split_mode.unique()) if "split_mode" in df.columns else "report"

    md_path = os.path.join(OUT_DIR, f"report_{mode_tag}_{ts}.md")
    tex_path = os.path.join(OUT_DIR, f"report_{mode_tag}_{ts}.tex")

    os.makedirs(OUT_DIR, exist_ok=True)

    md = generate_markdown(summary, tests, metrics, df, non_iid_df)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    tex = generate_latex(summary, tests, metrics, df, non_iid_df)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    # Also save summary CSV
    sum_path = os.path.join(OUT_DIR, f"report_summary_{ts}.csv")
    summary.to_csv(sum_path, index=False)
    if not tests.empty:
        sig_path = os.path.join(OUT_DIR, f"report_significance_{ts}.csv")
        tests.to_csv(sig_path, index=False)

    print(f"\n✅ Report: {md_path}")
    print(f"✅ LaTeX:  {tex_path}")
    print(f"✅ Summary: {sum_path}")


if __name__ == "__main__":
    main()
