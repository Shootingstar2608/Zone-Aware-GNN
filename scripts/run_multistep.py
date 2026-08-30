import os
import json
import torch
import pandas as pd
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train

def main():
    models = ["zone_full_tc", "zone_full_sinc", "gcn_gru", "stgcn"]
    # ===================================================================================================================================
    # 15p=3, 30p=6, 45p=9, 60p=12, 90p=18, 120p=24
    t_outs = [3, 6, 9, 12, 18, 24]
    # ===================================================================================================================================

    results = []

    out_dir = "data/results/multistep"
    os.makedirs(out_dir, exist_ok=True)

    for t_out in t_outs:
        print(f"\n{'#'*60}")
        print(f"### Running for T_out = {t_out} ({(t_out)*5} mins)")
        print(f"{'#'*60}\n")

        # ===================================================================================================================================
        # Create subfolder for specific time horizon
        t_out_dir = os.path.join(out_dir, f"T_{t_out}")
        os.makedirs(t_out_dir, exist_ok=True)
        train.OUT_DIR = t_out_dir  # Model weights for this T_out go here
        # ===================================================================================================================================
        
        # 1. Build graph
        print(f"Building graph for T_out={t_out}...")
        subprocess.run([sys.executable, "scripts/rebuild_graph_no_raw.py", str(t_out)], check=True)
        
        # 2. Load dataset and meta
        dataset_path = "data/processed/graph_dataset.pt"
        meta_path = "data/processed/meta.json"
        
        dataset = torch.load(dataset_path, weights_only=False)
        with open(meta_path) as f:
            meta = json.load(f)
            
        # 3. Train models
        for vname in models:
            if vname in train.ABLATION_VARIANTS:
                vcfg = train.ABLATION_VARIANTS[vname]
            else:
                vcfg = (False, False, False)
                
            print(f"\n--- Training {vname} (T_out={t_out}) ---")
            res = train.run_experiment(vname, meta, dataset, vcfg)
            res["T_out"] = t_out
            results.append(res)
            
    df = pd.DataFrame(results)
    
    # Reorder columns to make T_out and variant first
    cols = ["T_out", "variant"] + [c for c in df.columns if c not in ["T_out", "variant"]]
    df = df[cols]
    
    # =====
    csv_path = os.path.join(out_dir, "multistep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved multistep results to {csv_path}")
    
    # Generate Full Markdown Table (split by T_out)
    md_content = "# Multi-Step Forecasting Results\n\n"
    
    zone_cols = [c for c in df.columns if c.startswith("MAE_") and c != "MAE_multi_zone"]
    
    for t_out in df['T_out'].unique():
        horizon = f"{int(t_out) * 5} min"
        md_content += f"## Horizon {horizon} (T_out = {t_out})\n\n"
        
        # Build table header
        header = "| Model | Params | MAE | RMSE | MAPE (%) | MZ MAE | " + " | ".join([c.replace("MAE_", "").capitalize() for c in zone_cols]) + " |\n"
        separator = "|:---|---:|:---:|:---:|:---:|:---:|:---:" + "|:---:" * (len(zone_cols) - 1) + "|\n"
        md_content += header + separator
        
        # Add rows for this horizon
        df_t = df[df['T_out'] == t_out]
        for _, row in df_t.iterrows():
            params = f"{int(row.get('n_params', 0)):,}"
            mz = row.get('MAE_multi_zone', float('nan'))
            zone_vals = [f"{row.get(c, float('nan')):.4f}" for c in zone_cols]
            
            row_str = f"| `{row['variant']}` | {params} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['MAPE']:.2f}% | {mz:.4f} | " + " | ".join(zone_vals) + " |\n"
            md_content += row_str
            
        md_content += "\n"
        
    md_path = os.path.join(out_dir, "multistep_results.md")
    with open(md_path, "w") as f:
        f.write(md_content)
        
    print(f"✅ Saved detailed markdown to {md_path}")
    print(f"\nTraining completed! All outputs (CSV, MD, and Models) are grouped in '{out_dir}'.")
    # ======

if __name__ == "__main__":
    main()
