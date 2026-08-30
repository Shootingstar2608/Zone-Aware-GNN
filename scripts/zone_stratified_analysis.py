"""
zone_stratified_analysis.py
===========================
Zone-stratified evaluation + Non-IID visualization.

Pipeline:
  1. Load zone_labels + ablation_results + eda_jsd_results
  2. (Optional) Load .pt checkpoints nếu có torch → re-evaluate trên test split
  3. Tính MAE theo zone + đếm node theo zone
  4. Vẽ: zone count bar, zone-stratified MAE bar, JSD heatmap,
        zone similarity heatmap, JSD-Zone scatter, JSD boxplot

Chạy:  python scripts/zone_stratified_analysis.py
"""
import json, os, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    from scipy.stats import entropy, pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def entropy(pk, qk=None, base=None):
        pk = np.asarray(pk, dtype=float)
        qk = np.asarray(pk, dtype=float) if qk is None else np.asarray(qk, dtype=float)
        pk = pk / pk.sum() if pk.sum() > 0 else pk
        qk = qk / qk.sum() if qk.sum() > 0 else qk
        m = pk * np.log(pk / qk)
        m[~np.isfinite(m)] = 0
        return float(np.sum(m))
    def pearsonr(x, y):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        n = len(x); xm = x - x.mean(); ym = y - y.mean()
        denom = np.sqrt((xm*xm).sum() * (ym*ym).sum())
        if denom == 0: return 0.0, 1.0
        r = float((xm*ym).sum() / denom)
        if abs(r) >= 1: p = 0.0
        else:
            from math import sqrt
            t = r * sqrt((n-2) / max(1e-9, 1 - r*r))
            x_ = t*t / max(1, n-2)
            p_ = 1.0
            for k in range(20, 0, -1):
                p_ = 1 + x_*p_/(n-2+2*k-2)
            p_ = 1.0/p_ if p_ else 1.0
            p = min(1.0, p_)
        return r, p
    def spearmanr(x, y):
        rx = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
        ry = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
        return pearsonr(rx, ry)
    print("[info] scipy not available - using numpy-only fallback")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR, RAW_DIR, RESULTS_DIR = ROOT/"data", ROOT/"data/raw", ROOT/"data/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ZONE_TYPES = ["commercial","residential","industrial","school",
              "university","hospital","transport","park"]
ABLATION_VARIANTS = {
    "baseline_ahgnn":(False,False,False),
    "zone_concat":   (True, False,False),
    "zone_weight":   (True, True, False),
    "zone_full":     (True, True, True),
}

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[info] torch not available → using ablation_results.csv directly")

# ── LOADERS ────────────────────────────────────────────────────────────────
def load_zone_labels():
    return pd.read_csv(RAW_DIR/"zone_labels.csv", index_col="node")[ZONE_TYPES]

def load_jsd_data():
    with open(RESULTS_DIR/"eda_jsd_results.json") as f:
        eda = json.load(f)
    return eda["nodes"], np.array(eda["jsd_matrix"]), np.array(eda["zone_sim_matrix"])

def load_ablation():
    return pd.read_csv(RESULTS_DIR/"ablation_results.csv")

def compute_zone_group_stats(zdf):
    rows=[]
    N=len(zdf)
    for z in ZONE_TYPES:
        n=int(zdf[z].sum())
        rows.append({"zone_type":z,"num_nodes":n,"total_assignments":n,
                     "pct_nodes":round(100*n/N,1)})
    return pd.DataFrame(rows).sort_values("num_nodes",ascending=False)

def js_div(P,Q):
    e=1e-10; P=(P+e)/(P+e).sum(); Q=(Q+e)/(Q+e).sum()
    M=0.5*(P+Q)
    return 0.5*entropy(P,M)+0.5*entropy(Q,M)

def recompute_jsd(nodes, df_tt, bins=None):
    if bins is None: bins=np.linspace(0.9,3.0,50)
    h={}
    for n in nodes:
        sub=df_tt[df_tt["src_name"]==n]["congestion_ratio"].values
        h_,_=np.histogram(sub,bins=bins,density=True)
        h_=h_+1e-10; h[n]=h_/h_.sum()
    N=len(nodes); J=np.zeros((N,N))
    for i,u in enumerate(nodes):
        for j,v in enumerate(nodes):
            if i!=j: J[i,j]=js_div(h[u],h[v])
    return J

def compute_zone_cosine_sim(zdf):
    Z=zdf.values.astype(float)
    n=np.linalg.norm(Z,axis=1,keepdims=True); n[n==0]=1
    return (Z/n)@(Z/n).T

def collect_pair_stats(J,Z,nodes):
    iu=np.triu_indices(len(nodes),k=1)
    return pd.DataFrame({
        "node_u":[nodes[i] for i in iu[0]],
        "node_v":[nodes[j] for j in iu[1]],
        "jsd":J[iu], "zone_similarity":Z[iu],
        "same_primary":(Z[iu]>0.7).astype(int),
        "completely_diff":(Z[iu]==0).astype(int),
    })

# ── MODEL EVAL (optional) ──────────────────────────────────────────────────
def build_model_factory(variant, meta, *args):
    from models.zone_aware_gnn import ZoneAwareAHGNN
    from models.ah_gnn import AH_GNN
    from models.baselines import LSTMBaseline, GCNGRUBaseline, STGCNBaseline
    N,K,F_,T_in,T_out = meta["N"],meta["K"],meta["F"],meta["T_in"],meta["T_out"]
    in_ch=T_in*F_
    if variant=="baseline_ahgnn":
        return AH_GNN(N,in_ch,64,T_out,32,4,2)
    m=ZoneAwareAHGNN(N,K,in_ch,64,T_out,32,16,4,2)
    use_emb,use_w,use_a=args
    m.use_zone_weight=use_w; m.use_zone_adj=use_a
    if not use_emb:
        for p in m.zone_emb.parameters():
            p.requires_grad_(False); p.data.zero_()
    return m

try:
    _no_grad = torch.no_grad()
except NameError:
    class _no_grad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

def _safe_no_grad(func):
    def wrapper(*a, **kw):
        if HAS_TORCH:
            with torch.no_grad():
                return func(*a, **kw)
        return func(*a, **kw)
    return wrapper

@_safe_no_grad
def evaluate_zone_stratified(variant, ckpt, dataset, meta):
    use_emb,use_w,use_a=ABLATION_VARIANTS[variant]
    model=build_model_factory(variant,meta,use_emb,use_w,use_a)
    model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=False))
    model.eval()
    X,Y,TL,Z,A = dataset["X"],dataset["Y"],dataset["time_labels"],dataset["Z"],dataset["A"]
    from torch.utils.data import TensorDataset, random_split
    S=X.size(0); nt=int(S*0.7); nv=int(S*0.1); ns=S-nt-nv
    _,_,test_ds = random_split(TensorDataset(X,Y,TL),[nt,nv,ns],
        generator=torch.Generator().manual_seed(42))
    preds,trues=[],[]
    for Xb,Yb,Tb in test_ds:
        preds.append(model(Xb,Z,Tb,A)); trues.append(Yb)
    preds=torch.cat(preds); trues=torch.cat(trues)
    mae=(preds-trues).abs().mean().item()
    rmse=((preds-trues)**2).mean().sqrt().item()
    mask=trues.abs()>1e-5
    mape=((preds-trues).abs()/(trues.abs()+1e-8))[mask].mean().item()*100
    Znp=Z.cpu().numpy(); zm={}
    for k,z in enumerate(ZONE_TYPES):
        nm=Znp[:,k]==1
        if nm.sum()==0: continue
        zm[f"MAE_{z}"]=(preds[:,nm,:]-trues[:,nm,:]).abs().mean().item()
    mm=Znp.sum(1)>1
    if mm.sum()>0: zm["MAE_multi_zone"]=(preds[:,mm,:]-trues[:,mm,:]).abs().mean().item()
    np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"MAE":mae,"RMSE":rmse,"MAPE":mape,**zm,"variant":variant,"n_params":np_}

# ── VISUALIZATION ──────────────────────────────────────────────────────────
def plot_zone_count_bar(zs,out):
    fig,ax=plt.subplots(figsize=(10,5))
    clrs=sns.color_palette("viridis",len(zs))
    bars=ax.barh(zs["zone_type"],zs["num_nodes"],color=clrs,edgecolor="black")
    for b,n,p in zip(bars,zs["num_nodes"],zs["pct_nodes"]):
        ax.text(b.get_width()+0.15,b.get_y()+b.get_height()/2,
                f"{n} nodes ({p}%)",va="center",fontsize=10)
    ax.set_xlabel("Number of Nodes"); ax.invert_yaxis()
    ax.set_title("Zone Type Distribution — Multi-label TAZ",
                 fontsize=13,fontweight="bold")
    ax.grid(axis="x",linestyle="--",alpha=0.4)
    plt.tight_layout(); plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")

def plot_zone_stratified_mae(zmd,out):
    dp=zmd.melt(id_vars=["zone_type","n_nodes"],var_name="variant",value_name="MAE")
    fig,ax=plt.subplots(figsize=(12,6))
    sns.barplot(data=dp,x="zone_type",y="MAE",hue="variant",palette="Set2",ax=ax)
    ax.set_title("Zone-Stratified MAE — 4 Ablation Variants",
                 fontsize=13,fontweight="bold")
    ax.set_ylabel("MAE (test set)"); ax.set_xlabel("Zone type")
    ax.legend(title="Variant",bbox_to_anchor=(1.02,1),loc="upper left")
    for i,(zt,nn) in enumerate(zip(zmd["zone_type"],zmd["n_nodes"])):
        ax.text(i,ax.get_ylim()[1]*0.97,f"n={nn}",ha="center",va="top",
                fontsize=9,color="dimgray",
                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.7))
    plt.xticks(rotation=20,ha="right"); plt.tight_layout()
    plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")

def plot_jsd_heatmap(J,nodes,out):
    fig,ax=plt.subplots(figsize=(11,9))
    sns.heatmap(J,xticklabels=nodes,yticklabels=nodes,cmap="YlOrRd",annot=False,
                ax=ax,cbar_kws={"label":"JSD (nats)"})
    ax.set_title("JSD Heatmap — Phân bố PDF congestion_ratio giữa các node",
                 fontsize=13,fontweight="bold")
    plt.xticks(rotation=45,ha="right",fontsize=8)
    plt.yticks(fontsize=8); plt.tight_layout()
    plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")

def plot_zone_sim_heatmap(Z,nodes,out):
    fig,ax=plt.subplots(figsize=(11,9))
    sns.heatmap(Z,xticklabels=nodes,yticklabels=nodes,cmap="Blues",annot=False,
                ax=ax,vmin=0,vmax=1,cbar_kws={"label":"Cosine Similarity"})
    ax.set_title("Zone Label Cosine Similarity — Multi-hot TAZ giữa các node",
                 fontsize=13,fontweight="bold")
    plt.xticks(rotation=45,ha="right",fontsize=8)
    plt.yticks(fontsize=8); plt.tight_layout()
    plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")

def plot_jsd_zone_corr(df,out):
    fig,ax=plt.subplots(figsize=(9,6))
    sim=df[df["zone_similarity"]>0.5]
    mid=df[(df["zone_similarity"]>0)&(df["zone_similarity"]<=0.5)]
    dif=df[df["zone_similarity"]==0]
    ax.scatter(sim["zone_similarity"],sim["jsd"],color="#2ecc71",alpha=0.65,s=50,
               edgecolor="darkgreen",label=f"Similar (sim>0.5) n={len(sim)}")
    ax.scatter(mid["zone_similarity"],mid["jsd"],color="#f39c12",alpha=0.55,s=40,
               edgecolor="darkorange",label=f"Mid (0<sim≤0.5) n={len(mid)}")
    ax.scatter(dif["zone_similarity"],dif["jsd"],color="#e74c3c",alpha=0.7,s=55,
               marker="X",edgecolor="darkred",label=f"Different (sim=0) n={len(dif)}")
    sns.regplot(data=df,x="zone_similarity",y="jsd",scatter=False,
                color="steelblue",ax=ax,
                line_kws={"linewidth":2,"alpha":0.7,"linestyle":"--"})
    rp,pp=pearsonr(df["zone_similarity"],df["jsd"])
    rs,ps=spearmanr(df["zone_similarity"],df["jsd"])
    ax.text(0.04,0.96,
        f"Pearson r  = {rp:+.3f}  (p={pp:.2e})\n"
        f"Spearman ρ = {rs:+.3f}  (p={ps:.2e})\n"
        f"N pairs    = {len(df)}",
        transform=ax.transAxes,fontsize=10,va="top",
        bbox=dict(boxstyle="round,pad=0.5",facecolor="lightyellow",
                  edgecolor="gray",alpha=0.95))
    ax.set_xlabel("Zone Label Cosine Similarity")
    ax.set_ylabel("JSD — Phân kỳ phân phối congestion_ratio")
    ax.set_title("Correlation: Zone Similarity ↔ JSD (Non-IID evidence)",
                 fontsize=13,fontweight="bold")
    ax.grid(True,linestyle="--",alpha=0.4)
    ax.legend(loc="upper right",fontsize=10,framealpha=0.95)
    plt.tight_layout(); plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")

def plot_jsd_boxplot(df,out):
    df=df.copy()
    def cl(sim):
        if sim>0.5: return "Similar (sim>0.5)"
        if sim==0:  return "Different (sim=0)"
        return "Mid (0<sim≤0.5)"
    df["group"]=df["zone_similarity"].apply(cl)
    fig,ax=plt.subplots(figsize=(9,5))
    order=["Similar (sim>0.5)","Mid (0<sim≤0.5)","Different (sim=0)"]
    sns.boxplot(data=df,x="group",y="jsd",order=order,palette="Set2",ax=ax,width=0.55)
    sns.stripplot(data=df,x="group",y="jsd",order=order,color="black",alpha=0.4,size=3,ax=ax)
    ax.set_xlabel("Nhóm Zone Similarity"); ax.set_ylabel("JSD")
    ax.set_title("Phân bố JSD theo nhóm tương đồng zone — Non-IID evidence",
                 fontsize=13,fontweight="bold")
    ax.grid(axis="y",linestyle="--",alpha=0.4)
    plt.tight_layout(); plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  ✓ {out.name}")

# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("  Zone-Stratified Analysis — Model evaluation + Non-IID evidence")
    print("="*70)

    print("\n[1/6] Loading zone labels...")
    zdf=load_zone_labels()
    zs=compute_zone_group_stats(zdf)
    print(zs.to_string(index=False))
    zs.to_csv(RESULTS_DIR/"zone_group_stats.csv",index=False)

    print("\n[2/6] Loading JSD + zone similarity matrices...")
    nodes,Jmat,Zmat=load_jsd_data()

    print("\n[3/6] Loading ablation_results.csv...")
    ab=load_ablation()
    print(ab[["variant","MAE","RMSE","MAPE","MAE_multi_zone"]].to_string(index=False))

    print("\n[4/6] Re-evaluating models on test set...")
    eval_df=ab.copy()
    if HAS_TORCH and (RESULTS_DIR/"zone_full_best.pt").exists():
        with open(DATA_DIR/"processed/meta.json") as f: meta=json.load(f)
        ds=torch.load(DATA_DIR/"processed/graph_dataset.pt",weights_only=False)
        re=[]
        for v in ABLATION_VARIANTS:
            ck=RESULTS_DIR/f"{v}_best.pt"
            if ck.exists():
                print(f"  → Loading {v}...")
                re.append(evaluate_zone_stratified(v,ck,ds,meta))
        if re:
            eval_df=pd.DataFrame(re)
            eval_df.to_csv(RESULTS_DIR/"evaluation_test_set.csv",index=False)
    else:
        print("  [info] Using ablation_results.csv directly")

    print("\n[5/6] Building zone-stratified MAE table...")
    rows=[]
    for z in ZONE_TYPES:
        c=f"MAE_{z}"
        if c in eval_df.columns:
            r={"zone_type":z,"n_nodes":int(zs.loc[zs.zone_type==z,"num_nodes"].iloc[0])}
            for v in ABLATION_VARIANTS:
                sub=eval_df[eval_df.variant==v]
                if len(sub) and c in sub.columns:
                    r[v]=float(sub[c].iloc[0])
            rows.append(r)
    zmd=pd.DataFrame(rows)
    if not zmd.empty:
        zmd.to_csv(RESULTS_DIR/"zone_stratified_mae.csv",index=False)
        print(zmd.to_string(index=False))

    print("\n[6/6] Generating visualizations...")
    plot_zone_count_bar(zs, RESULTS_DIR/"zone_count_bar.png")
    if not zmd.empty and len(zmd.columns)>2:
        plot_zone_stratified_mae(zmd, RESULTS_DIR/"zone_stratified_mae.png")
    plot_jsd_heatmap(Jmat,nodes, RESULTS_DIR/"eda_jsd_heatmap.png")
    plot_zone_sim_heatmap(Zmat,nodes, RESULTS_DIR/"zone_similarity_heatmap.png")
    df_pairs=collect_pair_stats(Jmat,Zmat,nodes)
    df_pairs.to_csv(RESULTS_DIR/"jsd_zone_pairs.csv",index=False)
    plot_jsd_zone_corr(df_pairs, RESULTS_DIR/"eda_jsd_correlation.png")
    plot_jsd_boxplot(df_pairs, RESULTS_DIR/"jsd_boxplot_by_group.png")

    print("\n"+"="*70); print("  SUMMARY"); print("="*70)
    print(f"\n📊 Zone groups:")
    print(f"  - Total nodes       : {len(zdf)}")
    print(f"  - Multi-zone nodes  : {int((zdf.sum(1)>1).sum())} ({100*(zdf.sum(1)>1).mean():.1f}%)")
    print(f"  - Mean cardinality  : {zdf.sum(1).mean():.2f}")
    iu=np.triu_indices(len(nodes),k=1)
    print(f"\n📊 JSD off-diag: mean={Jmat[iu].mean():.4f}  max={Jmat[iu].max():.4f}  min={Jmat[iu].min():.4f}")
    print(f"📊 Zone sim off-diag: mean={Zmat[iu].mean():.4f}  pairs(sim=0)={(Zmat[iu]==0).sum()}  pairs(sim>0.5)={(Zmat[iu]>0.5).sum()}")
    mjs_sim=df_pairs[df_pairs.zone_similarity>0.5]["jsd"].mean()
    mjs_dif=df_pairs[df_pairs.zone_similarity==0]["jsd"].mean()
    print(f"\n🔬 Non-IID evidence:")
    print(f"  JSD(sim>0.5)={mjs_sim:.4f}  JSD(sim=0)={mjs_dif:.4f}  Δ={mjs_dif-mjs_sim:+.4f}")
    rp,pp=pearsonr(df_pairs["zone_similarity"],df_pairs["jsd"])
    print(f"\n📈 Correlation: Pearson r={rp:+.4f}  p={pp:.2e}  ({'significant' if pp<0.05 else 'not significant'})")
    print(f"\n📁 Output → {RESULTS_DIR}")
    print("="*70)

if __name__=="__main__":
    main()
