#!/usr/bin/env python3
"""分子层面可解释性 — 泡两个方向, 看哪个强。
方向1: 预测的细胞型特异表达是否恢复各细胞的经典分子程序(细胞型×程序 应块对角)。
方向2: 预后细胞(Fibro差/T·B好)的分子基础 — 预测丰度 ~ 分子程序 ~ 生存。
"""
import os, re, json, numpy as np, pandas as pd, torch, sys
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
sys.path.insert(0,"code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
import warnings; warnings.filterwarnings("ignore")
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN,CRUN="run/TCGA_HNSC","run/CPTAC_HNSC"          # RUN 仅用于共享数据(metadata/deconv bulk)
V2B="run/TCGA_HNSC_oro_v2b"; RES=f"{V2B}/results"  # 模型权重/基因表/细胞型 一律用 oro_v2b (8型)
CTS=list(pd.read_csv(f"{V2B}/cell_types_file.csv")["cell_types"])

# 经典分子程序(marker 基因)
PROG={"Proliferation":["MKI67","TOP2A","CCNB1","CDK1","BUB1","AURKA","PLK1","CCNA2","UBE2C","CDC20"],
 "ECM/CAF":["COL1A1","COL1A2","COL3A1","FN1","DCN","LUM","POSTN","FAP","PDGFRB","THBS2"],
 "TGF-β/myofib":["TGFB1","ACTA2","TAGLN","MYL9","TPM1","CTGF","SERPINE1"],
 "Cytotoxic-T":["CD8A","GZMB","GZMA","PRF1","NKG7","IFNG","CD3D","CD3E"],
 "B/plasma/TLS":["MS4A1","CD79A","CD79B","IGHG1","IGHM","CXCL13","CR2","FDCSP"],
 "Vascular":["PECAM1","VWF","CDH5","CLDN5","FLT1","KDR"],
 "Myeloid":["CD68","CD163","CSF1R","LYZ","AIF1","MRC1"],
 "Keratin/SCC":["KRT5","KRT14","KRT6A","SFN","TP63","DSG3","PERP"]}

# ---- CPTAC 预测细胞型表达 ----
cf=np.load(f"{CRUN}/features.pkl",allow_pickle=True); sids=[f[0] for f in cf]
CX=[torch.tensor(np.asarray(f[1]),dtype=torch.float32) for f in cf]
meta=pd.read_csv(f"{CRUN}/metadata.csv"); s2c=dict(zip(meta["Slide.ID"],meta["case"]))
def predict_ct(ct):
    gf=f"{V2B}/breast_{ct}_gene_file.csv"
    if not os.path.exists(gf): return None
    genes=list(pd.read_csv(gf)["gene"]); pred=np.zeros((len(CX),len(genes))); n=0
    for ik in range(5):
        mp=f"{RES}/{ct}_preds/result_{ik}_0/model_trained.pth"
        if not os.path.exists(mp): continue
        m=MLP_regression(768,768,len(genes),0.2,None); m.load_state_dict(torch.load(mp,map_location=dev)); m.to(dev).eval()
        with torch.no_grad(): pred+=np.vstack([m(x.to(dev)).cpu().numpy() for x in CX]); n+=1
    if n==0: return None
    pred/=n; dp=pd.DataFrame(pred,index=[s2c.get(s) for s in sids],columns=genes); return dp.groupby(level=0).mean()
print("predict cell-type expr (CPTAC)...")
preds={ct:predict_ct(ct) for ct in CTS}; preds={k:v for k,v in preds.items() if v is not None}

# ===== 方向1: 细胞型 × 程序 (每细胞型预测表达里, 各程序的平均表达水平) =====
def prog_level(dp, genes):
    g=[x for x in genes if x in dp.columns]
    if len(g)<3: return np.nan
    # 该程序基因在所有病人上的平均预测表达(log1pCPM), 再对该细胞型做跨程序标准化在下方
    return dp[g].mean(axis=1).mean()
M=pd.DataFrame(index=list(preds.keys()),columns=list(PROG.keys()),dtype=float)
for ct,dp in preds.items():
    for pn,gs in PROG.items(): M.loc[ct,pn]=prog_level(dp,gs)
# 每列(程序)跨细胞型 z-score -> 看哪个细胞型该程序最高
Mz=(M-M.mean())/(M.std()+1e-9)
print("\n=== 方向1: 细胞型×程序 (列z-score, 每程序最高的细胞型) ===")
for pn in PROG: print(f"  {pn:16s} 最高: {Mz[pn].idxmax():12s} (z={Mz[pn].max():.2f})")
M.to_csv("figures/paper/cache/interp_mol_celltype_prog.csv")

# ===== 方向2: 预后细胞的分子基础 (TCGA) =====
# TCGA OOF 预测丰度
feats=np.load(f"{V2B}/features.pkl",allow_pickle=True); tsids=[f[0] for f in feats]
tm=pd.read_csv(f"{RUN}/metadata.csv"); ts2c={str(r["Slide.ID"]):str(r["case"]) for _,r in tm.iterrows()}
tix=np.load("run/TCGA_HNSC_oro_v2b/splits/train_valid_test_idx.npz",allow_pickle=True)["test_idx"]
rows={}
for ik in range(5):
    pr=np.loadtxt(f"run/TCGA_HNSC_oro_v2b/results/cell_fraction_preds/result_{ik}_0/test_preds.txt")
    for j,gi in enumerate(tix[ik]): rows.setdefault(ts2c.get(tsids[gi],tsids[gi]),[]).append(pr[j])
CTS8=list(pd.read_csv(f"{V2B}/cell_types_file.csv")["cell_types"])  # 8型(cell_fraction)
assert CTS8==CTS, (CTS8,CTS)
tab=pd.DataFrame({c:np.mean(v,axis=0) for c,v in rows.items()}).T
assert tab.shape[1]==len(CTS8), (tab.shape,len(CTS8))
tab.columns=CTS8
# TCGA 真实 bulk RNA -> 程序分
trna=pd.read_csv(f"{RUN}/deconv/tcga_bulk_counts.tsv.gz",sep="\t",index_col=0).T
tcpm=np.log1p(trna.div(trna.sum(1),axis=0)*1e6)
def sigscore(mat,gs):
    g=[x for x in gs if x in mat.columns]; z=(mat[g]-mat[g].mean())/(mat[g].std()+1e-9); return z.mean(axis=1)
sv=pd.read_csv("run/TCGA_HNSC_oro_v2b/survival.csv",dtype={'case':str}).set_index("case"); sv.index=sv.index.astype(str)
zc=lambda x:(x-np.nanmean(x))/(np.nanstd(x)+1e-9)
print("\n=== 方向2: 预后细胞 预测丰度 ~ 分子程序 ~ 生存 (TCGA) ===")
# 每个(预后细胞, 相关程序): 预测丰度 vs 程序分 相关; 程序分 的生存HR
LINKS=[("Fibroblast","ECM/CAF"),("Fibroblast","TGF-β/myofib"),("T cell","Cytotoxic-T"),("B cell","B/plasma/TLS")]
res2=[]
for ct,pn in LINKS:
    prog=sigscore(tcpm,PROG[pn])
    com=[c for c in tab.index if c in prog.index]
    r=spearmanr(tab.loc[com,ct].values, prog.loc[com].values).correlation   # 预测丰度 vs 程序
    # 程序分 生存
    cs=[c for c in prog.index if c in sv.index]
    d=pd.DataFrame({"T":sv.loc[cs,"time_days"].values,"E":sv.loc[cs,"event"].values,"z":zc(prog.loc[cs].values)}).dropna()
    s=CoxPHFitter().fit(d,"T","E").summary.loc["z"]; hr,p=float(np.exp(s["coef"])),float(s["p"])
    res2.append({"cell":ct,"program":pn,"abund_vs_prog_r":float(r),"prog_HR":hr,"prog_p":p})
    print(f"  {ct:11s} ~ {pn:14s}: 预测丰度·程序 r={r:+.2f} | 程序生存 HR={hr:.2f} p={p:.3f}")

json.dump({"celltype_prog":M.to_dict(),"celltype_prog_z":Mz.to_dict(),"links":res2},
          open("figures/paper/cache/interp_molecular.json","w"),default=float)
print("\n已存 cache/interp_molecular.json")
