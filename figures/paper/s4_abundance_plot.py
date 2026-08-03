#!/usr/bin/env python3
"""S4 — 细胞丰度全账 + 双模态 marker 验证。
(a) 8型 内部CV Spearman + CPTAC迁移(vs θ)  (b) CPTAC 预测丰度 vs 独立蛋白/RNA marker。"""
import os, json, numpy as np, pandas as pd
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
from scipy.stats import spearmanr, t as tdist
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
    "font.size":8,"axes.linewidth":0.8,"axes.titlesize":9,"axes.labelsize":8.5,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
PAL={"prot":"#2c7fb8","risk":"#d95f0e","tcga":"#4477AA","cptac":"#CC6677","hancock":"#228833","grey":"#999999"}
def tag(ax,s): ax.text(-0.16,1.06,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top",ha="left")

internal=json.load(open("figures/paper/cache/abundance_internal.json"))
transfer=json.load(open("figures/paper/cache/transfer_cptac.json"))

NB=2000; SEED=0        # 所有 bootstrap: 2000 次重采样, 固定种子 0
def boot_spearman_ci(x,y,nb=NB,seed=SEED):
    """患者级 bootstrap 95% CI of Spearman rho."""
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=~(np.isnan(x)|np.isnan(y)); x=x[m]; y=y[m]; n=len(x)
    rng=np.random.RandomState(seed); bs=[]
    for _ in range(nb):
        i=rng.randint(0,n,n)
        if np.std(x[i])<1e-12 or np.std(y[i])<1e-12: continue
        r=spearmanr(x[i],y[i]).correlation
        if not np.isnan(r): bs.append(r)
    return float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))

# ---- (a) 内部CV: 5折各自的 Spearman -> t 型 95% CI ----
CTS_ORDER=list(pd.read_csv("run/TCGA_HNSC_oro_v2b/cell_types_file.csv")["cell_types"])
_fold_rho={ct:[] for ct in CTS_ORDER}
for ik in range(5):
    d=f"run/TCGA_HNSC_oro_v2b/results/cell_fraction_preds/result_{ik}_0"
    P=np.loadtxt(f"{d}/test_preds.txt"); L=np.loadtxt(f"{d}/test_labels.txt")
    for i,ct in enumerate(CTS_ORDER):
        _fold_rho[ct].append(float(spearmanr(P[:,i],L[:,i]).correlation))
def fold_ci(ct):
    v=np.array(_fold_rho[ct]); k=len(v)
    h=tdist.ppf(0.975,k-1)*v.std(ddof=1)/np.sqrt(k)
    return float(v.mean()-h),float(v.mean()+h)
INT_CI={ct:fold_ci(ct) for ct in CTS_ORDER}

# ---- (a) CPTAC 迁移: 患者级 bootstrap 95% CI ----
_ab=pd.read_csv("run/CPTAC_HNSC/pred_abundance.csv",index_col=0); _ab.index=_ab.index.astype(str)
_th=pd.read_csv("run/CPTAC_HNSC/deconv/bayesprism_oropharyngeal/theta.csv",index_col=0); _th.index=_th.index.astype(str)
_com=[c for c in _ab.index if c in _th.index]
TR_CI={ct:boot_spearman_ci(_ab.loc[_com,ct].values,_th.loc[_com,ct].values) for ct in CTS_ORDER}

# ---- 双模态 marker 验证(CPTAC 预测丰度 vs 蛋白/RNA marker 签名)----
pred=pd.read_csv("run/CPTAC_HNSC/pred_abundance.csv",index_col=0); pred.index=pred.index.astype(str)
import cptac
prot=cptac.Hnscc().get_proteomics("umich"); prot.columns=prot.columns.get_level_values(0)
prot=prot.loc[[s for s in prot.index if not str(s).endswith(".N")]]; prot=prot.loc[:,~prot.columns.duplicated()]
rna=pd.read_csv("run/CPTAC_HNSC/deconv/cptac_bulk_counts.tsv.gz",sep="\t",index_col=0).T
cpm=np.log1p(rna.div(rna.sum(1),axis=0)*1e6)
MK={"Malignant":["EPCAM","KRT5","KRT14","KRT6A","SFN","TP63"],"B cell":["MS4A1","CD19","CD79A","CD79B","BANK1"],
 "Dendritic":["CLEC9A","LILRA4","CD1C","ITGAX","BATF3","IRF8"],"Endothelial":["PECAM1","VWF","CDH5","CLDN5","FLT1","KDR"],
 "Fibroblast":["COL1A1","COL1A2","DCN","LUM","PDGFRB","FAP"],"Macrophage":["CD68","CD163","CSF1R","LYZ","AIF1","MRC1"],
 "Mast":["TPSAB1","TPSB2","CPA3","MS4A2","KIT"],"T cell":["CD3D","CD3E","CD2","CD8A","TRAC","LCK"]}
def sig(m,gs): a=[g for g in gs if g in m.columns]; return m[a].mean(axis=1)
def sp(a,b):
    idx=[i for i in a.index if i in b.index]; x=a.loc[idx].values; y=b.loc[idx].values
    mm=~(np.isnan(x)|np.isnan(y)); return spearmanr(x[mm],y[mm]).correlation
def sp_ci(a,b):
    idx=[i for i in a.index if i in b.index]
    return boot_spearman_ci(a.loc[idx].values,b.loc[idx].values)
mk={ct:{"prot":sp(pred[ct],sig(prot,MK[ct])),"rna":sp(pred[ct],sig(cpm,MK[ct])),
        "prot_ci":sp_ci(pred[ct],sig(prot,MK[ct])),"rna_ci":sp_ci(pred[ct],sig(cpm,MK[ct]))}
    for ct in pred.columns}

CTS=list(internal.keys())
fig=plt.figure(figsize=(6.2,4.2))

# 独立双模态 marker 验证(单面板)
axb=fig.add_subplot(111)
ordb=sorted(pred.columns,key=lambda c:-(min(mk[c]["prot"],mk[c]["rna"])))
y=np.arange(len(ordb))
for i,c in enumerate(ordb):
    p=mk[c]["prot"]; r=mk[c]["rna"]; both=min(p,r)>=0.3
    pc=mk[c]["prot_ci"]; rc=mk[c]["rna_ci"]
    axb.plot([p,r],[i-0.16,i+0.16],color="#ccc",lw=1.2,zorder=1)
    axb.plot(pc,[i-0.16,i-0.16],color=PAL["prot"],lw=1.4,zorder=2,solid_capstyle="butt")
    axb.plot(rc,[i+0.16,i+0.16],color=PAL["hancock"],lw=1.4,zorder=2,solid_capstyle="butt")
    axb.scatter([p],[i-0.16],s=42,color=PAL["prot"],zorder=3,label="protein" if i==0 else None,edgecolor="k",lw=0.3)
    axb.scatter([r],[i+0.16],s=42,color=PAL["hancock"],zorder=3,label="RNA" if i==0 else None,edgecolor="k",lw=0.3)
    if both: axb.text(-0.56,i,"✓",color="#1a7a1a",fontsize=10,ha="center",va="center",fontweight="bold")
axb.axvline(0.3,color="grey",ls="--",lw=0.8); axb.axvline(0,color="#ddd",lw=0.6)
axb.set_yticks(y); axb.set_yticklabels(ordb,fontsize=7.5)
axb.set_xlim(-0.62,0.75); axb.set_xlabel("Spearman correlation: predicted abundance vs cell-type marker level\n(marker genes averaged in CPTAC protein, or in RNA)\nwhiskers: patient bootstrap 95% CI (2000 resamples)")
axb.legend(loc="upper left",frameon=False,fontsize=6.8,markerscale=1.1)
axb.set_title("Does predicted abundance match each cell type's own marker genes?\nmarkers measured in CPTAC protein and RNA  (✓ = agrees with both, Spearman ≥0.3)",fontsize=8.3)
for e in ("pdf","png"): fig.savefig(f"figures/paper/S4_abundance.{e}",dpi=300,bbox_inches="tight")
plt.close(fig)
print("saved S4_abundance")
print("marker验证:", {c:(round(mk[c]['prot'],2),round(mk[c]['rna'],2)) for c in ordb})
