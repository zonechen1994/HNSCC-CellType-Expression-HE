#!/usr/bin/env python3
"""每种细胞的经典 marker: TCGA 的 RNA 预测相关 + CPTAC 的 预测-vs-实测蛋白 相关。
用于给 Fig2 b/g 挑选生物学干净、两层都能展示的示例基因。"""
import os, sys, json, numpy as np, pandas as pd, torch
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
sys.path.insert(0,"code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
from scipy.stats import pearsonr
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN,RES,CRUN="run/TCGA_HNSC_oro_v2b","run/TCGA_HNSC_oro_v2b/results","run/CPTAC_HNSC"
MARK={
 "Malignant":["KRT6C","KRT6B","DSG1","EPCAM","SFN","KRT5","KRT17","TP63","DSG3"],
 "Fibroblast":["FAP","PDGFRB","DCN","LUM","COL1A1","COL1A2","POSTN","SPARC","THY1"],
 "Endothelial":["CDH5","PECAM1","VWF","ENG","CLDN5","RAMP2"],
 "Macrophage":["MSR1","CD163","TYROBP","C1QA","C1QB","AIF1","FCGR3A","MRC1"],
 "T cell":["ZAP70","CD3D","CD3E","LCK","CD2","THEMIS"],
 "B cell":["MS4A1","CD79A","BANK1"],
 "Dendritic":["LAMP3","IRF8"],
 "Mast":["CPA3","TPSAB1","KIT","CMA1"],
}
# ---- CPTAC 特征 ----
cf=np.load(f"{CRUN}/features.pkl",allow_pickle=True); sids=[f[0] for f in cf]
CX=[torch.tensor(np.asarray(f[1]),dtype=torch.float32) for f in cf]
meta=pd.read_csv(f"{CRUN}/metadata.csv"); s2c=dict(zip(meta["Slide.ID"],meta["case"]))
def predict_ct(ct):
    genes=list(pd.read_csv(f"{RUN}/breast_{ct}_gene_file.csv")["gene"])
    pred=np.zeros((len(CX),len(genes))); n=0
    for ik in range(5):
        mp=f"{RES}/{ct}_preds/result_{ik}_0/model_trained.pth"
        if not os.path.exists(mp): continue
        m=MLP_regression(768,768,len(genes),0.2,None)
        m.load_state_dict(torch.load(mp,map_location=dev)); m.to(dev).eval()
        with torch.no_grad(): pred+=np.vstack([m(x.to(dev)).cpu().numpy() for x in CX])
        n+=1
    pred/=n
    return pd.DataFrame(pred,index=[s2c.get(s) for s in sids],columns=genes).groupby(level=0).mean()

# ---- 蛋白 ----
# 直接读本地 cptac 缓存(Zenodo 504 时绕开在线下载)
import cptac as _c
_p=os.path.join(os.path.dirname(_c.__file__),"data","umich-hnscc",
                "Report_abundance_groupby=protein_protNorm=MD_gu=2.tsv.gz")
_raw=pd.read_csv(_p,sep="\t")
_raw["sym"]=_raw["Index"].astype(str).str.split("|").str[-2]      # 基因符号
_pat=[c for c in _raw.columns if c.startswith(("C3L-","C3N-")) and c.endswith("-T")]  # 仅肿瘤
prot=_raw.set_index("sym")[_pat].apply(pd.to_numeric,errors="coerce").T
prot.index=[i[:-2] for i in prot.index]                            # C3L-00997-T -> C3L-00997
prot=prot.loc[:,~prot.columns.duplicated()]
print(f"[本地蛋白组] {prot.shape[0]} 病人 x {prot.shape[1]} 基因")

print(f"{'cell':12s} {'marker':10s} {'RNA r(TCGA)':>12s} {'Protein r(CPTAC)':>17s} {'n':>5s}")
out={}
for ct,mk in MARK.items():
    genes=list(pd.read_csv(f"{RUN}/breast_{ct}_gene_file.csv")["gene"])
    C=[np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/coef_slope.txt") for k in range(5)]
    P=predict_ct(ct)
    rows=[]
    for g in mk:
        if g not in genes or g not in prot.columns: continue
        j=genes.index(g); rrna=float(np.nanmean([c[j,1] for c in C]))
        com=[p for p in P.index if p in prot.index and pd.notna(prot.loc[p,g])]
        if len(com)<50: continue
        a=P.loc[com,g].values.astype(float); b=prot.loc[com,g].values.astype(float)
        if np.std(a)<1e-9 or np.std(b)<1e-9: continue
        rp,pp=pearsonr(a,b)
        rows.append({"gene":g,"r_rna":rrna,"r_prot":float(rp),"p_prot":float(pp),"n":len(com)})
    rows.sort(key=lambda r:-(r["r_rna"]+r["r_prot"]))
    out[ct]=rows
    for r in rows[:3]:
        star="*" if r["p_prot"]<0.05 else " "
        print(f"{ct:12s} {r['gene']:10s} {r['r_rna']:12.3f} {r['r_prot']:16.3f}{star} {r['n']:5d}")
    if not rows: print(f"{ct:12s} (无同时可用的 marker)")
json.dump(out,open("figures/paper/cache/marker_rna_protein.json","w"))
print("\n已存 cache/marker_rna_protein.json")
