#!/usr/bin/env python3
"""为 Fig2 b/g 生成同一批 marker 的两层数据:
  b: TCGA 折外预测 vs 反卷积观测 (每个 marker 在它自己的细胞型里)
  g: CPTAC 预测 vs 实测蛋白 (同样三个 marker)
三大区室各一个经典 marker: KRT6C(肿瘤) / POSTN(基质) / ZAP70(免疫)"""
import os, sys, json, numpy as np, pandas as pd, torch
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
sys.path.insert(0,"code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
from scipy.stats import pearsonr
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN,RES,CRUN="run/TCGA_HNSC_oro_v2b","run/TCGA_HNSC_oro_v2b/results","run/CPTAC_HNSC"

MARKERS=[("Malignant","KRT6C","Tumor"),
         ("Fibroblast","POSTN","Stroma"),
         ("T cell","ZAP70","Immune")]

# ================= (b) TCGA 折外: 预测 vs 反卷积观测 =================
sp=np.load(f"{RUN}/splits/train_valid_test_idx.npz",allow_pickle=True)
exb=[]
for ct,g,comp in MARKERS:
    genes=list(pd.read_csv(f"{RUN}/breast_{ct}_gene_file.csv")["gene"]); j=genes.index(g)
    pred,obs=[],[]
    for k in range(5):
        P=np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/test_preds.txt")[:,j]
        L=np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/test_labels.txt")[:,j]
        pred.append(P); obs.append(L)
    pred=np.concatenate(pred); obs=np.concatenate(obs)
    r=float(pearsonr(pred,obs)[0])
    exb.append({"ct":ct,"gene":g,"compartment":comp,"pcc":r,
                "pred":pred.tolist(),"obs":obs.tolist()})
    print(f"[b] {comp:7s} {ct:11s} {g:7s} r={r:.3f}  n={len(pred)}")
json.dump(exb,open("figures/paper/cache/expr_examples.json","w"))

# ================= (g) CPTAC: 预测 vs 实测蛋白 =================
cf=np.load(f"{CRUN}/features.pkl",allow_pickle=True); sids=[f[0] for f in cf]
CX=[torch.tensor(np.asarray(f[1]),dtype=torch.float32) for f in cf]
meta=pd.read_csv(f"{CRUN}/metadata.csv"); s2c=dict(zip(meta["Slide.ID"],meta["case"]))
import cptac as _c
_p=os.path.join(os.path.dirname(_c.__file__),"data","umich-hnscc",
                "Report_abundance_groupby=protein_protNorm=MD_gu=2.tsv.gz")
_raw=pd.read_csv(_p,sep="\t"); _raw["sym"]=_raw["Index"].astype(str).str.split("|").str[-2]
_pat=[c for c in _raw.columns if c.startswith(("C3L-","C3N-")) and c.endswith("-T")]
prot=_raw.set_index("sym")[_pat].apply(pd.to_numeric,errors="coerce").T
prot.index=[i[:-2] for i in prot.index]; prot=prot.loc[:,~prot.columns.duplicated()]

exg=[]
for ct,g,comp in MARKERS:
    genes=list(pd.read_csv(f"{RUN}/breast_{ct}_gene_file.csv")["gene"])
    pr=np.zeros((len(CX),len(genes))); n=0
    for ik in range(5):
        mp=f"{RES}/{ct}_preds/result_{ik}_0/model_trained.pth"
        m=MLP_regression(768,768,len(genes),0.2,None)
        m.load_state_dict(torch.load(mp,map_location=dev)); m.to(dev).eval()
        with torch.no_grad(): pr+=np.vstack([m(x.to(dev)).cpu().numpy() for x in CX])
        n+=1
    pr/=n
    P=pd.DataFrame(pr,index=[s2c.get(s) for s in sids],columns=genes).groupby(level=0).mean()
    com=[p for p in P.index if p in prot.index and pd.notna(prot.loc[p,g])]
    a=P.loc[com,g].values.astype(float); b=prot.loc[com,g].values.astype(float)
    r,pv=pearsonr(a,b)
    exg.append({"ct":ct,"gene":g,"compartment":comp,"r":float(r),"p":float(pv),
                "pred":a.tolist(),"prot":b.tolist(),"n":len(com)})
    print(f"[g] {comp:7s} {ct:11s} {g:7s} r={r:.3f} (P={pv:.1e}, n={len(com)})")
json.dump(exg,open("figures/paper/cache/marker_protein_examples.json","w"))
print("\n已存 expr_examples.json (panel b) 和 marker_protein_examples.json (panel g)")
