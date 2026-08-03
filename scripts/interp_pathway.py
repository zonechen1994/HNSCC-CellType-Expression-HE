#!/usr/bin/env python3
"""生物学可解释性: H&E预测的通路/程序活性 vs 真实(CPTAC外部)。
预测θ加权伪bulk表达 -> 50个Hallmark签名分 vs 真实bulk RNA签名分, 跨病人Spearman。
输出 cache/interp_pathway.json。"""
import os, re, json, numpy as np, pandas as pd, torch, sys
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
from scipy.stats import spearmanr
sys.path.insert(0,"code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN,RES,CRUN="run/TCGA_HNSC_oro_v2b","run/TCGA_HNSC_oro_v2b/results","run/CPTAC_HNSC"
ZDIR=f"{CRUN}/deconv/bayesprism_oropharyngeal"
CTS=list(pd.read_csv(f"{RUN}/cell_types_file.csv")["cell_types"])

cf=np.load(f"{CRUN}/features.pkl",allow_pickle=True); sids=[f[0] for f in cf]
CX=[torch.tensor(np.asarray(f[1]),dtype=torch.float32) for f in cf]
meta=pd.read_csv(f"{CRUN}/metadata.csv"); s2c=dict(zip(meta["Slide.ID"],meta["case"]))
def predict_ct(ct):
    gf=f"{RUN}/breast_{ct}_gene_file.csv"
    if not os.path.exists(gf): return None
    genes=list(pd.read_csv(gf)["gene"]); pred=np.zeros((len(CX),len(genes))); n=0
    for ik in range(5):
        mp=f"{RES}/{ct}_preds/result_{ik}_0/model_trained.pth"
        if not os.path.exists(mp): continue
        m=MLP_regression(768,768,len(genes),0.2,None); m.load_state_dict(torch.load(mp,map_location=dev)); m.to(dev).eval()
        with torch.no_grad(): pred+=np.vstack([m(x.to(dev)).cpu().numpy() for x in CX]); n+=1
    if n==0: return None
    pred/=n; dp=pd.DataFrame(pred,index=[s2c.get(s) for s in sids],columns=genes); return dp.groupby(level=0).mean()
print("predict + pseudobulk ...")
preds={ct:predict_ct(ct) for ct in CTS}; preds={k:v for k,v in preds.items() if v is not None}
theta=pd.read_csv(f"{ZDIR}/theta.csv",index_col=0); pts=preds["Malignant"].index
gu=sorted(set().union(*[set(v.columns) for v in preds.values()]))
pb=pd.DataFrame(0.0,index=pts,columns=gu)
for ct,dp in preds.items():
    if ct not in theta.columns: continue
    pb+=np.expm1(dp.reindex(pts)).reindex(columns=gu).fillna(0.0).values*theta.reindex(pts)[ct].values[:,None]
pbulk=np.log1p(pb)  # 预测伪bulk log1pCPM-ish

rna=pd.read_csv(f"{CRUN}/deconv/cptac_bulk_counts.tsv.gz",sep="\t",index_col=0).T
cpm=np.log1p(rna.div(rna.sum(1),axis=0)*1e6)  # 真实bulk

# Hallmark
import gseapy
HALL=gseapy.get_library("MSigDB_Hallmark_2020")
print(f"Hallmark集: {len(HALL)}")

def sig_scores(mat, geneset):
    g=[x for x in geneset if x in mat.columns]
    if len(g)<5: return None
    z=(mat[g]-mat[g].mean())/(mat[g].std()+1e-9)   # 跨病人z-score每基因
    return z.mean(axis=1)                            # 每病人签名分

pts_c=[p for p in pbulk.index if p in cpm.index]
# 类别(手工映射, 用于着色/thesis)
CAT={"proliferation":["E2F","G2M","MITOTIC","MYC"],"stroma/EMT":["EPITHELIAL_MESENCHYMAL","ANGIOGEN","APICAL","COAGULATION","MYOGENESIS"],
     "metabolic":["OXIDATIVE","GLYCOLYSIS","FATTY","ADIPOGEN","CHOLESTEROL","BILE","XENOBIOTIC","HEME","PEROXISOME"],
     "hypoxia/stress":["HYPOXIA","UNFOLDED","REACTIVE","UV_RESPONSE","DNA_REPAIR","P53","APOPTOSIS","PROTEIN_SECRETION"],
     "signaling":["TNFA","KRAS","MTORC","PI3K","WNT","NOTCH","HEDGEHOG","TGF_BETA","IL2","IL6","ESTROGEN","ANDROGEN","APICAL_SURFACE"],
     "immune":["INTERFERON","INFLAMMATORY","ALLOGRAFT","COMPLEMENT"]}
def cat_of(name):
    u=name.upper().replace(" ","_")
    for c,keys in CAT.items():
        if any(k in u for k in keys): return c
    return "other"

rows=[]; ex={}
for name,gs in HALL.items():
    a=sig_scores(pbulk.loc[pts_c],gs); b=sig_scores(cpm.loc[pts_c],gs)
    if a is None or b is None: continue
    rho=spearmanr(a.values,b.values).correlation
    rows.append({"pathway":name,"rho":float(rho),"cat":cat_of(name),"ngenes":int(len([x for x in gs if x in pbulk.columns]))})
rows=sorted(rows,key=lambda r:-r["rho"])
# 示例散点: 取前3(通常增殖/EMT/角化类)
for r in rows[:4]:
    gs=HALL[r["pathway"]]; a=sig_scores(pbulk.loc[pts_c],gs); b=sig_scores(cpm.loc[pts_c],gs)
    ex[r["pathway"]]={"pred":a.values.tolist(),"true":b.values.tolist(),"rho":r["rho"]}

json.dump({"pathways":rows,"examples":ex,"n":len(pts_c)},open("figures/paper/cache/interp_pathway.json","w"),default=float)
print(f"\n通路活性预测 (CPTAC, n={len(pts_c)}): 中位ρ={np.median([r['rho'] for r in rows]):.3f}")
print("Top10:")
for r in rows[:10]: print(f"  {r['rho']:+.3f}  [{r['cat']:14s}] {r['pathway']}")
print("Bottom5:")
for r in rows[-5:]: print(f"  {r['rho']:+.3f}  [{r['cat']:14s}] {r['pathway']}")
