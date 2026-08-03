#!/usr/bin/env python3
"""Fig 2e 式分析: 每种细胞, TCGA内部 well-predicted 基因(PCC>0.4) 与 CPTAC外部 well-predicted 基因(PCC>0.4) 的重叠。
- TCGA内部 per-gene PCC: 池化5折 OOF test_preds/test_labels, 逐基因 Pearson。
- CPTAC外部 per-gene PCC: 套模型到CPTAC特征 -> 预测表达 vs CPTAC反卷积Z(log1p CPM), 逐基因 Pearson。
- 重叠 |A∩B| + 超几何检验(基因宇宙=该细胞两队列共测基因)。
输出 experiments/ref_comparison/gene_overlap.log
"""
import os, re, sys, numpy as np, pandas as pd, torch
from scipy.stats import pearsonr, hypergeom
sys.path.insert(0, os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TR="run/TCGA_HNSC_oro_v2b"; RES=f"{TR}/results"; THR=0.4
CTS=list(pd.read_csv(f"{TR}/cell_types_file.csv")["cell_types"])
CZDIR="run/CPTAC_HNSC/deconv/bayesprism_oropharyngeal"
out=[]; P=lambda s:(out.append(str(s)),print(s))
_C={}
def feats(run):
    if run in _C: return _C[run]
    cf=np.load(f"{run}/features.pkl",allow_pickle=True)
    _C[run]=([f[0] for f in cf],[torch.tensor(np.asarray(f[1]),dtype=torch.float32) for f in cf]); return _C[run]
def apply_model(run,subdir,outdim):
    sids,X=feats(run); pr=np.zeros((len(X),outdim))
    for ik in range(5):
        m=MLP_regression(768,768,outdim,0.2,None)
        m.load_state_dict(torch.load(f"{RES}/{subdir}/result_{ik}_0/model_trained.pth",map_location=dev)); m.to(dev).eval()
        with torch.no_grad(): pr+=np.vstack([m(x.to(dev)).cpu().numpy() for x in X])
    return pr/5, sids
def s2c(run):
    m=pd.read_csv(f"{run}/metadata.csv"); return dict(zip(m["Slide.ID"],m["case"].astype(str)))
def genes_of(ct): return list(pd.read_csv(f"{TR}/breast_{ct}_gene_file.csv")["gene"])

def tcga_internal_pcc(ct):
    """池化5折 OOF, 逐基因 PCC。返回 Series(index=gene)。"""
    genes=genes_of(ct); Ps=[]; Ls=[]
    for ik in range(5):
        d=f"{RES}/{ct}_preds/result_{ik}_0"
        Ps.append(np.loadtxt(f"{d}/test_preds.txt")); Ls.append(np.loadtxt(f"{d}/test_labels.txt"))
    Pp=np.vstack(Ps); Ll=np.vstack(Ls)
    r=np.array([pearsonr(Pp[:,j],Ll[:,j])[0] if Pp[:,j].std()>1e-9 and Ll[:,j].std()>1e-9 else np.nan for j in range(len(genes))])
    return pd.Series(r,index=genes)

def cptac_external_pcc(ct):
    genes=genes_of(ct)
    zf=f"{CZDIR}/Z_{re.sub(r'[^A-Za-z0-9]+','_',ct)}.csv"
    if not os.path.exists(zf): return None
    pr,sids=apply_model("run/CPTAC_HNSC",f"{ct}_preds",len(genes)); mp=s2c("run/CPTAC_HNSC")
    dp=pd.DataFrame(pr,index=[mp.get(s) for s in sids],columns=genes).groupby(level=0).mean()
    Z=pd.read_csv(zf,index_col=0); lib=Z.sum(1).replace(0,np.nan); Zc=np.log1p(Z.div(lib,axis=0)*1e6).fillna(0.0)
    g=[x for x in genes if x in Zc.columns]; pts=[c for c in dp.index if c in Zc.index]
    A=dp.loc[pts,g].values; B=Zc.loc[pts,g].values
    r=np.array([pearsonr(A[:,j],B[:,j])[0] if A[:,j].std()>1e-9 and B[:,j].std()>1e-9 else np.nan for j in range(len(g))])
    return pd.Series(r,index=g)

P("="*90)
P("Fig2e式: TCGA内部 vs CPTAC外部 well-predicted 基因(PCC>0.4)重叠 + 超几何检验")
P("="*90)
P(f"{'细胞':12s} {'TCGA>0.4':>9s} {'CPTAC>0.4':>10s} {'共测基因':>8s} {'重叠∩':>7s} {'超几何p':>12s} {'Jaccard':>8s}")
summary=[]
for ct in CTS:
    ti=tcga_internal_pcc(ct); ce=cptac_external_pcc(ct)
    if ce is None: continue
    common=[g for g in ti.index if g in ce.index]      # 两队列共测基因 = 宇宙
    N=len(common)
    A=set(g for g in common if ti[g]>THR)              # TCGA well-predicted
    Bset=set(g for g in common if ce[g]>THR)           # CPTAC well-predicted
    ov=A & Bset; k=len(ov); K=len(A); n=len(Bset)
    p=hypergeom.sf(k-1, N, K, n) if K>0 and n>0 else np.nan
    jac=k/len(A|Bset) if (A|Bset) else 0
    P(f"{ct:12s} {K:9d} {n:10d} {N:8d} {k:7d} {p:12.2e} {jac:8.3f}")
    summary.append((ct,K,n,N,k,p))
P("\n注: 宇宙N=该细胞在TCGA与CPTAC都测到的基因数; 超几何 sf(k-1,N,K,n) 检验重叠是否显著高于随机。")
open("experiments/ref_comparison/gene_overlap.log","w").write("\n".join(out))
P("\nGENE_OVERLAP_DONE")
