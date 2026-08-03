#!/usr/bin/env python3
"""用 v2b(口咽HPV混合参考+放宽过滤)训练的模型, 重跑全部外部验证, 保证一篇一个参考。
输出 experiments/ref_comparison/v2b_external.log"""
import os, re, numpy as np, pandas as pd, torch, sys
from scipy.stats import pearsonr, spearmanr
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TR="run/TCGA_HNSC_oro_v2b"; RES=f"{TR}/results"
CTS=list(pd.read_csv(f"{TR}/cell_types_file.csv")["cell_types"])
out=[]; P=lambda s:(out.append(s),print(s))
_C={}
def feats(run):
    if run in _C: return _C[run]
    cf=np.load(f"{run}/features.pkl",allow_pickle=True)
    r=([f[0] for f in cf],[torch.tensor(np.asarray(f[1]),dtype=torch.float32) for f in cf]); _C[run]=r; return r
def apply_model(run,subdir,outdim):
    sids,X=feats(run); pr=np.zeros((len(X),outdim)); n=0
    for ik in range(5):
        mp=f"{RES}/{subdir}/result_{ik}_0/model_trained.pth"
        if not os.path.exists(mp): raise FileNotFoundError(f"缺 fold 模型: {mp} (需5折齐全)")  # 不静默容忍
        m=MLP_regression(768,768,outdim,0.2,None); m.load_state_dict(torch.load(mp,map_location=dev)); m.to(dev).eval()
        with torch.no_grad(): pr+=np.vstack([m(x.to(dev)).cpu().numpy() for x in X]); n+=1
    assert n==5, f"{subdir}: 只用了{n}/5折"
    return pr/n, sids
def s2c(run,z=None):
    m=pd.read_csv(f"{run}/metadata.csv"); d=dict(zip(m["Slide.ID"],m["case"].astype(str)))
    return {k:(v.zfill(z) if z else v) for k,v in d.items()}
def pred_ab(run,z=None):
    pr,sids=apply_model(run,"cell_fraction_preds",len(CTS)); mp=s2c(run,z)
    return pd.DataFrame(pr,index=[mp.get(s) for s in sids],columns=CTS).groupby(level=0).mean()
def cox(ab,svf,ct):
    sv=pd.read_csv(svf,dtype={'case':str}).set_index("case"); com=[c for c in ab.index if c in sv.index]
    x=ab.loc[com,ct].values.astype(float); zz=(x-x.mean())/(x.std()+1e-9)
    s=CoxPHFitter().fit(pd.DataFrame({"T":sv.loc[com,"time_days"].values,"E":sv.loc[com,"event"].values,"z":zz}),"T","E").summary.loc["z"]
    return np.exp(s["coef"]),np.exp(s["coef lower 95%"]),np.exp(s["coef upper 95%"]),s["p"],len(com)

P("="*70); P("v2b 外部验证(口咽HPV混合参考+放宽过滤) —— 一篇一个参考"); P("="*70)

# ---- HANCOCK 预后 ----
abH=pred_ab("run/HANCOCK",z=3)
P("\n[HANCOCK 预后] 预测丰度->生存")
for f,lab in [("run/HANCOCK/survival.csv","OS"),("run/HANCOCK/survival_dss.csv","DSS")]:
    for ct in ["T cell","Macrophage"]:
        hr,lo,hi,p,n=cox(abH,f,ct); P(f"  {lab} {ct:11s} HR={hr:.2f}({lo:.2f}-{hi:.2f}) p={p:.4f} n={n}")

# ---- HANCOCK HPV 外部 AUC ----
P("\n[HANCOCK HPV 外部AUC]")
hpvt=pd.read_csv("run/TCGA_HNSC/driver_mutations.csv",index_col=0)["HPV"]
tcases=pd.read_csv(f"{TR}/metadata.csv")["case"].tolist()
sp=np.load(f"{TR}/splits/train_valid_test_idx.npz",allow_pickle=True)
hpvh=pd.read_csv("run/HANCOCK/hpv.csv",dtype={'case':str}).set_index("case")["hpv_p16"].map({"positive":1,"negative":0})
gof=lambda ct:pd.read_csv(f"{TR}/breast_{ct}_gene_file.csv")["gene"].values
def ens():return VotingClassifier([("lr",make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))),("rf",RandomForestClassifier(200,random_state=0)),("svm",make_pipeline(StandardScaler(),SVC(probability=True,random_state=0)))],voting="soft")
def oof(ct):
    Pl,o=[],[]
    for ik in range(5):
        d=f"{RES}/{ct}_preds/result_{ik}_0"
        if not os.path.exists(f"{d}/test_preds.txt"): return None
        Pl.append(np.loadtxt(f"{d}/test_preds.txt")); o.extend(sp["test_idx"][ik].tolist())
    return pd.DataFrame(np.vstack(Pl),index=[tcases[i] for i in o],columns=gof(ct))
mpH=s2c("run/HANCOCK",3); ext_sum=None; ext_y=None; ne=0; int_sum=None; int_y=None; ni=0
for ct in CTS:
    Xt=oof(ct)
    if Xt is None: continue
    ct_t=[c for c in Xt.index if c in hpvt.index]; yt=hpvt.loc[ct_t].values.astype(int)
    if yt.sum()<8: continue
    Xtv=Xt.loc[ct_t].values
    o=np.zeros(len(yt))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xtv,yt): o[te]=ens().fit(Xtv[tr],yt[tr]).predict_proba(Xtv[te])[:,1]
    int_sum=o.copy() if int_sum is None else int_sum+o; int_y=yt; ni+=1
    pr,sids=apply_model("run/HANCOCK",f"{ct}_preds",len(gof(ct)))
    Xh=pd.DataFrame(pr,index=[str(mpH.get(s)).zfill(3) for s in sids],columns=gof(ct)).groupby(level=0).mean()
    ch=[c for c in Xh.index if c in hpvh.index]
    if len(ch)<20 or hpvh.loc[ch].sum()<5: continue
    ye=hpvh.loc[ch].values.astype(int); pe=ens().fit(Xtv,yt).predict_proba(Xh.loc[ch].values)[:,1]
    ext_sum=pe.copy() if ext_sum is None else ext_sum+pe; ext_y=ye; ne+=1
P(f"  内部集成AUC={roc_auc_score(int_y,int_sum/ni):.3f} | 外部HANCOCK集成AUC={roc_auc_score(ext_y,ext_sum/ne):.3f} (n={len(ext_y)},HPV+={int(ext_y.sum())})")

# ---- HANCOCK 免疫 vs IHC ----
P("\n[HANCOCK 免疫丰度 vs IHC]")
ihc=pd.read_csv("run/HANCOCK/ihc_density.csv",dtype={'case':str}).set_index("case")
for mk in ["CD3_density","CD8_density"]:
    com=[c for c in abH.index if c in ihc.index and pd.notna(ihc.loc[c,mk])]
    r=spearmanr(abH.loc[com,"T cell"].values.astype(float), ihc.loc[com,mk].values.astype(float))
    P(f"  预测T细胞 vs {mk}: Spearman={r.correlation:.3f} p={r.pvalue:.1e} n={len(com)}")

# ---- CPTAC 预后 ----
P("\n[CPTAC 预后] 预测丰度->生存")
abC=pred_ab("run/CPTAC_HNSC")
for ct in ["T cell"]:
    hr,lo,hi,p,n=cox(abC,"run/CPTAC_HNSC/survival.csv",ct); P(f"  T cell HR={hr:.2f}({lo:.2f}-{hi:.2f}) p={p:.4f} n={n}")

# ---- CPTAC 表达迁移(需口咽反卷积) ----
zdir="run/CPTAC_HNSC/deconv/bayesprism_oropharyngeal"
if os.path.exists(f"{zdir}/theta.csv"):
    P("\n[CPTAC 表达迁移] 预测表达 vs 口咽反卷积真值 per-gene PCC")
    for ct in CTS:
        zf=f"{zdir}/Z_{re.sub(r'[^A-Za-z0-9]+','_',ct)}.csv"
        if not os.path.exists(zf): continue
        genes=list(gof(ct)); pr,sids=apply_model("run/CPTAC_HNSC",f"{ct}_preds",len(genes))
        mpC=s2c("run/CPTAC_HNSC")
        dp=pd.DataFrame(pr,index=[mpC.get(s) for s in sids],columns=genes).groupby(level=0).mean()
        Z=pd.read_csv(zf,index_col=0); lib=Z.sum(1).replace(0,np.nan); Zc=np.log1p(Z.div(lib,axis=0)*1e6).fillna(0.0)
        g=[x for x in genes if x in Zc.columns]; pts=[c for c in dp.index if c in Zc.index]
        if len(g)<50 or len(pts)<20: continue
        A=dp.loc[pts,g].values; B=Zc.loc[pts,g].values
        rr=np.array([pearsonr(A[:,j],B[:,j])[0] if A[:,j].std()>1e-9 and B[:,j].std()>1e-9 else np.nan for j in range(len(g))]); rr=rr[~np.isnan(rr)]
        P(f"  {ct:11s} top1000中位PCC={np.median(np.sort(rr)[::-1][:1000]):.3f} (病人{len(pts)})")
else:
    P("\n[CPTAC 表达迁移] 口咽反卷积未完成, 跳过(稍后补)")
P("\nV2B_EXTERNAL_DONE")
open("experiments/ref_comparison/v2b_external.log","w").write("\n".join(out))
