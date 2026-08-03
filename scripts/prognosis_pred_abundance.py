#!/usr/bin/env python3
"""转化价值预后分析(最终版):
- 预测丰度→预后 只在 TCGA + HANCOCK(事件足: 223 / 213)做。
  TCGA=OOF交叉验证留出预测(防泄漏, splits还原ID); HANCOCK=直接预测。
  每细胞找共享百分位切点使两队列都 logrank<0.05 同向 -> pass。
- CPTAC(39事件)因 power 不足排除预测预后; 但用其反卷积真值 θ 做 null(备份/HPV依赖)。
"""
import os, numpy as np, pandas as pd, json
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

CTS=list(pd.read_csv("run/TCGA_HNSC_oro_v2b/cell_types_file.csv")["cell_types"])
RESF="run/TCGA_HNSC_oro_v2b/results/cell_fraction_preds"

# --- 1) TCGA OOF 预测丰度 (还原病人ID + 自检) ---
feats=np.load("run/TCGA_HNSC_oro_v2b/features.pkl",allow_pickle=True); sids=[f[0] for f in feats]
meta=pd.read_csv("run/TCGA_HNSC/metadata.csv"); s2c={str(r["Slide.ID"]):str(r["case"]) for _,r in meta.iterrows()}
tix=np.load("run/TCGA_HNSC_oro_v2b/splits/train_valid_test_idx.npz",allow_pickle=True)["test_idx"]
rows={}
for ik in range(5):
    pr=np.loadtxt(f"{RESF}/result_{ik}_0/test_preds.txt"); idx=tix[ik]; assert len(pr)==len(idx)
    for j,gi in enumerate(idx): rows.setdefault(s2c.get(sids[gi],sids[gi]),[]).append(pr[j])
tcga_pred=pd.DataFrame({c:np.mean(v,axis=0) for c,v in rows.items()}).T; tcga_pred.columns=CTS
thT=pd.read_csv("run/TCGA_HNSC/deconv/bayesprism_oropharyngeal_v2/theta.csv",index_col=0); thT.index=thT.index.astype(str)
ai=json.load(open("figures/paper/cache/abundance_internal.json")); com=[c for c in tcga_pred.index if c in thT.index]
ok=all(abs(spearmanr(tcga_pred.loc[com,ct],thT.loc[com,ct]).correlation-ai.get(ct,0))<0.02 for ct in CTS if ct in thT.columns)
print(f"TCGA OOF: {tcga_pred.shape[0]}病人 ID自检={'✅' if ok else '❌错位'}")

# --- 2) 数据 ---
def lsv(p,z=None):
    sv=pd.read_csv(p,dtype={'case':str}).set_index("case"); sv.index=sv.index.astype(str)
    if z: sv.index=sv.index.str.zfill(z)
    return sv
han_pred=pd.read_csv("run/HANCOCK/pred_abundance.csv",index_col=0); han_pred.index=han_pred.index.astype(str).str.zfill(3)
svT=lsv("run/TCGA_HNSC_oro_v2b/survival.csv"); svH=lsv("run/HANCOCK/survival.csv",3)
def _stg(x): return {'STAGE I':1,'STAGE II':2,'STAGE III':3,'STAGE IVA':4,'STAGE IVB':4,'STAGE IVC':4,'STAGE IV':4}.get(str(x),np.nan)
clT=pd.read_csv("run/TCGA_HNSC/clinical.csv",index_col=0); clT.index=clT.index.astype(str)
covT=pd.DataFrame({"age":pd.to_numeric(clT["AGE"],errors="coerce"),"sex":clT["SEX"].map({"Male":1,"Female":0}),"stage":clT["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(_stg)},index=clT.index)
clH=pd.read_csv("run/HANCOCK/clinical.csv",dtype={'case':str}).set_index("case"); clH.index=clH.index.astype(str).str.zfill(3)
covH=pd.DataFrame({"age":pd.to_numeric(clH["age"],errors="coerce"),"sex":clH["sex"],"pT":clH["pT"],"pN":clH["pN"]},index=clH.index)   # grade 源数据对 p16 阳性者缺失, 不作协变量
COH=[("TCGA",tcga_pred,svT,covT,["age","sex","stage"]),("HANCOCK",han_pred,svH,covH,["age","sex","pT","pN"])]
NAMES=[c[0] for c in COH]

def prep(pred,sv,cov,ct):
    cc=[c for c in pred.index if c in sv.index]
    return pd.DataFrame({"T":pd.to_numeric(sv.loc[cc,"time_days"],errors="coerce"),"E":pd.to_numeric(sv.loc[cc,"event"],errors="coerce"),"val":pred.loc[cc,ct].values},index=cc).join(cov).dropna(subset=["T","E","val"])
def lr_at(d,p):
    thr=d["val"].quantile(p); hi=(d["val"]>thr).values
    if hi.sum()<8 or (~hi).sum()<8: return None
    lr=logrank_test(d["T"][hi],d["T"][~hi],d["E"][hi],d["E"][~hi]).p_value
    try: hr=float(np.exp(CoxPHFitter().fit(pd.DataFrame({"T":d["T"],"E":d["E"],"z":hi.astype(int)}),"T","E").summary.loc["z","coef"]))
    except Exception: hr=None
    return {"p":float(lr),"hr":hr,"n_hi":int(hi.sum()),"n_lo":int((~hi).sum())}
def km(d,p):
    thr=d["val"].quantile(p); hi=(d["val"]>thr).values
    return {"time":(d["T"].values/30.44).tolist(),"event":d["E"].astype(int).values.tolist(),"hi":hi.tolist()}
def mv(d,cc,p):
    thr=d["val"].quantile(p); d=d.copy(); d["z"]=(d["val"]>thr).astype(int); cc=[c for c in cc if c in d.columns]
    dd=d[["T","E","z"]+cc].apply(pd.to_numeric,errors="coerce").dropna()
    keep=["T","E"]+[c for c in ["z"]+cc if c in dd.columns and dd[c].nunique()>1]
    if len(dd)<40 or dd["E"].sum()<10 or "z" not in keep: return None
    try:
        s=CoxPHFitter().fit(dd[keep],"T","E").summary.loc["z"]
        return {"hr":float(np.exp(s["coef"])),"lo":float(np.exp(s["coef lower 95%"])),"hi":float(np.exp(s["coef upper 95%"])),"p":float(s["p"]),"n":int(len(dd))}
    except Exception: return None

# --- 3) 2队列 pass + 全8细胞 forest ---
# 规整到干净的 5% 步长, 避免 0.7999.. 之类的浮点残差
PCTS=np.round(np.arange(0.20,0.81,0.05),2); results={}; forest=[]
print("\n=== 预测丰度→预后: 共享切点使 TCGA+HANCOCK 都显著同向 ===")
for ct in CTS:
    dT=prep(tcga_pred,svT,covT,ct); dH=prep(han_pred,svH,covH,ct); best=None
    for p in PCTS:
        rT=lr_at(dT,p); rH=lr_at(dH,p)
        if not rT or not rH or rT["hr"] is None or rH["hr"] is None: continue
        if rT["p"]<0.05 and rH["p"]<0.05 and (rT["hr"]-1)*(rH["hr"]-1)>0:
            # 判据: 使较弱队列的证据最强 (maximin)。相比取两队列显著性之积,
            # 它不允许一个队列的强 P 值补偿另一个队列的弱 P 值, 更保守。
            sc=-np.log10(max(rT["p"],rH["p"]))
            if best is None or sc>best["sc"]: best={"pct":float(p),"sc":sc}
    up=best["pct"] if best else 0.5
    e={"pass":best is not None,"pct":up}
    fr={"ct":ct,"pass":best is not None,"pct":up}
    for nm,pr,sv,cov,cc in COH:
        d=prep(pr,sv,cov,ct); r=lr_at(d,up); m=mv(d,cc,up)
        e[nm]={"km":km(d,up),"logrank_p":r["p"] if r else None,"uni_hr":r["hr"] if r else None,"mv":m,"n_hi":r["n_hi"] if r else 0,"n_lo":r["n_lo"] if r else 0}
        fr[nm+"_hr"]=r["hr"] if r else None; fr[nm+"_p"]=r["p"] if r else None
    results[ct]=e; forest.append(fr)
    print(f"  {ct:12s} {'✅' if best else '❌'} cut@{round(up*100)}%  TCGA p={e['TCGA']['logrank_p']:.3f} HR={e['TCGA']['uni_hr']:.2f} | HANCOCK p={e['HANCOCK']['logrank_p']:.3f} HR={e['HANCOCK']['uni_hr']:.2f}")

# --- 4) CPTAC θ (真值) null 备份 ---
thC=pd.read_csv("run/CPTAC_HNSC/deconv/bayesprism_oropharyngeal/theta.csv",index_col=0); thC.index=thC.index.astype(str)
svC=lsv("run/CPTAC_HNSC/survival.csv"); comC=[c for c in thC.index if c in svC.index]
zc=lambda x:(x-np.nanmean(x))/(np.nanstd(x)+1e-9); cptac_theta={}
print(f"\n=== CPTAC θ(真值) null 备份: {len(comC)}病人 {int(svC.loc[comC,'event'].sum())}事件 ===")
for ct in CTS:
    d=pd.DataFrame({"T":svC.loc[comC,"time_days"].values,"E":svC.loc[comC,"event"].values,"z":zc(thC.loc[comC,ct].values)}).dropna()
    s=CoxPHFitter().fit(d,"T","E").summary.loc["z"]
    cptac_theta[ct]={"hr":float(np.exp(s["coef"])),"lo":float(np.exp(s["coef lower 95%"])),"hi":float(np.exp(s["coef upper 95%"])),"p":float(s["p"])}
    if ct in ["T cell","B cell","Fibroblast"]: print(f"  {ct:12s} HR={cptac_theta[ct]['hr']:.2f} p={cptac_theta[ct]['p']:.3f}")

json.dump({"cells":results,"forest":forest,"cptac_theta":cptac_theta,
           "events":{"TCGA":int(svT["event"].sum()),"HANCOCK":int(svH["event"].sum()),"CPTAC":int(svC.loc[comC,"event"].sum())}},
          open("figures/paper/cache/prognosis_pred.json","w"),default=float)
print("\n通过(TCGA+HANCOCK复现):",[c for c in CTS if results[c]["pass"]])
