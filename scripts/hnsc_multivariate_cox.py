#!/usr/bin/env python3
"""完整多变量 Cox: T细胞/巨噬 丰度在调整临床因素(年龄/性别/分期/分级)与 HPV 后是否独立预后。
- TCGA: 真值theta丰度 + cBioPortal临床(AGE/SEX/AJCC stage/SUBTYPE=HPV)
- HANCOCK: 预测丰度 + StructuredData临床(age/sex/pT/pN/grade/p16)
- CPTAC: RNA真值θ丰度 + age/sex/stage(纯HPV阴性; 预测丰度未过迁移QC, 故用真值θ, 参考)
每种细胞输出 3 个模型: 单变量 / +临床 / +HPV。丰度预测缓存到 CSV。
输出 experiments/ref_comparison/multivariate_cox.log
"""
import os, sys, re, numpy as np, pandas as pd, torch
from lifelines import CoxPHFitter
sys.path.insert(0, os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TR="run/TCGA_HNSC_oro_v2b"; RES=f"{TR}/results"
CTS=list(pd.read_csv(f"{TR}/cell_types_file.csv")["cell_types"])
out=[]; P=lambda s:(out.append(str(s)),print(s))

def apply_ab(run):
    cf=np.load(f"{run}/features.pkl",allow_pickle=True)
    sids=[f[0] for f in cf]; X=[torch.tensor(np.asarray(f[1]),dtype=torch.float32) for f in cf]
    pr=np.zeros((len(X),len(CTS)))
    for ik in range(5):
        m=MLP_regression(768,768,len(CTS),0.2,None)
        m.load_state_dict(torch.load(f"{RES}/cell_fraction_preds/result_{ik}_0/model_trained.pth",map_location=dev)); m.to(dev).eval()
        with torch.no_grad(): pr+=np.vstack([m(x.to(dev)).cpu().numpy() for x in X])
    return pr/5, sids
def pred_ab(run,z=None):
    cache=f"{run}/pred_abundance.csv"
    if os.path.exists(cache):
        ab=pd.read_csv(cache,index_col=0); ab.index=ab.index.astype(str)
        if z: ab.index=ab.index.str.zfill(z)     # "001" 被读成 1 -> 补回
        return ab
    pr,sids=apply_ab(run); meta=pd.read_csv(f"{run}/metadata.csv")
    s2c={r["Slide.ID"]:(str(r["case"]).zfill(z) if z else str(r["case"])) for _,r in meta.iterrows()}
    ab=pd.DataFrame(pr,index=[s2c.get(s) for s in sids],columns=CTS).groupby(level=0).mean()
    ab.to_csv(cache); return ab

def zc(x): x=x.astype(float); return (x-np.nanmean(x))/(np.nanstd(x)+1e-9)

def fit_hr(df, var="z"):
    """稳健拟合, 返回 (HR,lo,hi,p,n) 或 None。df 数值化+去常数列, 防 lifelines 边缘崩。"""
    df=df.apply(pd.to_numeric,errors="coerce").dropna()
    if len(df)<40 or df["E"].sum()<10 or df["T"].nunique()<3: return None
    keep=["T","E"]+[c for c in df.columns if c not in ("T","E") and df[c].nunique()>1]
    if var not in keep: return None
    try:
        s=CoxPHFitter().fit(df[keep],"T","E").summary
    except Exception: return None
    r=s.loc[var]; hpv=s.loc["hpv"] if "hpv" in s.index else None
    return (np.exp(r['coef']),np.exp(r['coef lower 95%']),np.exp(r['coef upper 95%']),r['p'],len(df),
            (np.exp(hpv['coef']),hpv['p']) if hpv is not None else None)

def models(ab, surv, clin, hpv_col, cohort, covs):
    P(f"\n{'='*70}\n{cohort}\n{'='*70}")
    for ct in ["T cell","Macrophage"]:
        base=pd.DataFrame({"T":surv["time_days"],"E":surv["event"],"z":zc(ab[ct].reindex(surv.index))},index=surv.index)
        base=base.join(clin)
        P(f"\n[{ct}]")
        cc=[c for c in covs if c in base.columns and c!=hpv_col]
        r0=fit_hr(base[["T","E","z"]])
        if r0: P(f"  ① 单变量          HR={r0[0]:.2f} ({r0[1]:.2f}-{r0[2]:.2f}) p={r0[3]:.4f} n={r0[4]}")
        r1=fit_hr(base[["T","E","z"]+cc])
        if r1: P(f"  ② +临床({'+'.join(cc)})  HR={r1[0]:.2f} ({r1[1]:.2f}-{r1[2]:.2f}) p={r1[3]:.4f} n={r1[4]}")
        if hpv_col and hpv_col in base.columns:
            r2=fit_hr(base[["T","E","z"]+cc+[hpv_col]])
            if r2: P(f"  ③ +临床+HPV       HR={r2[0]:.2f} ({r2[1]:.2f}-{r2[2]:.2f}) p={r2[3]:.4f} n={r2[4]}"+(f"  [HPV自身 HR={r2[5][0]:.2f} p={r2[5][1]:.4f}]" if r2[5] else ""))

P("多变量 Cox — 丰度调整临床/HPV 后是否独立预后 (HR<1=保护)")

# ===== TCGA =====
theta=pd.read_csv("run/TCGA_HNSC/deconv/bayesprism_oropharyngeal_v2/theta.csv",index_col=0); theta.index=theta.index.astype(str)
sv=pd.read_csv(f"{TR}/survival.csv",dtype={'case':str}).set_index("case")
cl=pd.read_csv("run/TCGA_HNSC/clinical.csv",index_col=0)
stg={'STAGE I':1,'STAGE II':2,'STAGE III':3,'STAGE IVA':4,'STAGE IVB':4,'STAGE IVC':4,'STAGE IV':4}
clT=pd.DataFrame({"age":zc(cl["AGE"]),"sex":cl["SEX"].map({"Male":1,"Female":0}),
                  "stage":cl["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(stg),
                  "hpv":cl["SUBTYPE"].map(lambda x:1 if str(x).endswith("HPV+") else (0 if str(x).endswith("HPV-") else np.nan))},index=cl.index.astype(str))
models(theta, sv, clT, "hpv", "TCGA  (真值theta丰度, OS)", ["age","sex","stage"])

# ===== HANCOCK =====
abH=pred_ab("run/HANCOCK",z=3)
clH=pd.read_csv("run/HANCOCK/clinical.csv",dtype={'case':str}).set_index("case")
clH=pd.DataFrame({"age":zc(clH["age"]),"sex":clH["sex"],"pT":clH["pT"],"pN":clH["pN"],"hpv":clH["hpv_p16"]},index=clH.index)
for svf,lab in [("run/HANCOCK/survival.csv","OS"),("run/HANCOCK/survival_dss.csv","DSS")]:
    svH=pd.read_csv(svf,dtype={'case':str}).set_index("case")
    models(abH, svH, clH, "hpv", f"HANCOCK  (预测丰度, {lab})", ["age","sex","pT","pN"])

# ===== CPTAC (RNA真值θ丰度; 预测丰度在此免疫冷队列迁移不佳, 见迁移QC, 故预后用真值θ) =====
abC=pd.read_csv("run/CPTAC_HNSC/deconv/bayesprism_oropharyngeal/theta.csv",index_col=0); abC.index=abC.index.astype(str)
clC=pd.read_csv("run/CPTAC_HNSC/clinical.csv",index_col=0); clC.index=clC.index.astype(str)
gdc=pd.read_csv("run/CPTAC_HNSC/clinical_gdc.csv",dtype={'case':str}).set_index("case")
stgC={'Stage I':1,'Stage II':2,'Stage III':3,'Stage IV':4,'Stage IVA':4,'Stage IVB':4,'Stage IVC':4}
clCd=pd.DataFrame({"age":zc(gdc["age_gdc"]),"sex":clC["SEX"].map({"Male":1,"Female":0}),
                  "stage":gdc["stage"].map(stgC)})
svC=pd.read_csv("run/CPTAC_HNSC/survival.csv",dtype={'case':str}).set_index("case")
models(abC, svC, clCd, None, "CPTAC  (真值θ丰度, OS, HPV阴性)", ["age","sex","stage"])

P("\nMULTIVAR_COX_DONE")
open("experiments/ref_comparison/multivariate_cox.log","w").write("\n".join(out))
