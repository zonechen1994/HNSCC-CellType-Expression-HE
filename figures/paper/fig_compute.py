#!/usr/bin/env python3
"""Compute + cache ALL per-sample arrays for the 4-figure HNSC paper set.
Reuses the exact logic from build_notebook_hnsc.py. Heavy part = HPV external
(apply TCGA models to HANCOCK features on GPU0); everything else is cheap CPU
from on-disk files. Each block caches independently so a late failure keeps
earlier results. Run once; fig_plot.py then renders 300dpi PDFs from the cache.
"""
import os, re, json, glob, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))

TRAIN_RUN="run/TCGA_HNSC_oro_v2b"; CPTAC_RUN="run/CPTAC_HNSC"; HANCOCK_RUN="run/HANCOCK"
TCGA_THETA="run/TCGA_HNSC/deconv/bayesprism_oropharyngeal_v2/theta.csv"
CPTAC_ZDIR="run/CPTAC_HNSC/deconv/bayesprism_oropharyngeal"
HPV_TABLE="run/TCGA_HNSC/driver_mutations.csv"
TOPK=1000; PCC_THR=0.4
CACHE="figures/paper/cache"; os.makedirs(CACHE,exist_ok=True)
CTS=list(pd.read_csv(f"{TRAIN_RUN}/cell_types_file.csv")["cell_types"])
print("cell types:",CTS)
def dump(name,obj):
    with open(f"{CACHE}/{name}.json","w") as f: json.dump(obj,f,indent=1,default=float)
    print("  cached",name)

from scipy.stats import pearsonr, spearmanr
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# ---- gene list helper (aligned to coef_slope.txt / test_preds.txt columns) ----
def genes_of(ct): return list(pd.read_csv(f"{TRAIN_RUN}/breast_{ct}_gene_file.csv")["gene"])

# ================= 1. INTERNAL expression PCC (Fig2a, S2) =================
print("[1] internal expression PCC (coef_slope.txt)")
expr_pcc={}; expr_full={}; tcga_ng={}; gene_pcc={}
for ct in CTS:
    tcs=[]
    for ik in range(5):
        f=f"{TRAIN_RUN}/results/{ct}_preds/result_{ik}_0/coef_slope.txt"
        if os.path.exists(f) and os.path.getsize(f)>0:
            a=np.loadtxt(f)
            if a.ndim==2: tcs.append(a[:,1])           # col1 = held-out TEST per-gene PCC
    tc_full=np.nanmean(np.vstack(tcs),0)
    gene_pcc[ct]=tc_full
    tc=tc_full[~np.isnan(tc_full)]
    expr_pcc[ct]=float(np.median(np.sort(tc)[::-1][:TOPK]))
    expr_full[ct]=float(np.median(tc))
    tcga_ng[ct]=int((tc>PCC_THR).sum())
    print(f"  {ct:12s} top1000={expr_pcc[ct]:.3f} full={expr_full[ct]:.3f} >0.4:{tcga_ng[ct]}")
dump("expr_internal",{"top1000":expr_pcc,"full":expr_full,"ngenes":tcga_ng})

# CPTAC external top-1000 median PCC (from v2b_external.log, cell types validated on disk)
cptac_expr={"Malignant":0.417,"B cell":0.325,"Dendritic":0.377,"Endothelial":0.413,
            "Fibroblast":0.417,"Macrophage":0.358,"Mast":0.232,"T cell":0.380}
dump("expr_cptac",cptac_expr)

# ================= 2. example gene scatters (Fig2b) =================
print("[2] example high-PCC gene scatters")
def pool_gene(ct, gene):
    gl=genes_of(ct); j=gl.index(gene)
    P=[]; L=[]
    for ik in range(5):
        d=f"{TRAIN_RUN}/results/{ct}_preds/result_{ik}_0"
        p=np.loadtxt(f"{d}/test_preds.txt"); l=np.loadtxt(f"{d}/test_labels.txt")
        P.append(p[:,j]); L.append(l[:,j])
    return np.concatenate(P), np.concatenate(L)
# pick top gene from three well-predicted, biologically distinct cell types
examples=[]; used=set()
for ct in ["Malignant","Fibroblast","Macrophage"]:
    gl=genes_of(ct); order=np.argsort(-np.nan_to_num(gene_pcc[ct]))
    gene=None; r=np.nan
    for oi in order:
        g=gl[oi]
        if g not in used: gene=g; r=float(gene_pcc[ct][oi]); break
    used.add(gene)
    pr,ob=pool_gene(ct,gene)
    examples.append({"ct":ct,"gene":gene,"pcc":r,"pred":pr.tolist(),"obs":ob.tolist()})
    print(f"  {ct:12s} {gene}  PCC={r:.3f}  n={len(pr)}")
dump("expr_examples",examples)

# ================= 3. abundance Spearman TCGA CV (Fig3a) =================
print("[3] abundance Spearman (cell_fraction_preds)")
Ps=[]; Ls=[]
for ik in range(5):
    d=f"{TRAIN_RUN}/results/cell_fraction_preds/result_{ik}_0"
    Ps.append(np.loadtxt(f"{d}/test_preds.txt")); Ls.append(np.loadtxt(f"{d}/test_labels.txt"))
P=np.vstack(Ps); L=np.vstack(Ls)
ab_sp={ct: float(spearmanr(P[:,i],L[:,i]).correlation) for i,ct in enumerate(CTS)}
for ct in CTS: print(f"  {ct:12s} rho={ab_sp[ct]:.3f}")
dump("abundance_internal",ab_sp)

# ================= 4. transfer QC CPTAC (Fig3b) =================
print("[4] transfer QC (CPTAC pred vs theta)")
ab_c=pd.read_csv(f"{CPTAC_RUN}/pred_abundance.csv",index_col=0); ab_c.index=ab_c.index.astype(str)
th_c=pd.read_csv(f"{CPTAC_ZDIR}/theta.csv",index_col=0); th_c.index=th_c.index.astype(str)
com=[c for c in ab_c.index if c in th_c.index]
transfer={ct: float(spearmanr(ab_c.loc[com,ct],th_c.loc[com,ct]).correlation) for ct in CTS}
for ct in CTS: print(f"  {ct:12s} r={transfer[ct]:.3f}")
dump("transfer_cptac",transfer)

# ================= 5. IHC (Fig3c) =================
print("[5] IHC (HANCOCK pred T cell vs CD3/CD8)")
ab_h=pd.read_csv(f"{HANCOCK_RUN}/pred_abundance.csv",index_col=0); ab_h.index=ab_h.index.astype(str).str.zfill(3)
ihc=pd.read_csv(f"{HANCOCK_RUN}/ihc_density.csv",dtype={'case':str}).set_index("case")
ihc_out={}
for marker in ["CD3_density","CD8_density"]:
    c=[i for i in ab_h.index if i in ihc.index and pd.notna(ihc.loc[i,marker])]
    x=ab_h.loc[c,"T cell"].values.astype(float); y=ihc.loc[c,marker].values.astype(float)
    r=spearmanr(x,y)
    ihc_out[marker]={"x":x.tolist(),"y":y.tolist(),"rho":float(r.correlation),"p":float(r.pvalue),"n":len(c)}
    print(f"  {marker}: rho={r.correlation:.3f} p={r.pvalue:.1e} n={len(c)}")
dump("ihc",ihc_out)

# ================= 6. survival: HANCOCK OS/DSS Cox + KM; TCGA theta Cox (Fig4) =================
print("[6] survival Cox + KM")
def cox_hr(z,T,E):
    s=CoxPHFitter().fit(pd.DataFrame({"T":T,"E":E,"z":z}),"T","E").summary.loc["z"]
    return float(np.exp(s["coef"])),float(np.exp(s["coef lower 95%"])),float(np.exp(s["coef upper 95%"])),float(s["p"])
def zc(x): x=np.asarray(x,float); return (x-np.nanmean(x))/(np.nanstd(x)+1e-9)
surv={}
# HANCOCK OS + DSS per cell (predicted abundance)
for tag,svf in [("hancock_os",f"{HANCOCK_RUN}/survival.csv"),("hancock_dss",f"{HANCOCK_RUN}/survival_dss.csv")]:
    sv=pd.read_csv(svf,dtype={'case':str}).set_index("case")
    com=[c for c in ab_h.index if c in sv.index]
    surv[tag]={"n":len(com),"events":int(sv.loc[com,"event"].sum()),"hr":{}}
    for ct in CTS:
        surv[tag]["hr"][ct]=cox_hr(zc(ab_h.loc[com,ct].values),sv.loc[com,"time_days"].values,sv.loc[com,"event"].values)
    print(f"  {tag}: n={len(com)} T-cell HR={surv[tag]['hr']['T cell'][0]:.2f} Macro HR={surv[tag]['hr']['Macrophage'][0]:.2f}")
# TCGA theta OS per cell (real deconvolved)
thT=pd.read_csv(TCGA_THETA,index_col=0); thT.index=thT.index.astype(str)
svT=pd.read_csv(f"{TRAIN_RUN}/survival.csv",dtype={'case':str}).set_index("case")
comT=[c for c in thT.index if c in svT.index]
surv["tcga_os"]={"n":len(comT),"events":int(svT.loc[comT,"event"].sum()),"hr":{}}
for ct in thT.columns:
    if ct in CTS:
        surv["tcga_os"]["hr"][ct]=cox_hr(zc(thT.loc[comT,ct].values),svT.loc[comT,"time_days"].values,svT.loc[comT,"event"].values)
print(f"  tcga_os: n={len(comT)} T-cell HR={surv['tcga_os']['hr']['T cell'][0]:.2f}")
dump("survival_cox",surv)
# KM groups (HANCOCK T cell & Macrophage, OS)
svh=pd.read_csv(f"{HANCOCK_RUN}/survival.csv",dtype={'case':str}).set_index("case")
com=[c for c in ab_h.index if c in svh.index]
km_out={}
for ct in ["Macrophage","T cell"]:
    v=ab_h.loc[com,ct]; hi=(v>=v.median()).values
    Tm=svh.loc[com,"time_days"].values/30.44; E=svh.loc[com,"event"].values
    lr=logrank_test(Tm[hi],Tm[~hi],E[hi],E[~hi]).p_value
    km_out[ct]={"time":Tm.tolist(),"event":E.tolist(),"hi":hi.tolist(),"logrank_p":float(lr)}
    print(f"  KM {ct}: logrank p={lr:.4f}")
dump("km_hancock",km_out)
# immune abundance by HPV status (HANCOCK) — biology panel
hpv_h=pd.read_csv(f"{HANCOCK_RUN}/hpv.csv",dtype={'case':str}).set_index("case")["hpv_p16"].map({"positive":1,"negative":0})
byhpv={}
for ct in ["T cell","Dendritic","B cell","Macrophage"]:
    c=[i for i in ab_h.index if i in hpv_h.index and pd.notna(hpv_h.loc[i])]
    pos=ab_h.loc[[i for i in c if hpv_h.loc[i]==1],ct].values
    neg=ab_h.loc[[i for i in c if hpv_h.loc[i]==0],ct].values
    from scipy.stats import mannwhitneyu
    p=float(mannwhitneyu(pos,neg).pvalue)
    byhpv[ct]={"pos":pos.tolist(),"neg":neg.tolist(),"p":p}
    print(f"  {ct:12s} HPV+ vs -: median {np.median(pos):.3f}/{np.median(neg):.3f} p={p:.1e}")
dump("abundance_by_hpv",byhpv)

# ================= 7. multivariable nested Cox (Fig4 forest, S3) =================
print("[7] multivariable nested Cox")
def mvfit(df,var="z"):
    df=df.apply(pd.to_numeric,errors="coerce").dropna()
    if len(df)<40 or df["E"].sum()<10 or df["T"].nunique()<3: return None
    keep=["T","E"]+[c for c in df.columns if c not in("T","E") and df[c].nunique()>1]
    if var not in keep: return None
    try: s=CoxPHFitter().fit(df[keep],"T","E").summary
    except Exception: return None
    r=s.loc[var]
    return [float(np.exp(r['coef'])),float(np.exp(r['coef lower 95%'])),float(np.exp(r['coef upper 95%'])),float(r['p']),int(len(df))]
def hpv_term(df,var):
    df=df.apply(pd.to_numeric,errors="coerce").dropna()
    keep=["T","E"]+[c for c in df.columns if c not in("T","E") and df[c].nunique()>1]
    if var not in keep: return None
    try: s=CoxPHFitter().fit(df[keep],"T","E").summary
    except Exception: return None
    r=s.loc[var]; return [float(np.exp(r['coef'])),float(r['p'])]
_stg={'STAGE I':1,'STAGE II':2,'STAGE III':3,'STAGE IVA':4,'STAGE IVB':4,'STAGE IVC':4,'STAGE IV':4}
clT=pd.read_csv("run/TCGA_HNSC/clinical.csv",index_col=0); clT.index=clT.index.astype(str)
covT=pd.DataFrame({"age":zc(clT["AGE"]),"sex":clT["SEX"].map({"Male":1,"Female":0}),
    "stage":clT["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(_stg),
    "hpv":clT["SUBTYPE"].map(lambda x:1 if str(x).endswith("HPV+") else(0 if str(x).endswith("HPV-") else np.nan))},index=clT.index)
clH=pd.read_csv(f"{HANCOCK_RUN}/clinical.csv",dtype={'case':str}).set_index("case")
covH=pd.DataFrame({"age":zc(clH["age"]),"sex":clH["sex"],"pT":clH["pT"],"pN":clH["pN"],"grade":clH["grade"],"hpv":clH["hpv_p16"]},index=clH.index)
analyses=[("TCGA T cell (OS, real θ)",thT,f"{TRAIN_RUN}/survival.csv",covT,["age","sex","stage"],"T cell"),
          ("HANCOCK T cell (OS)",ab_h,f"{HANCOCK_RUN}/survival.csv",covH,["age","sex","pT","pN","grade"],"T cell"),
          ("HANCOCK Macrophage (OS)",ab_h,f"{HANCOCK_RUN}/survival.csv",covH,["age","sex","pT","pN","grade"],"Macrophage")]
MV=[]
for lab,ab,svf,cov,cc,ct in analyses:
    sv=pd.read_csv(svf,dtype={'case':str}).set_index("case")
    base=pd.DataFrame({"T":sv["time_days"],"E":sv["event"],"z":zc(ab[ct].reindex(sv.index).values)},index=sv.index).join(cov)
    ccx=[c for c in cc if c in base.columns]
    m1=mvfit(base[["T","E","z"]]); m2=mvfit(base[["T","E","z"]+ccx])
    m3=mvfit(base[["T","E","z"]+ccx+["hpv"]]) if "hpv" in base.columns else None
    hpvself=hpv_term(base[["T","E","z"]+ccx+["hpv"]],"hpv") if "hpv" in base.columns else None
    MV.append({"label":lab,"m1":m1,"m2":m2,"m3":m3,"hpv_self":hpvself})
    _s=lambda m: f"{m[0]:.2f}" if m else "NA"
    print(f"  {lab}: uni={_s(m1)} +clin={_s(m2)} +HPV={_s(m3)}")
dump("multivariable",MV)

# ================= 8. S1 reference / B-cell repair =================
print("[8] S1 baseline vs new reference")
base=pd.read_csv("experiments/ref_comparison/baseline_GSE103322.csv")
print(base.head().to_string())
base.to_csv(f"{CACHE}/s1_baseline.csv",index=False)
# new (oropharyngeal v2b) numbers = expr_pcc computed above; abundance from log parse handled in plot
print("  saved s1_baseline.csv")

print("\n=== CHEAP BLOCKS DONE ===")
if os.path.exists(f"{CACHE}/hpv_roc.npz") and os.path.exists(f"{CACHE}/hpv_percell_auc.json"):
    print("HPV ROC already cached -> skipping GPU block. (delete hpv_roc.npz to recompute)")
    import sys; sys.exit(0)
print("=== Starting HPV external (GPU) ===")
# ================= 9. HPV ROC internal + external (Fig4a) — GPU =================
import sys; sys.path.insert(0, os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/code/SLIDE_EX_code/SLIDE_EX_code/prediction")
import torch
DEV=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print("  device",DEV)
from model_MLP import MLP_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
_FC={}
def load_feats(rd):
    if rd in _FC: return _FC[rd]
    cf=np.load(f"{rd}/features.pkl",allow_pickle=True)
    sids=[f[0] for f in cf]; X=[torch.tensor(np.asarray(f[1]),dtype=torch.float32) for f in cf]
    _FC[rd]=(sids,X); return sids,X
def apply_model(rd,sub,outdim):
    sids,X=load_feats(rd); base=f"{TRAIN_RUN}/results/{sub}"; pr=np.zeros((len(X),outdim)); n=0
    for ik in range(5):
        mp=f"{base}/result_{ik}_0/model_trained.pth"
        m=MLP_regression(768,768,outdim,0.2,None); m.load_state_dict(torch.load(mp,map_location=DEV)); m.to(DEV).eval()
        with torch.no_grad(): pr+=np.vstack([m(x.to(DEV)).cpu().numpy() for x in X]); n+=1
    return pr/n,sids
def slide_to_case(rd):
    m=pd.read_csv(f"{rd}/metadata.csv"); return {r["Slide.ID"]:str(r["case"]) for _,r in m.iterrows()}
sp=np.load(f"{TRAIN_RUN}/splits/train_valid_test_idx.npz",allow_pickle=True)
tcga_cases=pd.read_csv(f"{TRAIN_RUN}/metadata.csv")["case"].tolist()
hpv_t=pd.read_csv(HPV_TABLE,index_col=0)["HPV"]
hpv_h=pd.read_csv(f"{HANCOCK_RUN}/hpv.csv",dtype={'case':str}).set_index("case")["hpv_p16"].map({"positive":1,"negative":0})
def ens(): return VotingClassifier([("lr",make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))),
  ("rf",RandomForestClassifier(n_estimators=300,random_state=0)),
  ("svm",make_pipeline(StandardScaler(),SVC(probability=True,random_state=0)))],voting="soft")
def tcga_oof_expr(ct):
    P,order=[],[]
    for ik in range(5):
        d=f"{TRAIN_RUN}/results/{ct}_preds/result_{ik}_0"
        if not os.path.exists(f"{d}/test_preds.txt"): return None
        P.append(np.loadtxt(f"{d}/test_preds.txt")); order.extend(sp["test_idx"][ik].tolist())
    return pd.DataFrame(np.vstack(P),index=[tcga_cases[i] for i in order],columns=genes_of(ct))
hpv_auc={}; hpv_int_auc={}
int_sum=None; int_y=None; ext_sum=None; ext_y=None; ni=0; ne=0
for ct in CTS:
    Xt=tcga_oof_expr(ct)
    if Xt is None: continue
    ct_t=[c for c in Xt.index if c in hpv_t.index]; yt=hpv_t.loc[ct_t].values.astype(int)
    if yt.sum()<8: continue
    Xtv=Xt.loc[ct_t].values
    oof=np.zeros(len(yt))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xtv,yt):
        oof[te]=ens().fit(Xtv[tr],yt[tr]).predict_proba(Xtv[te])[:,1]
    hpv_int_auc[ct]=float(roc_auc_score(yt,oof))
    if int_sum is None: int_sum=oof.copy(); int_y=yt; ni=1
    elif len(oof)==len(int_sum): int_sum+=oof; ni+=1
    pr,sids=apply_model(HANCOCK_RUN,f"{ct}_preds",len(genes_of(ct)))
    s2c=slide_to_case(HANCOCK_RUN)
    Xh=pd.DataFrame(pr,index=[str(s2c.get(s)).zfill(3) for s in sids],columns=genes_of(ct)).groupby(level=0).mean()
    ch=[c for c in Xh.index if c in hpv_h.index]
    if len(ch)<20 or hpv_h.loc[ch].sum()<5: continue
    ye=hpv_h.loc[ch].values.astype(int)
    pe=ens().fit(Xtv,yt).predict_proba(Xh.loc[ch].values)[:,1]
    hpv_auc[ct]=float(roc_auc_score(ye,pe))
    if ext_sum is None: ext_sum=pe.copy(); ext_y=ye; ne=1
    elif len(pe)==len(ext_sum): ext_sum+=pe; ne+=1
    print(f"  {ct:12s} int AUC={hpv_int_auc[ct]:.3f} ext AUC={hpv_auc[ct]:.3f} (n={len(ch)},HPV+={int(ye.sum())})")
int_prob=(int_sum/max(ni,1)); ext_prob=(ext_sum/max(ne,1))
def auc_ci(y,p,nb=2000,seed=0):
    y=np.asarray(y); p=np.asarray(p); rng=np.random.RandomState(seed); n=len(y); bs=[]
    for _ in range(nb):
        idx=rng.randint(0,n,n)
        if len(np.unique(y[idx]))<2: continue
        bs.append(roc_auc_score(y[idx],p[idx]))
    return float(roc_auc_score(y,p)),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
ai,il,ih=auc_ci(int_y,int_prob); ae,el,eh=auc_ci(ext_y,ext_prob)
print(f"  ENSEMBLE int AUC={ai:.3f} ({il:.2f}-{ih:.2f}) | ext AUC={ae:.3f} ({el:.2f}-{eh:.2f})")
fpr_i,tpr_i,_=roc_curve(int_y,int_prob); fpr_e,tpr_e,_=roc_curve(ext_y,ext_prob)
np.savez(f"{CACHE}/hpv_roc.npz",
    fpr_i=fpr_i,tpr_i=tpr_i,fpr_e=fpr_e,tpr_e=tpr_e,
    int_auc=ai,int_lo=il,int_hi=ih,ext_auc=ae,ext_lo=el,ext_hi=eh,
    int_n=len(int_y),int_pos=int(int_y.sum()),ext_n=len(ext_y),ext_pos=int(ext_y.sum()))
dump("hpv_percell_auc",{"internal":hpv_int_auc,"external":hpv_auc})
print("\n=== ALL DONE ===")
