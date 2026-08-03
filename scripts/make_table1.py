#!/usr/bin/env python3
"""生成 Table 1 (队列临床特征) + 跨队列检验, 输出 markdown。
数据: TCGA/CPTAC(main+gdc)/HANCOCK clinical + survival。跨队列检验仅作描述, 见脚注。"""
import pandas as pd, numpy as np
from scipy.stats import kruskal, chi2_contingency
import os
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))

tc = pd.read_csv("run/TCGA_HNSC/clinical.csv")
cp = pd.read_csv("run/CPTAC_HNSC/clinical.csv")          # AGE, SEX  (n=108)
cg = pd.read_csv("run/CPTAC_HNSC/clinical_gdc.csv")      # stage,pT,pN,age_gdc (n=105)
hn = pd.read_csv("run/HANCOCK/clinical.csv")

# ---- 归一化辅助 ----
def normT(x):
    if pd.isna(x): return None
    s = str(x).upper().replace("T", "")
    if s and s[0] in "1234": return "T"+s[0]
    return None
def normN(x):
    if pd.isna(x): return None
    s = str(x).upper().replace("N", "")
    if s and s[0] in "0123": return "N"+s[0]
    return None
def normStage(x):
    if pd.isna(x): return None
    s = str(x).upper().replace("STAGE", "").strip()
    for r in ["IVC","IVB","IVA","IV","III","II","I"]:
        if s.startswith(r): return "IV" if r.startswith("IV") else r
    return None

# ---- 各队列变量 ----
D = {}
# age
D['age'] = {'TCGA': pd.to_numeric(tc['AGE'],errors='coerce').dropna(),
            'CPTAC': pd.to_numeric(cp['AGE'],errors='coerce').dropna(),
            'HANCOCK': pd.to_numeric(hn['age'],errors='coerce').dropna()}
# sex  (TCGA Male/Female; CPTAC main SEX; HANCOCK 1=male 0=female)
sx_tc = tc['SEX'].map({'Male':'M','Female':'F'})
sx_cp = cp['SEX'].map({'Male':'M','Female':'F'}) if cp['SEX'].dtype==object else cp['SEX'].map({1:'M',0:'F'})
sx_hn = hn['sex'].map({1:'M',0:'F'})
D['sex'] = {'TCGA':sx_tc.dropna(),'CPTAC':sx_cp.dropna(),'HANCOCK':sx_hn.dropna()}
# T / N
D['T'] = {'TCGA':tc['PATH_T_STAGE'].map(normT).dropna(),
          'CPTAC':cg['pT'].map(normT).dropna(),
          'HANCOCK':hn['pT'].map(normT).dropna()}
D['N'] = {'TCGA':tc['PATH_N_STAGE'].map(normN).dropna(),
          'CPTAC':cg['pN'].map(normN).dropna(),
          'HANCOCK':hn['pN'].map(normN).dropna()}
# M (TCGA only)
D['M'] = {'TCGA':tc['PATH_M_STAGE'].map(lambda x: 'M1' if str(x).upper().startswith('M1')
                                        else ('M0' if str(x).upper().startswith('M0') else None)).dropna()}
# overall stage (TCGA + CPTAC)
D['stage'] = {'TCGA':tc['AJCC_PATHOLOGIC_TUMOR_STAGE'].map(normStage).dropna(),
              'CPTAC':cg['stage'].map(normStage).dropna()}
# grade / smoking / site (HANCOCK only)
D['grade'] = {'HANCOCK':hn['grade'].map({1:'G1',2:'G2',3:'G3'}).dropna()}
D['site']  = {'HANCOCK':hn['site'].dropna()}
# HPV (TCGA subtype; HANCOCK p16)
hpv_tc = tc['SUBTYPE'].map({'HNSC_HPV+':'Pos','HNSC_HPV-':'Neg'})
hpv_hn = hn['hpv_p16'].map({1:'Pos',0:'Neg'})
D['hpv'] = {'TCGA':hpv_tc.dropna(),'HANCOCK':hpv_hn.dropna()}

# ---- survival ----
def surv(path,evcol=None):
    s=pd.read_csv(path)
    tcol=[c for c in s.columns if 'time' in c.lower() or 'day' in c.lower()][0]
    ecol=evcol or [c for c in s.columns if c.lower() in ('event','status','death','vital')][0]
    m=(pd.to_numeric(s[tcol],errors='coerce')/30.44).dropna()
    ev=pd.to_numeric(s[ecol],errors='coerce')
    return len(s), int(ev.sum()), m.median(), m.min(), m.max()
SV={'TCGA':surv("run/TCGA_HNSC_oro_v2b/survival.csv"),
    'CPTAC':surv("run/CPTAC_HNSC/survival.csv"),
    'HANCOCK':surv("run/HANCOCK/survival.csv")}

# ---- 检验 ----
def kw(d):  # continuous, 3 groups
    grps=[v.values for v in d.values() if len(v)>0]
    if len(grps)<2: return None
    return kruskal(*grps).pvalue
def chi(d, cats):  # categorical
    cols={k:[ (v==c).sum() for c in cats] for k,v in d.items() if len(v)>0}
    tab=np.array([cols[k] for k in cols]).T
    tab=tab[:, (tab.sum(0)>0)]  # drop empty cohorts
    if tab.shape[1]<2 or tab.sum()==0: return None
    tab=tab[tab.sum(1)>0]
    try: return chi2_contingency(tab)[1]
    except: return None

def pf(p):
    if p is None: return "—"
    if p<0.001: return "<0.001"
    return f"{p:.3f}"

def npct(series, cats):
    n=len(series)
    return {c: f"{(series==c).sum()} ({100*(series==c).sum()/n:.1f}%)" for c in cats}, n

def cell(d, coh, cats):
    if coh not in d or len(d[coh])==0: return "NA"
    counts,n = npct(d[coh], cats)
    return counts, n

COH=['TCGA','CPTAC','HANCOCK']
rows=[]
def R(label,a,b,c,p=""): rows.append((label,a,b,c,p))

# N patients / survival / slides
R("**N patients (clinical)**","523","108","763")
R("N with survival data",str(SV['TCGA'][0]),str(SV['CPTAC'][0]),str(SV['HANCOCK'][0]))
R("N slides analysed (WSI)","442","110","701")
# age
a=D['age']
R("Age, median [IQR], yrs",
  f"{a['TCGA'].median():.0f} [{a['TCGA'].quantile(.25):.0f}-{a['TCGA'].quantile(.75):.0f}]",
  f"{a['CPTAC'].median():.0f} [{a['CPTAC'].quantile(.25):.0f}-{a['CPTAC'].quantile(.75):.0f}]",
  f"{a['HANCOCK'].median():.0f} [{a['HANCOCK'].quantile(.25):.0f}-{a['HANCOCK'].quantile(.75):.0f}]",
  pf(kw(a)))
# sex
sx=D['sex']; sc={k:npct(v,['M','F']) for k,v in sx.items()}
R("Sex, n (%)","","","",pf(chi(sx,['M','F'])))
R("  Male", sc['TCGA'][0]['M'], sc['CPTAC'][0]['M'], sc['HANCOCK'][0]['M'])
R("  Female", sc['TCGA'][0]['F'], sc['CPTAC'][0]['F'], sc['HANCOCK'][0]['F'])
# T
Tc={k:npct(v,['T1','T2','T3','T4']) for k,v in D['T'].items()}
R("T stage, n (%)","","","",pf(chi(D['T'],['T1','T2','T3','T4'])))
for t in ['T1','T2','T3','T4']:
    R(f"  {t}", Tc['TCGA'][0][t], Tc['CPTAC'][0][t], Tc['HANCOCK'][0][t])
# N
Nc={k:npct(v,['N0','N1','N2','N3']) for k,v in D['N'].items()}
R("N stage, n (%)","","","",pf(chi(D['N'],['N0','N1','N2','N3'])))
for t in ['N0','N1','N2','N3']:
    R(f"  {t}", Nc['TCGA'][0][t], Nc['CPTAC'][0][t], Nc['HANCOCK'][0][t])
# M (TCGA only)
Mc,_=npct(D['M']['TCGA'],['M0','M1'])
R("M stage (TCGA only), n (%)","","","")
R("  M0", Mc['M0'], "NA", "NA")
R("  M1", Mc['M1'], "NA", "NA")
# overall stage
St={k:npct(v,['I','II','III','IV']) for k,v in D['stage'].items()}
R("AJCC stage, n (%)","","","",pf(chi(D['stage'],['I','II','III','IV'])))
for t in ['I','II','III','IV']:
    R(f"  {t}", St['TCGA'][0][t], St['CPTAC'][0][t], "NA")
# grade (HANCOCK)
Gc,_=npct(D['grade']['HANCOCK'],['G1','G2','G3'])
R("Grade (HANCOCK only), n (%)","","","")
for g in ['G1','G2','G3']:
    R(f"  {g}","NA","NA",Gc[g])
# site (HANCOCK)
site=D['site']['HANCOCK']
R("Tumour site (HANCOCK only)","NA","NA",
  "; ".join(f"{c} {int((site==c).sum())}" for c in ['Oropharynx','Larynx','Oral_Cavity','Hypopharynx','CUP']))
# HPV
R("HPV / p16 status","","","",pf(chi(D['hpv'],['Pos','Neg'])))
hp={k:npct(v,['Pos','Neg']) for k,v in D['hpv'].items()}
R("  Positive", hp['TCGA'][0]['Pos'], "0 (by design)", hp['HANCOCK'][0]['Pos'])
R("  Negative", hp['TCGA'][0]['Neg'], "108 (by design)", hp['HANCOCK'][0]['Neg'])
R("  Unknown / not typed", str(36), "0", str(431))
# follow-up / events
R("Median follow-up, months",
  f"{SV['TCGA'][2]:.1f}", f"{SV['CPTAC'][2]:.1f}", f"{SV['HANCOCK'][2]:.1f}")
R("Deaths / events",
  f"{SV['TCGA'][1]} / {SV['TCGA'][0]}", f"{SV['CPTAC'][1]} / {SV['CPTAC'][0]}", f"{SV['HANCOCK'][1]} / {SV['HANCOCK'][0]}")

# ---- markdown ----
out=["# Table 1. Cohort characteristics","",
     "| Characteristic | TCGA-HNSC | CPTAC-HNSCC | HANCOCK | P value |",
     "|---|---|---|---|---|"]
for label,x,y,z,p in rows:
    out.append(f"| {label} | {x} | {y} | {z} | {p} |")
md="\n".join(out)
open("figures/paper/cache/table1.md","w").write(md)
print(md)
print("\n\n[saved figures/paper/cache/table1.md]")
PY
