#!/usr/bin/env python3
"""生成 Table 1 (队列临床特征), 输出 markdown, 复现 HNSC_cohort_table.md。

设计原则
--------
* 绝大多数格子从 repo 内真实数据实时计算 (见下方数据源)。
* CPTAC-HNSCC 队列锚定到 110 例有临床标注者: pred_abundance 里有 112 例,
  其中 2 例 (C3N-04151, C3N-04153) 只有切片/RNA、无任何临床标注, 从未进入
  任何验证分析, 按 "以临床为准" 剔除。故 N=110, WSI=382 (390-8), 预测=110。
* HANCOCK grade 用修正后的值: 原始 pathological_data.json 中 141 个 p16 阳性
  病人的 grading 字段被误填成列名 "hpv_association_p16", 这里置为缺失
  (与 fix_hancock_grade.py 一致), 避免 grade 沦为 p16 的代理。
* 三个格子的源不在 repo 内, 以带出处的常量写入 (见 EXTERNAL 段):
  TCGA race, TCGA grade, CPTAC race —— 均来自 cBioPortal / GDC 临床导出。

数据源
------
run/TCGA_HNSC/clinical.csv            age/sex/AJCC/pT/pN/pM/HPV(subtype)
run/TCGA_HNSC/gdc_site_smoking.csv    TCGA site + smoking
run/CPTAC_HNSC/clinical.csv           CPTAC age/sex (108)
run/CPTAC_HNSC/clinical_gdc.csv       CPTAC stage/pT/pN (105)
run/CPTAC_HNSC/gdc_tumor_grade.csv    CPTAC grade + case 集 (110)
run/CPTAC_HNSC/gdc_site_smoking.csv   CPTAC site + smoking + M (110)
run/CPTAC_HNSC/pred_abundance.csv     CPTAC 预测 case (112 -> 取 110)
run/CPTAC_HNSC/metadata.csv           CPTAC 切片 (390 -> 取 110 例的 382)
data/HANCOCK_clinical/StructuredData/pathological_data.json   HANCOCK grade/site/pT/pN/p16
data/HANCOCK_clinical/StructuredData/clinical_data.json       HANCOCK sex/smoking
run/{coh}/survival.csv (CPTAC/HANCOCK), run/TCGA_HNSC_oro_v2b/survival.csv (TCGA)
run/{TCGA_HNSC,HANCOCK}/metadata.csv  切片数
"""
import os, json
import pandas as pd, numpy as np
from scipy.stats import kruskal, chi2_contingency

os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))

# ------------------------------------------------------------------ 归一化
def normT(x):
    if pd.isna(x): return None
    s = str(x).upper().replace("T", "")
    return "T" + s[0] if s and s[0] in "1234" else None
def normN(x):
    if pd.isna(x): return None
    s = str(x).upper().replace("N", "")
    return "N" + s[0] if s and s[0] in "0123" else None
def normStage(x):
    if pd.isna(x): return None
    s = str(x).upper().replace("STAGE", "").strip()
    for r in ["IVC", "IVB", "IVA", "IV", "III", "II", "I"]:
        if s.startswith(r): return "IV" if r.startswith("IV") else r
    return None
def normM(x):
    s = str(x).upper()
    if s.startswith("M1"): return "M1"
    if s.startswith("M0"): return "M0"
    return None
def site_bucket(s):
    s = str(s).lower()
    if ("larynx" in s or "glottis" in s) and "hypo" not in s: return "Larynx"
    if "hypopharynx" in s:                                     return "Hypopharynx"
    if any(k in s for k in ["oropharynx", "tonsil", "base of tongue"]): return "Oropharynx"
    if any(k in s for k in ["tongue", "floor of mouth", "lip", "gum", "cheek",
                            "mouth", "palate", "retromolar", "oral cavity"]): return "Oral cavity"
    return "Other / unspecified"
def smoking_ever(series):
    """从自由文本吸烟史归一到 Ever / Never (None=未知)。"""
    def m(x):
        s = str(x).lower()
        if any(k in s for k in ["non-smoker", "lifelong", "never"]): return "Never"
        if "smoker" in s or "former" in s or "reformed" in s:         return "Ever"
        return None
    return series.map(m)

# ------------------------------------------------------------------ 载入
tc  = pd.read_csv("run/TCGA_HNSC/clinical.csv")
tcs = pd.read_csv("run/TCGA_HNSC/gdc_site_smoking.csv")
cp  = pd.read_csv("run/CPTAC_HNSC/clinical.csv")
cg  = pd.read_csv("run/CPTAC_HNSC/clinical_gdc.csv")
cgr = pd.read_csv("run/CPTAC_HNSC/gdc_tumor_grade.csv")
css = pd.read_csv("run/CPTAC_HNSC/gdc_site_smoking.csv")
hnp = pd.DataFrame(json.load(open("data/HANCOCK_clinical/StructuredData/pathological_data.json")))
hnc = pd.DataFrame(json.load(open("data/HANCOCK_clinical/StructuredData/clinical_data.json")))
hn  = pd.read_csv("run/HANCOCK/clinical.csv")

# CPTAC 临床锚定集 = 110 (任意临床标注的并集), 剔除 2 例只有切片者
cptac_clin = (set(cp["patientId"].astype(str)) | set(cg["case"].astype(str)) |
              set(cgr["case"].astype(str)) | set(css["case"].astype(str)))
assert len(cptac_clin) == 110, len(cptac_clin)

# ------------------------------------------------------------------ 计数结构
# COUNTS[var][cohort] = {category: n};  RAWAGE[cohort] = Series
def vc(series, cats):
    s = series.dropna()
    return {c: int((s == c).sum()) for c in cats}

CATS = {
    "Sex": ["Male", "Female"],
    "Race": ["White", "Black", "Asian", "Other"],
    "T stage": ["T1", "T2", "T3", "T4"],
    "N stage": ["N0", "N1", "N2", "N3"],
    "M stage": ["M0", "M1"],
    "AJCC stage": ["I", "II", "III", "IV"],
    "Histologic grade": ["G1", "G2", "G3", "G4"],
    "Tumour site": ["Oral cavity", "Oropharynx", "Larynx", "Hypopharynx", "Other / unspecified"],
    "HPV / p16 status": ["Positive", "Negative"],
    "Smoking status": ["Ever", "Never"],
}
COUNTS = {k: {} for k in CATS}

# ---- TCGA (从数据) ----
COUNTS["Sex"]["TCGA"]        = vc(tc["SEX"], ["Male", "Female"])
COUNTS["T stage"]["TCGA"]    = vc(tc["PATH_T_STAGE"].map(normT), CATS["T stage"])
COUNTS["N stage"]["TCGA"]    = vc(tc["PATH_N_STAGE"].map(normN), CATS["N stage"])
COUNTS["M stage"]["TCGA"]    = vc(tc["PATH_M_STAGE"].map(normM), CATS["M stage"])
COUNTS["AJCC stage"]["TCGA"] = vc(tc["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(normStage), CATS["AJCC stage"])
COUNTS["Tumour site"]["TCGA"]= vc(tcs["site_organ"].map(site_bucket), CATS["Tumour site"])
COUNTS["Smoking status"]["TCGA"] = vc(smoking_ever(tcs["smoke"]), CATS["Smoking status"])
COUNTS["HPV / p16 status"]["TCGA"] = vc(tc["SUBTYPE"].map({"HNSC_HPV+": "Positive", "HNSC_HPV-": "Negative"}),
                                        CATS["HPV / p16 status"])

# ---- CPTAC (从数据, 各变量按其源天然分母) ----
COUNTS["Sex"]["CPTAC"]        = vc(cp["SEX"], ["Male", "Female"])
COUNTS["T stage"]["CPTAC"]    = vc(cg["pT"].map(normT), CATS["T stage"])
COUNTS["N stage"]["CPTAC"]    = vc(cg["pN"].map(normN), CATS["N stage"])
COUNTS["M stage"]["CPTAC"]    = vc(css["M"].map(normM), CATS["M stage"])
COUNTS["AJCC stage"]["CPTAC"] = vc(cg["stage"].map(normStage), CATS["AJCC stage"])
COUNTS["Histologic grade"]["CPTAC"] = vc(cgr["tumor_grade"], CATS["Histologic grade"])
COUNTS["Tumour site"]["CPTAC"]= vc(css["site_organ"].map(site_bucket), CATS["Tumour site"])
COUNTS["Smoking status"]["CPTAC"] = vc(smoking_ever(css["smoke"]), CATS["Smoking status"])
# CPTAC HPV: HPV-negative by design; 见格式化时特判

# ---- HANCOCK (从数据) ----
hnp["case"] = hnp["patient_id"].astype(str).str.zfill(3)
hnc["case"] = hnc["patient_id"].astype(str).str.zfill(3)
COUNTS["Sex"]["HANCOCK"]      = vc(hn["sex"].map({1: "Male", 0: "Female"}), ["Male", "Female"])
COUNTS["T stage"]["HANCOCK"]  = vc(hn["pT"].map(normT), CATS["T stage"])
COUNTS["N stage"]["HANCOCK"]  = vc(hn["pN"].map(normN), CATS["N stage"])
# HANCOCK grade: 丢弃非法 "hpv_association_p16" (置缺失)
COUNTS["Histologic grade"]["HANCOCK"] = vc(hnp["grading"].where(hnp["grading"].isin(["G1", "G2", "G3", "G4"])),
                                           CATS["Histologic grade"])
sitemap = {"Oral_Cavity": "Oral cavity", "Oropharynx": "Oropharynx", "Larynx": "Larynx",
           "Hypopharynx": "Hypopharynx", "CUP": "Other / unspecified"}
COUNTS["Tumour site"]["HANCOCK"] = vc(hn["site"].map(sitemap), CATS["Tumour site"])
COUNTS["Smoking status"]["HANCOCK"] = vc(hnc["smoking_status"].map(
    {"smoker": "Ever", "former": "Ever", "non-smoker": "Never"}), CATS["Smoking status"])
COUNTS["HPV / p16 status"]["HANCOCK"] = vc(hn["hpv_p16"].map({1: "Positive", 0: "Negative"}),
                                          CATS["HPV / p16 status"])

# ------------------------------------------------------------------ EXTERNAL 常量
# 以下三块源不在 repo (cBioPortal / GDC 临床导出), 以发表值写入并注明出处。
# TCGA race  : cBioPortal hnsc_tcga 'Race Category'          (denom 508)
COUNTS["Race"]["TCGA"]  = {"White": 448, "Black": 47, "Asian": 11, "Other": 2}
# TCGA grade : cBioPortal hnsc_tcga / GDC neoplasm grade     (denom 402)
COUNTS["Histologic grade"]["TCGA"] = {"G1": 51, "G2": 246, "G3": 100, "G4": 5}
# CPTAC race : cBioPortal ohnca_cptac_gdc 'Race Category' 子集 (denom 105)
COUNTS["Race"]["CPTAC"] = {"White": 97, "Black": 0, "Asian": 8, "Other": 0}

# ------------------------------------------------------------------ 连续 / 生存
def agearr(f, col): return pd.to_numeric(pd.read_csv(f)[col], errors="coerce").dropna()
AGE = {"TCGA": agearr("run/TCGA_HNSC/clinical.csv", "AGE"),
       "CPTAC": agearr("run/CPTAC_HNSC/clinical.csv", "AGE"),
       "HANCOCK": agearr("run/HANCOCK/clinical.csv", "age")}
def surv(f):
    s = pd.read_csv(f)
    tcol = [c for c in s.columns if "time" in c.lower() or "day" in c.lower()][0]
    ecol = [c for c in s.columns if c.lower() in ("event", "status", "death", "vital")][0]
    m = (pd.to_numeric(s[tcol], errors="coerce") / 30.44).dropna()
    return len(s), int(pd.to_numeric(s[ecol], errors="coerce").sum()), m.median()
SV = {"TCGA": surv("run/TCGA_HNSC_oro_v2b/survival.csv"),
      "CPTAC": surv("run/CPTAC_HNSC/survival.csv"),
      "HANCOCK": surv("run/HANCOCK/survival.csv")}

# 表头计数
N_WSI  = {"TCGA": len(pd.read_csv("run/TCGA_HNSC/metadata.csv")),          # 442
          "CPTAC": int(pd.read_csv("run/CPTAC_HNSC/metadata.csv")["case"].astype(str).isin(cptac_clin).sum()),  # 382
          "HANCOCK": len(pd.read_csv("run/HANCOCK/metadata.csv"))}         # 701
N_PRED = {"TCGA": N_WSI["TCGA"],                                           # 一slide一例, 全预测
          "CPTAC": int(pd.read_csv("run/CPTAC_HNSC/pred_abundance.csv")["Unnamed: 0"].astype(str).isin(cptac_clin).sum()),
          "HANCOCK": len(pd.read_csv("run/HANCOCK/pred_abundance.csv"))}
N_CLIN = {"TCGA": len(tc), "CPTAC": len(cptac_clin), "HANCOCK": len(hn)}

# ------------------------------------------------------------------ P 值
def pf(p):
    if p is None: return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"
def cat_p(var):
    cohs = [c for c in ["TCGA", "CPTAC", "HANCOCK"] if c in COUNTS[var]]
    tab = np.array([[COUNTS[var][c][cat] for cat in CATS[var]] for c in cohs]).T
    tab = tab[:, tab.sum(0) > 0]
    tab = tab[tab.sum(1) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2: return None
    return chi2_contingency(tab)[1]
def age_p():
    grps = [v.values for v in AGE.values() if len(v) > 0]
    return kruskal(*grps).pvalue

# 不做检验的变量 (仅描述): grade (不同分级体系), M (仅 TCGA/CPTAC 有且高度缺失)
P = {"Age": pf(age_p()), "Sex": pf(cat_p("Sex")), "Race": pf(cat_p("Race")),
     "T stage": pf(cat_p("T stage")), "N stage": pf(cat_p("N stage")), "M stage": "",
     "AJCC stage": pf(cat_p("AJCC stage")), "Histologic grade": "",
     "Tumour site": pf(cat_p("Tumour site")), "HPV / p16 status": pf(cat_p("HPV / p16 status")),
     "Smoking status": pf(cat_p("Smoking status"))}

# ------------------------------------------------------------------ 组表
def cell(var, coh, cat):
    if coh not in COUNTS[var]: return "NA"
    d = sum(COUNTS[var][coh].values())
    if d == 0: return "NA"
    n = COUNTS[var][coh][cat]
    return f"{n} ({100 * n / d:.1f}%)"

rows = []
def R(label, a, b, c, p=""): rows.append((label, a, b, c, p))

R("**N patients (clinical)**", N_CLIN["TCGA"], N_CLIN["CPTAC"], N_CLIN["HANCOCK"])
R("N with survival data", SV["TCGA"][0], SV["CPTAC"][0], SV["HANCOCK"][0])
R("N slides analysed (WSI)", N_WSI["TCGA"], N_WSI["CPTAC"], N_WSI["HANCOCK"])
R("N cases with model predictions", N_PRED["TCGA"], N_PRED["CPTAC"], N_PRED["HANCOCK"])
R("Age, median [IQR], yrs",
  *[f"{AGE[c].median():.0f} [{AGE[c].quantile(.25):.0f}-{AGE[c].quantile(.75):.0f}]" for c in ["TCGA", "CPTAC", "HANCOCK"]],
  P["Age"])

def block(var, label, cohorts=("TCGA", "CPTAC", "HANCOCK"), na_for=()):
    R(f"{label}, n (%)", "", "", "", P[var])
    disp = {"Race": {"White": "White", "Black": "Black", "Asian": "Asian", "Other": "Other"}}
    for cat in CATS[var]:
        vals = []
        for coh in ("TCGA", "CPTAC", "HANCOCK"):
            vals.append("NA" if coh in na_for else cell(var, coh, cat))
        R(f"  {cat}", *vals)

block("Sex", "Sex")
block("Race", "Race", na_for=("HANCOCK",))
block("T stage", "T stage")
block("N stage", "N stage")
# M stage: 仅 TCGA/CPTAC
R("M stage, n (%)", "", "", "", P["M stage"])
for cat in CATS["M stage"]:
    R(f"  {cat}", cell("M stage", "TCGA", cat), cell("M stage", "CPTAC", cat), "NA")
block("AJCC stage", "AJCC stage", na_for=("HANCOCK",))
# grade: G4 仅 TCGA 有; CPTAC/HANCOCK 分级体系止于 G3, G4 记 NA
R("Histologic grade, n (%)", "", "", "", P["Histologic grade"])
for cat in CATS["Histologic grade"]:
    cptac_g = "NA" if cat == "G4" else cell("Histologic grade", "CPTAC", cat)
    han_g   = "NA" if cat == "G4" else cell("Histologic grade", "HANCOCK", cat)
    R(f"  {cat}", cell("Histologic grade", "TCGA", cat), cptac_g, han_g)
block("Tumour site", "Tumour site")
# HPV: CPTAC by design + Unknown 行
R("HPV / p16 status", "", "", "", P["HPV / p16 status"])
hpv_unknown = {"TCGA": len(tc) - sum(COUNTS["HPV / p16 status"]["TCGA"].values()),
               "CPTAC": 0,
               "HANCOCK": len(hn) - sum(COUNTS["HPV / p16 status"]["HANCOCK"].values())}
R("  Positive", cell("HPV / p16 status", "TCGA", "Positive"), "0 (by design)",
  cell("HPV / p16 status", "HANCOCK", "Positive"))
R("  Negative", cell("HPV / p16 status", "TCGA", "Negative"), f"{N_CLIN['CPTAC']} (by design)",
  cell("HPV / p16 status", "HANCOCK", "Negative"))
R("  Unknown / not typed", hpv_unknown["TCGA"], hpv_unknown["CPTAC"], hpv_unknown["HANCOCK"])
block("Smoking status", "Smoking status")
R("Median follow-up, months", f"{SV['TCGA'][2]:.1f}", f"{SV['CPTAC'][2]:.1f}", f"{SV['HANCOCK'][2]:.1f}")
R("Deaths / N with follow-up", f"{SV['TCGA'][1]} / {SV['TCGA'][0]}", f"{SV['CPTAC'][1]} / {SV['CPTAC'][0]}",
  f"{SV['HANCOCK'][1]} / {SV['HANCOCK'][0]}")

# ------------------------------------------------------------------ 输出 markdown
out = ["# Table 1. Cohort characteristics", "",
       "| Characteristic | TCGA-HNSC | CPTAC-HNSCC | HANCOCK | P value |",
       "|---|---|---|---|---|"]
for label, a, b, c, p in rows:
    out.append(f"| {label} | {a} | {b} | {c} | {p} |")
md = "\n".join(out) + "\n"
open("figures/paper/cache/table1.md", "w").write(md)
print(md)
print("[saved figures/paper/cache/table1.md]")
