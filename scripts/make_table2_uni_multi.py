#!/usr/bin/env python3
"""Table 2. 每个队列的单因素与多因素 Cox, 所有变量并列。

采用临床研究的常规呈现方式: 左列为每个变量单独拟合的单因素结果, 右列为多因素结果。
每个细胞型各自拟合一个多因素模型 (临床协变量 + HPV + 该细胞型丰度), 三个细胞型不放进
同一个模型 —— 它们的预测丰度彼此相关 (HANCOCK 中 T 与 B 为 0.65, T 与成纤维为 -0.56),
同时纳入会互相抵消, 且事件数不足以支撑那么多变量。临床协变量的多因素列取自仅含
临床变量与 HPV 的基线模型。丰度按主分析选定的切点二分, 不重新搜索。
"""
import os, json, numpy as np, pandas as pd
from lifelines import CoxPHFitter
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
TR = "run/TCGA_HNSC_oro_v2b"
CTS = list(pd.read_csv(f"{TR}/cell_types_file.csv")["cell_types"])
PR = json.load(open("figures/paper/cache/prognosis_pred.json"))["cells"]
PASS = [c for c in ["T cell", "B cell", "Fibroblast"] if PR[c]["pass"]]
PCT = {c: PR[c]["pct"] for c in PASS}

sp = np.load(f"{TR}/splits/train_valid_test_idx.npz", allow_pickle=True)
meta = pd.read_csv(f"{TR}/metadata.csv"); cases = meta["case"].astype(str).tolist()
P, order = [], []
for k in range(5):
    P.append(np.loadtxt(f"{TR}/results/cell_fraction_preds/result_{k}_0/test_preds.txt"))
    order.extend(sp["test_idx"][k].tolist())
predT = pd.DataFrame(np.vstack(P), index=[cases[i] for i in order], columns=CTS).groupby(level=0).mean()
predH = pd.read_csv("run/HANCOCK/pred_abundance.csv", index_col=0)
predH.index = predH.index.astype(str).str.zfill(3)

def lsv(p, z=None):
    s = pd.read_csv(p, dtype={"case": str}).set_index("case")
    if z: s.index = s.index.str.zfill(z)
    return s

_stg = {'STAGE I':1,'STAGE II':2,'STAGE III':3,'STAGE IVA':4,'STAGE IVB':4,'STAGE IVC':4,'STAGE IV':4}
clT = pd.read_csv("run/TCGA_HNSC/clinical.csv", index_col=0); clT.index = clT.index.astype(str)
covT = pd.DataFrame({"Age (per 10 years)": pd.to_numeric(clT["AGE"], errors="coerce") / 10,
                     "Male sex": clT["SEX"].map({"Male":1,"Female":0}),
                     "AJCC stage (per level)": clT["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(_stg),
                     "HPV positive": clT["SUBTYPE"].map(lambda x: 1 if str(x).endswith("HPV+")
                                                        else (0 if str(x).endswith("HPV-") else np.nan))},
                    index=clT.index)
clH = pd.read_csv("run/HANCOCK/clinical.csv", dtype={"case":str}).set_index("case")
clH.index = clH.index.astype(str).str.zfill(3)
covH = pd.DataFrame({"Age (per 10 years)": pd.to_numeric(clH["age"], errors="coerce") / 10,
                     "Male sex": clH["sex"], "pT stage (per level)": clH["pT"],
                     "pN stage (per level)": clH["pN"], "HPV positive (p16)": clH["hpv_p16"]},
                    index=clH.index)

COH = [("TCGA-HNSC", predT, lsv(f"{TR}/survival.csv"), covT,
        ["Age (per 10 years)", "Male sex", "AJCC stage (per level)"], "HPV positive"),
       ("HANCOCK", predH, lsv("run/HANCOCK/survival.csv", 3), covH,
        ["Age (per 10 years)", "Male sex", "pT stage (per level)", "pN stage (per level)"], "HPV positive (p16)")]

def fmt(r):
    if r is None: return "—", "—"
    p = f"{r['p']:.3f}" if r["p"] >= 0.001 else "<0.001"
    if r["p"] < 0.05: p = f"**{p}**"
    return f"{r['hr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})", p

def fit(df, var):
    d = df.apply(pd.to_numeric, errors="coerce").dropna()
    keep = ["T","E"] + [c for c in d.columns if c not in ("T","E") and d[c].nunique() > 1]
    if len(d) < 40 or d["E"].sum() < 10 or var not in keep: return None, len(d), int(d["E"].sum())
    try:
        s = CoxPHFitter().fit(d[keep], "T", "E").summary.loc[var]
        return ({"hr": float(np.exp(s["coef"])), "lo": float(np.exp(s["coef lower 95%"])),
                 "hi": float(np.exp(s["coef upper 95%"])), "p": float(s["p"])}, len(d), int(d["E"].sum()))
    except Exception: return None, len(d), int(d["E"].sum())

L = ["# Table 2. Univariate and multivariable Cox regression for overall survival", ""]
for cname, pred, sv, cov, clin, hpvcol in COH:
    cc = [c for c in pred.index if c in sv.index]
    base = pd.DataFrame({"T": pd.to_numeric(sv.loc[cc,"time_days"], errors="coerce"),
                         "E": pd.to_numeric(sv.loc[cc,"event"], errors="coerce")}, index=cc).join(cov)
    ab = {}
    for ct in PASS:
        v = pred.loc[cc, ct]
        ab[f"{ct} abundance, high vs low"] = (v > v.quantile(PCT[ct])).astype(int)
    base = base.assign(**ab)
    clin_hpv = clin + [hpvcol]
    base_c  = base[["T","E"] + clin]                 # 基线模型: 仅临床
    base_ch = base[["T","E"] + clin_hpv]             # 基线模型: 临床 + HPV
    def nev(df):
        d = df.apply(pd.to_numeric, errors="coerce").dropna(); return len(d), int(d["E"].sum())
    n1, e1 = nev(base_c); n2, e2 = nev(base_ch)
    L += [f"**{cname}**", "",
          f"| Variable | Univariate HR (95% CI) | P | + clinical HR (95% CI) | P | + clinical + HPV HR (95% CI) | P |",
          "|---|---|---|---|---|---|---|"]
    for v in clin_hpv:
        u, _, _ = fit(base[["T","E",v]], v)
        m1, _, _ = fit(base_c, v) if v in clin else (None, 0, 0)
        m2, _, _ = fit(base_ch, v)
        uh, up = fmt(u); a1, p1 = fmt(m1); a2, p2 = fmt(m2)
        L.append(f"| {v} | {uh} | {up} | {a1} | {p1} | {a2} | {p2} |")
    for v in ab:
        u, _, _ = fit(base[["T","E",v]], v)
        m1, _, _ = fit(base[["T","E"] + clin + [v]], v)
        m2, _, _ = fit(base[["T","E"] + clin_hpv + [v]], v)
        uh, up = fmt(u); a1, p1 = fmt(m1); a2, p2 = fmt(m2)
        L.append(f"| **{v}** | {uh} | {up} | {a1} | {p1} | {a2} | {p2} |")
    L += ["", f"Patients entering each model: univariate, every patient with that variable; "
              f"+ clinical, n = {n1} with {e1} deaths; + clinical + HPV, n = {n2} with {e2} deaths. "
              f"Clinical variables are reported from the corresponding baseline model without any cell type; "
              f"each cell type was added to that baseline one at a time, so its estimate comes from its own model.", ""]
L += ["**Notes.** Cell type abundance was predicted from the H&E slide and dichotomised at the cutoff selected in the cross-cohort "
      f"replication analysis ({', '.join(f'{c} at the {round(PCT[c]*100)}th percentile' for c in PASS)}); the cutoffs were not re-optimised here. "
      "Age is modelled per 10 years and stage per category level. Hazard ratios below 1 indicate better survival. P values below 0.05 are in bold. "
      "Adding HPV status restricts the analysis to patients with a recorded HPV or p16 result, which in HANCOCK removes about half of the deaths, so the last pair of columns is the most stringent and the least powered of the three. "
      "The three cell types were not entered into a single model together: their predicted abundances are correlated (in HANCOCK, Spearman 0.65 "
      "between T cell and B cell and -0.56 between T cell and fibroblast), so a joint model splits one shared signal between competing terms. "
      "In HANCOCK, HPV status was recorded for only part of the cohort, so the multivariable models are fitted on that subgroup; tumour grade was "
      "not used because the source pathology file leaves it unrecorded for every p16-positive patient."]
open("HNSC_submission/HNSC_cox_table.md","w").write("\n".join(L) + "\n")
print("\n".join(L))
