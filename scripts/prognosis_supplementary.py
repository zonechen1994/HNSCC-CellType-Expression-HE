#!/usr/bin/env python3
"""S6 的补充内容: 疾病特异生存与调整 HPV 后的模型。

Figure 4 已经给出总生存的 Kaplan-Meier 与调整临床变量的多因素 Cox。本脚本计算
Figure 4 未覆盖的两项, 用于替换与 Figure 4 重复的旧 S6:

  1. 嵌套模型序列  单因素 -> +临床 -> +临床+HPV
     HPV 是头颈癌最强的预后因素, 需要说明丰度的预后价值不是 HPV 的替身
  2. HANCOCK 的疾病特异生存 (DSS), 作为总生存之外的第二个终点

细胞型与切点沿用主分析的结果, 不重新搜索。
"""
import os, json, numpy as np, pandas as pd
from lifelines import CoxPHFitter
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
TR = "run/TCGA_HNSC_oro_v2b"
CTS = list(pd.read_csv(f"{TR}/cell_types_file.csv")["cell_types"])
PR = json.load(open("figures/paper/cache/prognosis_pred.json"))
PASS = [ct for ct in CTS if PR["cells"][ct]["pass"]]
PCT = {ct: PR["cells"][ct]["pct"] for ct in PASS}
print("参与细胞型与切点:", {k: f"{int(v*100)}%" for k, v in PCT.items()})

# ---- 预测丰度 ----
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
covT = pd.DataFrame({"age": pd.to_numeric(clT["AGE"], errors="coerce"),
                     "sex": clT["SEX"].map({"Male":1,"Female":0}),
                     "stage": clT["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(_stg),
                     "hpv": clT["SUBTYPE"].map(lambda x: 1 if str(x).endswith("HPV+")
                                               else (0 if str(x).endswith("HPV-") else np.nan))}, index=clT.index)
clH = pd.read_csv("run/HANCOCK/clinical.csv", dtype={"case":str}).set_index("case")
clH.index = clH.index.astype(str).str.zfill(3)
covH = pd.DataFrame({"age": pd.to_numeric(clH["age"], errors="coerce"), "sex": clH["sex"],
                     "pT": clH["pT"], "pN": clH["pN"],   # grade 在源数据中对 p16 阳性者缺失, 故不作协变量
                     "hpv": clH["hpv_p16"]}, index=clH.index)

COH = [("TCGA", predT, lsv(f"{TR}/survival.csv"), covT, ["age","sex","stage"], "OS"),
       ("HANCOCK", predH, lsv("run/HANCOCK/survival.csv", 3), covH, ["age","sex","pT","pN"], "OS"),
       ("HANCOCK", predH, lsv("run/HANCOCK/survival_dss.csv", 3), covH, ["age","sex","pT","pN"], "DSS")]

def fit(pred, sv, cov, ct, pct, covars):
    cc = [c for c in pred.index if c in sv.index]
    d = pd.DataFrame({"T": pd.to_numeric(sv.loc[cc,"time_days"], errors="coerce"),
                      "E": pd.to_numeric(sv.loc[cc,"event"], errors="coerce"),
                      "val": pred.loc[cc, ct].values}, index=cc).join(cov).dropna(subset=["T","E","val"])
    d["z"] = (d["val"] > d["val"].quantile(pct)).astype(int)
    out = {}
    for name, cols in [("univariate", []), ("+ clinical", covars), ("+ clinical + HPV", covars + ["hpv"])]:
        dd = d[["T","E","z"] + cols].apply(pd.to_numeric, errors="coerce").dropna()
        keep = ["T","E"] + [c for c in ["z"] + cols if c in dd.columns and dd[c].nunique() > 1]
        if len(dd) < 40 or dd["E"].sum() < 10 or "z" not in keep: out[name] = None; continue
        try:
            s = CoxPHFitter().fit(dd[keep], "T", "E").summary.loc["z"]
            out[name] = {"hr": float(np.exp(s["coef"])), "lo": float(np.exp(s["coef lower 95%"])),
                         "hi": float(np.exp(s["coef upper 95%"])), "p": float(s["p"]),
                         "n": int(len(dd)), "events": int(dd["E"].sum())}
        except Exception: out[name] = None
    return out

RES = {}
print(f"\n{'细胞型':11s} {'队列/终点':16s} {'模型':18s} {'HR (95% CI)':>22s} {'P':>9s} {'n':>5s} {'事件':>5s}")
for ct in PASS:
    RES[ct] = {}
    for cohort, pred, sv, cov, covars, endp in COH:
        key = f"{cohort} {endp}"
        RES[ct][key] = fit(pred, sv, cov, ct, PCT[ct], covars)
        for mname, r in RES[ct][key].items():
            if r is None:
                print(f"{ct:11s} {key:16s} {mname:18s} {'拟合失败':>22s}"); continue
            star = "*" if r["p"] < 0.05 else " "
            print(f"{ct:11s} {key:16s} {mname:18s} "
                  f"{r['hr']:6.2f} ({r['lo']:.2f}-{r['hi']:.2f}){'':>4s} {r['p']:8.4f}{star} {r['n']:5d} {r['events']:5d}")
    print()
json.dump({"cutoffs": PCT, "models": RES}, open("figures/paper/cache/prognosis_supplementary.json", "w"))
print("已存 figures/paper/cache/prognosis_supplementary.json")
