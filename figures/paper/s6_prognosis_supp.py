#!/usr/bin/env python3
"""S6: 预后的补充分析 —— 调整 HPV 后的嵌套模型, 以及疾病特异生存。

Figure 4 已经给出总生存的 KM 曲线与调整临床变量的多因素 Cox, 因此 S6 不再重复,
改为承担两件 Figure 4 未覆盖的事:
  (a) 嵌套模型 单因素 -> +临床 -> +临床+HPV, 两个队列的总生存
  (b) HANCOCK 的疾病特异生存, 同样的三层模型
数据来自 scripts/prognosis_supplementary.py。
"""
import os, json, numpy as np
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 8, "axes.linewidth": 0.8})
PAL = {"prot": "#2f7ab8", "risk": "#d1603d", "grey": "#8a8a8a"}
D = json.load(open("figures/paper/cache/prognosis_supplementary.json"))
M, PCT = D["models"], D["cutoffs"]
CELLS = ["T cell", "B cell", "Fibroblast"]
MODELS = ["univariate", "+ clinical", "+ clinical + HPV"]

def forest(ax, keys, title):
    rows = []
    for ct in CELLS:
        for key in keys:
            for mn in MODELS:
                r = M[ct].get(key, {}).get(mn)
                rows.append((ct, key, mn, r))
    rows = rows[::-1]
    y = np.arange(len(rows))
    for i, (ct, key, mn, r) in enumerate(rows):
        if r is None: continue
        c = PAL["prot"] if r["hr"] < 1 else PAL["risk"]
        sig = r["p"] < 0.05
        ax.plot([r["lo"], r["hi"]], [y[i], y[i]], color=c, lw=1.5, alpha=0.9, solid_capstyle="round")
        ax.scatter(r["hr"], y[i], s=44 if sig else 30, color=c if sig else "white",
                   edgecolors=c, lw=1.2, zorder=3)
        ax.text(r["hi"] * 1.06, y[i], f"{r['hr']:.2f}" + ("*" if sig else ""),
                va="center", fontsize=6.2, color=c)
    ax.axvline(1, color="0.35", ls="--", lw=0.9)
    ax.set_xscale("log"); ax.set_yticks(y)
    labs = []
    multi = len(keys) > 1
    for ct, key, mn, r in rows:
        n = f"  n={r['n']}, {r['events']} ev" if r else ""
        pre = (key.split()[0] + " · ") if multi else ""
        labs.append(f"{pre}{mn}{n}")
    ax.set_yticklabels(labs, fontsize=5.9)
    # 左侧细胞型/队列分组标注
    step = len(MODELS) * len(keys)
    for j, ct in enumerate(CELLS):
        lo = len(rows) - (j + 1) * step; hi = len(rows) - j * step - 1
        ax.text(-0.60, (lo + hi) / 2, f"{ct}\ncut {round(PCT[ct]*100)}%", transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=6.8, fontweight="bold")
        if j: ax.axhline(hi + 0.5, color="0.85", lw=0.7)
    if len(keys) > 1:
        for j in range(len(CELLS)):
            for k in range(1, len(keys)):
                ax.axhline(len(rows) - j * step - k * len(MODELS) - 0.5, color="0.93", lw=0.6)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(title, fontsize=8.6, pad=7)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", ls=":", alpha=0.4)

fig = plt.figure(figsize=(11.4, 7.6))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.78, left=0.235, right=0.965, top=0.86, bottom=0.145)

axa = fig.add_subplot(gs[0, 0])
forest(axa, ["TCGA OS", "HANCOCK OS"], "a   Overall survival, nested models")
axa.set_xlabel("Hazard ratio with 95% CI (log scale)\nfilled marker: P < 0.05", fontsize=7.4)

axb = fig.add_subplot(gs[0, 1])
forest(axb, ["HANCOCK DSS"], "b   Disease-specific survival (HANCOCK)")
axb.set_xlabel("Hazard ratio with 95% CI (log scale)\nfilled marker: P < 0.05", fontsize=7.4)

fig.suptitle("Supplementary: the predicted-abundance prognosis adjusted for HPV status, and disease-specific survival",
             fontsize=10, y=0.955)
fig.text(0.5, 0.055,
         "Cell types and cutoffs are those selected in the main analysis and are not re-optimised here. Each cell type is shown as three nested Cox models: the dichotomised abundance alone, "
         "with the clinical covariates, and with HPV status added.\nAdding HPV restricts the analysis to patients with a known HPV or p16 status, which reduces the sample size "
         "(in HANCOCK from 550 to 284 patients and from 161 to 71 events), so the intervals widen even where the hazard ratio is little changed.",
         ha="center", va="top", fontsize=6.8, color="0.35", linespacing=1.5)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/paper/S6_prognosis_supp.{ext}", dpi=300, bbox_inches="tight")
print("saved S6_prognosis_supp")
