#!/usr/bin/env python3
"""Figure 5 panel b: 细胞型身份程序在 5 折之间的一致性 (TCGA-HNSC 内部)。

5 折的模型在不同训练子集上独立训练, 各自只对自己的留出病人预测, 因此折间一致性
是真正的稳健性证据 (不同于 CPTAC, 那里两个队列由同一套模型打分)。

每个细胞型取其在合并结果中 NES 最高的通路, 画出该通路在 5 折中各自的 NES。
"""
import os, json, numpy as np, pandas as pd, re
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 8})

T = json.load(open("figures/paper/cache/gsea_celltype_gobp.json"))
PF = json.load(open("figures/paper/cache/gsea_celltype_gobp_perfold.json"))
ORDER = ["Malignant", "Fibroblast", "Endothelial", "Macrophage", "Dendritic", "Mast", "T cell", "B cell"]
SHORT = lambda s: re.sub(r"\s*\(GO:\d+\)$", "", s)

# 每个细胞型的身份程序 = 合并结果里 NES 最高的通路
sig_pw = {}
for ct in ORDER:
    sig = {k: v for k, v in T[ct].items() if v["FDR"] < 0.05 and v["NES"] > 0}
    pool = sig if sig else T[ct]
    sig_pw[ct] = max(pool.items(), key=lambda kv: kv[1]["NES"])[0]

COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
fig.subplots_adjust(left=0.44, right=0.965, top=0.86, bottom=0.16)
y = np.arange(len(ORDER))[::-1]
for i, ct in enumerate(ORDER):
    pw = sig_pw[ct]
    v = [PF[str(k)][ct].get(pw, {}).get("NES", np.nan) for k in range(5)]
    v = np.array(v, dtype=float)
    ok = np.isfinite(v)
    if ok.sum():
        ax.hlines(y[i], np.nanmin(v), np.nanmax(v), color="0.8", lw=3.4, zorder=1)
    for k in range(5):
        if np.isfinite(v[k]):
            ax.scatter(v[k], y[i], s=26, color=COLS[k], zorder=3, edgecolors="none",
                       label=f"fold {k+1}" if i == 0 else None)
    ax.scatter(T[ct][pw]["NES"], y[i], s=46, facecolors="none", edgecolors="0.25",
               lw=1.0, zorder=4, label="all folds pooled" if i == 0 else None)
ax.axvline(0, color="0.6", lw=0.7, ls=":")
ax.set_yticks(y)
ax.set_yticklabels([f"{ct}\n{SHORT(sig_pw[ct])[:34]}" for ct in ORDER], fontsize=6.8)
ax.set_xlabel("GSEA normalized enrichment score of that cell type's top program", fontsize=8)
ax.legend(loc="lower right", fontsize=6.6, frameon=False, ncol=2, handletextpad=0.3)
ax.grid(axis="x", ls=":", alpha=0.45)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

# 折间一致性
sel = []
for ct in ORDER:
    sig = {k: v for k, v in T[ct].items() if v["FDR"] < 0.05 and v["NES"] > 0}
    pool = sig if len(sig) >= 3 else T[ct]
    for t, _ in sorted(pool.items(), key=lambda kv: -kv[1]["NES"])[:3]:
        if t not in sel: sel.append(t)
V = {k: np.array([[PF[str(k)][c].get(p, {}).get("NES", np.nan) for c in ORDER] for p in sel]).ravel()
     for k in range(5)}
rs = []
for i in range(5):
    for j in range(i + 1, 5):
        m = np.isfinite(V[i]) & np.isfinite(V[j])
        rs.append(spearmanr(V[i][m], V[j][m]).correlation)
ax.set_title("Each cell type's identity program is recovered in every fold\n"
             f"mean pairwise Spearman across folds {np.mean(rs):.2f} "
             f"(range {min(rs):.2f} to {max(rs):.2f}, {len(sel)} pathways x {len(ORDER)} cell types)",
             fontsize=8.6, pad=8)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/paper/Fig5_panelb_perfold.{ext}", dpi=300, bbox_inches="tight")
print(f"saved Fig5_panelb_perfold | 折间 Spearman 平均 {np.mean(rs):.3f}, 范围 {min(rs):.3f}-{max(rs):.3f}")
for ct in ORDER:
    v = [PF[str(k)][ct].get(sig_pw[ct], {}).get("NES", np.nan) for k in range(5)]
    print(f"  {ct:12s} {SHORT(sig_pw[ct])[:44]:46s} " + " ".join(f"{x:+.2f}" if np.isfinite(x) else "  na" for x in v))
