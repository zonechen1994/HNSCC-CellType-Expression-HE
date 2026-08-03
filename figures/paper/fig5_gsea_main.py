#!/usr/bin/env python3
"""Figure 5 (主图), 两个面板, 全部基于 TCGA-HNSC。

(a) 细胞型 x GO 通路的 NES 热图。基因按细胞型特异性打分排序 (该细胞型预测表达
    减 其余细胞型均值), gseapy prerank 对 GO biological process 富集。
    行序按"该通路在哪个细胞型最富集"分块, 形成对角结构。
(b) 每个细胞型的身份程序在 5 折中各自的 NES。5 折的模型在不同训练子集上独立
    训练, 各自只预测自己的留出病人, 因此折间一致性是真正的稳健性证据。

CPTAC 不出现在本图中: 两个队列由同一套 TCGA 训练的模型打分, 且打分前跨病人取了
均值, 两幅热图必然近乎相同 (186 个格子中 97% 差值 < 0.3, 而色阶跨度为 6),
不构成独立验证。
"""
import os, json, re, numpy as np, pandas as pd
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 8,
    "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8})

T = json.load(open("figures/paper/cache/gsea_celltype_gobp.json"))
PF = json.load(open("figures/paper/cache/gsea_celltype_gobp_perfold.json"))
ORDER = [c for c in ["Malignant", "Fibroblast", "Endothelial", "Macrophage",
                     "Dendritic", "Mast", "T cell", "B cell"] if c in T]
SHORT = lambda s: re.sub(r"\s*\(GO:\d+\)$", "", s)

sel = []
for ct in ORDER:
    sig = {k: v for k, v in T[ct].items() if v["FDR"] < 0.05 and v["NES"] > 0}
    pool = sig if len(sig) >= 3 else T[ct]
    for t, _ in sorted(pool.items(), key=lambda kv: -kv[1]["NES"])[:3]:
        if t not in sel: sel.append(t)

M = pd.DataFrame(np.nan, index=sel, columns=ORDER)
F = pd.DataFrame(1.0, index=sel, columns=ORDER)
for ct in ORDER:
    for pw in sel:
        M.loc[pw, ct] = T[ct].get(pw, {}).get("NES", np.nan)
        F.loc[pw, ct] = T[ct].get(pw, {}).get("FDR", 1.0)
key = M.idxmax(axis=1).map({c: i for i, c in enumerate(ORDER)})
sel = list(M.assign(_k=key, _n=M.max(axis=1)).sort_values(["_k", "_n"], ascending=[True, False]).index)
M, F = M.loc[sel], F.loc[sel]

fig = plt.figure(figsize=(15.2, 8.6))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.76], wspace=1.16,
              left=0.245, right=0.985, top=0.895, bottom=0.155)

# ---------------- (a) 热图 ----------------
axa = fig.add_subplot(gs[0, 0])
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#e8e8e8")
im = axa.imshow(np.ma.masked_invalid(M.values), cmap=cmap, vmin=-3, vmax=3, aspect="auto")
axa.set_xticks(range(len(ORDER)))
axa.set_xticklabels(ORDER, rotation=35, ha="right", fontsize=8.5)
axa.set_yticks(range(len(sel)))
axa.set_yticklabels([SHORT(s) for s in sel], fontsize=7.2)
for i in range(len(sel)):
    for j in range(len(ORDER)):
        if F.values[i, j] < 0.05:
            axa.text(j, i, "•", ha="center", va="center", fontsize=8.5,
                     color="white" if abs(M.values[i, j]) > 1.8 else "#333")
axa.set_xticks(np.arange(-.5, len(ORDER), 1), minor=True)
axa.set_yticks(np.arange(-.5, len(sel), 1), minor=True)
axa.grid(which="minor", color="white", lw=0.7); axa.tick_params(which="minor", length=0)
axa.set_title("a   Cell type-resolved pathway enrichment", fontsize=10, pad=9, loc="left")

pa = axa.get_position()
cax = fig.add_axes([pa.x1 + 0.006, pa.y0 + 0.30 * pa.height, 0.010, 0.34 * pa.height])
cb = fig.colorbar(im, cax=cax)
cb.set_label("GSEA NES\n(+ enriched, - depleted)", fontsize=7.2)
cb.ax.tick_params(labelsize=7)

# ---------------- (b) 逐折一致性 ----------------
axb = fig.add_subplot(gs[0, 1])
sig_pw = {}
for ct in ORDER:
    sig = {k: v for k, v in T[ct].items() if v["FDR"] < 0.05 and v["NES"] > 0}
    pool = sig if sig else T[ct]
    sig_pw[ct] = max(pool.items(), key=lambda kv: kv[1]["NES"])[0]
COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
y = np.arange(len(ORDER))[::-1]
for i, ct in enumerate(ORDER):
    v = np.array([PF[str(k)][ct].get(sig_pw[ct], {}).get("NES", np.nan) for k in range(5)], float)
    if np.isfinite(v).any():
        axb.hlines(y[i], np.nanmin(v), np.nanmax(v), color="0.82", lw=4.0, zorder=1)
    for k in range(5):
        if np.isfinite(v[k]):
            axb.scatter(v[k], y[i], s=30, color=COLS[k], zorder=3, edgecolors="none",
                        label=f"fold {k+1}" if i == 0 else None)
    axb.scatter(T[ct][sig_pw[ct]]["NES"], y[i], s=64, facecolors="none", edgecolors="0.2",
                lw=1.1, zorder=4, label="all folds pooled" if i == 0 else None)
axb.set_yticks(y)
def _wrap(t, w=30):
    t = SHORT(t); out, line = [], ""
    for word in t.split():
        if len(line) + len(word) + 1 <= w: line = (line + " " + word).strip()
        else: out.append(line); line = word
    out.append(line); return "\n".join(out[:2]) + ("..." if len(out) > 2 else "")
axb.set_yticklabels([f"{ct}\n{_wrap(sig_pw[ct])}" for ct in ORDER], fontsize=6.8)
axb.set_xlabel("GSEA normalized enrichment score of each cell type's top program", fontsize=8.5)
axb.set_xlim(1.4, 4.35)
# 图例放在左下角空白区 (T cell 与 B cell 两行的点都在 x>3.1, 该区域无数据)
axb.legend(loc="lower left", bbox_to_anchor=(0.015, 0.01), fontsize=6.6, frameon=False,
           ncol=2, handletextpad=0.3, columnspacing=0.9, labelspacing=0.35)
axb.grid(axis="x", ls=":", alpha=0.45)
for s in ["top", "right"]: axb.spines[s].set_visible(False)

V = {k: np.array([[PF[str(k)][c].get(p, {}).get("NES", np.nan) for c in ORDER] for p in sel]).ravel()
     for k in range(5)}
rs = []
for i in range(5):
    for j in range(i + 1, 5):
        m = np.isfinite(V[i]) & np.isfinite(V[j])
        rs.append(spearmanr(V[i][m], V[j][m]).correlation)
axb.set_title("b   The same programs are recovered in every cross-validation fold",
              fontsize=10, pad=9, loc="left")

fig.suptitle("The H&E-based model resolves pathway activity one cell type at a time",
             fontsize=12.5, y=0.965)
fig.text(0.5, 0.058,
         "TCGA-HNSC, n = 442. Genes were ranked within each cell type by a specificity score, that cell type's predicted expression minus the mean of the other cell types, and tested by\n"
         "gseapy prerank against the GO biological-process collection. In (a) dots mark a false discovery rate below 0.05 and grey cells were not testable in that cell type; rows are ordered so that\n"
         f"each pathway sits with the cell type in which it is most enriched. In (b) each fold's model was trained on a different subset and scored only its own held-out patients; the mean pairwise\n"
         f"Spearman correlation between folds was {np.mean(rs):.2f} (range {min(rs):.2f} to {max(rs):.2f}) across the {len(sel)} pathways and {len(ORDER)} cell types shown in (a).",
         ha="center", va="top", fontsize=7.4, color="0.32", linespacing=1.55)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/paper/Fig5_gsea_main.{ext}", dpi=300, bbox_inches="tight")
print(f"saved Fig5_gsea_main | {len(sel)} pathways x {len(ORDER)} cell types | 折间 Spearman {np.mean(rs):.3f} ({min(rs):.3f}-{max(rs):.3f})")
