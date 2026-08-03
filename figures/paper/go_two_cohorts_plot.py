#!/usr/bin/env python3
"""细胞型 GO(BP) 富集聚类热图, TCGA 与 CPTAC 各一张。

画法对齐 SLIDE-EX (Wang et al., npj Precis Oncol 2026, doi:10.1038/s41698-026-01419-9) 的 Fig. 5:
行列均做层次聚类并画聚类树, 通路名在右侧, 细胞型在底部竖排,
色条标注 Scaled Odds Ratio (0-1, 按细胞型即列归一), 蓝-白-红配色。
数据来自 scripts/go_enrichment_two_cohorts.py。
"""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
C, OUT = "figures/paper/cache/go_two_cohorts", "figures/paper"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "axes.linewidth": 0.6})

def load(coh):
    M = pd.read_csv(f"{C}/go_{coh}_oddsratio.csv", index_col=0)
    M.index = [re.sub(r"\s*\(GO:\d+\)$", "", p) for p in M.index]
    return M.div(M.max(axis=0).replace(0, np.nan), axis=1).fillna(0.0)

def order(M, axis):
    D = M.values if axis == 0 else M.values.T
    if D.shape[0] < 3: return list(range(D.shape[0])), None
    Z = linkage(pdist(D, "euclidean"), method="average")
    return dendrogram(Z, no_plot=True)["leaves"], Z

def panel(fig, sub, M, title, letter=None):
    """在 sub (SubplotSpec) 内画 行树 + 列树 + 热图, 复刻 clustermap 版式"""
    ri, rZ = order(M, 0); ci, cZ = order(M, 1)
    M = M.iloc[ri, ci]
    inner = GridSpec(2, 2, figure=fig, height_ratios=[0.13, 1], width_ratios=[0.11, 1],
                     hspace=0.02, wspace=0.02,
                     left=sub[0], right=sub[1], bottom=sub[2], top=sub[3])
    axcol = fig.add_subplot(inner[0, 1]); axrow = fig.add_subplot(inner[1, 0])
    axh = fig.add_subplot(inner[1, 1])
    for ax, Z, orient in ((axcol, cZ, "top"), (axrow, rZ, "left")):
        if Z is not None:
            dendrogram(Z, ax=ax, orientation=orient, color_threshold=0,
                       above_threshold_color="0.45", no_labels=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    if rZ is not None: axrow.invert_yaxis()
    im = axh.imshow(M.values, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    axh.set_yticks(range(M.shape[0])); axh.set_yticklabels(M.index, fontsize=6.8)
    axh.yaxis.set_ticks_position("right"); axh.yaxis.set_label_position("right")
    axh.set_xticks(range(M.shape[1]))
    axh.set_xticklabels(M.columns, fontsize=8, rotation=90)
    axh.tick_params(length=2, pad=1.5)
    axcol.set_title(title, fontsize=9.5, pad=4)
    if letter is not None:
        fig.text(sub[0]-0.03, sub[3]+0.03, letter, fontsize=16, fontweight="bold", va="bottom", ha="left")
    return im

T, P = load("TCGA"), load("CPTAC")
fig = plt.figure(figsize=(14.0, 7.4))
im = panel(fig, (0.055, 0.325, 0.10, 0.86), T, "TCGA-HNSC, cross-validation", "a")
panel(fig, (0.625, 0.885, 0.10, 0.86), P, "CPTAC-HNSCC, external validation", "b")
cax = fig.add_axes([0.012, 0.72, 0.011, 0.14])
cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
cb.set_label("Scaled Odds Ratio", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
cax.yaxis.set_ticks_position("left"); cax.yaxis.set_label_position("left")
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/Fig_GO_two_cohorts.{ext}", dpi=300, bbox_inches="tight")
print(f"saved Fig_GO_two_cohorts | TCGA {T.shape} CPTAC {P.shape}")

# ---- 基因集重叠单独成图, 不混进上面的聚类热图 ----
# Optional panel: needs the per-patient expression predictions (run/.../results, ~4.6 GB),
# which are not bundled. Skip gracefully if absent; Figure S6 itself is already saved above.
import sys as _sys
from scipy.stats import pearsonr
TR, RES = "run/TCGA_HNSC_oro_v2b", "run/TCGA_HNSC_oro_v2b/results"
if not os.path.exists(f"{RES}/Malignant_preds/result_0_0/test_preds.txt"):
    print("skipped gene-set overlap panel: per-patient predictions not bundled (see DATA.md)")
    _sys.exit(0)
CTS = list(pd.read_csv(f"{TR}/cell_types_file.csv")["cell_types"])
S = {}
for ct in CTS:
    g = np.array(pd.read_csv(f"{TR}/breast_{ct}_gene_file.csv")["gene"])
    Pp = np.vstack([np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/test_preds.txt") for k in range(5)])
    Ll = np.vstack([np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/test_labels.txt") for k in range(5)])
    r = np.array([pearsonr(Pp[:, j], Ll[:, j])[0] if Pp[:, j].std() > 1e-9 and Ll[:, j].std() > 1e-9 else 0
                  for j in range(len(g))])
    S[ct] = set(g[r > 0.4])
keep = [c for c in CTS if len(S[c]) >= 100]
J = np.array([[len(S[a] & S[b]) / max(1, len(S[a] | S[b])) for b in keep] for a in keep])
f2, ax = plt.subplots(figsize=(4.3, 3.9))
ax.imshow(J, cmap="Purples", vmin=0, vmax=1)
ax.set_xticks(range(len(keep))); ax.set_xticklabels(keep, rotation=90, fontsize=8)
ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep, fontsize=8)
for i in range(len(keep)):
    for j in range(len(keep)):
        ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if J[i, j] > 0.55 else "0.2")
ax.set_title("Overlap of well-predicted gene sets\nbetween cell types (Jaccard index)", fontsize=9, pad=6)
for ext in ("pdf", "png"):
    f2.savefig(f"{OUT}/Fig_GO_geneset_overlap.{ext}", dpi=300, bbox_inches="tight")
print("saved Fig_GO_geneset_overlap")
