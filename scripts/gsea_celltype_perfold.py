#!/usr/bin/env python3
"""细胞型通路 GSEA 的逐折一致性 (TCGA-HNSC 内部)。

Figure 5 的 panel b。CPTAC 的一致性不构成证据, 因为两个队列由同一套模型打分;
而 5 折交叉验证的 5 个模型是在不同的训练子集上独立训练的, 各自只对自己的
留出病人做预测。因此折与折之间的一致性才真正反映结果不依赖于某一次划分。

对每一折: 用该折的模型对该折的留出病人预测 -> 计算细胞型特异性排序 ->
prerank GSEA (GO biological process) -> 得到该折的 NES。
"""
import os, json, numpy as np, pandas as pd
import gseapy
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
RUN, RES = "run/TCGA_HNSC_oro_v2b", "run/TCGA_HNSC_oro_v2b/results"
CTS = list(pd.read_csv(f"{RUN}/cell_types_file.csv")["cell_types"])
sp = np.load(f"{RUN}/splits/train_valid_test_idx.npz", allow_pickle=True)
meta = pd.read_csv(f"{RUN}/metadata.csv"); cases = meta["case"].astype(str).tolist()
LIB = "GO_Biological_Process_2023"

def fold_profile(ct, k):
    """第 k 折的模型对其留出病人的预测, 取跨病人均值"""
    g = list(pd.read_csv(f"{RUN}/breast_{ct}_gene_file.csv")["gene"])
    P = np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/test_preds.txt")
    idx = [cases[i] for i in sp["test_idx"][k].tolist()]
    return pd.DataFrame(P, index=idx, columns=g).groupby(level=0).mean().mean(axis=0)

OUT = {}
for k in range(5):
    prof = {c: fold_profile(c, k) for c in CTS}
    allg = pd.DataFrame(prof).dropna(how="all")
    print(f"\n===== fold {k}  ({allg.shape[0]} 基因) =====", flush=True)
    OUT[k] = {}
    for ct in CTS:
        others = [c for c in CTS if c != ct]
        spec = (allg[ct] - allg[others].mean(axis=1)).dropna().sort_values(ascending=False)
        pre = gseapy.prerank(rnk=spec.reset_index(), gene_sets=LIB, min_size=10, max_size=500,
                             permutation_num=1000, seed=42, no_plot=True, outdir=None, threads=8)
        r = pre.res2d.copy()
        r["NES"] = pd.to_numeric(r["NES"], errors="coerce")
        r["FDR"] = pd.to_numeric(r["FDR q-val"], errors="coerce")
        r = r.dropna(subset=["NES"])
        OUT[k][ct] = {x["Term"]: {"NES": float(x["NES"]), "FDR": float(x["FDR"])} for _, x in r.iterrows()}
        top = r.sort_values("NES", ascending=False).iloc[0]
        print(f"  {ct:12s} top: {top['Term'][:50]:52s} NES{top['NES']:+.2f}", flush=True)

json.dump({str(k): v for k, v in OUT.items()},
          open("figures/paper/cache/gsea_celltype_gobp_perfold.json", "w"))
print("\n已存 figures/paper/cache/gsea_celltype_gobp_perfold.json")

# 一致性: 用主图那批通路, 计算折间两两 Spearman
T = json.load(open("figures/paper/cache/gsea_celltype_gobp.json"))
sel = []
for ct in CTS:
    sig = {a: b for a, b in T[ct].items() if b["FDR"] < 0.05 and b["NES"] > 0}
    pool = sig if len(sig) >= 3 else T[ct]
    for t, _ in sorted(pool.items(), key=lambda kv: -kv[1]["NES"])[:3]:
        if t not in sel: sel.append(t)
from scipy.stats import spearmanr
V = {k: np.array([[OUT[k][c].get(p, {}).get("NES", np.nan) for c in CTS] for p in sel]).ravel()
     for k in range(5)}
rs = []
print("\n折间两两 Spearman (主图那 24 条通路 x 8 细胞型):")
for i in range(5):
    for j in range(i + 1, 5):
        m = np.isfinite(V[i]) & np.isfinite(V[j])
        r = spearmanr(V[i][m], V[j][m]).correlation
        rs.append(r); print(f"  fold {i} vs fold {j}: {r:.3f}")
print(f"\n平均 {np.mean(rs):.3f}, 范围 {min(rs):.3f} - {max(rs):.3f}")
