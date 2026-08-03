#!/usr/bin/env python3
"""
Generic label formatter: BayesPrism outputs -> SLIDE-EX training labels, for ANY cancer.
Cell types are auto-detected from theta.csv columns (no hardcoding).
Keeps the 'breast_' filename prefix because the training code expects that literal token.

Usage: python gen_format_labels.py <bayesprism_dir> <run_proj_dir> [min_nonzero_frac] [min_var]
  e.g. python gen_format_labels.py run/TCGA_LUAD/deconv/bayesprism run/TCGA_LUAD 0.5 1e-5
  后两个=基因过滤阈值(可选): 非零样本比例下限、方差下限; 不传则用论文默认 0.5 / 1e-5
"""
import os, re, sys
import numpy as np, pandas as pd
BP, PROJ = sys.argv[1], sys.argv[2]
MIN_NONZERO_FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5   # 基因过滤: 非零样本≥该比例
MIN_VAR          = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5  # 基因过滤: 方差≥该值
meta = pd.read_csv(f"{PROJ}/metadata.csv")
order = meta["case"].tolist()
safe = lambda ct: re.sub(r"[^A-Za-z0-9]+","_",ct)

theta = pd.read_csv(f"{BP}/theta.csv", index_col=0).reindex(order)
CELL_TYPES = list(theta.columns)
theta.to_pickle(f"{PROJ}/breast_cell_frac_merged.pkl")
pd.DataFrame({"cell_types": CELL_TYPES}).to_csv(f"{PROJ}/cell_types_file.csv", index=False)
print(f"cell types ({len(CELL_TYPES)}):", CELL_TYPES)
print(f"abundances: {theta.shape} -> breast_cell_frac_merged.pkl")

for ct in CELL_TYPES:
    z = pd.read_csv(f"{BP}/Z_{safe(ct)}.csv", index_col=0).reindex(order)
    libsize = z.sum(axis=1).replace(0, np.nan)
    logexp = np.log1p(z.div(libsize, axis=0)*1e6).fillna(0.0)        # log(CPM+1)
    keep = ((z>0).mean(axis=0) >= MIN_NONZERO_FRAC) & (logexp.var(axis=0) >= MIN_VAR)  # 论文式基因过滤(阈值可配)
    logexp = logexp.loc[:, keep]
    logexp.to_pickle(f"{PROJ}/breast_{ct}_log_exp_merged.pkl")
    pd.DataFrame({"gene": logexp.columns}).to_csv(f"{PROJ}/breast_{ct}_gene_file.csv", index=False)
    print(f"{ct}: kept {int(keep.sum())}/{len(keep)} genes")
print("labels ->", PROJ)
