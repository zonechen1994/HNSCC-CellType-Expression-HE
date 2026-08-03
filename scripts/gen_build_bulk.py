#!/usr/bin/env python3
"""
Generic bulk count-matrix builder. Same as 10_build_bulk_matrix.py but parameterized.
Usage: python gen_build_bulk.py <rnaseq_dir> <run_dir>
  e.g. python gen_build_bulk.py data/TCGA-LUAD/rnaseq run/TCGA_LUAD
Output: <run_dir>/deconv/tcga_bulk_counts.tsv.gz  (gene symbol x case, unstranded counts)
"""
import os, sys, numpy as np, pandas as pd
RNADIR, RUN = sys.argv[1], sys.argv[2]
OUT = f"{RUN}/deconv"; os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(f"{RUN}/metadata.csv")
cases, files = meta["case"].tolist(), meta["rnaseq_file"].tolist()
SKIP = {"N_unmapped","N_multimapping","N_noFeature","N_ambiguous"}
def read_one(p):
    syms, cnt = [], []
    for line in open(p):
        if line.startswith("#"): continue
        x = line.rstrip("\n").split("\t")
        if x[0]=="gene_id" or x[0] in SKIP or len(x)<4: continue
        syms.append(x[1]); cnt.append(int(x[3]))
    return syms, np.array(cnt, dtype=np.int32)
ref_syms, c0 = read_one(f"{RNADIR}/{files[0]}")
mat = np.zeros((len(ref_syms), len(files)), dtype=np.int32); mat[:,0]=c0
for j in range(1,len(files)):
    s,c = read_one(f"{RNADIR}/{files[j]}")
    if j<5 and s!=ref_syms: raise SystemExit("gene order mismatch")
    mat[:,j]=c
    if j%200==0: print(f"  {j}/{len(files)}")
df = pd.DataFrame(mat, index=ref_syms, columns=cases).groupby(level=0).sum()
df.to_csv(f"{OUT}/tcga_bulk_counts.tsv.gz", sep="\t", compression="gzip")
print(f"bulk: {df.shape[0]} genes x {df.shape[1]} samples -> {OUT}/tcga_bulk_counts.tsv.gz")
