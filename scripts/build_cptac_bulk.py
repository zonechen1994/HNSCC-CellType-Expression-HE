#!/usr/bin/env python3
"""从下载的 CPTAC STAR gene counts 拼 bulk 计数矩阵 (gene symbol x case, unstranded)."""
import os, csv, gzip, sys
import numpy as np, pandas as pd
ROOT=os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61")
RNADIR=f"{ROOT}/data/CPTAC_HNSC_rnaseq/files"
MAN=f"{ROOT}/data/CPTAC_HNSC_rnaseq/manifest.tsv"
OUT=f"{ROOT}/run/CPTAC_HNSC/deconv"; os.makedirs(OUT,exist_ok=True)
man=pd.read_csv(MAN,sep="\t")
def read_counts(fp):
    # STAR augmented tsv: header + 4 stat rows(N_*), cols: gene_id,gene_name,gene_type,unstranded,...
    genes=[]; vals=[]
    with open(fp) as f:
        r=csv.reader(f,delimiter="\t")
        header=next(r)
        while header and header[0].startswith("#"):   # 跳过 "# gene-model" 注释行
            header=next(r)
        gi=header.index("gene_name"); ci=header.index("unstranded")
        for row in r:
            if not row or row[0].startswith("N_"): continue
            genes.append(row[gi]); vals.append(int(row[ci]))
    return genes, np.array(vals)
cols={}; ref_genes=None; ok=0; miss=[]
for _,rw in man.iterrows():
    case=rw["case"]; fp=os.path.join(RNADIR, f"{case}.tsv")
    if not os.path.exists(fp) or os.path.getsize(fp)==0: miss.append(case); continue
    g,v=read_counts(fp)
    if ref_genes is None: ref_genes=g
    if g!=ref_genes:  # 不同顺序则按symbol对齐
        s=pd.Series(v,index=g).groupby(level=0).sum()
        v=s.reindex(pd.Index(ref_genes)).fillna(0).values
    cols[case]=v; ok+=1
print(f"读入 {ok} 个样本, 缺 {len(miss)}: {miss[:5]}")
df=pd.DataFrame(cols,index=ref_genes)
# 同symbol合并(gene_name有重复)
df=df.groupby(level=0).sum()
df.index.name=None
df.to_csv(f"{OUT}/cptac_bulk_counts.tsv.gz",sep="\t",compression="gzip")
open(f"{OUT}/bulk_samples.txt","w").write("\n".join(df.columns))
print(f"bulk 矩阵: {df.shape[0]} genes x {df.shape[1]} samples -> {OUT}/cptac_bulk_counts.tsv.gz")
