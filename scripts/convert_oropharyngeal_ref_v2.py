import scanpy as sc, numpy as np, scipy.io, scipy.sparse as sp, pandas as pd
np.random.seed(0)
print("载入 h5ad...",flush=True)
a=sc.read_h5ad("data/refs/Oropharyngeal_HPVmix/oropharyngeal.h5ad")
a=a[a.obs["Sample.type"].isin(["HPV+ HNSCC","HPV- HNSCC"])].copy()
# 映射(与基线9类一致); NK 和 mural 一律丢弃(不合并)
M={'epithelial cell':'Malignant','fibroblast':'Fibroblast',
   'CD4-positive, alpha-beta T cell':'T cell','CD8-positive, alpha-beta T cell':'T cell',
   'regulatory T cell':'T cell','mature alpha-beta T cell':'T cell',
   'endothelial cell':'Endothelial','endothelial cell of lymphatic vessel':'Endothelial',
   'B cell':'B cell','plasma cell':'B cell','mast cell':'Mast',
   'myeloid cell':'Macrophage','plasmacytoid dendritic cell':'Dendritic'}
a.obs["ct9"]=a.obs["cell_type"].map(M); a=a[a.obs["ct9"].notna()].copy()   # NK/mural 丢
print("映射后自然分布:",dict(a.obs["ct9"].value_counts()),flush=True)
# 按比例抽样: 每类同一比例, 总数~20000, 保留自然比例(不拉平)
frac=min(1.0, 20000/a.n_obs)
idx=[]
for ct,g in a.obs.groupby("ct9",observed=True):
    ii=g.index.tolist(); k=max(30,int(round(len(ii)*frac)))
    idx+= list(np.random.choice(ii,min(k,len(ii)),replace=False))
a=a[idx].copy()
print(f"比例抽样(frac={frac:.2f})后:",a.n_obs,dict(a.obs["ct9"].value_counts()),flush=True)
Xc=a.raw[a.obs_names].X; Xc=Xc.tocsr() if sp.issparse(Xc) else sp.csr_matrix(Xc)
sym=a.raw.var["feature_name"].astype(str).values
good=np.array([bool(s) and not s.startswith("ENSG") for s in sym]); Xc=Xc[:,good]; sym=sym[good]
uniq,inv=np.unique(sym,return_inverse=True)
A=sp.csr_matrix((np.ones(len(sym)),(inv,np.arange(len(sym)))),shape=(len(uniq),len(sym)))
G=A @ Xc.T.tocsr()
print("genes×cells:",G.shape,flush=True)
D="data/refs/Oropharyngeal_HPVmix"
scipy.io.mmwrite(f"{D}/ref_counts_v2.mtx", G)
open(f"{D}/ref_genes_v2.txt","w").write("\n".join(uniq))
open(f"{D}/ref_barcodes_v2.txt","w").write("\n".join(a.obs_names))
pd.DataFrame({"barcode":a.obs_names,"Cell_type":a.obs["ct9"].values}).to_csv(f"{D}/annotation_v2.txt",sep="\t",index=False)
print("DONE",flush=True)
