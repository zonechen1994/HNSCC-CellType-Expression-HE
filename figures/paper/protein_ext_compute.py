#!/usr/bin/env python3
"""计算并缓存 CPTAC-HNSCC 蛋白组外部验证 supp 图所需数据 -> cache/protein_ext.json
H&E预测细胞型表达(TCGA训) -> CPTAC -> vs 实测bulk蛋白(TMT umich) per-gene PCC(跨病人)。
比法: Malignant代理 + theta加权伪bulk。分层: 实测mRNA-protein天花板 & 预测-vs-RNA好坏。"""
import os, re, json, numpy as np, pandas as pd, torch, sys
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
from scipy.stats import pearsonr
sys.path.insert(0, "code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN, RES, CRUN = "run/TCGA_HNSC_oro_v2b", "run/TCGA_HNSC_oro_v2b/results", "run/CPTAC_HNSC"
ZDIR = f"{CRUN}/deconv/bayesprism_oropharyngeal"
CACHE = "figures/paper/cache"
CTS = list(pd.read_csv(f"{RUN}/cell_types_file.csv")["cell_types"])
slug = lambda ct: re.sub(r'[^A-Za-z0-9]+', '_', ct)

cf = np.load(f"{CRUN}/features.pkl", allow_pickle=True)
sids = [f[0] for f in cf]
CX = [torch.tensor(np.asarray(f[1]), dtype=torch.float32) for f in cf]
meta = pd.read_csv(f"{CRUN}/metadata.csv"); s2c = dict(zip(meta["Slide.ID"], meta["case"]))

def predict_ct(ct):
    gf = f"{RUN}/breast_{ct}_gene_file.csv"
    if not os.path.exists(gf): return None
    genes = list(pd.read_csv(gf)["gene"])
    pred = np.zeros((len(CX), len(genes))); n = 0
    for ik in range(5):
        mp = f"{RES}/{ct}_preds/result_{ik}_0/model_trained.pth"
        if not os.path.exists(mp): continue
        m = MLP_regression(768, 768, len(genes), 0.2, None)
        m.load_state_dict(torch.load(mp, map_location=dev)); m.to(dev).eval()
        with torch.no_grad():
            pred += np.vstack([m(x.to(dev)).cpu().numpy() for x in CX]); n += 1
    if n == 0: return None
    pred /= n
    dp = pd.DataFrame(pred, index=[s2c.get(s) for s in sids], columns=genes)
    return dp.groupby(level=0).mean()

print("predicting cell types ...")
preds = {ct: predict_ct(ct) for ct in CTS}
preds = {k: v for k, v in preds.items() if v is not None}
mal = preds["Malignant"]

# theta 加权伪bulk
theta = pd.read_csv(f"{ZDIR}/theta.csv", index_col=0)
pts_all = mal.index
gene_union = sorted(set().union(*[set(v.columns) for v in preds.values()]))
pb_lin = pd.DataFrame(0.0, index=pts_all, columns=gene_union)
for ct, dp in preds.items():
    if ct not in theta.columns: continue
    lin = np.expm1(dp.reindex(pts_all))
    w = theta.reindex(pts_all)[ct].values[:, None]
    pb_lin += lin.reindex(columns=gene_union).fillna(0.0).values * w
pbulk = np.log1p(pb_lin)

# 蛋白
import cptac
prot = cptac.Hnscc().get_proteomics("umich")
prot.columns = prot.columns.get_level_values(0)
prot = prot.loc[[s for s in prot.index if not str(s).endswith(".N")]]
prot = prot.loc[:, ~prot.columns.duplicated()]

# 实测 mRNA-protein 天花板(实测bulk RNA vs 蛋白)
rna = pd.read_csv(f"{CRUN}/deconv/cptac_bulk_counts.tsv.gz", sep="\t", index_col=0).T
cpm = np.log1p(rna.div(rna.sum(1), axis=0) * 1e6)
def per_gene_pcc(A, B, minn=30):
    pts = [p for p in A.index if p in B.index]
    genes = [g for g in A.columns if g in B.columns]
    A2, B2 = A.loc[pts, genes], B.loc[pts, genes]
    out = {}
    for g in genes:
        a, b = A2[g].values, B2[g].values
        m = ~np.isnan(b)
        if m.sum() >= minn and np.nanstd(a[m]) > 1e-9 and np.nanstd(b[m]) > 1e-9:
            out[g] = pearsonr(a[m], b[m])[0]
    return out
ceil = per_gene_pcc(cpm, prot)                       # gene -> r (实测RNA vs 蛋白)

# 预测 vs 蛋白
mal_prot = per_gene_pcc(mal, prot)
pb_prot = per_gene_pcc(pbulk, prot)

# 预测 vs RNA(反卷积真值, Malignant)——"预测好"分层
Zm = pd.read_csv(f"{ZDIR}/Z_Malignant.csv", index_col=0)
Zmc = np.log1p(Zm.div(Zm.sum(1).replace(0, np.nan), axis=0) * 1e6).fillna(0.0)
mal_rna = per_gene_pcc(mal, Zmc)

hi = {g for g, r in ceil.items() if r > 0.4}          # 天花板高
well = {g for g, r in mal_rna.items() if r > 0.4}     # 预测好(malignant)

def strata_stats(dd, genesets):
    out = {}
    for name, gs in genesets.items():
        vals = np.array([dd[g] for g in dd if (gs is None or g in gs)])
        out[name] = {"median": float(np.median(vals)), "n": int(len(vals)),
                     "frac04": float((vals > 0.4).mean())}
    return out

STRAT = {"All": None, "High ceiling": hi, "Well-predicted": well, "Both": hi & well}
ladder_pb = strata_stats(pb_prot, STRAT)
ladder_mal = strata_stats(mal_prot, STRAT)

# 示例基因: 双高集里 预测(malignant) vs 蛋白 r 最高的3个
both = [g for g in mal_prot if g in hi and g in well]
both_sorted = sorted(both, key=lambda g: -mal_prot[g])
examples = []
for g in both_sorted[:6]:
    pts = [p for p in mal.index if p in prot.index]
    a = mal.loc[pts, g].values; b = prot.loc[pts, g].values
    m = ~np.isnan(b)
    if m.sum() < 60: continue
    examples.append({"gene": g, "pred": a[m].tolist(), "prot": b[m].tolist(),
                     "r": mal_prot[g], "ceil": ceil[g], "rna": mal_rna[g]})
    if len(examples) == 3: break

# panel c 用: 每基因 (天花板r, 预测vs蛋白r) —— 用伪bulk
scatter_genes = [g for g in pb_prot if g in ceil]
cache = {
    "ceiling_all": [ceil[g] for g in ceil],
    "ceil_median": float(np.median(list(ceil.values()))),
    "ladder_pb": ladder_pb, "ladder_mal": ladder_mal,
    "dist_all": [pb_prot[g] for g in pb_prot],
    "dist_both": [mal_prot[g] for g in both],
    "sc_ceil": [ceil[g] for g in scatter_genes],
    "sc_pred": [pb_prot[g] for g in scatter_genes],
    "sc_well": [int(g in well) for g in scatter_genes],
    "examples": examples,
    "n_patients": int(len([p for p in mal.index if p in prot.index])),
}
os.makedirs(CACHE, exist_ok=True)
json.dump(cache, open(f"{CACHE}/protein_ext.json", "w"))
print("cached protein_ext.json | pb All med=%.3f  Both(mal) med=%.3f  ceil med=%.3f" %
      (ladder_pb["All"]["median"], ladder_mal["Both"]["median"], cache["ceil_median"]))
print("examples:", [(e["gene"], round(e["r"],2)) for e in examples])
