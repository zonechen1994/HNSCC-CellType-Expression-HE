#!/usr/bin/env python3
"""细胞型特异的 GO(BP) 富集, 在 TCGA 与 CPTAC 中各做一遍并比较一致性。

做法对齐 SLIDE-EX (Wang et al., npj Precis Oncol 2026, doi:10.1038/s41698-026-01419-9):
gseapy 的 enrichr 过表达分析, GO_Biological_Process_2023 基因集,
每个细胞型取 top-5 显著通路, 取并集画聚类热图。

与之前版本的区别: 之前只做了 TCGA∩CPTAC 的交集基因集(仅 4 个细胞型有显著通路)。
这里改为两个队列各自的 well-predicted 基因集分别富集, 再看同一细胞型的通路是否复现。

well-predicted 定义: TCGA 用 5 折交叉验证的逐基因 Pearson r>0.4;
CPTAC 用 5 折集成模型的预测表达与该队列反卷积表达的逐基因 Pearson r>0.4。
"""
import os, re, sys, json, numpy as np, pandas as pd, torch
from scipy.stats import pearsonr
import gseapy as gp
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
sys.path.insert(0, "code/SLIDE_EX_code/SLIDE_EX_code/prediction")
from model_MLP import MLP_regression
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TR, RES = "run/TCGA_HNSC_oro_v2b", "run/TCGA_HNSC_oro_v2b/results"
CZDIR = "run/CPTAC_HNSC/deconv/bayesprism_oropharyngeal"
THR, MIN_GENES, TOPN = 0.4, 100, 5   # 基因数下限提到 100: 几十个基因的富集结果不稳定
GO_LIB = "GO_Biological_Process_2023"
CTS = list(pd.read_csv(f"{TR}/cell_types_file.csv")["cell_types"])

_C = {}
def feats(run):
    if run not in _C:
        cf = np.load(f"{run}/features.pkl", allow_pickle=True)
        _C[run] = ([f[0] for f in cf], [torch.tensor(np.asarray(f[1]), dtype=torch.float32) for f in cf])
    return _C[run]

def apply_model(run, subdir, outdim):
    sids, X = feats(run); pr = np.zeros((len(X), outdim))
    for ik in range(5):
        m = MLP_regression(768, 768, outdim, 0.2, None)
        m.load_state_dict(torch.load(f"{RES}/{subdir}/result_{ik}_0/model_trained.pth", map_location=dev))
        m.to(dev).eval()
        with torch.no_grad():
            pr += np.vstack([m(x.to(dev)).cpu().numpy() for x in X])
    return pr / 5, sids

def genes_of(ct): return list(pd.read_csv(f"{TR}/breast_{ct}_gene_file.csv")["gene"])

def tcga_pcc(ct):
    genes = genes_of(ct)
    Pp = np.vstack([np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/test_preds.txt") for k in range(5)])
    Ll = np.vstack([np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/test_labels.txt") for k in range(5)])
    r = [pearsonr(Pp[:, j], Ll[:, j])[0] if Pp[:, j].std() > 1e-9 and Ll[:, j].std() > 1e-9 else np.nan
         for j in range(len(genes))]
    return pd.Series(r, index=genes)

def cptac_pcc(ct):
    genes = genes_of(ct)
    zf = f"{CZDIR}/Z_{re.sub(r'[^A-Za-z0-9]+','_',ct)}.csv"
    if not os.path.exists(zf): return None
    pr, sids = apply_model("run/CPTAC_HNSC", f"{ct}_preds", len(genes))
    mp = pd.read_csv("run/CPTAC_HNSC/metadata.csv")
    mp = dict(zip(mp["Slide.ID"], mp["case"].astype(str)))
    dp = pd.DataFrame(pr, index=[mp.get(s) for s in sids], columns=genes).groupby(level=0).mean()
    Z = pd.read_csv(zf, index_col=0)
    Zc = np.log1p(Z.div(Z.sum(1).replace(0, np.nan), axis=0) * 1e6).fillna(0.0)
    g = [x for x in genes if x in Zc.columns]; pts = [c for c in dp.index if c in Zc.index]
    A = dp.loc[pts, g].values; B = Zc.loc[pts, g].values
    r = [pearsonr(A[:, j], B[:, j])[0] if A[:, j].std() > 1e-9 and B[:, j].std() > 1e-9 else np.nan
         for j in range(len(g))]
    return pd.Series(r, index=g)

print("== 两个队列各自的 well-predicted 基因集 (r>0.4) ==", flush=True)
SETS = {"TCGA": {}, "CPTAC": {}}
for ct in CTS:
    ti = tcga_pcc(ct); ce = cptac_pcc(ct)
    SETS["TCGA"][ct] = [g for g in ti.index if ti[g] > THR]
    SETS["CPTAC"][ct] = [] if ce is None else [g for g in ce.index if ce[g] > THR]
    print(f"  {ct:12s} TCGA={len(SETS['TCGA'][ct]):5d}  CPTAC={len(SETS['CPTAC'][ct]):5d}", flush=True)

print(f"\n== GO(BP) 富集, 每队列每细胞型独立做 ==", flush=True)
ENR = {"TCGA": {}, "CPTAC": {}}
for coh in ("TCGA", "CPTAC"):
    print(f"-- {coh} --", flush=True)
    for ct, gl in SETS[coh].items():
        if len(gl) < MIN_GENES:
            print(f"  {ct:12s} 基因 {len(gl)} < {MIN_GENES}, 跳过", flush=True); continue
        try:
            e = gp.enrichr(gene_list=gl, gene_sets=[GO_LIB], organism="human", outdir=None, no_plot=True)
            t = e.results.sort_values("Adjusted P-value")
            ts = t[t["Adjusted P-value"] < 0.05]
            if len(ts) == 0:
                print(f"  {ct:12s} 显著通路 0", flush=True); continue
            ENR[coh][ct] = ts
            print(f"  {ct:12s} 显著通路 {len(ts):4d}  top: {ts.iloc[0]['Term'][:52]}", flush=True)
        except Exception as ex:
            print(f"  {ct:12s} Enrichr 失败: {ex}", flush=True)

def matrix(enr):
    terms = list(dict.fromkeys([x for ct, t in enr.items() for x in t.head(TOPN)["Term"]]))
    M = pd.DataFrame(0.0, index=terms, columns=list(enr.keys()))
    for ct, t in enr.items():
        ts = t.drop_duplicates("Term").set_index("Term")
        for term in terms:
            if term in ts.index: M.loc[term, ct] = float(ts.loc[term, "Odds Ratio"])
    return M

os.makedirs("figures/paper/cache/go_two_cohorts", exist_ok=True)
for coh in ("TCGA", "CPTAC"):
    if not ENR[coh]: continue
    M = matrix(ENR[coh])
    M.to_csv(f"figures/paper/cache/go_two_cohorts/go_{coh}_oddsratio.csv")
    full = pd.concat([t.head(TOPN).assign(cell=ct) for ct, t in ENR[coh].items()])
    full.to_csv(f"figures/paper/cache/go_two_cohorts/go_{coh}_top.csv", index=False)
    print(f"\n{coh} 矩阵: {M.shape[0]} 通路 x {M.shape[1]} 细胞型 -> {list(M.columns)}")

# ---- 跨队列一致性 ----
print("\n== 跨队列一致性: TCGA 的 top-5 通路在 CPTAC 中是否也显著 ==")
print(f"{'cell type':13s} {'TCGA top5 在CPTAC显著':>22s} {'CPTAC显著通路数':>16s}")
rows = []
for ct in CTS:
    if ct not in ENR["TCGA"]: continue
    t5 = list(ENR["TCGA"][ct].head(TOPN)["Term"])
    if ct not in ENR["CPTAC"]:
        print(f"{ct:13s} {'CPTAC 无显著通路':>22s}"); rows.append(dict(cell=ct, n_top=len(t5), n_repl=0, repl=None)); continue
    cset = set(ENR["CPTAC"][ct]["Term"])
    k = sum(1 for x in t5 if x in cset)
    rows.append(dict(cell=ct, n_top=len(t5), n_repl=k, repl="; ".join([x for x in t5 if x in cset])))
    print(f"{ct:13s} {f'{k}/{len(t5)}':>22s} {len(cset):16d}")
pd.DataFrame(rows).to_csv("figures/paper/cache/go_two_cohorts/cross_cohort_replication.csv", index=False)
print("\n已存 figures/paper/cache/go_two_cohorts/")
