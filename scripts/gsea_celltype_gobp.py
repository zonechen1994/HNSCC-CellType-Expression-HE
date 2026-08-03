#!/usr/bin/env python3
"""细胞型特异性排序的 prerank GSEA, 换成 GO Biological Process 基因集。

动机: MSigDB Hallmark 只有 50 条, 免疫相关的仅 Interferon alpha/gamma、Complement、
Inflammatory Response、Allograft Rejection, 全部偏髓系和 T 细胞, 没有任何 B 细胞程序,
所以 B 细胞在 Hallmark 上不可能出现正富集。GO_BP 含 B Cell Receptor Signaling Pathway、
B Cell Activation、Immunoglobulin Production 等, 可以检验 B 细胞是否真的没有正向信号。

排序方式与主图一致: 该细胞型预测表达 减 其余细胞型的均值 (TCGA 折外预测)。
"""
import os, json, numpy as np, pandas as pd
import gseapy
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
RUN, RES = "run/TCGA_HNSC_oro_v2b", "run/TCGA_HNSC_oro_v2b/results"
CTS = list(pd.read_csv(f"{RUN}/cell_types_file.csv")["cell_types"])
sp = np.load(f"{RUN}/splits/train_valid_test_idx.npz", allow_pickle=True)
meta = pd.read_csv(f"{RUN}/metadata.csv"); cases = meta["case"].astype(str).tolist()

USE_CPTAC = os.environ.get("COHORT","TCGA") == "CPTAC"

def profile_cptac(ct):
    import torch, sys
    sys.path.insert(0,"code/SLIDE_EX_code/SLIDE_EX_code/prediction")
    from model_MLP import MLP_regression
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = list(pd.read_csv(f"{RUN}/breast_{ct}_gene_file.csv")["gene"])
    cf = np.load("run/CPTAC_HNSC/features.pkl", allow_pickle=True)
    sids = [x[0] for x in cf]
    X = [torch.tensor(np.asarray(x[1]), dtype=torch.float32) for x in cf]
    m2 = pd.read_csv("run/CPTAC_HNSC/metadata.csv"); s2c = dict(zip(m2["Slide.ID"], m2["case"]))
    pr = np.zeros((len(X), len(g))); n = 0
    for ik in range(5):
        mm = MLP_regression(768, 768, len(g), 0.2, None)
        mm.load_state_dict(torch.load(f"{RES}/{ct}_preds/result_{ik}_0/model_trained.pth", map_location=dev))
        mm.to(dev).eval()
        with torch.no_grad(): pr += np.vstack([mm(x.to(dev)).cpu().numpy() for x in X])
        n += 1
    d = pd.DataFrame(pr / n, index=[s2c.get(x) for x in sids], columns=g)
    return d.groupby(level=0).mean().mean(axis=0)

def profile(ct):
    g = list(pd.read_csv(f"{RUN}/breast_{ct}_gene_file.csv")["gene"])
    P, order = [], []
    for k in range(5):
        P.append(np.loadtxt(f"{RES}/{ct}_preds/result_{k}_0/test_preds.txt"))
        order.extend(sp["test_idx"][k].tolist())
    d = pd.DataFrame(np.vstack(P), index=[cases[i] for i in order], columns=g)
    return d.groupby(level=0).mean().mean(axis=0)

print(f"载入 {'CPTAC 外部' if USE_CPTAC else 'TCGA 折外'} 预测的细胞型表达谱 ...")
prof = {c: (profile_cptac(c) if USE_CPTAC else profile(c)) for c in CTS}
allg = pd.DataFrame(prof).dropna(how="all")
print(f"基因 {allg.shape[0]} x 细胞型 {allg.shape[1]}")

LIB = "GO_Biological_Process_2023"
BKEY = ("b cell", "immunoglobulin", "humoral", "lymphocyte")
OUT = {}
for ct in CTS:
    others = [c for c in CTS if c != ct]
    spec = (allg[ct] - allg[others].mean(axis=1)).dropna().sort_values(ascending=False)
    try:
        pre = gseapy.prerank(rnk=spec.reset_index(), gene_sets=LIB, min_size=10, max_size=500,
                             permutation_num=1000, seed=42, no_plot=True, outdir=None, threads=8)
        r = pre.res2d.copy()
        r["NES"] = pd.to_numeric(r["NES"], errors="coerce")
        r["FDR"] = pd.to_numeric(r["FDR q-val"], errors="coerce")
        r = r.dropna(subset=["NES"])
        OUT[ct] = {x["Term"]: {"NES": float(x["NES"]), "FDR": float(x["FDR"])} for _, x in r.iterrows()}
        pos = r[r["FDR"] < 0.05].sort_values("NES", ascending=False)
        print(f"\n### {ct}  ({len(r)} 条通路, FDR<0.05 共 {int((r['FDR']<0.05).sum())} 条)")
        print("   最强正富集:")
        for _, x in pos.head(4).iterrows():
            print(f"     {x['Term'][:58]:60s} NES{x['NES']:+.2f} FDR{x['FDR']:.3f}")
        if ct == "B cell":
            hit = r[r["Term"].str.lower().str.contains("|".join(BKEY))].sort_values("NES", ascending=False)
            print("   -- B 细胞相关通路的表现 --")
            for _, x in hit.head(10).iterrows():
                print(f"     {x['Term'][:58]:60s} NES{x['NES']:+.2f} FDR{x['FDR']:.3f}")
    except Exception as e:
        print(f"  {ct}: 失败 {e}")

json.dump(OUT, open(f"figures/paper/cache/gsea_celltype_gobp{'_cptac' if USE_CPTAC else ''}.json", "w"))
print("\n已存 gsea_celltype_gobp" + ("_cptac" if USE_CPTAC else "") + ".json")
