#!/usr/bin/env python3
"""Compute the uncertainty caches that fig_plot.py needs but fig_compute.py never stored.

Blocks
  cheap : per-fold full-gene median PCC (S2)  +  per-fold abundance Spearman (Fig3a)
  cptac : patient-level bootstrap 95% CI of the CPTAC external top-1000 median PCC (Fig2c/2d)
  hpv   : bootstrap 95% CI of the per cell type external HANCOCK HPV AUC (Fig4b)

Convention for the 5-fold CI is identical to the Fig2a reference implementation
already in expr_internal_folds.json: the gene set is fixed once on the fold-mean
per-gene PCC, the statistic is then recomputed inside each held-out fold, and the
interval is mean +/- t(0.975, 4) * sd / sqrt(5).

usage: python fig_ci_compute.py [cheap] [cptac] [hpv]
"""
import os, re, sys, json, numpy as np, pandas as pd
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
from scipy import stats

TRAIN_RUN = "run/TCGA_HNSC_oro_v2b"
CPTAC_RUN = "run/CPTAC_HNSC"
HANCOCK_RUN = "run/HANCOCK"
CPTAC_ZDIR = "run/CPTAC_HNSC/deconv/bayesprism_oropharyngeal"
HPV_TABLE = "run/TCGA_HNSC/driver_mutations.csv"
CACHE = "figures/paper/cache"
TOPK = 1000
NBOOT = 2000
SEED = 0
CTS = list(pd.read_csv(f"{TRAIN_RUN}/cell_types_file.csv")["cell_types"])


def dump(name, obj):
    with open(f"{CACHE}/{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print("  cached", name)


def t_ci(v):
    """mean +/- t(0.975, n-1) * sd/sqrt(n) over the fold replicates."""
    v = np.asarray(v, float)
    n = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return m, sd, m - h, m + h


def genes_of(ct):
    return list(pd.read_csv(f"{TRAIN_RUN}/breast_{ct}_gene_file.csv")["gene"])


def colwise_pcc(A, B):
    """Pearson r for every column pair of two (n x g) matrices."""
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    num = (Ac * Bc).sum(0)
    den = np.sqrt((Ac ** 2).sum(0)) * np.sqrt((Bc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r


# ----------------------------------------------------------------- cheap block
def cheap():
    print("[cheap-1] per-fold full-gene + top-1000 median PCC (S2, Fig2a cross-check)")
    out = {}
    for ct in CTS:
        A = np.vstack([np.loadtxt(f"{TRAIN_RUN}/results/{ct}_preds/result_{ik}_0/coef_slope.txt")[:, 1]
                       for ik in range(5)])
        m = np.nanmean(A, 0)
        ok = np.isfinite(m)
        idx_top = np.argsort(-np.nan_to_num(m, nan=-9))[:TOPK]
        pf_full = [float(np.median(A[k, ok])) for k in range(5)]
        pf_top = [float(np.median(A[k, idx_top])) for k in range(5)]
        fm, fsd, flo, fhi = t_ci(pf_full)
        tm, tsd, tlo, thi = t_ci(pf_top)
        out[ct] = {"full_median": float(np.median(m[ok])), "full_per_fold": pf_full,
                   "full_fold_mean": fm, "full_sd": fsd, "full_ci_lo": flo, "full_ci_hi": fhi,
                   "top1000_median": float(np.median(m[idx_top])), "top1000_per_fold": pf_top,
                   "top1000_fold_mean": tm, "top1000_sd": tsd, "top1000_ci_lo": tlo, "top1000_ci_hi": thi}
        print(f"  {ct:12s} full={out[ct]['full_median']:.3f} ({flo:.3f}-{fhi:.3f})  "
              f"top1000={out[ct]['top1000_median']:.3f} ({tlo:.3f}-{thi:.3f})")
    dump("expr_internal_folds_full", out)

    print("[cheap-2] per-fold abundance Spearman (Fig3a)")
    ab = {}
    per_fold = {ct: [] for ct in CTS}
    Ps, Ls = [], []
    for ik in range(5):
        d = f"{TRAIN_RUN}/results/cell_fraction_preds/result_{ik}_0"
        P = np.loadtxt(f"{d}/test_preds.txt")
        L = np.loadtxt(f"{d}/test_labels.txt")
        Ps.append(P)
        Ls.append(L)
        for i, ct in enumerate(CTS):
            per_fold[ct].append(float(stats.spearmanr(P[:, i], L[:, i]).correlation))
    P = np.vstack(Ps)
    L = np.vstack(Ls)
    for i, ct in enumerate(CTS):
        pooled = float(stats.spearmanr(P[:, i], L[:, i]).correlation)
        m, sd, lo, hi = t_ci(per_fold[ct])
        ab[ct] = {"pooled": pooled, "per_fold": per_fold[ct], "fold_mean": m,
                  "sd": sd, "ci_lo": lo, "ci_hi": hi, "n_per_fold": int(Ps[0].shape[0])}
        print(f"  {ct:12s} pooled={pooled:.3f}  fold CI {lo:.3f}-{hi:.3f}")
    dump("abundance_internal_folds", ab)


# ---------------------------------------------------------------- cptac block
def _apply_model(run, subdir, outdim, dev):
    import torch
    sys.path.insert(0, os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/code/SLIDE_EX_code/SLIDE_EX_code/prediction")
    from model_MLP import MLP_regression
    sids, X = _feats(run)
    pr = np.zeros((len(X), outdim))
    for ik in range(5):
        mp = f"{TRAIN_RUN}/results/{subdir}/result_{ik}_0/model_trained.pth"
        m = MLP_regression(768, 768, outdim, 0.2, None)
        m.load_state_dict(torch.load(mp, map_location=dev))
        m.to(dev).eval()
        with torch.no_grad():
            pr += np.vstack([m(x.to(dev)).cpu().numpy() for x in X])
    return pr / 5, sids


_FC = {}


def _feats(run):
    import torch
    if run in _FC:
        return _FC[run]
    cf = np.load(f"{run}/features.pkl", allow_pickle=True)
    _FC[run] = ([f[0] for f in cf], [torch.tensor(np.asarray(f[1]), dtype=torch.float32) for f in cf])
    return _FC[run]


def _s2c(run, z=None):
    m = pd.read_csv(f"{run}/metadata.csv")
    d = dict(zip(m["Slide.ID"], m["case"].astype(str)))
    return {k: (v.zfill(z) if z else v) for k, v in d.items()}


def cptac():
    """Patient-level bootstrap of the CPTAC external top-1000 median per-gene PCC.

    The point estimate reproduces the number hardcoded in fig_compute.py
    (predicted expression vs oropharyngeal-deconvolved Z, log1p CPM).
    """
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[cptac] device", dev)
    rng = np.random.RandomState(SEED)
    out = {}
    for ct in CTS:
        zf = f"{CPTAC_ZDIR}/Z_{re.sub(r'[^A-Za-z0-9]+', '_', ct)}.csv"
        if not os.path.exists(zf):
            print(f"  {ct:12s} no Z file, skipped")
            continue
        genes = genes_of(ct)
        pr, sids = _apply_model(CPTAC_RUN, f"{ct}_preds", len(genes), dev)
        mp = _s2c(CPTAC_RUN)
        dp = pd.DataFrame(pr, index=[mp.get(s) for s in sids], columns=genes).groupby(level=0).mean()
        Z = pd.read_csv(zf, index_col=0)
        lib = Z.sum(1).replace(0, np.nan)
        Zc = np.log1p(Z.div(lib, axis=0) * 1e6).fillna(0.0)
        g = [x for x in genes if x in Zc.columns]
        pts = [c for c in dp.index if c in Zc.index]
        if len(g) < 50 or len(pts) < 20:
            print(f"  {ct:12s} too few genes/patients, skipped")
            continue
        A = dp.loc[pts, g].values.astype(float)
        B = Zc.loc[pts, g].values.astype(float)
        r = colwise_pcc(A, B)
        keep = np.isfinite(r)
        A = A[:, keep]; B = B[:, keep]; r = r[keep]
        # gene set fixed once on the full sample, exactly as the Fig2a 5-fold reference
        # does; resampling patients inside a re-selected top-1000 would inherit the
        # winner's curse of the selection and bias the interval upward.
        top = np.argsort(-r)[:TOPK]
        point = float(np.median(r[top]))
        At = A[:, top]; Bt = B[:, top]
        n = len(pts)
        bs = []
        for _ in range(NBOOT):
            idx = rng.randint(0, n, n)
            rb = colwise_pcc(At[idx], Bt[idx])
            bs.append(np.nanmedian(rb))
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        out[ct] = {"point": point, "ci_lo": lo, "ci_hi": hi, "n_patients": n,
                   "n_genes": int(len(r)), "n_boot": len(bs), "seed": SEED,
                   "boot_median": float(np.median(bs))}
        print(f"  {ct:12s} top1000 median PCC={point:.3f}  bootstrap 95% CI {lo:.3f}-{hi:.3f}  "
              f"(n={n} patients, {len(bs)} resamples)")
        np.savez(f"{CACHE}/expr_cptac_pergene_{re.sub(r'[^A-Za-z0-9]+', '_', ct)}.npz",
                 pcc=r, top_idx=top)
    dump("expr_cptac_ci", out)


# ------------------------------------------------------------------ hpv block
def hpv():
    """Bootstrap 95% CI of the per cell type external HANCOCK HPV AUC (Fig4b).

    Replays the exact classifier of fig_compute.py block 9 and bootstraps the
    external patients; the point AUC must reproduce cache/hpv_percell_auc.json.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[hpv] device", dev)

    def ens():
        return VotingClassifier([("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                                 ("rf", RandomForestClassifier(n_estimators=300, random_state=0)),
                                 ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)))],
                                voting="soft")
    sp = np.load(f"{TRAIN_RUN}/splits/train_valid_test_idx.npz", allow_pickle=True)
    tcga_cases = pd.read_csv(f"{TRAIN_RUN}/metadata.csv")["case"].tolist()
    hpv_t = pd.read_csv(HPV_TABLE, index_col=0)["HPV"]
    hpv_h = pd.read_csv(f"{HANCOCK_RUN}/hpv.csv", dtype={'case': str}).set_index("case")["hpv_p16"] \
        .map({"positive": 1, "negative": 0})

    def tcga_oof(ct):
        P, order = [], []
        for ik in range(5):
            d = f"{TRAIN_RUN}/results/{ct}_preds/result_{ik}_0"
            if not os.path.exists(f"{d}/test_preds.txt"):
                return None
            P.append(np.loadtxt(f"{d}/test_preds.txt"))
            order.extend(sp["test_idx"][ik].tolist())
        return pd.DataFrame(np.vstack(P), index=[tcga_cases[i] for i in order], columns=genes_of(ct))

    out = {}
    for ct in CTS:
        Xt = tcga_oof(ct)
        if Xt is None:
            continue
        ct_t = [c for c in Xt.index if c in hpv_t.index]
        yt = hpv_t.loc[ct_t].values.astype(int)
        if yt.sum() < 8:
            continue
        Xtv = Xt.loc[ct_t].values
        pr, sids = _apply_model(HANCOCK_RUN, f"{ct}_preds", len(genes_of(ct)), dev)
        s2c = _s2c(HANCOCK_RUN, 3)
        Xh = pd.DataFrame(pr, index=[str(s2c.get(s)).zfill(3) for s in sids],
                          columns=genes_of(ct)).groupby(level=0).mean()
        ch = [c for c in Xh.index if c in hpv_h.index]
        if len(ch) < 20 or hpv_h.loc[ch].sum() < 5:
            continue
        ye = hpv_h.loc[ch].values.astype(int)
        pe = ens().fit(Xtv, yt).predict_proba(Xh.loc[ch].values)[:, 1]
        auc = float(roc_auc_score(ye, pe))
        rng = np.random.RandomState(SEED)
        n = len(ye)
        bs = []
        for _ in range(NBOOT):
            idx = rng.randint(0, n, n)
            if len(np.unique(ye[idx])) < 2:
                continue
            bs.append(roc_auc_score(ye[idx], pe[idx]))
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        out[ct] = {"auc": auc, "ci_lo": lo, "ci_hi": hi, "n": n, "n_pos": int(ye.sum()),
                   "n_boot": len(bs), "seed": SEED}
        print(f"  {ct:12s} ext AUC={auc:.3f}  bootstrap 95% CI {lo:.3f}-{hi:.3f}  (n={n})")
    dump("hpv_percell_ci", out)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["cheap"]
    for t in todo:
        globals()[t]()
