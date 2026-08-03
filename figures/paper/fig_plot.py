#!/usr/bin/env python3
"""Render the 4 main + 3 supplementary HNSC figures as 300dpi PDF (+PNG preview)
from the cache produced by fig_compute.py. English labels, embedded fonts."""
import os, json, numpy as np, pandas as pd
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Arial","Helvetica"],
    "font.size":8,"axes.linewidth":0.8,"axes.titlesize":9,"axes.labelsize":8.5,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
CACHE="figures/paper/cache"; OUT="figures/paper"
PAL={"prot":"#2c7fb8","risk":"#d95f0e","tcga":"#4477AA","cptac":"#CC6677",
     "hancock":"#228833","hpv":"#AA3377","grey":"#999999"}
def L(n): return json.load(open(f"{CACHE}/{n}.json"))
def LQ(n):
    """Optional cache: returns None instead of raising when the file is absent."""
    try: return json.load(open(f"{CACHE}/{n}.json"))
    except Exception: return None

# ---------- shared uncertainty helpers (house style for the whole manuscript) ----------
from scipy import stats as _st
ERRKW=dict(lw=0.8,ecolor="0.25")      # matches the Fig2a reference implementation
CAPSIZE=2.5
BAND_ALPHA=0.30                        # band is narrow at large n; raised so it reads under the fit line
def fitband(ax,x,y,color,lw=1.2,alpha=BAND_ALPHA,npts=120,zorder=3,ls="-",label=None):
    """OLS fit line + shaded 95% confidence band of the fitted line.
    Band = yhat +/- t(0.975, n-2) * s * sqrt(1/n + (x-xbar)^2/Sxx), i.e. the CI of
    the regression line itself (not a +/- 1 SD prediction band)."""
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]; n=x.size
    if n<3: return None
    z=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),npts); yh=np.polyval(z,xs)
    ax.plot(xs,yh,color=color,lw=lw,ls=ls,zorder=zorder,label=label)
    resid=y-np.polyval(z,x); s=np.sqrt(float((resid**2).sum())/(n-2))
    Sxx=float(((x-x.mean())**2).sum())
    if Sxx>0:
        se=s*np.sqrt(1.0/n+(xs-x.mean())**2/Sxx); t=float(_st.t.ppf(0.975,n-2))
        ax.fill_between(xs,yh-t*se,yh+t*se,color=color,alpha=alpha,lw=0,zorder=zorder-1)
        # 细边界线: n 大时带宽只有 y 轴的百分之几, 仅靠填充会被拟合线盖住
        for yb in (yh-t*se, yh+t*se):
            ax.plot(xs,yb,color=color,lw=0.45,alpha=0.85,zorder=zorder-1)
    return z
def err_from_ci(vals,los,his):
    """Asymmetric matplotlib yerr/xerr anchored on the plotted point estimate."""
    lo=[max(0.0,float(v)-float(l)) for v,l in zip(vals,los)]
    hi=[max(0.0,float(h)-float(v)) for v,h in zip(vals,his)]
    return [lo,hi]
def save(fig,name):
    for ext in ("pdf","png"): fig.savefig(f"{OUT}/{name}.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig); print("saved",name)
# consistent cell order = descending internal expression PCC
ei=L("expr_internal"); ORDER=sorted(ei["top1000"],key=lambda c:-ei["top1000"][c])
def panel_tag(ax,s):
    ax.text(-0.16,1.06,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top",ha="left")
from scipy.stats import pearsonr as _prsn
def r_ci_p(x,y,nb=2000,seed=0):
    x=np.asarray(x,float); y=np.asarray(y,float); _m=~(np.isnan(x)|np.isnan(y)); x,y=x[_m],y[_m]; n=len(x)
    r,pp=_prsn(x,y); _rng=np.random.RandomState(seed); bs=[]
    for _ in range(nb):
        i=_rng.randint(0,n,n)
        if np.std(x[i])>0 and np.std(y[i])>0: bs.append(_prsn(x[i],y[i])[0])
    _lo,_hi=np.percentile(bs,[2.5,97.5]); return r,_lo,_hi,pp
def _pfmt(pp):
    if pp>=0.001: return f"P = {pp:.3f}"
    ss=f"{pp:.0e}".replace("e-0","e-").replace("e+0","e+"); return f"P = {ss}"

# ==================== FIGURE 2 — cell-type expression prediction ====================
def fig2():
    ei=L("expr_internal"); ec=L("expr_cptac"); ex=L("expr_examples")
    fig=plt.figure(figsize=(12.4,13.0))
    # 4 行: (0) 两个柱状图 + 迁移森林图; (1) TCGA RNA 散点; (2) CPTAC 蛋白散点;
    # (3) 蛋白分布 + 基因重叠韦恩图。b 与 g 上下同宽同高, 便于直接对照同样三个基因。
    gs=gridspec.GridSpec(4,3,figure=fig,hspace=0.62,wspace=0.30,
                         height_ratios=[1,0.90,0.90,0.85])
    x=np.arange(len(ORDER))
    # (a) internal top-1000 PCC bar
    axa=fig.add_subplot(gs[0,0])
    _fold=L("expr_internal_folds")
    _lo=[max(0,ei["top1000"][c]-_fold[c]["ci_lo"]) if c in _fold else 0 for c in ORDER]
    _hi=[max(0,_fold[c]["ci_hi"]-ei["top1000"][c]) if c in _fold else 0 for c in ORDER]
    axa.bar(x,[ei["top1000"][c] for c in ORDER],color=PAL["tcga"],width=0.7,
            yerr=[_lo,_hi],capsize=2.5,error_kw=dict(lw=0.8,ecolor="0.25"))
    axa.axhline(0.4,color="grey",ls="--",lw=0.9)
    axa.set_xticks(x); axa.set_xticklabels(ORDER,rotation=40,ha="right")
    axa.set_ylabel("Top-1000 median PCC"); axa.set_ylim(0,0.6)
    axa.set_title("Internal cross-validation (TCGA-HNSC)",fontsize=9,pad=12)
    axa.text(0.5,1.015,"error bars: 95% CI, patient bootstrap (1000 resamples)",transform=axa.transAxes,
             ha="center",va="bottom",fontsize=6.2,color="0.4"); panel_tag(axa,"a")
    for i,c in enumerate(ORDER):
        _t=_fold[c]["ci_hi"] if c in _fold else ei["top1000"][c]
        axa.text(i,_t+0.012,f"{ei['top1000'][c]:.2f}",ha="center",fontsize=6.5)
    # (b) example gene scatters (nested 1x3, top-right)
    sub=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[2,:],wspace=0.26)
    for k,e in enumerate(ex):
        axb=fig.add_subplot(sub[0,k])
        p=np.array(e["pred"]); o=np.array(e["obs"])
        axb.scatter(o,p,s=9,alpha=0.55,color=PAL["tcga"],edgecolors="white",linewidths=0.2,zorder=1)
        lo=min(o.min(),p.min()); hi=max(o.max(),p.max())
        # identity reference (grey dashed) kept, now visually distinct from the OLS fit
        axb.plot([lo,hi],[lo,hi],color=PAL["grey"],lw=0.9,ls="--",zorder=2)
        fitband(axb,o,p,PAL["risk"],lw=0.9)   # OLS fit + shaded 95% CI band (as in panel g)
        _r,_rlo,_rhi,_rp=r_ci_p(o,p)   # r + bootstrap 95% CI + P, 从实际散点算保证标签=点
        _cp=e.get("compartment","")
        axb.set_title(f"{e['gene']}\n{_cp} ({e['ct']})",fontsize=7.2,fontweight="bold")
        axb.text(0.05,0.985,f"r = {_r:.2f} ({_rlo:.2f}\u2013{_rhi:.2f})\n{_pfmt(_rp)}",transform=axb.transAxes,fontsize=5.6,va="top",ha="left",
                 bbox=dict(boxstyle="round,pad=0.28",fc="#ffffff",ec=PAL["tcga"],lw=0.7))
        axb.set_xlabel("Deconvolved expr (observed)",fontsize=7)
        if k==0: axb.set_ylabel("Predicted expr",fontsize=7)
        axb.tick_params(labelsize=6.5)
        if k==0:
            panel_tag(axb,"e")
            from matplotlib.lines import Line2D
            axb.legend(handles=[Line2D([0],[0],color=PAL["grey"],lw=0.9,ls="--",label="y = x"),
                                Line2D([0],[0],color=PAL["risk"],lw=1.2,label="OLS fit, 95% CI")],
                       loc="lower right",frameon=False,fontsize=5.2,handlelength=1.2,
                       borderpad=0.05,labelspacing=0.2,handletextpad=0.4)
    # (c) CPTAC external PCC bar (same axis as a)
    axc=fig.add_subplot(gs[0,1])
    _cci=LQ("expr_cptac_ci")
    _cv=[ec[c] for c in ORDER]
    if _cci:
        _clo=[_cci[c]["ci_lo"] if c in _cci else ec[c] for c in ORDER]
        _chi=[_cci[c]["ci_hi"] if c in _cci else ec[c] for c in ORDER]
        axc.bar(x,_cv,color=PAL["cptac"],width=0.7,
                yerr=err_from_ci(_cv,_clo,_chi),capsize=CAPSIZE,error_kw=ERRKW)
        _ctitle="External validation (CPTAC-HNSCC, n=110)"
    else:
        axc.bar(x,_cv,color=PAL["cptac"],width=0.7); _chi=_cv
        _ctitle="External validation (CPTAC-HNSCC, n=110)"
    axc.axhline(0.4,color="grey",ls="--",lw=0.9)
    axc.set_xticks(x); axc.set_xticklabels(ORDER,rotation=40,ha="right")
    axc.set_ylabel("Top-1000 median PCC"); axc.set_ylim(0,0.6)
    axc.set_title(_ctitle,fontsize=9,pad=12)
    axc.text(0.5,1.015,"error bars: 95% CI, patient bootstrap (2000 resamples)",transform=axc.transAxes,
             ha="center",va="bottom",fontsize=6.2,color="0.4"); panel_tag(axc,"b")
    for i,c in enumerate(ORDER):
        axc.text(i,max(ec[c],_chi[i])+0.012,f"{ec[c]:.2f}",ha="center",fontsize=6.5)
    # (d) dumbbell: per-cell internal vs external (no label overlap)
    axd=fig.add_subplot(gs[0,2])
    dorder=sorted(ORDER,key=lambda c:ei["top1000"][c])   # ascending -> best on top
    yy=np.arange(len(dorder))
    for i,c in enumerate(dorder):
        yi=ei["top1000"][c]; ye=ec[c]
        axd.plot([ye,yi],[i,i],color="#cccccc",lw=1.4,zorder=1)
    _iv=[ei["top1000"][c] for c in dorder]; _ev=[ec[c] for c in dorder]
    # horizontal 95% CI whiskers: both cohorts by patient-level bootstrap (see cache B field)
    _ilo=[_fold[c]["ci_lo"] if c in _fold else _iv[i] for i,c in enumerate(dorder)]
    _ihi=[_fold[c]["ci_hi"] if c in _fold else _iv[i] for i,c in enumerate(dorder)]
    axd.errorbar(_iv,yy,xerr=err_from_ci(_iv,_ilo,_ihi),fmt="none",
                 ecolor=PAL["tcga"],elinewidth=0.8,capsize=2.0,zorder=2)
    if _cci:
        _elo=[_cci[c]["ci_lo"] if c in _cci else _ev[i] for i,c in enumerate(dorder)]
        _ehi=[_cci[c]["ci_hi"] if c in _cci else _ev[i] for i,c in enumerate(dorder)]
        axd.errorbar(_ev,yy,xerr=err_from_ci(_ev,_elo,_ehi),fmt="none",
                     ecolor=PAL["cptac"],elinewidth=0.8,capsize=2.0,zorder=2)
    else:
        _elo=_ev; _ehi=_ev
    axd.scatter(_iv,yy,s=34,color=PAL["tcga"],zorder=3,label="TCGA internal")
    axd.scatter(_ev,yy,s=34,color=PAL["cptac"],zorder=3,label="CPTAC external")
    axd.axvline(0.4,color="grey",ls="--",lw=0.9)
    axd.set_yticks(yy); axd.set_yticklabels(dorder)
    axd.set_xlim(min(0.15,min(_ilo+_elo)-0.02),max(0.6,max(_ihi+_ehi)+0.02))
    axd.set_xlabel("Top-1000 median PCC\nwhiskers: 95% CI, patient bootstrap\n(TCGA 1000, CPTAC 2000 resamples)",
                   fontsize=7.5)
    axd.set_title("Per-cell transfer (modest decay)"); axd.legend(loc="lower right",frameon=False)
    panel_tag(axd,"c")
    # (e) cross-cohort well-predicted gene overlap (Fig2e-style Venn stack)
    from matplotlib_venn import venn2, venn2_circles
    from matplotlib.patches import Patch
    ov=L("gene_overlap"); evens=["Fibroblast","Malignant","Macrophage"]
    esub=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,:],wspace=0.18)
    for k,ct in enumerate(evens):
        axe=fig.add_subplot(esub[0,k]); d=ov[ct]
        sub=(d["tcga_only"],d["cptac_only"],d["shared"])
        v=venn2(subsets=sub,set_labels=("",""),ax=axe)
        for pid,col in [("10",PAL["tcga"]),("01",PAL["cptac"]),("11","#8B7BA8")]:
            pt=v.get_patch_by_id(pid)
            if pt: pt.set_color(col); pt.set_alpha(0.62); pt.set_edgecolor("none")
        venn2_circles(subsets=sub,ax=axe,lw=0.7,color="#555")
        for lab in v.subset_labels:
            if lab: lab.set_fontsize(6.5)
        axe.set_title(f"{ct}   {_pfmt(d['p'])}",fontsize=7)
        if k==0:
            panel_tag(axe,"d")
            axe.legend(handles=[Patch(fc=PAL["tcga"],alpha=0.62,ec="none",label="TCGA-HNSC (internal)"),
                                Patch(fc=PAL["cptac"],alpha=0.62,ec="none",label="CPTAC-HNSCC (external)")],
                       loc="lower center",bbox_to_anchor=(0.5,-0.30),fontsize=6.2,frameon=False,
                       handlelength=1.1,labelspacing=0.3)
        if k==2:
            axe.text(0.5,-0.14,"robustly-predicted genes (PCC > 0.4)\nhypergeometric test",
                     transform=axe.transAxes,ha="center",va="top",fontsize=6,color="#555")
    # ===== (f,g) protein-level validation row =====
    from scipy.stats import gaussian_kde
    C2=L("protein_simple")
    prow=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[3,:],wspace=0.26)
    _mk=LQ("marker_protein_examples") or C2["examples"]
    for k,e in enumerate(_mk):
        axg=fig.add_subplot(prow[0,k])
        p=np.array(e["pred"]); o=np.array(e["prot"])
        axg.scatter(p,o,s=9,alpha=0.55,color=PAL["cptac"],edgecolors="white",linewidths=0.2)
        fitband(axg,p,o,PAL["risk"],lw=0.9)   # OLS fit + t-based 95% CI band of the fitted line
        _cp=e.get("compartment","")
        axg.set_title(f"{e['gene']}"+(f"\n{_cp} ({e['ct']})" if _cp else ""),fontsize=7.2,fontweight="bold")
        _gr,_glo,_ghi,_gp=r_ci_p(p,o)
        axg.text(0.05,0.985,f"r = {_gr:.2f} ({_glo:.2f}\u2013{_ghi:.2f})\n{_pfmt(_gp)}",transform=axg.transAxes,fontsize=5.6,va="top",ha="left",
                 bbox=dict(boxstyle="round,pad=0.28",fc="#ffffff",ec=PAL["cptac"],lw=0.7))
        axg.set_xlabel("Predicted expr",fontsize=7)
        if k==0: axg.set_ylabel("Measured protein",fontsize=7); panel_tag(axg,"f")
        if k==2:
            axg.text(0.97,0.03,"line: OLS fit\nband: 95% CI",transform=axg.transAxes,
                     fontsize=5.6,va="bottom",ha="right",color=PAL["risk"])
        axg.tick_params(labelsize=6.5)
    fig.suptitle("Figure 2. Deep learning predicts cell type-specific gene expression, internally, externally, and at the protein level",
                 fontsize=10.5,y=0.99)
    save(fig,"Fig2_expression")

# ==================== FIGURE 3 — abundance + transfer QC + IHC ====================
def fig3():
    ab=L("abundance_internal"); tr=L("transfer_cptac"); ihc=L("ihc")
    fig=plt.figure(figsize=(9.2,6.6))
    gs=gridspec.GridSpec(2,2,figure=fig,height_ratios=[1,1.1],hspace=0.5,wspace=0.32)
    # (a) wide: abundance Spearman bar
    axa=fig.add_subplot(gs[0,:])
    order_ab=sorted(ab,key=lambda c:-ab[c]); x=np.arange(len(order_ab))
    cols=[PAL["hancock"] if ab[c]>=0.3 else PAL["risk"] for c in order_ab]
    _abf=LQ("abundance_internal_folds"); _av=[ab[c] for c in order_ab]
    if _abf:
        _alo=[_abf[c]["ci_lo"] if c in _abf else _av[i] for i,c in enumerate(order_ab)]
        _ahi=[_abf[c]["ci_hi"] if c in _abf else _av[i] for i,c in enumerate(order_ab)]
        axa.bar(x,_av,color=cols,width=0.6,yerr=err_from_ci(_av,_alo,_ahi),
                capsize=CAPSIZE,error_kw=ERRKW)
        _atitle=("Cell type abundance prediction, TCGA cross-validation\n"
                 "(error bars: 95% CI, patient bootstrap, 1000 resamples)")
    else:
        axa.bar(x,_av,color=cols,width=0.6); _ahi=_av
        _atitle="Cell type abundance prediction, TCGA cross-validation"
    axa.set_xticks(x); axa.set_xticklabels(order_ab,rotation=25,ha="right")
    axa.set_ylabel("Spearman (pred vs deconv.)"); axa.set_title(_atitle,fontsize=9)
    for i,c in enumerate(order_ab):
        axa.text(i,max(ab[c],_ahi[i])+0.012,f"{ab[c]:.2f}",ha="center",fontsize=6.5)
    panel_tag(axa,"a")
    # (b) transfer QC heatmap strip
    axb=fig.add_subplot(gs[1,0])
    order_tr=sorted(tr,key=lambda c:-tr[c])
    M=np.array([[tr[c]] for c in order_tr])
    im=axb.imshow(M,cmap="RdYlGn",vmin=-0.1,vmax=0.6,aspect="auto")
    axb.set_yticks(range(len(order_tr))); axb.set_yticklabels(order_tr)
    axb.set_xticks([]); axb.set_title("Transfer QC: predicted vs true θ\n(CPTAC, HPV-neg / immune-cold)")
    for i,c in enumerate(order_tr):
        axb.text(0,i,f"{tr[c]:.2f}",ha="center",va="center",fontsize=7,
                 color="white" if tr[c]<0.1 else "black")
    cb=fig.colorbar(im,ax=axb,fraction=0.12,pad=0.04); cb.set_label("Spearman",fontsize=7)
    axb.text(0.5,-0.09,"T cell / Mast collapse: no dynamic range",transform=axb.transAxes,
             ha="center",fontsize=6.3,color=PAL["risk"])
    panel_tag(axb,"b")
    # (c) IHC scatters (CD3, CD8)
    sub=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,1],wspace=0.45)
    for k,marker in enumerate(["CD3_density","CD8_density"]):
        axc=fig.add_subplot(sub[0,k])
        d=ihc[marker]; xv=np.array(d["x"]); yv=np.array(d["y"])
        axc.scatter(xv,yv,s=7,alpha=0.45,color=PAL["hancock"],edgecolors="none")
        # OLS fit line + shaded 95% CI band (house style, as in Fig2 panel g)
        fitband(axc,xv,yv,PAL["risk"],lw=1.0)
        axc.set_title(f"{marker.split('_')[0]}\nr={d['rho']:.2f}, p<1e-20, n={d['n']}",fontsize=7.5)
        axc.set_xlabel("Predicted T-cell abundance",fontsize=7)
        if k==0: axc.set_ylabel("IHC density",fontsize=7)
        axc.tick_params(labelsize=6.5)
        if k==0: panel_tag(axc,"c")
        if k==1:
            axc.text(0.5,-0.30,"line: OLS fit, shaded band: 95% CI",transform=axc.transAxes,
                     fontsize=6,va="top",ha="center",color="#555")
    fig.suptitle("Figure 3. Cell-type abundance: predictable, transfer-QC gated, IHC-validated",
                 fontsize=10.5,y=0.98)
    save(fig,"Fig3_abundance")

# ==================== FIGURE 4 — merged HPV axis (subtyping + immune survival) ====
def fig4():
    roc=np.load(f"{CACHE}/hpv_roc.npz"); pca=L("hpv_percell_auc")
    km=L("km_hancock"); byhpv=L("abundance_by_hpv"); MV=L("multivariable"); surv=L("survival_cox")
    fig=plt.figure(figsize=(9.6,7.2))
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.42)
    # (a) ROC internal + external
    axa=fig.add_subplot(gs[0,0])
    axa.plot(roc["fpr_i"],roc["tpr_i"],color=PAL["tcga"],lw=1.6,
             label=f"TCGA internal\nAUC={float(roc['int_auc']):.2f} ({float(roc['int_lo']):.2f}-{float(roc['int_hi']):.2f})")
    axa.plot(roc["fpr_e"],roc["tpr_e"],color=PAL["hancock"],lw=1.6,
             label=f"HANCOCK external\nAUC={float(roc['ext_auc']):.2f} ({float(roc['ext_lo']):.2f}-{float(roc['ext_hi']):.2f})")
    axa.plot([0,1],[0,1],color="grey",ls="--",lw=0.8)
    axa.set_xlabel("1 - specificity"); axa.set_ylabel("Sensitivity")
    axa.set_title("HPV classification from\npredicted expression"); axa.legend(loc="lower right",frameon=False,fontsize=6.5)
    panel_tag(axa,"a")
    # (b) per-cell external HPV AUC
    axb=fig.add_subplot(gs[0,1])
    ext=pca["external"]; ob=sorted(ext,key=lambda c:-ext[c]); x=np.arange(len(ob))
    _hci=LQ("hpv_percell_ci"); _hv=[ext[c] for c in ob]
    if _hci:
        _hlo=[_hci[c]["ci_lo"] if c in _hci else _hv[i] for i,c in enumerate(ob)]
        _hhi=[_hci[c]["ci_hi"] if c in _hci else _hv[i] for i,c in enumerate(ob)]
        axb.bar(x,_hv,color=PAL["hpv"],width=0.7,yerr=err_from_ci(_hv,_hlo,_hhi),
                capsize=CAPSIZE,error_kw=ERRKW)
        axb.set_xlabel("error bars: 95% CI, patient bootstrap, 2000 resamples",fontsize=6.3)
    else:
        axb.bar(x,_hv,color=PAL["hpv"],width=0.7)
    _btitle="HPV signal is ecosystem-wide\n(every compartment decodes it)"
    axb.axhline(0.5,color="grey",ls="--",lw=0.8); axb.set_ylim(0,1)
    axb.set_xticks(x); axb.set_xticklabels(ob,rotation=40,ha="right")
    axb.set_ylabel("External HPV AUC (HANCOCK)")
    axb.set_title(_btitle)
    panel_tag(axb,"b")
    # (c) predicted immune abundance by HPV status
    axc=fig.add_subplot(gs[0,2])
    cells=["T cell","Dendritic"]; positions=[]; data=[]; cols=[]
    for i,ct in enumerate(cells):
        data.append(byhpv[ct]["neg"]); positions.append(i*3+0.6); cols.append(PAL["cptac"])
        data.append(byhpv[ct]["pos"]); positions.append(i*3+1.4); cols.append(PAL["tcga"])
    bp=axc.boxplot(data,positions=positions,widths=0.65,patch_artist=True,showfliers=False)
    for patch,c in zip(bp["boxes"],cols): patch.set_facecolor(c); patch.set_alpha(0.75)
    for med in bp["medians"]: med.set_color("black")
    axc.set_xticks([0.6,1.4,3.6,4.4]); axc.set_xticklabels(["HPV−","HPV+","HPV−","HPV+"],fontsize=6.5)
    ylo,yhi=axc.get_ylim(); axc.set_ylim(ylo,yhi*1.18)
    for i,ct in enumerate(cells):
        axc.text(i*3+1.0,ylo+(yhi-ylo)*0.02,f"p={byhpv[ct]['p']:.0e}",ha="center",fontsize=6.2,color="#444")
        axc.text(i*3+1.0,-0.16,ct,transform=axc.get_xaxis_transform(),ha="center",fontsize=7,fontweight="bold")
    axc.set_ylabel("Predicted abundance")
    axc.set_title("HPV+ tumors are lymphoid/DC-rich",pad=8)
    panel_tag(axc,"c")
    # (d),(e) KM curves
    from lifelines import KaplanMeierFitter
    for idx,(ct,tag) in enumerate([("Macrophage","d"),("T cell","e")]):
        axk=fig.add_subplot(gs[1,idx])
        g=km[ct]; T=np.array(g["time"]); E=np.array(g["event"]); hi=np.array(g["hi"])
        kmf=KaplanMeierFitter()
        for m,lab,col in [(hi,f"{ct} high",PAL["prot"]),(~hi,f"{ct} low",PAL["risk"])]:
            kmf.fit(T[m],E[m],label=lab)
            kmf.plot_survival_function(ax=axk,color=col,ci_show=True,ci_alpha=BAND_ALPHA,lw=1.5)
        hr=surv["hancock_os"]["hr"][ct]
        pstr="<0.001" if g['logrank_p']<0.001 else f"{g['logrank_p']:.3f}"
        axk.set_xlabel("Months\n(shaded bands: 95% CI)",fontsize=7.5)
        if idx==0: axk.set_ylabel("Overall survival")
        axk.set_title(f"HANCOCK {ct} → OS\nHR={hr[0]:.2f} ({hr[1]:.2f}-{hr[2]:.2f}), logrank p={pstr}",fontsize=7.3)
        axk.legend(loc="lower left",frameon=False,fontsize=6.8); axk.set_ylim(0,1.02)
        panel_tag(axk,tag)
    # (f) multivariable nested Cox forest — the HPV-driven punchline
    axf=fig.add_subplot(gs[1,2])
    mcol={0:PAL["tcga"],1:PAL["grey"],2:PAL["hpv"]}; mmark={0:"o",1:"s",2:"D"}
    mlab={0:"univariate",1:"+clinical",2:"+HPV"}; off={0:0.24,1:0.0,2:-0.24}; seen=set()
    labs=[m["label"] for m in MV][::-1]
    for yi,m in enumerate(MV[::-1]):
        for mi,key in enumerate(["m1","m2","m3"]):
            v=m[key]
            if not v: continue
            y=yi+off[mi]; col=PAL["prot"] if v[0]<1 else PAL["risk"]
            axf.plot([v[1],v[2]],[y,y],color=col,lw=1.4,zorder=2)
            lab=mlab[mi] if mi not in seen else None; seen.add(mi)
            axf.scatter([v[0]],[y],s=42,c=[col],marker=mmark[mi],edgecolor="k",lw=0.5,zorder=3,label=lab)
            axf.text(v[2]+0.02,y,f"{v[0]:.2f}{'*' if v[3]<0.05 else ''}",va="center",fontsize=6,color=col)
    axf.axvline(1,color="k",ls="--",lw=0.8); axf.set_yticks(range(len(labs)))
    axf.set_yticklabels(labs,fontsize=6.5)
    axf.set_xlabel("HR per SD with 95% CI (HR<1 protective)",fontsize=7.5)
    axf.set_xlim(0.55,1.35); axf.set_ylim(-0.55,len(labs)-1+0.85)  # headroom on top for legend strip
    axf.legend(loc="upper center",ncol=3,fontsize=6,frameon=False,
               handletextpad=0.25,columnspacing=0.9,borderpad=0.1)
    axf.set_title("Immune-survival attenuates after\nHPV adjustment (HPV-driven)",fontsize=7.3)
    panel_tag(axf,"f")
    fig.suptitle("Figure 4. One HPV axis: H&E infers HPV status, and explains the HPV-driven immune-survival association",
                 fontsize=10,y=0.985)
    save(fig,"Fig4_HPV_survival")

# ==================== SUPPLEMENTARY ====================
def s1():
    base=pd.read_csv(f"{CACHE}/s1_baseline.csv"); ei=L("expr_internal")
    # baseline csv columns unknown-safe: find cell + top1000 col
    fig,axes=plt.subplots(1,2,figsize=(9,4));
    cols=list(base.columns); print("S1 baseline cols:",cols)
    # heuristic column detection
    cell_col=cols[0]
    pcc_col=[c for c in cols if "pcc" in c.lower() or "1000" in c.lower()]
    ax=axes[0]
    if pcc_col:
        pc=pcc_col[0]; cells=base[cell_col].tolist()
        newv=[ei["top1000"].get(c,np.nan) for c in cells]
        x=np.arange(len(cells)); w=0.4
        ax.bar(x-w/2,base[pc].values,w,label="Baseline (GSE103322)",color=PAL["grey"])
        # 95% CI (patient bootstrap, 1000 resamples) is available for the v2b series only; the
        # cached baseline is a single scalar per cell type with no interval, so it stays bare
        _fold=LQ("expr_internal_folds")
        if _fold:
            _lo=[_fold[c]["ci_lo"] if c in _fold else newv[i] for i,c in enumerate(cells)]
            _hi=[_fold[c]["ci_hi"] if c in _fold else newv[i] for i,c in enumerate(cells)]
            ax.bar(x+w/2,newv,w,label="Oropharyngeal ref (v2b)",
                   color=PAL["tcga"],yerr=err_from_ci(newv,_lo,_hi),capsize=CAPSIZE,error_kw=ERRKW)
        else:
            ax.bar(x+w/2,newv,w,label="Oropharyngeal ref (v2b)",color=PAL["tcga"])
        ax.set_xticks(x); ax.set_xticklabels(cells,rotation=40,ha="right")
        ax.set_ylabel("Top-1000 median PCC")
        ax.set_ylim(0,0.72)   # headroom so the legend clears the new error bars
        ax.legend(frameon=False,fontsize=7.5,loc="upper center",ncol=2,
                  columnspacing=1.2,handlelength=1.4)
        ax.set_title("a. Reference swap rescues B-cell / dendritic prediction",fontsize=8.5)
        ax.text(0.5,-0.42,"error bars: 95% CI, patient bootstrap (1000 resamples; shown for the\n"
                          "oropharyngeal reference only, the cached baseline stores one scalar per cell type)",
                transform=ax.transAxes,ha="center",va="top",fontsize=6,color="#555")
    axes[1].axis("off")
    axes[1].text(0.02,0.95,"Reference construction (B-cell repair)",fontsize=9,fontweight="bold",va="top")
    txt=("• Baseline GSE103322 atlas has only 138 B cells → BayesPrism\n"
         "  deconvolves B cells to ~0% in 93% of TCGA tumors → labels\n"
         "  near-zero → B-cell prediction collapses (PCC 0.085, 0 genes>0.4).\n\n"
         "• Fix: oropharyngeal HPV-mixed atlas (3332 B cells) + relaxed\n"
         "  gene filter (nonzero ≥0.2). B cell 0.085→0.34, Dendritic\n"
         "  0.16→0.35; T cell small accepted cost 0.47→0.40.\n\n"
         "• Single reference used uniformly for TCGA and CPTAC\n"
         "  (no per-cell-type cherry-picking).")
    axes[1].text(0.02,0.82,txt,fontsize=7,va="top",family="monospace")
    fig.suptitle("Single-cell reference choice / B-cell repair",fontsize=10,y=1.02)
    fig.tight_layout(); save(fig,"S1_reference")

def s2():
    ei=L("expr_internal"); ec=L("expr_cptac")
    fig,ax=plt.subplots(figsize=(7,4.2)); x=np.arange(len(ORDER)); w=0.38
    _ff=LQ("expr_internal_folds_full")
    _tv=[ei["top1000"][c] for c in ORDER]; _fv=[ei["full"][c] for c in ORDER]
    _terr=_ferr=None
    if _ff:
        _terr=err_from_ci(_tv,[_ff[c]["top1000_ci_lo"] for c in ORDER],[_ff[c]["top1000_ci_hi"] for c in ORDER])
        _ferr=err_from_ci(_fv,[_ff[c]["full_ci_lo"] for c in ORDER],[_ff[c]["full_ci_hi"] for c in ORDER])
    ax.bar(x-w/2,_tv,w,label="Top-1000 (selected)",color=PAL["tcga"],alpha=0.55,
           yerr=_terr,capsize=CAPSIZE,error_kw=ERRKW)
    ax.bar(x+w/2,_fv,w,label="Full-gene median (unbiased)",color=PAL["tcga"],
           yerr=_ferr,capsize=CAPSIZE,error_kw=ERRKW)
    ax.set_xticks(x); ax.set_xticklabels(ORDER,rotation=40,ha="right")
    ax.set_ylabel("Median PCC (TCGA internal)")
    ax.set_xlabel("error bars: 95% CI across 5 cross-validation folds",fontsize=7.5)
    ax.legend(frameon=False)
    ax.set_title("Full-gene median PCC, an unbiased complement to the top-1000 summary")
    fig.tight_layout(); save(fig,"S2_fullgene_pcc")

def s3():
    MV=L("multivariable")
    fig,ax=plt.subplots(figsize=(7.5,4.2))
    mcol={0:PAL["tcga"],1:PAL["grey"],2:PAL["hpv"]}; mmark={0:"o",1:"s",2:"D"}
    mlab={0:"① univariate",1:"② + clinical",2:"③ + HPV"}; off={0:0.24,1:0.0,2:-0.24}; seen=set()
    labs=[m["label"] for m in MV][::-1]
    for yi,m in enumerate(MV[::-1]):
        for mi,key in enumerate(["m1","m2","m3"]):
            v=m[key]
            if not v: continue
            y=yi+off[mi]; col=PAL["prot"] if v[0]<1 else PAL["risk"]
            ax.plot([v[1],v[2]],[y,y],color=col,lw=1.6,zorder=2)
            lab=mlab[mi] if mi not in seen else None; seen.add(mi)
            ax.scatter([v[0]],[y],s=60,c=[col],marker=mmark[mi],edgecolor="k",lw=0.5,zorder=3,label=lab)
            ax.text(v[2]+0.02,y,f"HR={v[0]:.2f}{'*' if v[3]<0.05 else ''} (n={v[4]})",va="center",fontsize=6.5,color=col)
        hs=m.get("hpv_self")
        if hs: ax.text(1.55,yi+off[2],f"[HPV term HR={hs[0]:.2f}{'*' if hs[1]<0.05 else ''}]",va="center",fontsize=6,color=PAL["hpv"])
    ax.axvline(1,color="k",ls="--",lw=0.8); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_xlim(0.5,1.95); ax.set_ylim(-0.7,len(labs)-0.3)
    ax.set_xlabel("HR per SD with 95% CI (HR<1 protective)")
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=3,frameon=False,title="model")
    ax.set_title("Multivariable Cox: immune abundance is stage-independent\nbut attenuates after HPV adjustment (HPV-driven)",fontsize=9,pad=12)
    fig.tight_layout(); save(fig,"S3_multivariable_cox")

# ==================== S4 — cross-cohort well-predicted gene overlap (Venn) ==========
def _pfmt(p):
    """Hypergeometric p as 'P < 10^-300' / 'P = 4×10^-265'."""
    if p<=0 or p<1e-300: return "$P < 10^{-300}$"
    import math
    e=int(math.floor(math.log10(p))); m=p/10**e
    return f"$P = {m:.1f}\\times10^{{{e}}}$"

def s4():
    from matplotlib_venn import venn2, venn2_circles
    ov=L("gene_overlap")
    # four robustly-predicted cell types (all hypergeometric P < 1e-80), descending overlap
    cells=["Fibroblast","Malignant","Endothelial","Macrophage"]
    fig,axes=plt.subplots(2,2,figsize=(7.4,6.2)); axes=axes.ravel()
    for ax,ct in zip(axes,cells):
        d=ov[ct]
        v=venn2(subsets=(d["tcga_only"],d["cptac_only"],d["shared"]),
                set_labels=("TCGA-HNSC","CPTAC-HNSCC"),ax=ax)
        for pid,col in [("10",PAL["tcga"]),("01",PAL["cptac"]),("11","#8B7BA8")]:
            patch=v.get_patch_by_id(pid)
            if patch: patch.set_color(col); patch.set_alpha(0.62); patch.set_edgecolor("none")
        venn2_circles(subsets=(d["tcga_only"],d["cptac_only"],d["shared"]),ax=ax,lw=0.8,color="#555")
        for lab in v.set_labels:
            if lab: lab.set_fontsize(7.5); lab.set_fontweight("bold")
        for lab in v.subset_labels:
            if lab: lab.set_fontsize(8)
        jac=d["shared"]/(d["tcga_only"]+d["cptac_only"]+d["shared"])
        ax.set_title(f"{ct}\n{_pfmt(d['p'])}   (Jaccard {jac:.2f})",fontsize=8.5)
    fig.suptitle("Supplementary Figure S4. Robustly-predicted genes (PCC > 0.4) overlap between cohorts\n"
                 "TCGA-HNSC cross-validation vs CPTAC-HNSCC external validation (two-sided hypergeometric test)",
                 fontsize=9,y=1.02)
    fig.tight_layout(); save(fig,"S4_gene_overlap")

# ==================== FIGURE 5 — GO enrichment of cross-cohort robustly-predicted genes ====
def fig5():
    import re, seaborn as sns
    M=pd.read_csv(f"{CACHE}/fig5_go_matrix.csv",index_col=0)
    M.index=[re.sub(r"\s*\(GO:\d+\)$","",p) for p in M.index]      # drop GO id for readability
    g=sns.clustermap(M,cmap="RdBu_r",vmin=0,vmax=1,figsize=(7.4,5.8),
                     linewidths=0.5,linecolor="#e5e5e5",
                     dendrogram_ratio=(0.13,0.17),col_cluster=True,row_cluster=True,
                     cbar_pos=(0.02,0.82,0.03,0.15),
                     cbar_kws={"label":"Scaled odds ratio","ticks":[0,0.5,1]})
    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
    plt.setp(g.ax_heatmap.get_yticklabels(),fontsize=7.2,rotation=0)
    plt.setp(g.ax_heatmap.get_xticklabels(),fontsize=8.5,rotation=35,ha="right",fontweight="bold")
    g.ax_heatmap.yaxis.set_ticks_position("right"); g.ax_heatmap.yaxis.set_label_position("right")
    g.figure.suptitle("Figure 5. GO(BP) enrichment of cross-cohort robustly-predicted genes\n"
                      "(PCC > 0.4 in both TCGA-HNSC CV and CPTAC-HNSCC); odds ratio scaled per cell type",
                      fontsize=9.5,y=1.02,x=0.55)
    for ext in ("pdf","png"): g.savefig(f"{OUT}/Fig5_GO_enrichment.{ext}",dpi=300,bbox_inches="tight")
    plt.close(g.figure); print("saved Fig5_GO_enrichment")

if __name__=="__main__":
    import sys
    todo=sys.argv[1:] or ["fig2","fig3","fig4","fig5","s1","s2","s3"]  # s4() optional all-8 supp
    for t in todo:
        try: globals()[t]()
        except Exception as e:
            import traceback; print(f"!! {t} failed:",e); traceback.print_exc()
