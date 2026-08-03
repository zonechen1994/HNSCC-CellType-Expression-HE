#!/usr/bin/env python3
"""Fig 4 — 临床应用(HPV 分类 + 预后分层 merge)。
上行 HPV: (a)ROC (b)per-cell AUC (c)HPV±免疫丰度
下行 预后: (d-f)T/B/Fibro 代表KM  右(g)多因素forest。"""
import os, json, numpy as np
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd
plt.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
    "font.size":8,"axes.linewidth":0.8,"axes.titlesize":9,"axes.labelsize":8.5,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
PAL={"prot":"#2c7fb8","risk":"#d95f0e","tcga":"#4477AA","cptac":"#CC6677","hancock":"#228833","hpv":"#AA3377","grey":"#999999"}
def tag(ax,s): ax.text(-0.2,1.07,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top",ha="left")
def pf(p): return "<0.001" if p<0.001 else f"{p:.3f}"
C="figures/paper/cache"
roc=np.load(f"{C}/hpv_roc.npz"); pca=json.load(open(f"{C}/hpv_percell_auc.json")); byhpv=json.load(open(f"{C}/abundance_by_hpv.json"))
PR=json.load(open(f"{C}/prognosis_pred.json"))

fig=plt.figure(figsize=(13.6,10.4))
gs=gridspec.GridSpec(3,4,figure=fig,hspace=0.85,wspace=0.48,width_ratios=[1,1,1,1.05],bottom=0.075)

# ===== 上行 HPV =====
axa=fig.add_subplot(gs[0,0])
axa.plot(roc["fpr_i"],roc["tpr_i"],color=PAL["tcga"],lw=1.7,
         label=f"TCGA int. AUC={float(roc['int_auc']):.2f} ({float(roc['int_lo']):.2f}-{float(roc['int_hi']):.2f})")
axa.plot(roc["fpr_e"],roc["tpr_e"],color=PAL["hancock"],lw=1.7,
         label=f"HANCOCK ext. AUC={float(roc['ext_auc']):.2f} ({float(roc['ext_lo']):.2f}-{float(roc['ext_hi']):.2f})")
axa.plot([0,1],[0,1],color="grey",ls="--",lw=0.8); axa.set_xlabel("1 − specificity"); axa.set_ylabel("Sensitivity")
axa.set_title("HPV classification\nfrom predicted expression",fontsize=8)
axa.legend(loc="lower right",frameon=False,fontsize=5.6,title="AUC (bootstrap 95% CI)",title_fontsize=5.8)
axa.get_legend().get_title().set_ha("left"); tag(axa,"a")
axb=fig.add_subplot(gs[0,1])
ext=pca["external"]; ob=sorted(ext,key=lambda c:-ext[c]); x=np.arange(len(ob))
# 每型 AUC 的患者级 bootstrap 95% CI (2000 resamples, seed 0), 见 hpv_percell_ci.py
try: pcci=json.load(open(f"{C}/hpv_percell_auc_ci.json"))
except FileNotFoundError: pcci={}
yv=[ext[c] for c in ob]
err=np.array([[max(0.0,ext[c]-pcci[c]["lo"]) if c in pcci else 0.0 for c in ob],
              [max(0.0,pcci[c]["hi"]-ext[c]) if c in pcci else 0.0 for c in ob]])
axb.bar(x,yv,color=PAL["hpv"],width=0.72,
        yerr=err if err.any() else None,capsize=2.5,
        error_kw=dict(lw=0.8,ecolor="0.25"))
for _i in range(len(ob)):
    axb.text(x[_i], yv[_i]+(err[1][_i] if err.any() else 0)+0.02, f"{yv[_i]:.2f}", ha="center", fontsize=5.8, color="0.2")
axb.axhline(0.5,color="grey",ls="--",lw=0.8); axb.set_ylim(0,1)
axb.set_xticks(x); axb.set_xticklabels(ob,rotation=42,ha="right",fontsize=6.5); axb.set_ylabel("External HPV AUC")
axb.set_title("HPV signal is\necosystem-wide",fontsize=8)
if err.any(): axb.set_xlabel("error bars: patient bootstrap 95% CI (2000 resamples)",fontsize=5.6,color="0.25",labelpad=2)
tag(axb,"b")
axc=fig.add_subplot(gs[0,2])
cells=["T cell","Dendritic"]; positions=[]; data=[]; cols=[]
for i,ct in enumerate(cells):
    data.append(byhpv[ct]["neg"]); positions.append(i*3+0.6); cols.append(PAL["cptac"])
    data.append(byhpv[ct]["pos"]); positions.append(i*3+1.4); cols.append(PAL["tcga"])
bp=axc.boxplot(data,positions=positions,widths=0.65,patch_artist=True,showfliers=False)
for patch,c in zip(bp["boxes"],cols): patch.set_facecolor(c); patch.set_alpha(0.75)
for med in bp["medians"]: med.set_color("black")
axc.set_xticks([0.6,1.4,3.6,4.4]); axc.set_xticklabels(["−","+","−","+"],fontsize=7)
ylo,yhi=axc.get_ylim(); axc.set_ylim(ylo,yhi*1.18)
for i,ct in enumerate(cells): axc.text(i*3+1.0,-0.15,ct,transform=axc.get_xaxis_transform(),ha="center",fontsize=7,fontweight="bold")
axc.set_ylabel("Predicted abundance"); axc.set_title("HPV+ tumours are\nlymphoid/DC-rich",fontsize=8); tag(axc,"c")


def at_risk_table(ax, T, E, hi, cols, labels, ticks, fs=5.3):
    """在 KM 图下方标注各时点仍在随访的人数, 并在图例中给出 n 与事件数。
    风险人数表是 KM 图的常规要求 (REMARK)。"""
    ax.set_xticks(ticks)
    off = -0.235
    ax.text(0.0, off + 0.055, "No. at risk (cumulative deaths)", transform=ax.transAxes,
            fontsize=fs, fontweight="bold", ha="left", va="top", color="0.25")
    for i, (m, lab, col) in enumerate(zip([hi, ~hi], labels, cols)):
        y = off - i * 0.075
        for t in ticks:
            n = int((T[m] >= t).sum()); ev = int(E[m][T[m] <= t].sum())
            # 首个刻度左对齐, 否则居中的数字会探到轴线左侧压住行标签
            ax.text(t, y, f"{n} ({ev})", transform=ax.get_xaxis_transform(),
                    fontsize=fs, ha="left" if t == ticks[0] else "center", va="top", color=col)
        ax.text(-0.015, y, lab, transform=ax.transAxes, fontsize=fs,
                ha="right", va="top", color=col, fontweight="bold")
    ax.text(0.5, off - 2 * 0.075 - 0.045, "shaded bands: 95% CI", transform=ax.transAxes,
            fontsize=fs - 0.3, ha="center", va="top", color="0.45")
    return ax

# ===== 下行 预后 代表 KM (选更显著队列) =====
KMSEL=[("T cell","TCGA","d",1),("B cell","TCGA","e",1),("Fibroblast","TCGA","f",1),
       ("T cell","HANCOCK","g",2),("B cell","HANCOCK","h",2),("Fibroblast","HANCOCK","i",2)]
for k,(cell,coh,lt,row) in enumerate(KMSEL):
    k=k%3
    ax=fig.add_subplot(gs[row,k]); d=PR["cells"][cell][coh]; km=d["km"]
    T=np.array(km["time"]); E=np.array(km["event"]); hi=np.array(km["hi"],bool)
    kmf=KaplanMeierFitter()
    for m,lab,col in [(hi,"high",PAL["prot"]),(~hi,"low",PAL["risk"])]:
        kmf.fit(T[m],E[m],label=f"{lab}")
        kmf.plot_survival_function(ax=ax,color=col,ci_show=True,ci_alpha=0.13,
                                   ci_force_lines=False,lw=1.5)
    _nh,_eh=int(hi.sum()),int(E[hi].sum()); _nl,_el=int((~hi).sum()),int(E[~hi].sum())
    _cx=CoxPHFitter().fit(pd.DataFrame({"T":T,"E":E,"z":hi.astype(int)}),"T","E").summary.loc["z"]
    _khr=np.exp(_cx["coef"]); _klo=np.exp(_cx["coef lower 95%"]); _khi=np.exp(_cx["coef upper 95%"])
    ax.set_title(f"{cell}, {coh}\nHR={_khr:.2f} (95% CI {_klo:.2f}\u2013{_khi:.2f}), p={pf(d['logrank_p'])}",fontsize=6.8)
    ax.set_xlabel("Months",fontsize=7.5,labelpad=1.5); ax.set_ylim(0,1.02)
    h,l=ax.get_legend_handles_labels()
    keep=[(hh,ll) for hh,ll in zip(h,l) if not str(ll).startswith("_")]
    _lab={"high":f"high  n={_nh}, {_eh} deaths","low":f"low   n={_nl}, {_el} deaths"}
    ax.legend([hh for hh,_ in keep],[_lab.get(str(ll),str(ll)) for _,ll in keep],
              loc="lower left",frameon=False,fontsize=5.8)
    _tmax=int(T.max()); _step=50*max(1,int(np.ceil(_tmax/50/4)))
    _tk=list(range(0,_tmax+1,_step))
    at_risk_table(ax,T,E,hi,[PAL["prot"],PAL["risk"]],["high","low"],_tk)
    if k==0: ax.set_ylabel("Overall survival\n(" + coh + ")",fontsize=8)
    tag(ax,lt)

# ===== forest (右列跨三行) =====
axg=fig.add_subplot(gs[:,3]); rows=[]
for cell in ["T cell","B cell","Fibroblast"]:
    for coh,col in [("TCGA",PAL["tcga"]),("HANCOCK",PAL["hancock"])]:
        mv=PR["cells"][cell][coh]["mv"]
        if mv: rows.append((f"{cell}\n{coh}",mv["hr"],mv["lo"],mv["hi"],mv["p"]))
rows=rows[::-1]; y=np.arange(len(rows))
for i,(lab,hr,lo,hi,p) in enumerate(rows):
    c=PAL["prot"] if hr<1 else PAL["risk"]
    axg.plot([lo,hi],[i,i],color=c,lw=1.6,zorder=2); axg.scatter([hr],[i],s=44,color=c,zorder=3,edgecolor="k",lw=0.4)
    axg.text(hi*1.04,i,f"{hr:.2f}{'*' if p<0.05 else ''}",va="center",fontsize=6.4,color=c)
axg.axvline(1,color="k",ls="--",lw=0.8); axg.set_yticks(y); axg.set_yticklabels([r[0] for r in rows],fontsize=6.6)
axg.set_xscale("log"); axg.xaxis.set_minor_locator(plt.NullLocator())
axg.set_xticks([0.3,0.5,1,2]); axg.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FixedFormatter(["0.3","0.5","1","2"]))
axg.set_xlim(0.28,2.6); axg.set_xlabel("Multivariable HR with 95% CI\n(adj. clinical) HR<1 protective",fontsize=7.3)
axg.set_title("Prognosis: predicted abundance\nstratifies survival (TCGA+HANCOCK)",fontsize=7.6); tag(axg,"j")

fig.suptitle("Figure 4. Clinical applications: H&E-inferred expression classifies HPV status and stratifies prognosis",
             fontsize=10.5,y=0.995,fontweight="bold")
for e in ("pdf","png"): fig.savefig(f"figures/paper/Fig4_clinical.{e}",dpi=300,bbox_inches="tight")
plt.close(fig); print("saved Fig4_clinical")
