#!/usr/bin/env python3
"""预后主图(H&E预测丰度 → 生存): T/B/Fibroblast × TCGA/HANCOCK 的 KM + 多因素 forest。
附图: 其余5细胞 forest + Macrophage 单队列陷阱。读 cache/prognosis_pred.json。"""
import os, json, numpy as np
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd
plt.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Arial","Helvetica"],
    "font.size":8,"axes.linewidth":0.8,"axes.titlesize":9,"axes.labelsize":8.5,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
PAL={"prot":"#2c7fb8","risk":"#d95f0e","tcga":"#4477AA","cptac":"#CC6677",
     "hancock":"#228833","hpv":"#AA3377","grey":"#999999"}
C=json.load(open("figures/paper/cache/prognosis_pred.json"))
def tag(ax,s): ax.text(-0.20,1.06,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top",ha="left")
def pfmt(p): return "<0.001" if p<0.001 else f"{p:.3f}"

def uni_ci(cell,cohort):
    """单因素 Cox(二分丰度)的 HR 95% CI, 由缓存的 KM 生存数据重算(HR 与缓存 uni_hr 完全一致)。"""
    v=C["cells"].get(cell,{}).get(cohort)
    if not v or not v.get("km"): return None
    km=v["km"]
    df=pd.DataFrame({"T":km["time"],"E":km["event"],"x":np.array(km["hi"],int)})
    if df["x"].nunique()<2 or df["E"].sum()==0: return None
    s=CoxPHFitter().fit(df,"T","E").summary.loc["x"]
    return float(np.exp(s["coef lower 95%"])),float(np.exp(s["coef upper 95%"]))

def km_panel(ax,cell,cohort,letter=None):
    d=C["cells"][cell][cohort]; km=d["km"]
    T=np.array(km["time"]); E=np.array(km["event"]); hi=np.array(km["hi"],bool)
    kmf=KaplanMeierFitter()
    for m,lab,col in [(hi,"high",PAL["prot"]),(~hi,"low",PAL["risk"])]:
        if m.sum()==0: continue
        kmf.fit(T[m],E[m],label=f"{lab}  n={int(m.sum())}, {int(E[m].sum())} deaths")
        kmf.plot_survival_function(ax=ax,color=col,lw=1.6,ci_show=True,ci_alpha=0.15,ci_no_lines=True,ci_legend=False)
    _cx=CoxPHFitter().fit(pd.DataFrame({"T":T,"E":E,"z":hi.astype(int)}),"T","E").summary.loc["z"]
    hr=np.exp(_cx["coef"]); _ulo=np.exp(_cx["coef lower 95%"]); _uhi=np.exp(_cx["coef upper 95%"]); p=d["logrank_p"]
    ax.set_title(f"{cohort}, {cell}\nHR={hr:.2f} (95% CI {_ulo:.2f}\u2013{_uhi:.2f}), logrank p={pfmt(p)}",fontsize=7.0)
    ax.set_xlabel("Months",fontsize=7.5,labelpad=1.5); ax.set_ylim(0,1.02)
    ax.legend(loc="lower left",frameon=False,fontsize=5.9)
    # 风险人数表 (REMARK 要求): 各时点仍在随访的人数
    _tmax=int(T.max()); _step=50*max(1,int(np.ceil(_tmax/50/4)))
    ticks=list(range(0,_tmax+1,_step))
    ax.set_xticks(ticks); off=-0.30; fs=5.3
    ax.text(0.0,off+0.055,"No. at risk (cumulative deaths)",transform=ax.transAxes,fontsize=fs,
            fontweight="bold",ha="left",va="top",color="0.25")
    for i,(m,lab,col) in enumerate(zip([hi,~hi],["high","low"],[PAL["prot"],PAL["risk"]])):
        y=off-i*0.075
        for t in ticks:
            ax.text(t,y,f"{int((T[m]>=t).sum())} ({int(E[m][T[m]<=t].sum())})",transform=ax.get_xaxis_transform(),
                    fontsize=fs,ha="left" if t==ticks[0] else "center",va="top",color=col)
        ax.text(-0.015,y,lab,transform=ax.transAxes,fontsize=fs,ha="right",va="top",
                color=col,fontweight="bold")
    ax.text(0.5,off-2*0.075-0.045,"shaded bands: 95% CI",transform=ax.transAxes,
            fontsize=fs-0.3,ha="center",va="top",color="0.45")
    if letter: tag(ax,letter)

# ==================== MAIN FIGURE ====================
PASS=["T cell","B cell","Fibroblast"]
fig=plt.figure(figsize=(11.6,9.2))
gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.95,wspace=0.34,width_ratios=[1,1,1.15],bottom=0.075)
letters=iter("abcdef")
for i,cell in enumerate(PASS):
    axT=fig.add_subplot(gs[i,0]); km_panel(axT,cell,"TCGA",next(letters))
    if i==0: axT.set_ylabel("Overall survival",fontsize=8)
    axH=fig.add_subplot(gs[i,1]); km_panel(axH,cell,"HANCOCK",next(letters))

# forest (col2, span all rows): 多因素 HR ± CI, 3细胞 × 2队列
axf=fig.add_subplot(gs[:,2])
rows=[]
for cell in PASS:
    for coh,col in [("TCGA",PAL["tcga"]),("HANCOCK",PAL["hancock"])]:
        mv=C["cells"][cell][coh]["mv"]
        if mv: rows.append((f"{cell}\n{coh}",mv["hr"],mv["lo"],mv["hi"],mv["p"],col))
rows=rows[::-1]; y=np.arange(len(rows))
for i,(lab,hr,lo,hi,p,col) in enumerate(rows):
    c = PAL["prot"] if hr<1 else PAL["risk"]
    axf.plot([lo,hi],[i,i],color=c,lw=1.6,zorder=2)
    axf.scatter([hr],[i],s=46,color=c,zorder=3,edgecolor="k",lw=0.4)
    axf.text(hi*1.04,i,f"{hr:.2f}{'*' if p<0.05 else ''}",va="center",fontsize=6.4,color=c)
axf.axvline(1,color="k",ls="--",lw=0.8)
axf.set_yticks(y); axf.set_yticklabels([r[0] for r in rows],fontsize=6.6)
axf.set_xscale("log"); axf.xaxis.set_minor_locator(plt.NullLocator())
axf.set_xticks([0.3,0.5,1,2]); axf.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FixedFormatter(["0.3","0.5","1","2"]))
axf.set_xlabel("Multivariable HR with 95% CI (adj. clinical)\nHR<1 protective",fontsize=7.5)
axf.set_title("Adjusted prognosis (both cohorts)",fontsize=8)
axf.set_xlim(0.28,2.6); tag(axf,"g")
fig.suptitle("H&E-predicted immune/stromal abundance stratifies survival, replicated in TCGA and HANCOCK",
             fontsize=10.5,y=0.995,fontweight="bold")
fig.text(0.5,0.962,"Shaded bands in (a to f): 95% CI of the survival function; whiskers in (g): 95% CI of the hazard ratio",
         ha="center",va="top",fontsize=7.4,color="0.3")
for ext in ("pdf","png"): fig.savefig(f"figures/paper/Fig_prognosis.{ext}",dpi=300,bbox_inches="tight")
plt.close(fig); print("saved Fig_prognosis")

# ==================== SUPP ====================
from matplotlib.lines import Line2D
ev=C.get("events",{})
figS=plt.figure(figsize=(8.6,7.6))
gsS=gridspec.GridSpec(2,2,figure=figS,hspace=0.55,wspace=0.42,height_ratios=[1.15,1])
# (a) forest 全8细胞 uni HR TCGA vs HANCOCK
axa=figS.add_subplot(gsS[0,0])
allc=list(C["forest"]); lab=[f["ct"] for f in allc][::-1]; yy=np.arange(len(lab))
for i,f in enumerate(allc[::-1]):
    for coh,hr,p,col,off in [("TCGA",f["TCGA_hr"],f["TCGA_p"],PAL["tcga"],0.16),
                             ("HANCOCK",f["HANCOCK_hr"],f["HANCOCK_p"],PAL["hancock"],-0.16)]:
        if hr is None: continue
        ci=uni_ci(f["ct"],coh)
        if ci: axa.plot([ci[0],ci[1]],[i+off,i+off],color=col,lw=1.6,zorder=2,solid_capstyle="butt")
        mk="o" if (p is not None and p<0.05) else "x"
        axa.scatter([hr],[i+off],s=36,color=col,marker=mk,zorder=3,edgecolor="k" if mk=="o" else col,lw=0.4)
axa.axvline(1,color="k",ls="--",lw=0.8); axa.set_yticks(yy); axa.set_yticklabels(lab,fontsize=7)
axa.set_xscale("log"); axa.xaxis.set_minor_locator(plt.NullLocator())
axa.set_xticks([0.3,0.5,1,2,3]); axa.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FixedFormatter(["0.3","0.5","1","2","3"]))
axa.set_xlim(0.30,3.2); axa.set_ylim(-1.5,len(lab)-0.4)
axa.set_xlabel("Univariate HR (dichotomized), whiskers 95% CI\n● sig(p<0.05)  ✕ ns",fontsize=7)
axa.set_title("All 8 cell types, only B/T/Fibroblast\nreplicate in BOTH cohorts",fontsize=7.8)
axa.legend(handles=[Line2D([],[],marker='o',color=PAL["tcga"],ls='',label=f'TCGA ({ev.get("TCGA","?")} events)'),
                    Line2D([],[],marker='o',color=PAL["hancock"],ls='',label=f'HANCOCK ({ev.get("HANCOCK","?")} events)')],
           loc="lower right",frameon=False,fontsize=6.2); tag(axa,"a")
# (b) CPTAC θ null forest (备份: HPV−, 事件少)
axb=figS.add_subplot(gsS[0,1])
cpt=C["cptac_theta"]; cells3=["Fibroblast","B cell","T cell"]; y3=np.arange(len(cells3))
for i,ct in enumerate(cells3):
    d=cpt[ct]; axb.plot([d["lo"],d["hi"]],[i,i],color=PAL["grey"],lw=1.6,zorder=2)
    axb.scatter([d["hr"]],[i],s=46,color=PAL["grey"],zorder=3,edgecolor="k",lw=0.4)
    axb.text(d["hi"]*1.03,i,f"HR={d['hr']:.2f} (95% CI {d['lo']:.2f}\u2013{d['hi']:.2f})\np={d['p']:.2f}",va="center",fontsize=5.5,color="#555")
axb.axvline(1,color="k",ls="--",lw=0.8); axb.set_yticks(y3); axb.set_yticklabels(cells3,fontsize=7.2)
axb.set_xscale("log"); axb.xaxis.set_minor_locator(plt.NullLocator())
axb.set_xticks([0.5,1,2]); axb.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FixedFormatter(["0.5","1","2"]))
axb.set_xlim(0.4,2.6)
axb.set_xlabel("CPTAC ground-truth θ\ncontinuous HR per SD, whiskers 95% CI",fontsize=7)
axb.set_title(f"CPTAC (HPV−, {ev.get('CPTAC','?')} events):\nno association even with true abundance",fontsize=7.6)
tag(axb,"b")
# (d,e) Macrophage 单队列陷阱
for k,(coh,letter) in enumerate([("TCGA","c"),("HANCOCK","d")]):
    axm=figS.add_subplot(gsS[1,k]); km_panel(axm,"Macrophage",coh,letter)
    if k==0: axm.set_ylabel("Overall survival",fontsize=8)
figS.suptitle("Supplementary: cross-cohort replication filter, CPTAC power/HPV, and the single-cohort trap",
              fontsize=9.8,y=0.99,fontweight="bold")
figS.text(0.5,0.955,"Forest whiskers in (a) and (b): 95% CI of the hazard ratio; shaded bands in (c) and (d): 95% CI of the survival function",
          ha="center",va="top",fontsize=7.2,color="0.3")
for ext in ("pdf","png"): figS.savefig(f"figures/paper/S_prognosis_other.{ext}",dpi=300,bbox_inches="tight")
plt.close(figS); print("saved S_prognosis_other")
