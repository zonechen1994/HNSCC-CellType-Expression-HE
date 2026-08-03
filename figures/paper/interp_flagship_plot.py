#!/usr/bin/env python3
"""旗舰生物学可解释性主图(v2): 分子身份 + 通路验证(无空间叠加, 空间归 Fig4)。
(a) 细胞型×分子程序 块对角(8型全列, B/DC 基因太少标灰)
(b) 通路活性验证: 50 Hallmark 预测vs实测 r 排序(CPTAC外部)
(c) top 通路示例散点  (d) 能力边界: 形态表征 vs 沉默"""
import os, json, numpy as np, pandas as pd
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from scipy import stats
plt.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
    "font.size":8,"axes.linewidth":0.8,"axes.titlesize":9,"axes.labelsize":8.5,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
def tag(ax,s): ax.text(-0.15,1.06,s,transform=ax.transAxes,fontsize=13,fontweight="bold",va="top",ha="left")

RISK="#d95f0e"   # 与 fig_plot.py / protein_ext_plot.py 回归带同色

def rho_ci(rho,n):
    """Spearman r 的 Fisher z 95% CI (se=1.06/sqrt(n-3))。"""
    r=np.asarray(rho,float); se=1.06/np.sqrt(n-3); z=np.arctanh(np.clip(r,-0.999999,0.999999))
    return np.tanh(z-1.96*se),np.tanh(z+1.96*se)

def mean_ci(v):
    """均值的 t 分布 95% CI 半宽。"""
    v=np.asarray(v,float); n=len(v)
    return 0.0 if n<2 else float(stats.t.ppf(0.975,n-1)*v.std(ddof=1)/np.sqrt(n))

def fit_band(ax,x,y,color=RISK,alpha=0.12,lw=1.2):
    """回归线 + 回归线的 t 分布 95% 置信带 (yhat ± t*s*sqrt(1/n+(x-xbar)^2/Sxx))。"""
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(x)
    z=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),200); yh=np.polyval(z,xs)
    resid=y-np.polyval(z,x); s=np.sqrt((resid**2).sum()/max(n-2,1))
    xb=x.mean(); Sxx=((x-xb)**2).sum()
    se=s*np.sqrt(1.0/n+(xs-xb)**2/Sxx); t=stats.t.ppf(0.975,max(n-2,1))
    ax.fill_between(xs,yh-t*se,yh+t*se,color=color,alpha=max(alpha,0.30),linewidth=0,zorder=1)
    for _yb in (yh-t*se, yh+t*se):   # 带宽随 n 收缩, 加细边界线保证可见
        ax.plot(xs,_yb,color=color,lw=0.45,alpha=0.85,zorder=1)
    ax.plot(xs,yh,color=color,lw=lw,zorder=2)

MOL=json.load(open("figures/paper/cache/interp_molecular.json"))
PATH=json.load(open("figures/paper/cache/interp_pathway.json"))
CATCOL={"proliferation":"#8856a7","stroma/EMT":"#e6550d","immune":"#31a354","hypoxia/stress":"#3182bd",
        "signaling":"#756bb1","metabolic":"#636363","other":"#bdbdbd"}

fig=plt.figure(figsize=(12.6,7.4))
gs=gridspec.GridSpec(2,2,figure=fig,height_ratios=[1,1],width_ratios=[1.15,1],hspace=0.5,wspace=0.34)

# ---- (a) 通路活性验证: 50 Hallmark r 排序 (左列跨两行) ----
axb=fig.add_subplot(gs[:,0])
rr=sorted(PATH["pathways"],key=lambda r:r["rho"]); y=np.arange(len(rr))
rv=np.array([r["rho"] for r in rr]); lo_b,hi_b=rho_ci(rv,PATH["n"])
axb.barh(y,rv,color=[CATCOL[r["cat"]] for r in rr],height=0.82,
         xerr=[rv-lo_b,hi_b-rv],error_kw=dict(lw=0.4,ecolor="0.55",capsize=0,zorder=1))
axb.set_yticks([]); axb.axvline(0.4,color="grey",ls="--",lw=0.8); axb.set_xlim(-0.32,0.96)
axb.set_xlabel(f"Predicted vs measured pathway-activity r (CPTAC, n={PATH['n']})\nerror bars: 95% CI (Fisher z)",fontsize=7.5)
axb.set_ylim(-1,len(rr))
# 干净 top/bottom 注释(不逐条叠)
axb.annotate("top: Allograft rej. · EMT · Angiogenesis\n(immune / stroma, r 0.80 to 0.85)",
             xy=(0.93,len(rr)-2),xytext=(0.28,len(rr)-9),fontsize=6,color="#333",
             bbox=dict(fc="white",ec="none",alpha=0.82,pad=1.2),zorder=6,
             arrowprops=dict(arrowstyle="->",color="#888",lw=0.8))
axb.annotate("bottom: OxPhos · ROS · p53\n(metabolic, morphologically silent)",
             xy=(0.13,1),xytext=(0.30,7),fontsize=6,color="#333",
             bbox=dict(fc="white",ec="none",alpha=0.82,pad=1.2),zorder=6,
             arrowprops=dict(arrowstyle="->",color="#888",lw=0.8))
axb.legend(handles=[Patch(fc=CATCOL[c],label=c) for c in ["immune","stroma/EMT","proliferation","signaling","hypoxia/stress","metabolic"]],
           loc="center right",frameon=True,framealpha=0.82,edgecolor="none",fontsize=6,handlelength=1.0)
axb.set_title("Pathway-activity validation: H&E recovers\nfunctional program activity (50 Hallmarks)",fontsize=8.2); tag(axb,"a")

# ---- (c) top 通路示例散点 ----
sub=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,1],wspace=0.18)
ex=PATH["examples"]; names=list(ex.keys())[:3]
for k,nm in enumerate(names):
    ax=fig.add_subplot(sub[0,k]); e=ex[nm]; p=np.array(e["pred"]); o=np.array(e["true"])
    ax.scatter(p,o,s=9,alpha=0.5,color="#e6550d" if("Mesench"in nm or"Angio"in nm)else"#31a354",edgecolors="none")
    fit_band(ax,p,o,lw=1.2)
    short=nm.replace(" Signaling","").replace("Epithelial Mesenchymal Transition","EMT").replace(" Response","")[:16]
    _elo,_ehi=rho_ci(e['rho'],PATH['n']); _ep=stats.spearmanr(e['pred'],e['true'])[1]
    _eps=("P < 1e-300" if not (_ep>0) else (f"P = {_ep:.3f}" if _ep>=0.001 else "P = "+f"{_ep:.0e}".replace("e-0","e-").replace("e+0","e+")))
    ax.set_title(f"{short}\nr={e['rho']:.2f} ({_elo:.2f}\u2013{_ehi:.2f})\n{_eps}",fontsize=6.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("predicted\nline: OLS fit, shaded: 95% CI" if k==1 else "predicted",fontsize=6.3)
    if k==0: ax.set_ylabel("measured",fontsize=6.5); tag(ax,"c")

# ---- (b) 能力边界 ----
axd=fig.add_subplot(gs[0,1])
cats=["immune","stroma/EMT","proliferation","signaling","hypoxia/stress","metabolic"]
valsd=[[r["rho"] for r in PATH["pathways"] if r["cat"]==c] for c in cats]
means=[np.mean(v) for v in valsd]; errs=[mean_ci(v) for v in valsd]; ncat=[len(v) for v in valsd]
xd=np.arange(len(cats)); axd.bar(xd,means,color=[CATCOL[c] for c in cats],width=0.72,
                                 yerr=errs,capsize=2.5,error_kw=dict(lw=0.8,ecolor="0.25"))
axd.axhline(0.4,color="grey",ls="--",lw=0.8)
for i,(m,e) in enumerate(zip(means,errs)): axd.text(i,m+e+0.02,f"{m:.2f}",ha="center",fontsize=6.6)
axd.set_xticks(xd); axd.set_xticklabels([f"{l}\n(n={n})" for l,n in zip(
    ["immune","stroma/\nEMT","prolif.","signaling","hypoxia/\nstress","metabolic"],ncat)],fontsize=6.8)
axd.set_ylabel("Mean pathway r",fontsize=7.5); axd.set_ylim(0,1.0)
axd.set_title("Capability boundary: morphologically-manifest programs\npredict; morphologically-silent metabolic states do not\n(error bars: 95% CI across pathways in the category)",fontsize=7.8); tag(axd,"b")

fig.suptitle("Pathway-activity validation: H&E recovers functional program activity and its capability boundary",
             fontsize=10.2,y=0.995,fontweight="bold")
for e in ("pdf","png"): fig.savefig(f"figures/paper/Fig_molecular_flagship.{e}",dpi=300,bbox_inches="tight")
plt.close(fig); print("saved Fig_molecular_flagship v2")
