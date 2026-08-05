#!/usr/bin/env python3
"""渲染 CPTAC-HNSCC 蛋白组外部验证 supplementary figure (S_protein)。
读 cache/protein_ext.json,风格对齐 fig_plot.py,打磨为出版级 supp 图。"""
import os, json, numpy as np
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde, pearsonr, t as tdist
plt.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Arial","Helvetica"],
    "font.size":8,"axes.linewidth":0.8,"axes.titlesize":9,"axes.labelsize":8.5,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
CACHE="figures/paper/cache"; OUT="figures/paper"
PAL={"prot":"#2c7fb8","risk":"#d95f0e","tcga":"#4477AA","cptac":"#CC6677",
     "hancock":"#228833","hpv":"#AA3377","grey":"#999999"}
# 分层递进配色(灰->浅蓝->中蓝->强调红): 表示"条件越严 -> 信号越强"
LADDER=["#b9b9b9","#9ecae1","#4292c6","#08519c"]
C=json.load(open(f"{CACHE}/protein_ext.json"))
NP=C["n_patients"]
def tag(ax,s):
    ax.text(-0.20,1.07,s,transform=ax.transAxes,fontsize=13,fontweight="bold",va="top",ha="left")
def rbox(ax,x,y,txt,fc="#ffffff",ec="#bbbbbb",fs=7):
    ax.text(x,y,txt,transform=ax.transAxes,fontsize=fs,va="top",ha="left",
            bbox=dict(boxstyle="round,pad=0.32",fc=fc,ec=ec,lw=0.7))
def fit_ci(x,y,npts=120):
    """OLS fit + t-based 95% CI of the fitted line: yhat ± t(0.975,n-2)·s·sqrt(1/n+(x-x̄)²/Sxx)."""
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(x)
    z=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),npts); yh=np.polyval(z,xs)
    resid=y-np.polyval(z,x); s=np.sqrt((resid**2).sum()/(n-2))
    xb=x.mean(); sxx=((x-xb)**2).sum()
    se=s*np.sqrt(1.0/n+(xs-xb)**2/sxx)
    tt=tdist.ppf(0.975,n-2)
    return xs,yh,yh-tt*se,yh+tt*se
def boot_median_ci(v,nb=2000,seed=0):
    """Gene-level bootstrap 95% CI of the median (fixed seed, 2000 resamples)."""
    v=np.asarray(v,float); v=v[~np.isnan(v)]; n=len(v)
    rng=np.random.RandomState(seed)
    bs=np.median(v[rng.randint(0,n,(nb,n))],axis=1)
    return float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))

fig=plt.figure(figsize=(12.8,8.4))
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.50,wspace=0.36,height_ratios=[1,1])

# ================= (a) 天花板分布 =================
axa=fig.add_subplot(gs[0,0])
cc=np.array(C["ceiling_all"]); med=C["ceil_median"]
n,bins,patches=axa.hist(cc,bins=42,edgecolor="white",linewidth=0.3)
# 用渐变给柱上色(越靠右越蓝)
norm=(0.5*(bins[:-1]+bins[1:])-cc.min())/(cc.max()-cc.min())
cmap=plt.cm.Blues
for p,t in zip(patches,norm): p.set_facecolor(cmap(0.25+0.6*t))
axa.axvline(med,color=PAL["risk"],lw=1.4,ls="--")
axa.text(med+0.02,axa.get_ylim()[1]*0.93,f"median {med:.2f}",color=PAL["risk"],fontsize=7.2,fontweight="bold")
axa.axvline(0.4,color="#555",lw=0.9,ls=":")
axa.text(0.405,axa.get_ylim()[1]*0.985,f"{(cc>0.4).mean()*100:.0f}% of genes > 0.4",
         fontsize=6.4,color="#555",ha="left",va="top")
# 标出示例基因(panel e)在天花板上的位置: 错开高度的水平标签
ymax=axa.get_ylim()[1]
ex_sorted=sorted(C["examples"],key=lambda e:e["ceil"])
for i,e in enumerate(ex_sorted):
    yt=ymax*(0.30+0.11*i)
    axa.plot([e["ceil"],e["ceil"]],[0,yt*0.9],color=PAL["cptac"],lw=0.7,alpha=0.8)
    axa.annotate(e["gene"],xy=(e["ceil"],yt*0.9),xytext=(e["ceil"]+0.03,yt),
                 fontsize=6,color=PAL["cptac"],ha="left",va="center",
                 arrowprops=dict(arrowstyle="-",color=PAL["cptac"],lw=0.6))
axa.text(0.60,ymax*0.20,"panel e\nexamples",fontsize=5.6,color=PAL["cptac"],ha="left",style="italic")
axa.set_xlabel("Measured mRNA-protein PCC (per gene)")
axa.set_ylabel("Number of genes")
axa.set_title(f"Measured mRNA-protein correlation per gene\nCPTAC bulk RNA vs protein  (n={NP})",fontsize=8.2)
tag(axa,"a")

# ================= (b) 分层阶梯: Malignant vs 伪bulk 分组柱 =================
axb=fig.add_subplot(gs[0,1])
steps=["All","High ceiling","Well-predicted","Both"]
labels=["All\ngenes","High RNA–\nprot. r","Well-\npredicted","Both"]
lm=C["ladder_mal"]; lp=C["ladder_pb"]
x=np.arange(len(steps)); w=0.38
mv=[lm[s]["median"] for s in steps]; pv=[lp[s]["median"] for s in steps]
# --- 每根柱的 gene-level bootstrap 95% CI of the median (2000 resamples, seed 0) ---
_sx=np.array(C["sc_ceil"]); _sy=np.array(C["sc_pred"]); _sw=np.array(C["sc_well"])
PBGENES={"All":_sy,"High ceiling":_sy[_sx>0.4],"Well-predicted":_sy[_sw==1],
         "Both":_sy[(_sx>0.4)&(_sw==1)]}
try: MALGENES=json.load(open(f"{CACHE}/protein_ext_ladder_genes.json"))["mal"]
except FileNotFoundError: MALGENES=None
def _err(vals,genesets):
    if genesets is None: return None
    lo=[];hi=[]
    for s,v in zip(steps,vals):
        a,b=boot_median_ci(genesets[s]); lo.append(max(0.0,v-a)); hi.append(max(0.0,b-v))
    return np.array([lo,hi])
em=_err(mv,MALGENES); ep=_err(pv,PBGENES)
EKW=dict(capsize=2.5,error_kw=dict(lw=0.8,ecolor="0.25"))
b1=axb.bar(x-w/2,mv,w,color=PAL["cptac"],edgecolor="white",lw=0.5,label="Malignant prediction",yerr=em,**EKW)
b2=axb.bar(x+w/2,pv,w,color=PAL["prot"],edgecolor="white",lw=0.5,label="Whole-tumor profile",yerr=ep,**EKW)
mtop=[v+(em[1][i] if em is not None else 0) for i,v in enumerate(mv)]
ptop=[v+(ep[1][i] if ep is not None else 0) for i,v in enumerate(pv)]
for xi,v,tp in zip(x-w/2,mv,mtop): axb.text(xi,tp+0.008,f"{v:.2f}",ha="center",fontsize=6.6,fontweight="bold")
for xi,v,tp in zip(x+w/2,pv,ptop): axb.text(xi,tp+0.008,f"{v:.2f}",ha="center",fontsize=6.6,fontweight="bold")
axb.axhline(med,color=PAL["risk"],lw=1.1,ls="--")
axb.text(len(steps)-0.55,med+0.006,"median mRNA–protein r",color=PAL["risk"],fontsize=6.2,ha="right")
tick=[f"{labels[i]}\nn={lm[steps[i]]['n']}" for i in range(len(steps))]
axb.set_xticks(x); axb.set_xticklabels(tick,fontsize=6.8)
axb.tick_params(axis="x",length=0,pad=3)
axb.set_ylim(0,0.50); axb.set_ylabel("Predicted vs protein  median PCC")
axb.set_title("Signal concentrates where expected",fontsize=8.6)
if em is not None or ep is not None:
    axb.set_xlabel("error bars: gene-level bootstrap 95% CI of the median (2000 resamples)",
                   fontsize=5.8,color="0.25",labelpad=2)
axb.legend(loc="upper left",frameon=False,fontsize=6.5,handlelength=1.2)
# 递进箭头提示
axb.annotate("",xy=(3.0,0.455),xytext=(0.15,0.20),
             arrowprops=dict(arrowstyle="->",color="#999",lw=1.0,
                             connectionstyle="arc3,rad=-0.22"))
axb.text(1.35,0.395,"increasing specificity",fontsize=6,color="#999",ha="center",
         style="italic",rotation=14)
tag(axb,"b")

# ================= (c) 天花板约束 (hexbin + well-predicted) =================
axd=fig.add_subplot(gs[0,2])
sx=np.array(C["sc_ceil"]); sy=np.array(C["sc_pred"]); sw=np.array(C["sc_well"])
hb=axd.hexbin(sx,sy,gridsize=42,cmap="Greys",mincnt=1,linewidths=0,extent=(-0.55,1.0,-0.45,0.85))
axd.scatter(sx[sw==1],sy[sw==1],s=6,alpha=0.6,color=PAL["cptac"],edgecolors="none",
            label="well-predicted genes")
xs,yh,clo,chi=fit_ci(sx,sy)
axd.plot(xs,yh,color=PAL["risk"],lw=1.5)
axd.fill_between(xs,clo,chi,color=PAL["risk"],alpha=0.30,linewidth=0)
for _yb in (clo, chi):   # 带宽随 n 收缩, 加细边界线保证可见
    axd.plot(xs, _yb, color=PAL["risk"], lw=0.45, alpha=0.85, zorder=1)
def _pf(pp):
    if not (pp>0): return "P < 1e-300"
    if pp>=0.001: return f"P = {pp:.3f}"
    return "P = "+f"{pp:.0e}".replace("e-0","e-").replace("e+0","e+")
rr,_rcp=pearsonr(sx,sy)
_ng=len(sx); _z=np.arctanh(np.clip(rr,-0.999,0.999)); _se=1/np.sqrt(max(_ng-3,1))
_rlo=np.tanh(_z-1.96*_se); _rhi=np.tanh(_z+1.96*_se)
rbox(axd,0.03,0.96,f"r = {rr:.3f}\n95% CI {_rlo:.3f}–{_rhi:.3f}\n{_pf(_rcp)}",fc="#fff5ec",ec=PAL["risk"])
axd.axhline(0,color="#ccc",lw=0.6); axd.axvline(0.4,color="#777",lw=0.8,ls=":")
axd.set_xlabel("Measured mRNA-protein PCC\nline: OLS fit, band: 95% CI of the fit")
axd.set_ylabel("Predicted (whole-tumor)\nvs protein PCC")
axd.set_title("Prediction does best where mRNA and protein agree",fontsize=8.2)
axd.legend(loc="lower right",frameon=False,fontsize=6.3,markerscale=1.6)
axd.set_xlim(-0.55,1.0); axd.set_ylim(-0.45,0.85)
cb=fig.colorbar(hb,ax=axd,fraction=0.045,pad=0.02); cb.set_label("genes / bin",fontsize=6); cb.ax.tick_params(labelsize=6)
tag(axd,"c")

# ================= (d) 示例基因散点 (双高, per-patient) =================
sub=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,:],wspace=0.40)
for k,e in enumerate(C["examples"]):
    ax=fig.add_subplot(sub[0,k])
    p=np.array(e["pred"]); o=np.array(e["prot"])
    ax.scatter(p,o,s=11,alpha=0.55,color=PAL["cptac"],edgecolors="white",linewidths=0.2)
    # 拟合 + 回归线的 t 型 95% 置信带
    xs,yh,clo,chi=fit_ci(p,o)
    ax.plot(xs,yh,color=PAL["risk"],lw=1.3)
    ax.fill_between(xs,clo,chi,color=PAL["risk"],alpha=0.30,linewidth=0)
    for _yb in (clo, chi):   # 带宽随 n 收缩, 加细边界线保证可见
        ax.plot(xs, _yb, color=PAL["risk"], lw=0.45, alpha=0.85, zorder=1)
    ax.set_title(f"{e['gene']}",fontsize=8.5,fontweight="bold")
    _m=~(np.isnan(p)|np.isnan(o)); _n=int(_m.sum()); _mp=pearsonr(p[_m],o[_m])[1]
    _z=np.arctanh(np.clip(e['r'],-0.999,0.999)); _se=1/np.sqrt(max(_n-3,1))
    _rlo=np.tanh(_z-1.96*_se); _rhi=np.tanh(_z+1.96*_se)
    rbox(ax,0.04,0.97,f"r = {e['r']:.3f}\n95% CI {_rlo:.3f}–{_rhi:.3f}\n{_pf(_mp)}",fc="#fbeff2",ec=PAL["cptac"],fs=5.6)
    ax.set_xlabel("Predicted expression (log CPM)\nband: 95% CI of the fit",fontsize=7)
    if k==0:
        ax.set_ylabel("Measured protein (z-score)",fontsize=7); tag(ax,"d")
    ax.tick_params(labelsize=6.5)

fig.suptitle("Protein-level orthogonal validation of H&E-inferred expression  (CPTAC-HNSCC)",
             fontsize=11,y=0.998,fontweight="bold")
for ext in ("pdf","png"):
    fig.savefig(f"{OUT}/S_protein_validation.{ext}",dpi=300,bbox_inches="tight")
plt.close(fig)
print("saved S_protein_validation")
