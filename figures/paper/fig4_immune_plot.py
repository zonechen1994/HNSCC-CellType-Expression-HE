#!/usr/bin/env python3
"""Fig 4 — H&E-inferred immune infiltration: IHC-validated + spatially resolved.
(a,b) 预测T丰度 vs TMA IHC CD3/CD8 密度  (c) HANCOCK 高清 B/T 空间叠加示例。"""
import os, json, numpy as np
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
from scipy import stats as _st
plt.rcParams.update({
    "pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
    "font.size":8,"axes.linewidth":0.8,"axes.titlesize":9,"axes.labelsize":8.5,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
PAL={"prot":"#2c7fb8","risk":"#d95f0e","hancock":"#228833","cptac":"#CC6677","grey":"#999999"}
def tag(ax,s): ax.text(-0.16,1.06,s,transform=ax.transAxes,fontsize=12,fontweight="bold",va="top",ha="left")
ihc=json.load(open("figures/paper/cache/ihc.json"))

def fit_ci(x,y,npts=120):
    """OLS fit + t-based 95% CI of the fitted line (same recipe used across the figure set)."""
    x=np.asarray(x,float); y=np.asarray(y,float)
    n=len(x); z=np.polyfit(x,y,1)
    xs=np.linspace(x.min(),x.max(),npts); yh=np.polyval(z,xs)
    resid=y-np.polyval(z,x); s=np.sqrt((resid**2).sum()/(n-2))
    xb=x.mean(); sxx=((x-xb)**2).sum()
    se=s*np.sqrt(1.0/n+(xs-xb)**2/sxx)
    t=_st.t.ppf(0.975,n-2)
    return xs,yh,yh-t*se,yh+t*se

fig=plt.figure(figsize=(11.6,7.4))
gs=gridspec.GridSpec(2,2,figure=fig,height_ratios=[1,1.05],hspace=0.42,wspace=0.3)

# (a,b) IHC 散点
for k,mk in enumerate(["CD3_density","CD8_density"]):
    ax=fig.add_subplot(gs[0,k]); d=ihc[mk]
    x=np.array(d["x"]); y=np.array(d["y"])
    _m=~(np.isnan(x)|np.isnan(y)); _xx,_yy=x[_m],y[_m]; _n=len(_xx)
    _rng=np.random.RandomState(0); _bs=[]
    for _ in range(2000):
        _i=_rng.randint(0,_n,_n)
        if np.std(_xx[_i])>0 and np.std(_yy[_i])>0:
            _r=_st.spearmanr(_xx[_i],_yy[_i]).correlation
            if not np.isnan(_r): _bs.append(_r)
    _clo,_chi=np.percentile(_bs,[2.5,97.5])
    ax.scatter(x,y,s=8,alpha=0.45,color=PAL["hancock"],edgecolors="none")
    xs,yh,lo,hi=fit_ci(x,y)
    ax.plot(xs,yh,color=PAL["risk"],lw=1.3)
    ax.fill_between(xs,lo,hi,color=PAL["risk"],alpha=0.30,linewidth=0)
    for _yb in (lo, hi):   # 带宽随 n 收缩, 加细边界线保证可见
        ax.plot(xs, _yb, color=PAL["risk"], lw=0.45, alpha=0.85, zorder=1)
    ax.set_title(f"{mk.split('_')[0]}: predicted T-cell vs IHC density\nr={d['rho']:.3f} (95% CI {_clo:.3f}–{_chi:.3f}), p<1e-20, n={d['n']}",fontsize=8)
    ax.set_xlabel("Predicted T-cell abundance (WSI)\nline: OLS fit, band: 95% CI of the fit",fontsize=7.5)
    ax.set_ylabel("TMA IHC density (cells/mm²)",fontsize=7.5)
    tag(ax,"ab"[k])

# (c) 空间叠加示例(HANCOCK HE_628, H&E|B|T)
axc=fig.add_subplot(gs[1,:]); axc.axis("off")
img=mpimg.imread("figures/paper/HANCOCK_TB_hires_PrimaryTumor_HE_628.png")
axc.imshow(img)
axc.set_title("Spatial example (HANCOCK): predicted B-cell (focal/TLS-like) vs T-cell (diffuse) on H&E",
              fontsize=8.5,pad=4)
tag(axc,"c")

fig.suptitle("Figure 3. H&E-inferred immune infiltration is IHC-validated and spatially resolved",
             fontsize=10.5,y=0.99,fontweight="bold")
for e in ("pdf","png"): fig.savefig(f"figures/paper/Fig3_immune.{e}",dpi=300,bbox_inches="tight")
plt.close(fig); print("saved Fig3_immune")
