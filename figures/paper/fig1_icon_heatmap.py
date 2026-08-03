#!/usr/bin/env python3
"""Figure 1 面板 (c) 的热图 icon —— 纯示意图形, 不含任何真实数值。

Figure 1 是 study design, 只需要传达"每个细胞型各自富集到属于自己的程序"这个
概念, 因此用程式化的对角块结构, 不使用实际的 NES 数据。

输出两个版本的矢量 PDF (无坐标轴、无文字、背景透明), 可直接置入版式软件:
  square  8 列 x 8 行,  对角线单格,   方形, 适合小 icon
  block   8 列 x 16 行, 对角线双格块, 竖长, 更像一张真实的热图
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

def schematic(ncol=8, rows_per_block=1, seed=7):
    """对角块为暖色, 其余为冷色, 叠加轻微起伏以免显得死板"""
    rng = np.random.default_rng(seed)
    nrow = ncol * rows_per_block
    M = rng.normal(-1.25, 0.32, (nrow, ncol))          # 底色: 偏冷, 收敛一些
    for j in range(ncol):                               # 对角块: 偏暖
        r0 = j * rows_per_block
        M[r0:r0 + rows_per_block, j] = rng.normal(2.75, 0.12, rows_per_block)
    # 相邻细胞型之间留一点弱正相关, 让图看起来自然
    for j in range(ncol - 1):
        r0 = j * rows_per_block
        M[r0:r0 + rows_per_block, j + 1] += rng.normal(0.55, 0.15, rows_per_block)
    return np.clip(M, -3, 3)

def draw(M, name, w, h, lw=1.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=lw)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.9); s.set_color("#3a3a3a")
    fig.subplots_adjust(0, 0, 1, 1)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/paper/{name}.{ext}", dpi=600, bbox_inches="tight",
                    pad_inches=0.02, transparent=True)
    plt.close(fig); print(f"saved {name}  ({M.shape[0]} 行 x {M.shape[1]} 列)")

draw(schematic(8, 1), "Fig1_icon_heatmap_square", 1.9, 1.9)
draw(schematic(8, 2), "Fig1_icon_heatmap_block", 1.9, 3.2, lw=0.9)
print("\n纯示意图形, 不含任何真实 NES 数值。PDF 为矢量, 背景透明。")
