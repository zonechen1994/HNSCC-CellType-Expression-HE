#!/usr/bin/env python3
"""修正 HANCOCK 临床表中 grade 字段的解析错误。

HANCOCK 原始文件 data/HANCOCK_clinical/StructuredData/pathological_data.json 中,
141 个 p16 阳性病人的 grading 字段被写成了字符串 "hpv_association_p16", 即列名本身
被误填进了取值。原先的解析把这个非法值映射成了数字 1, 结果全部 p16 阳性病人都变成
grade 1, 使 grade 成为 p16 状态的完美代理变量, 污染了以 grade 为协变量的 Cox 模型。

本脚本把这些病人的 grade 置为缺失, 并重写 run/HANCOCK/clinical.csv (原文件另存备份)。
其余字段保持原样。
"""
import os, json, shutil, numpy as np, pandas as pd
os.chdir(os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))
SRC = "data/HANCOCK_clinical/StructuredData/pathological_data.json"
CSV = "run/HANCOCK/clinical.csv"

raw = pd.DataFrame(json.load(open(SRC)))
raw["case"] = raw["patient_id"].astype(str).str.zfill(3)
BAD = "hpv_association_p16"
n_bad = int((raw["grading"] == BAD).sum())
gmap = {"G1": 1.0, "G2": 2.0, "G3": 3.0, "G4": 4.0}
grade_fixed = raw["grading"].map(gmap)          # 非法值与缺失都会变成 NaN
raw["grade_fixed"] = grade_fixed
print(f"原始 grading 取值:\n{raw['grading'].value_counts(dropna=False).to_string()}")
print(f"\n非法值 '{BAD}' 的病人数: {n_bad}  -> 置为缺失")
print(f"修正后 grade 分布:\n{raw['grade_fixed'].value_counts(dropna=False).to_string()}")

cl = pd.read_csv(CSV, dtype={"case": str}); cl["case"] = cl["case"].astype(str).str.zfill(3)
before = cl.set_index("case")["grade"]
fixed = raw.set_index("case")["grade_fixed"]
chg = cl.set_index("case").index.map(lambda c: (before.get(c), fixed.get(c)))
n_changed = sum(1 for a, b in chg if (pd.notna(a) != pd.notna(b)) or (pd.notna(a) and pd.notna(b) and a != b))
if not os.path.exists(CSV + ".bak_gradebug"):
    shutil.copy(CSV, CSV + ".bak_gradebug"); print(f"\n已备份 -> {CSV}.bak_gradebug")
cl["grade"] = cl["case"].map(fixed)
cl.to_csv(CSV, index=False)
print(f"已重写 {CSV}, 改动 {n_changed} 行")

chk = pd.read_csv(CSV, dtype={"case": str})
print("\n修正后 grade x p16 交叉表:")
print(pd.crosstab(chk["grade"], chk["hpv_p16"], dropna=False).to_string())
print(f"\ngrade 缺失: {int(chk['grade'].isna().sum())} / {len(chk)}")
