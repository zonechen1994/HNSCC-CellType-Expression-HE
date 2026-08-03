#!/usr/bin/env python3
"""
Build 5x5 nested-CV index splits for SLIDE-EX, matching how utils.load_dataset reads them:
    d = np.load(path2split, allow_pickle=True)
    train_idx = d["train_idx"][ik_fold][il_fold]
    valid_idx = d["valid_idx"][ik_fold][il_fold]
    test_idx  = d["test_idx"][ik_fold]

Outer 5-fold -> test_idx[k] (held-out 20%). Inner 5-fold on the remaining 80%
-> train_idx[k][l] (64%) and valid_idx[k][l] (16%). Indices are positions into
the metadata.csv / features.pkl ordering. Seed 42, shuffled.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 项目真实目录 run/<OUT_NAME>；可由命令行传入(换癌种/换数据集都不用改脚本)
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/run/TCGA_BRCA_OUTPUT"
meta = pd.read_csv(os.path.join(ROOT, "metadata.csv"))
n = len(meta)
SEED = 42
print(f"n samples: {n}")

outer = KFold(n_splits=5, shuffle=True, random_state=SEED)

test_idx = []        # [k] -> array
train_idx = []       # [k][l] -> array
valid_idx = []       # [k][l] -> array

for k, (outer_train, outer_test) in enumerate(outer.split(np.arange(n))):
    test_idx.append(np.array(outer_test, dtype=np.int64))
    inner = KFold(n_splits=5, shuffle=True, random_state=SEED + 1 + k)
    tr_k, va_k = [], []
    for l, (tr_rel, va_rel) in enumerate(inner.split(outer_train)):
        # map inner relative positions back to absolute sample indices
        tr_k.append(outer_train[tr_rel].astype(np.int64))
        va_k.append(outer_train[va_rel].astype(np.int64))
    train_idx.append(tr_k)
    valid_idx.append(va_k)

# store as object arrays so [k][l] indexing works with ragged sizes
out = os.path.join(ROOT, "splits", "train_valid_test_idx.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out,
         train_idx=np.array(train_idx, dtype=object),
         valid_idx=np.array(valid_idx, dtype=object),
         test_idx=np.array(test_idx, dtype=object))

# sanity check: for each outer fold, train+valid+test partition all n with no overlap
for k in range(5):
    te = set(test_idx[k].tolist())
    for l in range(5):
        tr = set(train_idx[k][l].tolist())
        va = set(valid_idx[k][l].tolist())
        assert not (tr & va), "train/valid overlap"
        assert not (tr & te) and not (va & te), "test leak into train/valid"
        assert tr | va | te == set(range(n)), "not a full partition"
    print(f"outer {k}: test={len(te)}  inner0 train={len(train_idx[k][0])} valid={len(valid_idx[k][0])}")

print(f"\nWrote {out}")
