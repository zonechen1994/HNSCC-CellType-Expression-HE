#!/usr/bin/env python3
"""Redraw every figure and table in the paper from the bundled cache.

Usage:
    export HNSC_ROOT=/path/to/HNSC_code      # or run from inside the package
    python make_figures.py

Nothing here needs a GPU, the trained models, or the whole-slide images. Every number is read
from figures/paper/cache/, which holds the outputs of the analyses described in the Methods.
Outputs are written next to the scripts in figures/paper/ and are byte-identical in content to
the figures in the manuscript.
"""
import os, subprocess, sys, time

ROOT = os.environ.get("HNSC_ROOT") or os.path.dirname(os.path.abspath(__file__))
os.environ["HNSC_ROOT"] = ROOT
PY = sys.executable

STEPS = [
    ("Figure 2, S1",     "figures/paper/fig_plot.py"),
    ("Figure 3",         "figures/paper/fig4_immune_plot.py"),
    ("Figure 4",         "figures/paper/fig4_clinical_plot.py"),
    ("Figure 5",         "figures/paper/fig5_gsea_main.py"),
    ("Figure S2",        "figures/paper/protein_ext_plot.py"),
    ("Figure S3",        "figures/paper/s4_abundance_plot.py"),
    ("Figure S4",        "figures/paper/prognosis_plot.py"),
    ("Figure S5",        "figures/paper/interp_flagship_plot.py"),
    ("Figure S6",        "figures/paper/go_two_cohorts_plot.py"),
]

print(f"HNSC_ROOT = {ROOT}\n")
ok = fail = 0
t0 = time.time()
for label, script in STEPS:
    path = os.path.join(ROOT, script)
    print(f"  {label:18s} {script:44s} ", end="", flush=True)
    r = subprocess.run([PY, path], cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        print("ok"); ok += 1
    else:
        print("FAILED"); fail += 1
        tail = [l for l in r.stderr.strip().split("\n") if l.strip()][-1:]
        for l in tail: print(f"      {l[:120]}")
print(f"\n{ok} of {ok+fail} steps completed in {time.time()-t0:.0f} s.")
print(f"Figures were written to {os.path.join(ROOT,'figures/paper')}.")
sys.exit(1 if fail else 0)
