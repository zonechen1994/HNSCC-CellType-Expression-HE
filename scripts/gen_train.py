#!/usr/bin/env python3
"""
Generic queue-based training launcher (any cancer/project). Reads cell types from
<project>/cell_types_file.csv. GPU 0/1 only. Resume-safe.

Usage: python gen_train.py <PROJECT> <mode> <per_gpu> [gpus]
  gpus    : 逗号分隔的 GPU 列表, 如 "0,1" 或 "0,1,2"; 默认 "0,1"
  PROJECT : e.g. TCGA_LUAD   (输入/输出都在真实目录 run/<PROJECT>/, 不需要任何软链接)
  mode    : 5x1 | 5x5
  per_gpu : concurrent jobs per GPU

说明: 训练脚本 1main_*.py 用相对路径 ./<PROJECT>/ 读输入、写 results。这里让它以
cwd=run 运行(脚本用绝对路径定位, 其同目录模块经 sys.path[0] 自动可导入), 于是
./<PROJECT>/ == run/<PROJECT>/ —— 真实目录, 不再依赖 prediction/<PROJECT> 软链接。
"""
import os, sys, subprocess, time, pandas as pd
from collections import deque

PROJECT = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "5x1"
per_gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 6
# 第4个参数=要用的GPU(逗号分隔, 如 "0,1" 或 "0,1,2")；不传默认 0,1
GPUS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1").split(",")]
il_folds = [0] if mode == "5x1" else [0,1,2,3,4]
max_conc = per_gpu*len(GPUS)

PRED = os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/code/SLIDE_EX_code/SLIDE_EX_code/prediction"  # 训练脚本.py 所在目录(真实)
RUN_ROOT = os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/run"        # 所有项目的真实根; 训练以此为 cwd
RUN = f"{RUN_ROOT}/{PROJECT}"                 # 本项目真实目录(输入+输出都在这)
PROJ_DIR = RUN                                # 结果就写在 run/<PROJECT>/results, 无软链接
LOG = f"{RUN}/logs_train"; os.makedirs(LOG, exist_ok=True)
CELL_TYPES = list(pd.read_csv(f"{RUN}/cell_types_file.csv")["cell_types"])

cs_done = lambda ct,ik,il: os.path.exists(f"{PROJ_DIR}/results/{ct}_preds/result_{ik}_{il}/model_trained.pth")
ab_done = lambda ik,il: os.path.exists(f"{PROJ_DIR}/results/cell_fraction_preds/result_{ik}_{il}/model_trained.pth")

jobs = []
for ct in CELL_TYPES:
    for ik in range(5):
        for il in il_folds:
            if not cs_done(ct,ik,il): jobs.append(("cs",ct,ik,il))
for ik in range(5):
    for il in il_folds:
        if not ab_done(ik,il): jobs.append(("ab",ik,il))
print(f"PROJECT={PROJECT} mode={mode} cells={len(CELL_TYPES)} jobs={len(jobs)} max_conc={max_conc}", flush=True)

def launch(job, gpu):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    env["OMP_NUM_THREADS"]="1"; env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    if job[0]=="cs":
        _,ct,ik,il = job; tag=f"cs_{ct.replace(' ','_')}_{ik}_{il}"
        cmd=["python","-u",f"{PRED}/1main_cs_exp_regression.py",PROJECT,str(ik),str(il),ct]
    else:
        _,ik,il = job; tag=f"ab_{ik}_{il}"
        cmd=["python","-u",f"{PRED}/1main_abundance_regression.py",PROJECT,str(ik),str(il)]
    log=open(f"{LOG}/{tag}.log","w")
    # cwd=RUN_ROOT: 训练脚本里的 ./<PROJECT>/ 解析为 run/<PROJECT>/(真实目录, 无软链接)
    return {"p":subprocess.Popen(cmd,cwd=RUN_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),
            "tag":tag,"gpu":gpu,"log":log,"t0":time.time()}

q=deque(jobs); running=[]; load={g:0 for g in GPUS}; done=0; total=len(jobs)
while q or running:
    while q and len(running)<max_conc:
        gpu=min(GPUS, key=lambda g: load[g])   # 选当前负载最轻的 GPU
        if load[gpu]>=per_gpu: break
        r=launch(q.popleft(),gpu); load[gpu]+=1; running.append(r)
        print(f"[start] {r['tag']} gpu{gpu} (run {len(running)} q {len(q)})", flush=True)
    time.sleep(5); still=[]
    for r in running:
        if r["p"].poll() is None: still.append(r)
        else:
            r["log"].close(); load[r["gpu"]]-=1; done+=1
            print(f"[done {done}/{total}] {r['tag']} {'OK' if r['p'].returncode==0 else 'FAIL'} {(time.time()-r['t0'])/60:.1f}min", flush=True)
    running=still
print("ALL_TRAINING_DONE", flush=True)
