"""
Parallel CTransPath feature extraction for SLIDE-EX (faithful to 1main_processing.py).

A single worker loads the CTransPath model ONCE on its assigned GPU, then processes
its shard of slides (indices [shard::nshards] of metadata.csv). For each slide it
reproduces the paper's pipeline exactly:
  read_region(level=0) -> RGB -> resize to 512 -> evaluate_tile(edge) -> Macenko norm
  -> CTransPath features -> save feature/{Slide.ID}.pkl

Differences from 1main_processing.py (speed/scale only, NOT the tiles/features):
  * model loaded once per worker (not once per slide)
  * QC mask PDF generation skipped (it is only a visualization, unused downstream)
  * skips slides whose feature pkl already exists (resume-safe)

Usage:
  python extract_features_parallel.py <project> <shard> <nshards>
  GPU is chosen by env CUDA_VISIBLE_DEVICES set by the launcher.
"""
import os, sys, time, pickle
import numpy as np
import torch
from PIL import Image
import openslide

# keep each worker single-threaded so 144 cores aren't oversubscribed
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)
import cv2
cv2.setNumThreads(0)

from func.utils_preprocessing import evaluate_tile, init_random_seed
from func import utils_color_norm
from ctrans_model import CTransPath
import torchvision.transforms as T

# ---- fixed config (identical to 1main_processing.py) ----
MAG_ASSUMED = 20
MAG_SELECTED = 20
TILE_SIZE = 512
BATCH_SIZE = 64  # feature values are batch-invariant (eval-mode BN/LN); smaller batch caps GPU mem under many parallel workers
EDGE_MAG_THRSH = 15
EDGE_FRACTION_THRSH = 0.5
# 组织块上限: 下游对图块特征取均值(model_MLP: torch.mean(x,dim=0)), 随机2500块的均值≈全部块均值,
# 且TCGA训练中位数~2382块, 封顶让巨型片(HANCOCK 40x, 上万块)与TCGA更一致且大幅提速. 0=不封顶.
MAX_TILES = int(os.environ.get("MAX_TILES", "0"))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
color_norm = utils_color_norm.macenko_normalizer()

_tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_model():
    model = CTransPath(num_classes=0)
    model.to(device)
    weights = os.environ.get("CTRANSPATH_WEIGHTS", os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/models/ctranspath.pth")
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.eval()
    return model


def tiles_to_features(model, tiles_list):
    if len(tiles_list) == 0:
        return np.zeros((0, 768), dtype=np.float32)
    tensors = torch.cat([_tf(t).unsqueeze(0) for t in tiles_list], dim=0).to(device)
    feats = []
    with torch.no_grad():
        for s in range(0, tensors.shape[0], BATCH_SIZE):
            feats.append(model(tensors[s:s + BATCH_SIZE]).detach().cpu().numpy())
    return np.concatenate(feats)


def process_slide(model, slide_path, out_pkl):
    slide = openslide.OpenSlide(slide_path)
    if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
        mag_max = int(float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]))  # 有些切片(HANCOCK)倍率是浮点字符串如"82.44"
    else:
        mag_max = MAG_ASSUMED
    downsampling = int(mag_max / MAG_SELECTED)
    if downsampling < 1:
        downsampling = 1
    px0, py0 = slide.level_dimensions[0]
    tile_size0 = int(TILE_SIZE * downsampling)
    n_rows, n_cols = int(py0 / tile_size0), int(px0 / tile_size0)

    coords = [(row, col) for row in range(n_rows) for col in range(n_cols)]
    if MAX_TILES > 0:                       # 打乱后凑够 MAX_TILES 个组织块就停(省Macenko+读取), 种子固定可复现
        rng = np.random.default_rng(abs(hash(os.path.basename(slide_path))) % (2**32))
        rng.shuffle(coords)

    tiles_list = []
    for row, col in coords:
        tile = slide.read_region((col * tile_size0, row * tile_size0), 0,
                                 [tile_size0, tile_size0]).convert("RGB")
        if tile.size[0] == tile_size0 and tile.size[1] == tile_size0:
            tile = tile.resize((TILE_SIZE, TILE_SIZE))
            arr = np.array(tile)
            if evaluate_tile(arr, EDGE_MAG_THRSH, EDGE_FRACTION_THRSH) == 1:
                tiles_list.append(Image.fromarray(color_norm.transform(arr)))
                if MAX_TILES > 0 and len(tiles_list) >= MAX_TILES:
                    break
    slide.close()

    features = tiles_to_features(model, tiles_list)
    with open(out_pkl, "wb") as f:
        pickle.dump(features, f)
    return n_rows * n_cols, len(tiles_list), features.shape


def main():
    import pandas as pd
    project = sys.argv[1]
    shard = int(sys.argv[2])
    nshards = int(sys.argv[3])
    init_random_seed(42)

    path2project = f"./{project}/"
    meta = pd.read_csv(path2project + "metadata.csv")
    path2slide = path2project + "slides/"
    path2feat = path2project + "feature/"
    os.makedirs(path2feat, exist_ok=True)

    idx = list(range(shard, len(meta), nshards))
    model = build_model()
    print(f"[shard {shard}/{nshards}] gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"slides={len(idx)} device={device}", flush=True)

    for n, i in enumerate(idx):
        sid = meta["Slide.ID"].values[i]
        fname = str(meta["Filename"].values[i])
        out_pkl = f"{path2feat}{sid}.pkl"
        if os.path.exists(out_pkl):
            print(f"[shard {shard}] ({n+1}/{len(idx)}) SKIP {sid}", flush=True)
            continue
        t0 = time.time()
        try:
            slide_path = fname if os.path.isabs(fname) else f"{path2slide}{fname}"
            ntot, nsel, shp = process_slide(model, slide_path, out_pkl)
            print(f"[shard {shard}] ({n+1}/{len(idx)}) {sid} tiles={nsel}/{ntot} "
                  f"feat={shp} {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[shard {shard}] ({n+1}/{len(idx)}) FAIL {sid}: {e}", flush=True)


if __name__ == "__main__":
    main()
