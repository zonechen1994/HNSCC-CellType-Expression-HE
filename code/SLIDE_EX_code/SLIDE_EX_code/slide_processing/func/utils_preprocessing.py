import os
import numpy as np
import cv2
import torch
from torch import nn
import torchvision
from torchvision.models import resnet50

import torchvision.transforms as transforms
from ctrans_model import CTransPath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tile_transform(tiles_list, data_mean, data_std):
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std)
    ])

    # Data transform
    n_tiles = len(tiles_list)
    print("n_tiles:", n_tiles)

    tiles = []
    for i in range(n_tiles):
        # Apply the transformation to each image only once
        transformed_tile = data_transform(tiles_list[i]).unsqueeze(0)
        # Move the transformed tile to the specified device
        transformed_tile = transformed_tile.to(device)
        tiles.append(transformed_tile)
    
    tiles = torch.cat(tiles, dim=0)
    print("tiles.shape:", tiles.shape)
    tiles_list = 0

    return tiles


def tiles2features(tiles_list, model_name, batch_size):
    # model config
    if model_name == "ctrans":
        path2model = os.environ.get("CTRANSPATH_WEIGHTS", os.environ.get("HNSC_ROOT",os.environ.get("HNSC_ROOT", "/data1/zqchen/npo_61"))+"/models/ctranspath.pth")
        model = CTransPath(num_classes=0)
        model.to(device)
        model.load_state_dict(torch.load(path2model)['model'])
        data_mean=[0.485, 0.456, 0.406] ; data_std = [0.229, 0.224, 0.225]
    else:
        raise Exception("Model name not valid")

    model.eval()
    
    # tile transform
    tiles = tile_transform(tiles_list, data_mean, data_std)

    # extract features from tiles
    n_tiles = tiles.shape[0]
    features = []
    for idx_start in range(0, n_tiles, batch_size):
        idx_end = idx_start + min(batch_size, n_tiles - idx_start)

        with torch.no_grad():
            y = model(tiles[idx_start:idx_end])

        if model_name == "vit":
            y = y.last_hidden_state[:, 0]

        features.append(y.detach().cpu().numpy())

    features = np.concatenate(features)
    print("features.shape:", features.shape)
    
    return features


def evaluate_tile(img_np, edge_mag_thrsh, edge_fraction_thrsh):

    select = 1  # initial value
    
    tile_size = img_np.shape[0]
        
    # 0) exclude if edge_mag > 0.5
    img_gray=cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)

    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely1 = cv2.convertScaleAbs(sobely)

    mag = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)

    unique, counts = np.unique(mag, return_counts=True)

    edge_mag = counts[np.argwhere(unique < edge_mag_thrsh)].sum()/(tile_size*tile_size)

    if edge_mag > edge_fraction_thrsh:
        select = 0
    
    return select

##================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
