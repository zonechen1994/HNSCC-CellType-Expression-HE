import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys
import openslide
from PIL import Image
import os
import pickle
import torch

from func.utils_preprocessing import *
from func import utils_color_norm
color_norm = utils_color_norm.macenko_normalizer()


# Initialize Script ==================================================

# check available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

model_name = "ctrans"

init_random_seed(random_seed=42)

try:
    project = sys.argv[1]
    i_slide = int(sys.argv[2])

except:
    project = 'TCGA_BRCA_OUTPUT'
    i_slide = 3

print(f"project: {project}, i_slide: {i_slide}")

mag_assumed = 20 ###################################################
evaluate_edge = True
evaluate_color = False
extract_pretrained_features = True

mag_selected = 20
tile_size = 512
mask_downsampling = 16
batch_size = 128

# filepaths
path2project = f"./{project}/"  # TODO: substitute with your own filepath
print(f"path2project: {path2project}")

path2slide = path2project + "slides/"  # TODO: substitute with your own filepath
print("path2slide:", path2slide)
path2meta = path2project + "metadata.csv"  # TODO: substitute with your own filepath
print("path2meta:", path2meta)

path2mask = path2project + "mask/"
path2features =  path2project + "feature/"
print(f"path2mask: {path2mask}")
print(f"path2features: {path2features}")

os.makedirs(path2mask, exist_ok=True)
os.makedirs(path2features, exist_ok=True)

metadata = pd.read_csv(path2meta)


# Evaluate Tile ==============================================================================
edge_mag_thrsh = 15
edge_fraction_thrsh = 0.5

slide_file_names = metadata['Filename'].values
slide_names = metadata['Slide.ID'].values

slide_file_name = str(slide_file_names[i_slide])
slide_name = slide_names[i_slide]
print(f"slide_file_name: {slide_file_name}, slide_name: {slide_name}")

slide = openslide.OpenSlide(f"{path2slide}{slide_file_name}")

# magnification max
if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
    mag_max = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    print("mag_max:", mag_max)
    mag_original = mag_max
else:
    print(f"[WARNING] mag not found, assuming: {mag_assumed}")
    mag_max = mag_assumed
    mag_original = 0

# downsample_level
downsampling = int(int(mag_max)/mag_selected)
print(f"downsampling: {downsampling}")

# slide size at largest level (level=0)
px0, py0 = slide.level_dimensions[0]
tile_size0 = int(tile_size*downsampling)
print(f"px0: {px0}, py0: {py0}, tile_size0: {tile_size0}")

n_rows,n_cols = int(py0/tile_size0), int(px0/tile_size0)
print(f"n_rows: {n_rows}, n_cols: {n_cols}")

n_tiles_total = n_rows*n_cols
print(f"n_tiles_total: {n_tiles_total}")

mask_tile_size = int(np.ceil(tile_size/mask_downsampling))
print(f"mask_tile_size: {mask_tile_size}")


img_mask = np.full((int((n_rows)*mask_tile_size),int((n_cols)*mask_tile_size),3),255).astype(np.uint8)
mask = np.full((int((n_rows)*mask_tile_size),int((n_cols)*mask_tile_size),3),255).astype(np.uint8)

i_tile = 0
tiles_list = []
for row in range(n_rows):
    print(f"row: {row}/{n_rows}")
    for col in range(n_cols):
        
        tile = slide.read_region((col*tile_size0, row*tile_size0),\
                                 level=0, size=[tile_size0, tile_size0]) # RGBA image
        tile = tile.convert("RGB")
        
        if tile.size[0] == tile_size0 and tile.size[1] == tile_size0:
            # downsample to target tile size
            tile = tile.resize((tile_size, tile_size))

            mask_tile = np.array(tile.resize((mask_tile_size, mask_tile_size)))
            
            img_mask[int(row*mask_tile_size):int((row+1)*mask_tile_size),\
                     int(col*mask_tile_size):int((col+1)*mask_tile_size),:] = mask_tile

            tile = np.array(tile)

            # evaluate tile
            select = evaluate_tile(tile, edge_mag_thrsh, edge_fraction_thrsh)

            if select == 1:
                tile_norm = Image.fromarray(color_norm.transform(tile))

                mask_tile_norm = np.array(tile_norm.resize((mask_tile_size, mask_tile_size)))

                mask[int(row*mask_tile_size):int((row+1)*mask_tile_size),\
                     int(col*mask_tile_size):int((col+1)*mask_tile_size),:] = mask_tile_norm    

                tiles_list.append(tile_norm)

        i_tile += 1


# Plot: draw color lines on the mask ============================================
line_color = [0,255,0]

n_tiles = len(tiles_list)

img_mask[:,::mask_tile_size,:] = line_color
img_mask[::mask_tile_size,:,:] = line_color
mask[:,::mask_tile_size,:] = line_color
mask[::mask_tile_size,:,:] = line_color

fig, ax = plt.subplots(1,2,figsize=(40,20))
ax[0].imshow(img_mask)
ax[1].imshow(mask)

ax[0].set_title(f"{slide_name}, mag_original: {mag_original}, mag_assumed: {mag_assumed}")
ax[1].set_title(f"n_rows: {n_rows}, n_cols: {n_cols}, n_tiles_total: {n_tiles_total}, n_tiles_selected: {n_tiles}")

plt.tight_layout(h_pad=0.4, w_pad=0.5)
plt.savefig(f"{path2mask}{slide_name}.pdf", format="pdf", dpi=50)
plt.close()

img_mask = 0 ; mask = 0

print("completed cleaning")


# Extract Features ===============================================

features = tiles2features(tiles_list, model_name, batch_size)

slide_file = f"{path2features}{slide_name}.pkl"

with open(slide_file, 'wb') as file:
    pickle.dump(features, file)
