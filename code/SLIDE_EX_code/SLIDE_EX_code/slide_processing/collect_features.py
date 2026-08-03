import sys, os
import numpy as np
import pandas as pd
import pickle


project = sys.argv[1]
path2project = f"./{project}/"  # TODO: substitute with your own filepath
path2features = path2project + "feature/"
path2meta = path2project + "metadata.csv"  # TODO: substitute with your own filepath
meta = pd.read_csv(path2meta)

slide_names = meta['Slide.ID'].values

n_slides = len(slide_names)
print(f"n_slides: {n_slides}")

# compile features
features = []

for i_slide in range(n_slides):
    slide_name = slide_names[i_slide]
    path_2_feature = f'{path2features}{slide_name}.pkl'
    if not os.path.exists(path_2_feature):
        raise Exception(f"{i_slide} not found")
    x = np.load(path_2_feature, allow_pickle=True)
    features.append((slide_name, x))

out_file = f'{path2project}features.pkl'

with open(out_file, 'wb') as f:
    pickle.dump(features, f)

print("--- completed collecting features--- ")
