import numpy as np
import pandas as pd
import torch
import os,sys,time

from model_MLP import *
from utils import *


# Initialize Script ==================================================

# check available device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

init_random_seed(random_seed=42)

n_inputs = 768
n_hiddens = 768
dropout = 0.2
batch_size = 32
learning_rate = 0.0001

try:
    project = sys.argv[1]
    ik_fold = int(sys.argv[2])
    il_fold = int(sys.argv[3])
    cell_type = sys.argv[4]
    max_epochs,patience = 500,50

except:
    print("test on PC")
    project = "codefacs_abundance"
    ik_fold = 0
    il_fold = 0
    cell_type = 'CAFs'
    max_epochs,patience = 1,20
    
if cell_type == "Cancer":
    cell_type = "Cancer Epithelial"
if cell_type == "Normal":
    cell_type = "Normal Epithelial"

print("project:", project)
print("ik_fold:", ik_fold)
print("il_fold:", il_fold)
print("max_epochs: {}, patience: {}".format(max_epochs, patience))

# filepaths
path2project = f"./{project}/"  # TODO: substitute with your own filepath
path2features = f'{path2project}features.pkl'
path2target = f"{path2project}breast_{cell_type}_log_exp_merged.pkl"  # TODO: substitute with your own filepath
path2split = f"{path2project}splits/train_valid_test_idx.npz"  # TODO: substitute with your own filepath; for 5-fold nested cross-validation split
path2gene_file = f"{path2project}breast_{cell_type}_gene_file.csv"  # TODO: substitute with your own filepath

genes = pd.read_csv(path2gene_file)
genes = genes["gene"].values

print("genes:", genes)
print("len(genes):", len(genes))

# create result directory
result_dir = f"{path2project}results/{cell_type}_preds/result_%s_%s"%(ik_fold, il_fold)
os.makedirs(result_dir,exist_ok=True)


# Load Data and Initialize Model ==================================================
train_set, valid_set, test_set = load_dataset(path2features, path2target, path2split, genes, \
                                              ik_fold, il_fold)

bias_init = torch.nn.Parameter(torch.Tensor(np.mean([sample[1].detach().cpu().numpy()\
                         for sample in train_set], axis=0)).to(device))
n_outputs = len(genes)
print(f"num classes/outputs: {n_outputs}")

model = MLP_regression(n_inputs, n_hiddens, n_outputs, dropout, bias_init)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train Model ==================================================
print(" --- fit --- ")

model,train_loss,train_coef,train_slope,\
valid_loss,valid_coef,valid_slope,valid_labels,valid_preds = \
fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size)

analyze_result(result_dir,genes,model,train_loss,train_coef,train_slope, \
               valid_loss,valid_coef,valid_slope,valid_labels, valid_preds, test_set)

print("--- completed ---")
