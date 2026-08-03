import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset,Dataset
from scipy.stats import pearsonr
from scipy.stats import norm


##===================================================================================================
class SlideClassDataset(Dataset):
    # input: features_list[n_slides](slide_name, features[n_tiles,512])
    # target[n_slides, n_genes]
    
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets  # gene_expression_values
        self.dim = self.features[0][1].shape[1]

    def __getitem__(self, index):
        sample = torch.Tensor(self.features[index][1]).float()
        # NOTE: original code did int(self.targets[index]), which truncates and
        # breaks regression (targets are gene/cell-type VECTORS, not scalars).
        # Return the raw target as a tensor: a float vector for regression, or a
        # scalar for the direct-classification path; callers cast as needed.
        target = torch.as_tensor(np.asarray(self.targets[index]))
        return sample, target

    def __len__(self):
        return len(self.features)


##===================================================================================================
def load_dataset(path2features, path2target, path2split, genes, ik_fold, il_fold):
    # load image feature
    features = np.load(path2features, allow_pickle=True)
    print("len(features):", len(features))    

    # load target file
    print(f"target_file: {path2target}")
    target_df = pd.read_pickle(path2target)
    if isinstance(genes, str) and genes == "pCR.RD":
        response_dict = {'pCR': 1, 'RD': 0}
        target_df['pCR_binary'] = target_df['pCR.RD'].apply(lambda x: response_dict[x] if x in response_dict.keys() else np.nan)
        targets = target_df['pCR_binary'].values
    else:
        targets = target_df[genes].values

    # load_train_valid_test_idx:
    train_valid_test_idx = np.load(path2split, allow_pickle=True)

    train_idx = train_valid_test_idx["train_idx"][ik_fold][il_fold]
    valid_idx = train_valid_test_idx["valid_idx"][ik_fold][il_fold]
    test_idx = train_valid_test_idx["test_idx"][ik_fold]

    dataset = SlideClassDataset(features, targets)

    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, valid_set, test_set


##===================================================================================================
def compute_coefs(labels, preds):
    return np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])

def compute_slope(labels, preds):
    return np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])

def compute_coef_slope(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    return coef, slope


##===================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
