import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

# PyTorch Dataset Object
class dataset(Dataset):
    def __init__(self, data, labels, device, transform = None, target_transform = None):
        self.data = data
        self.labels = labels
        self.x = torch.tensor(self.data.to_numpy(), dtype = torch.float).to(device)
        self.y = torch.tensor(self.labels.to_numpy(), dtype = torch.int).to(device)
        self.len = self.data.shape[0]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.x[idx]
        label = self.y[idx]

        if self.transform:
            self.transform(sample)
        if self.target_transform:
            self.target_transform(label)
        
        return sample, label

    def get_labels(self):
        return self.labels
    
def calc_loss(model, data, Criterion):
    z = model(data.x)
    loss = float(Criterion(z,data.y.view(-1).long()).data)
    return loss/len(data)

def accuracy(y_test, y_pred):
    y_test = y_test.view(-1).cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    return np.mean(y_test == y_pred)

def score(y_test, y_pred, A):    
    y_test = y_test.view(-1).cpu().long().numpy()
    y_pred = y_pred.cpu().numpy()
    S = 0.0
    for i in range(0, y_test.shape[0]):
        S -= A[y_test[i], y_pred[i]]
    return S/y_test.shape[0]

def import_data(path):
    x = pd.read_csv(path)
    class dummy:
        def __init__(self, x):
            self.x = torch.Tensor(x.to_numpy())
    d = dummy(x)
    return d

def convert_data(data, device):
    class dummy:
        def __init__(self, x):
            self.x = torch.Tensor(x.to_numpy()).to(device)
    d = dummy(data)
    return d

def import_df(path_x, path_y):
    x = pd.read_csv(path_x)
    y = pd.read_csv(path_y)
    return x,y