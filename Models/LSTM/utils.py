import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import time

def calc_loss(model, data, Criterion):
    loss = 0
    for i in data.UniqueNames:
        output, hidden = model(data.datasets[i].unx)
        loss += float(Criterion(output, data.datasets[i].y.view(-1).long()).data)
    return loss/len(data.UniqueNames)

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

def import_data(device, x_path, y_path):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    return RNN_dataset(device, X, y)


class dataset(Dataset):
    def __init__(self, data, labels, device, transform = None, target_transform = None, y_float = False, rnn = False):
        self.data = data
        self.labels = labels
        self.x = torch.tensor(self.data, dtype = torch.float).to(device)

        if rnn:
           self.unx = self.x.unsqueeze(-1).permute(2,0,1)
        if y_float:
            self.y = torch.tensor(self.labels, dtype = torch.float).to(device)
        else:
            self.y = torch.tensor(self.labels, dtype = torch.int).to(device)
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
    

class RNN_dataset:
    def __init__(self, device, X, y):
        self.X = torch.Tensor(X.to_numpy()).to(device)
        self.y = torch.Tensor(y.to_numpy()).to(device)
        self.device = device

        self.datasets, self.UniqueNames = self.well_split(X, y)

    def well_split(self, X, y):
        X = pd.concat ([X,y], axis = 1)
        
        UniqueNames = X.WELL.unique()
        DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}

        for key in DataFrameDict.keys():
            DataFrameDict[key] = X[:][X.WELL == key]

        X_dict = {}
        Y_dict = {}
        datasets = {}

        for i in UniqueNames:
            data = DataFrameDict[i].sort_values('DEPTH_MD')
            X_dict[i] = data.drop(['FORCE_2020_LITHOFACIES_LITHOLOGY', 'WELL'], axis =1).to_numpy()
            Y_dict[i] = data['FORCE_2020_LITHOFACIES_LITHOLOGY'].to_numpy()
            datasets[i] = dataset(X_dict[i], Y_dict[i], self.device, rnn = True)
        return datasets, UniqueNames

def samples(data):
    sum = 0
    for name in data.UniqueNames:
        sum += len(data.datasets[name])
    return sum

def get_windows(num, inp):
    samples = inp.shape[1]
    dim = inp.shape[2]
    matrix = torch.zeros([samples,num,dim])

    for i in range(samples-(num-1)):
            matrix[i+(num-1)] = inp[0][i:i+num,:]

    return matrix