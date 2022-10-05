import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd 

lithology_keys = {30000: 'Sandstone',
                 65030: 'Sandstone/Shale',
                 65000: 'Shale',
                 80000: 'Marl',
                 74000: 'Dolomite',
                 70000: 'Limestone',
                 70032: 'Chalk',
                 88000: 'Halite',
                 86000: 'Anhydrite',
                 99000: 'Tuff',
                 90000: 'Coal',
                 93000: 'Basement'}

lithology_numbers = {30000: 0,
                 65030: 1,
                 65000: 2,
                 80000: 3,
                 74000: 4,
                 70000: 5,
                 70032: 6,
                 88000: 7,
                 86000: 8,
                 99000: 9,
                 90000: 10,
                 93000: 11}

Categorical = ["GROUP", "FORMATION", "WELL"]
Numerical = ['DEPTH_MD', 'X_LOC', 'Y_LOC', 'Z_LOC','CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF','DTC', 'SP', 'BS', 'ROP', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC','ROPA', 'RXO']
y_name = "FORCE_2020_LITHOFACIES_LITHOLOGY" 

def accuracy(y_test, y_pred):
    y_test = y_test.view(-1).cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    return np.mean(y_test == y_pred)


def score(y_test, y_pred, A):    
    y_test = y_test.view(-1).long().cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    S = 0.0
    for i in range(0, y_test.shape[0]):
        S -= A[y_test[i], y_pred[i]]
    return S/y_test.shape[0]


def add_noise(data, mu, sigma):
    noise = np.random.normal(mu, sigma, [data.shape[0],data.shape[1]-2])
    noise_ = np.append(noise,np.zeros((data.shape[0],2)), axis = 1)
    
    return data + noise_