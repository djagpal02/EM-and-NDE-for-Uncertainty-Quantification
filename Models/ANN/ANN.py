from torch._C import device
from Modules import Net
from train import train
from predict import predict
import torch
import numpy as np
import utils

class dummy_args():
    def __init__(self,layers, activation, dropout, optim, milestones, class_weights, lr, weight_decay, n_epoch, batchsize, momentum, balance, savemodel, savemodelroot, run_name, active_log ):
        self.layers           = layers       
        self.activation       = activation     
        self.dropout          = dropout             
        self.optim            = optim            
        self.milestones       = milestones       
        self.class_weights    = class_weights    
        self.lr               = lr               
        self.weight_decay     = weight_decay     
        self.n_epoch          = n_epoch          
        self.batchsize        = batchsize        
        self.momentum         = momentum
        self.balance          = balance
        self.savemodel        = savemodel         
        self.savemodelroot    = savemodelroot    
        self.run_name         = run_name         
        self.active_log       = active_log      

class ANN:
    def __init__(self, Layers, activation, dropout = 0, device = None):
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.layers = Layers
        self.activation = activation
        self.dropout = dropout
        self.__model = Net( Layers, activation, dropout).to(self.device)

    def fit(self, x, y, xt = None, yt = None, optim = 'adamw', milestones = [], class_weights = 'bal', lr = 0.001, weight_decay = 0.1, n_epoch = 100, batchsize = 1024, momentum = 0.1, balance = False, savemodel = True, savemodelroot = './bestmodels/ANN', run_name = 'testrun', active_log = True):
        opt = dummy_args(self.layers, self.activation, self.dropout, optim, milestones, class_weights, lr, weight_decay, n_epoch, batchsize, momentum, balance, savemodel, savemodelroot, run_name, active_log)
        self.model = train(self.__model, opt, self.device, x, y, xt, yt)
        print('Model Fitting Complete')

    def get_params(self):
        params = self.__model.parameters
        return params
    
    def state_dict(self):
        return self.__model.state_dict()

    def predict(self, x):
        y = predict(self.__model, utils.convert_data(x, self.device))
        return y
    
    def score(self, x, y):
        y_pred = self.predict(x)
        y_pred = y_pred.cpu().numpy()
        y = y.to_numpy().reshape(-1)
        return np.mean(y == y_pred)

    def set_params(self, params):
        self.__model.load_state_dict(params)
        print('Parameters Loaded')

