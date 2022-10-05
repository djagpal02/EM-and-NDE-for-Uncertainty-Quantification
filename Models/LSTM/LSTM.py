from Modules import torch_LSTM
from train import train
from predict import predict
import utils 
import torch
import numpy as np

class dummy_args():
    def __init__(self,window_size, input_size, output_size, hidden_dim, n_layers, optim, milestones, class_weights, lr, weight_decay, n_epoch, batchsize, momentum, savemodel, savemodelroot, run_name, active_log ):
        self.window_size      = window_size
        self.input_size       = input_size       
        self.output_size      = output_size      
        self.hidden_dim       = hidden_dim       
        self.n_layers         = n_layers                
        self.optim            = optim            
        self.milestones       = milestones       
        self.class_weights    = class_weights    
        self.lr               = lr               
        self.weight_decay     = weight_decay     
        self.n_epoch          = n_epoch          
        self.batchsize        = batchsize        
        self.momentum         = momentum
        self.savemodel        = savemodel         
        self.savemodelroot    = savemodelroot    
        self.run_name         = run_name         
        self.active_log       = active_log      

class LSTM:
    def __init__(self, window_size = 1, input_size = 26, output_size = 12, hidden_dim = 16, n_layers = 1, device = None):
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.window_size = window_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.__model = torch_LSTM(window_size, input_size, output_size, hidden_dim, n_layers, self.device).to(self.device)

    def fit(self, x, y, xt = None, yt = None, optim = 'adamw', milestones = [], class_weights = 'bal', lr = 0.001, weight_decay = 0.1, n_epoch = 100, batchsize = 128, momentum = 0.1, savemodel = True, savemodelroot = './bestmodels/RNN', run_name = 'testrun', active_log = True):
        opt = dummy_args(self.window_size, self.input_size, self.output_size, self.hidden_dim, self.n_layers, optim, milestones, class_weights, lr, weight_decay, n_epoch, batchsize, momentum, savemodel, savemodelroot, run_name, active_log )
        self.model = train(self.__model, opt, self.device, x, y, xt, yt)
        print('Model Fitting Complete')

    def get_params(self):
        params = self.__model.parameters
        return params
    
    def state_dict(self):
        return self.__model.state_dict()

    def predict(self, x, y):
        data = utils.RNN_dataset(self.device, x, y)
        y = predict(self.__model, data)
        return y
    
    def score(self, x, y):
        y_pred = self.predict(x, y)
        y_pred = y_pred.cpu().numpy()
        y = y.to_numpy().reshape(-1)
        return np.mean(y == y_pred)

    def set_params(self, params):
        self.__model.load_state_dict(params)
        print('Parameters Loaded')

