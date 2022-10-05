import torch
import argparse
import Modules
import utils
import numpy as np


def predict(model, data, vector = False):
    model.eval()
    if vector:
        return model.softmax(model.forward(data.x))
    else:
        return torch.Tensor.argmax(model.forward(data.x), axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Layers', default=[26,12], type=list)
    parser.add_argument('--activation', default=None, type=str, choices = ['relu', 'leaky_relu', 'sigmoid', 'tanh'])
    parser.add_argument('--dropout', default=0, type=float)

    parser.add_argument('--from_file', required=True, type=str)
    parser.add_argument('--data_file', required=True, type=str)


    opt = parser.parse_args()
    
    model = Modules.Net(opt.Layers, opt.activation, opt.dropout)
    print(model.parameters)
    
    if opt.from_file != '.':
        model.load_state_dict(torch.load(opt.from_file))
    
    data = utils.import_data(opt.data_file)

    prediction = predict(model, data)

    print(prediction)

    np.savetxt('prediction.txt', prediction.numpy())