import torch
import argparse
import Modules
import utils
import numpy as np

def predict(model, data):
    model.eval()
    tensors = []
    for i in data.UniqueNames:
        tensors.append(torch.Tensor.argmax(model.forward(data.datasets[i].unx)[0], axis=1))

    return torch.cat(tuple(tensors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=26, type=int)
    parser.add_argument('--output_size', default=12, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--input_type', default='direct', type=str, choices=['pred','direct'])

    parser.add_argument('--from_file', required=True, type=str)
    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--label_file', required=True, type=str)

    opt = parser.parse_args()

    model = Modules.torch_RNN(opt.input_size, opt.output_size, opt.hidden_dim, opt.n_layers, torch.device("cpu"))
    print(model.parameters)
    
    if opt.from_file != '.':
        model.load_state_dict(torch.load(opt.from_file))
    
    data = utils.import_data(torch.device("cpu"),opt.data_file, opt.label_file)

    prediction = predict(model, data)

    print(prediction)

    np.savetxt('prediction.txt', prediction.numpy())