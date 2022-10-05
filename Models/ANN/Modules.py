import torch.nn as nn
import torch

class Net(nn.Module):
    
    # Constructor
    def __init__(self, Layers, activation, dropout = 0):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        self.softmax = nn.Softmax(dim = 1)

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
         
        # input is a list of neurons per layer, loop adds said layers to model with He initialisation
        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            if activation == 'relu' or activation == 'leaky_relu':
                torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            elif activation == 'sigmoid' or activation == 'tanh':
                torch.nn.init.xavier_uniform_(linear.weight, gain=1.0)
            self.hidden.append(linear)

        self.dropout = nn.Dropout(dropout)
    # Prediction
    def forward(self, activation):
        # Get Number of Layers
        L = len(self.hidden)
        # Loop over int 0-L and layers
        for (l, linear_transform) in zip(range(L), self.hidden):
            # If current layer is not last layer (L-1) apply transformation and acivation function
            if l < L - 1:
                activation = self.dropout(activation)
                activation = self.activation_fn(linear_transform(activation))
            # If current layer is last layer then only apply linear transformation
            else:
                activation = linear_transform(activation)
        return activation 



    


