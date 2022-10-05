import torch.nn as nn
import torch
import utils

class torch_LSTM(nn.Module):
    
    # Constructor
    def __init__(self,window_size, input_size, output_size, hidden_dim, n_layers, device):
        super(torch_LSTM, self).__init__()
        self.device = device
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.window_size = window_size
        
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        
    # Prediction
    def forward(self, x):
        windows = utils.get_windows(self.window_size, x).to(self.device)
        
        batch_size = windows.size(0)
        
        hidden = self.init_hidden(batch_size)
        cell_state = self.init_hidden(batch_size)

        out, hidden = self.rnn(windows, (hidden,cell_state))

        hidden = self.fc(hidden[0][len(hidden[0])-1])
        
        return hidden, out 
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden

