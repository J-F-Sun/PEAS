"""The only trainable component: tanh hidden layers, linear scalar output."""
from torch import nn

class TransientMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = int(config['hidden_dim'])
        layers = [nn.Linear(int(config['input_dim']),width), nn.Tanh()]
        for _ in range(int(config['num_hidden_layers'])-1):
            layers += [nn.Linear(width,width),nn.Tanh()]
        layers.append(nn.Linear(width,1))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)
