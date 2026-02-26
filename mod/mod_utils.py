import torch
from torch import nn
import numpy as np


### normalize to [-1,1] # TODO: check what happens if the x,y exceed the bounds, is it okay ? should we use a true x_min, x_max from the dataset ?
def normalize_coords_siren(inputs, x_min=-60, x_max=80, y_min=-40, y_max=20):
    x = 2 * (inputs[:, 0] - x_min) / (x_max - x_min) - 1
    y = 2 * (inputs[:, 1] - y_min) / (y_max - y_min) - 1
    
    total_seconds_day = 24 * 60 * 60  # 86400
    seconds_since_midnight = (inputs[:, 2] % total_seconds_day)
    t = seconds_since_midnight / total_seconds_day  # normalize time to [0, 1] within a day
    
    t = t * 2 - 1  # rescale time to [-1, 1]
    
    return torch.stack([x, y, t], dim=-1)


def normalize_coords_space(inputs, x_min=-60, x_max=80, y_min=-40, y_max=20):
    x = (inputs[:, 0] - x_min) / (x_max - x_min)
    y = (inputs[:, 1] - y_min) / (y_max - y_min)

    x = x * 2 - 1
    y = y * 2 - 1

    return torch.stack([x, y], dim=-1)


class Sine(nn.Module):
    def __init__(self, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, input):
        return torch.sin(self.omega_0 * input)
    
    
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False, use_bias=True):
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.act = Sine(omega_0=omega_0)
        
        with torch.no_grad():
            if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                                np.sqrt(6 / self.in_features) / self.omega_0)

        if use_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        return self.act(self.linear(input))   