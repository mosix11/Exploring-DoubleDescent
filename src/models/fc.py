import torch
import torch.nn as nn
import torch.nn.functional as F


class FC1(nn.Module):
    
    def __init__(
        self,
        input_dim=784,
        hidden_dim=512,
        ouput_dim=10
    ):
        super().__init__()
        
        self.h1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, ouput_dim, bias=True)
        
        
        
    def forward(self, x: torch.Tensor):
        
        x = self.h1(x)
        x = self.out(x)
        
        return x