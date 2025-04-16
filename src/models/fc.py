import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class FC1(nn.Module):
    
    def __init__(
        self,
        input_dim=784,
        hidden_dim=512,
        ouput_dim=10,
        loss_fn=None,
        metric=None
    ):
        super().__init__()
        
        self.h1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, ouput_dim, bias=True)
        
        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn
        if metric:
            self.metric = metric
        
    def training_step(self, x, y, use_amp=False):
        with autocast('cuda', enabled=use_amp):
            preds = self(x)
            loss = self.loss_fn(y, preds)
        if self.metric:
            met = self.metric(y, preds)
            return loss, met
        else: return loss, None
        
    def validation_step(self, x, y, use_amp=False):
        with torch.no_grad():
            with autocast('cuda', enabled=use_amp):
                preds = self(x)
                loss = self.loss_fn(y, preds)
        if self.metric:
            met = self.metric(y, preds)
            return loss, met
        else: return loss, None
    
    
    def predict(self, x):
        with torch.no_grad():
            preds = self(x)
        return preds
    
    def forward(self, x: torch.Tensor):
        x = self.h1(x)
        x = self.out(x)
        return x