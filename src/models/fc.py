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
        weight_init=None,
        loss_fn=None,
        metric=None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = ouput_dim
        self.h1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, ouput_dim, bias=True)
        
        
        if weight_init:
            self.apply(weight_init)
        
        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn
        if metric:
            self.metric = metric
            
    
    def reuse_weights(self, old_state: dict):
        """
        Load weights from a smaller FC1 model state_dict into this wider model.
        Copies the first `old_hidden` neurons exactly, and leaves the rest of the weights as they are initialized.

        Args:
            old_model_or_state: state dict of an FC1 instance.
            init_std: standard deviation for normal init of new weights.
        """
        # Sizes
        old_h = old_state['h1.weight'].shape[0]
        new_h = self.hidden_dim

        # Ensure new_h >= old_h
        if new_h < old_h:
            raise ValueError(f"New hidden_dim ({new_h}) must be >= old hidden_dim ({old_h})")

        with torch.no_grad():
            # 1) Copy and init h1 weights
            #   - copy the first old_h rows
            self.h1.weight.data[:old_h, :].copy_(old_state['h1.weight'])
            self.h1.bias.data[:old_h].copy_(old_state['h1.bias'])

            # 2) Copy and init out weights
            #   - copy the first old_h columns
            self.out.weight.data[:, :old_h].copy_(old_state['out.weight'])
            self.out.bias.data.copy_(old_state['out.bias'])

    def training_step(self, x, y, use_amp=False):        
        with autocast('cuda', enabled=use_amp):
            preds = self(x)
            if isinstance(self.loss_fn, torch.nn.MSELoss):
                y_onehot = F.one_hot(y, num_classes=self.output_dim).float()
                loss = self.loss_fn(preds, y_onehot)
            else:
                loss = self.loss_fn(preds, y)
        if self.metric:
            met = self.metric(preds, y)
            return loss, met
        else: return loss, None
        
    def validation_step(self, x, y, use_amp=False):
        with torch.no_grad():
            with autocast('cuda', enabled=use_amp):
                preds = self(x)
                if isinstance(self.loss_fn, torch.nn.MSELoss):
                    y_onehot = F.one_hot(y, num_classes=self.output_dim).float()
                    loss = self.loss_fn(preds, y_onehot)
                else:
                    loss = self.loss_fn(preds, y)
        if self.metric:
            met = self.metric(preds, y)
            return loss, met
        else: return loss, None
    
    
    def predict(self, x):
        with torch.no_grad():
            preds = self(x)
        return preds
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.h1(x))
        x = self.out(x)
        return x
    
    def get_identifier(self):
        return f"fc1|h{self.hidden_dim}"