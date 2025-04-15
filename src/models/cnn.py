import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))


class FiveCNN(nn.Module):
    
    
    def __init__(
        self,
        num_channels=64,
        num_classes=10
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            # Layer 0
            nn.Conv2d(3, num_channels, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            # Layer 1
            nn.Conv2d(num_channels, num_channels*2, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(num_channels*2, num_channels*4, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(num_channels*4, num_channels*8, kernel_size=3,
                    stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_channels*8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.MaxPool2d(4),
            Flatten(),
            nn.Linear(num_channels*8, num_classes, bias=True)
        )

        
    
    
    def forward(self, x):
        
        return self.net(x)