import torch
from torch.utils.data import Dataset

class LabelRemapper(Dataset):
        """
        Wraps any dataset whose __getitem__ returns (x, y)
        and remaps y via a provided dict mapping_orig2new.
        """
        def __init__(self, base_dataset: Dataset, mapping_orig2new: dict):
            self.base = base_dataset
            self.map = mapping_orig2new

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, y = self.base[idx]
            return x, self.map[y]