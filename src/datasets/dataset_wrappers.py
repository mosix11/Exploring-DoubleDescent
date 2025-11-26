import torch
from torch.utils.data import Dataset, Subset
import warnings
import numpy as np
from typing import Optional, List, Tuple, Union
import math
import numpy as np
from PIL import Image
import hashlib

class DatasetWithIndex(Dataset):
    
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if len(data) == 3:
            x, y, is_noisy = data
            return x, y, idx, is_noisy 
        elif len(data) == 2:
            x, y = data
            return x, y, idx
        else:
            raise RuntimeError('Data structure unknown!')

class LabelRemapper(Dataset):
    """
    Wraps any dataset whose __getitem__ returns (x, y)
    and remaps y via a provided dict mapping_orig2new.
    """
    def __init__(self, dataset: Dataset, mapping_orig2new: dict):
        super().__init__()
        self.dataset = dataset
        self.map = mapping_orig2new

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if len(data) == 3:
            x, y, is_noisy = data
            key = y.item() if isinstance(y, torch.Tensor) else y
            return x, self.map[key], is_noisy
        elif len(data) == 2:
            x, y = data
            key = y.item() if isinstance(y, torch.Tensor) else y
            return x, self.map[key]
        else:
            raise RuntimeError('Data structure unknown!')





class NoisyClassificationDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_name: str = None,
        noise_type: str = 'symmetric',  # 'symmetric', 'asymmetric', 'constant', 'IC', 'T_matrix'
        T_mat: np.ndarray = None,       # only for noise_type='T_matrix'
        noise_rate: float | int = 0.0,
        num_classes: int = None,
        target_class: int = None,       # Only for 'constant' noise
        class_swap: Tuple[int, int] | List[int] = None,  # Only for IC
        available_labels: list = None,
        seed=None,
        generator=None,
        input_noise: bool = False       # <<< NEW: replace inputs with full uniform noise if True
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.noise_type = noise_type
        self.class_swap = class_swap
        self.input_noise = input_noise   # <<< NEW
        

        # --- Handle noise_rate normalization ---
        dataset_len = len(self.dataset)
        if isinstance(noise_rate, float):
            if not (0.0 <= noise_rate <= 1.0):
                raise ValueError("If noise_rate is float, it must be between 0 and 1.")
            self.noise_rate = noise_rate
            self.num_noisy_samples = int(noise_rate * dataset_len)
        elif isinstance(noise_rate, int):
            if not (0 <= noise_rate <= dataset_len):
                raise ValueError(f"If noise_rate is int, it must be between 0 and {dataset_len}.")
            self.noise_rate = noise_rate / dataset_len
            self.num_noisy_samples = noise_rate
        else:
            raise TypeError("noise_rate must be either float (fraction) or int (count).")

        if available_labels is None:
            self.available_labels = list(range(num_classes))
        else:
            self.available_labels = available_labels

        self.num_classes = num_classes

        if noise_type == 'constant':
            if target_class is not None:
                self.target_class = target_class
            else:
                raise ValueError('For constant noise, the target class should be specified!')
        elif noise_type == 'T_matrix':
            if T_mat is not None:
                self.T_mat = T_mat
            else:
                raise ValueError('For generating noise based on Transition Matrix, pass `T_mat`.')

        if seed and not generator:
            generator = torch.Generator().manual_seed(seed)
        elif seed and generator:
            generator = generator.manual_seed(seed)

        self.seed = seed
        self.generator = generator

        self.noisy_labels = None
        self.is_noisy_flags = None

        self.return_clean_labels = False

        self._add_noise_to_labels()

    def switch_to_clean_lables(self):
        self.return_clean_labels = True

    def switch_to_noisy_lables(self):
        self.return_clean_labels = False

    def replace_labels(self, new_labels):
        self.noisy_labels = new_labels
        for idx, orig_lbl in enumerate(self.original_labels):
            self.is_noisy_flags[idx] = torch.tensor(1, dtype=torch.float32) if orig_lbl != new_labels[idx] else torch.tensor(0, dtype=torch.float32)

    def get_original_labels(self):
        return self.original_labels

    def __len__(self):
        return len(self.dataset)

    def _get_original_labels(self):
        """
        Traverses wrapped datasets (Subset) to get the original labels
        from the base dataset, correctly composing indices from Subsets.
        """
        current_dataset = self.dataset
        indices_chain = []

        while isinstance(current_dataset, Subset):
            indices_chain.append(current_dataset.indices)
            current_dataset = current_dataset.dataset

        base_dataset = current_dataset

        if hasattr(base_dataset, 'targets'):
            if isinstance(base_dataset.targets, list):
                base_labels = torch.tensor(base_dataset.targets, dtype=torch.long)
            else:
                base_labels = torch.as_tensor(base_dataset.targets, dtype=torch.long)
        else:
            warnings.warn("Base dataset has no .targets attribute. Iterating to extract labels (slow).")
            base_labels = torch.tensor([label for _, label in base_dataset], dtype=torch.long)

        if not indices_chain:
            return base_labels

        indices_chain.reverse()
        composed_indices = indices_chain[0]
        for i in range(1, len(indices_chain)):
            next_level_indices = indices_chain[i]
            composed_indices = [composed_indices[j] for j in next_level_indices]

        original_labels = base_labels[torch.tensor(composed_indices)]
        return original_labels

    def _add_noise_to_labels(self):
        original_labels = self._get_original_labels()
        self.original_labels = original_labels
        noisy_labels = original_labels.clone()
        self.is_noisy_flags = torch.zeros(len(original_labels), dtype=torch.float32)

        if self.num_noisy_samples == 0:
            self.noisy_labels = noisy_labels
            return

        if self.noise_type == 'symmetric':
            noisy_indices = torch.randperm(len(original_labels), generator=self.generator)[:self.num_noisy_samples]
            for idx in noisy_indices:
                current_label = original_labels[idx].item()
                possible_flips = [l for l in self.available_labels if l != current_label]
                if possible_flips:
                    rand_idx = torch.randint(0, len(possible_flips), (1,), generator=self.generator).item()
                    noisy_labels[idx] = possible_flips[rand_idx]
                    self.is_noisy_flags[idx] = torch.tensor(1, dtype=torch.float32)

        elif self.noise_type == 'constant':
            if self.num_noisy_samples > 0:
                noisy_indices = torch.randperm(len(original_labels), generator=self.generator)[:self.num_noisy_samples]
                noisy_labels[noisy_indices] = self.target_class
                self.is_noisy_flags[noisy_indices] = torch.tensor(1, dtype=torch.float32)

        elif self.noise_type == 'IC':
            if self.dataset_name == 'MNIST':
                swap_pairs = [tuple(self.class_swap)] if self.class_swap else [(1, 7)]
            elif self.dataset_name == 'CIFAR10':
                swap_pairs = [tuple(self.class_swap)] if self.class_swap else [(3, 5)]
            elif self.dataset_name == 'CIFAR100':
                swap_pairs = [tuple(self.class_swap)] if self.class_swap else [(47, 52)]
            else:
                raise ValueError(f"IC noise not implemented for dataset '{self.dataset_name}'.")

            for a, b in swap_pairs:
                idx_a = (original_labels == a).nonzero(as_tuple=True)[0]
                idx_b = (original_labels == b).nonzero(as_tuple=True)[0]
                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue
                num_to_swap = min(int(self.noise_rate * len(idx_a)),
                                  int(self.noise_rate * len(idx_b)))
                if num_to_swap > 0:
                    perm_a = torch.randperm(len(idx_a), generator=self.generator)[:num_to_swap]
                    perm_b = torch.randperm(len(idx_b), generator=self.generator)[:num_to_swap]
                    swap_a = idx_a[perm_a]
                    swap_b = idx_b[perm_b]
                    noisy_labels[swap_a] = b
                    noisy_labels[swap_b] = a
                    self.is_noisy_flags[swap_a] = torch.tensor(1, dtype=torch.float32)
                    self.is_noisy_flags[swap_b] = torch.tensor(1, dtype=torch.float32)

        elif self.noise_type == 'asymmetric':
            if self.dataset_name is None:
                raise ValueError("To inject asymmetric noise, specify dataset_name.")

            noise_map = {}
            if self.dataset_name == 'MNIST':
                noise_map = {7: 1, 2: 7, 5: 6, 6: 5, 3: 8}
            elif self.dataset_name == 'CIFAR10':
                noise_map = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}
            elif self.dataset_name == 'CIFAR100':
                coarse_labels = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
                super_class_map = [[] for _ in range(20)]
                for i, c in enumerate(coarse_labels):
                    super_class_map[c].append(i)
                for super_class in super_class_map:
                    for i in range(len(super_class)):
                        source_label = super_class[i]
                        target_label = super_class[(i + 1) % len(super_class)]
                        noise_map[source_label] = target_label
            elif self.dataset_name == 'Clothing1M':
                noise_map = {
                    2: 4, 4: 2, 9: 6, 12: 11, 8: 12, 13: 12, 6: 8, 10: 2
                }
            else:
                raise ValueError(f"Asymmetric noise not implemented for dataset '{self.dataset_name}'.")

            for source_label, target_label in noise_map.items():
                class_indices = (original_labels == source_label).nonzero(as_tuple=True)[0]
                if len(class_indices) == 0:
                    continue
                num_to_flip = int(self.noise_rate * len(class_indices))
                if num_to_flip > 0:
                    perm = torch.randperm(len(class_indices), generator=self.generator)
                    indices_to_flip_in_class = class_indices[perm[:num_to_flip]]
                    noisy_labels[indices_to_flip_in_class] = target_label
                    self.is_noisy_flags[indices_to_flip_in_class] = torch.tensor(1, dtype=torch.float32)

        elif self.noise_type == 'T_matrix':
            if not isinstance(self.T_mat, np.ndarray):
                raise TypeError("T_mat must be a numpy.ndarray of shape (num_classes, num_classes).")
            K = self.num_classes
            if self.T_mat.shape != (K, K):
                raise ValueError(f"T_mat must have shape ({K}, {K}), got {self.T_mat.shape}.")
            T = self.T_mat
            tol_row = 1e-6
            tol_diag = 1e-9
            row_sums = T.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=tol_row):
                raise ValueError(f"T_mat rows must sum to 1 (±{tol_row}). Got {row_sums}.")
            if not np.all(np.abs(np.diag(T)) <= tol_diag):
                raise ValueError(f"T_mat diagonal must be ~0 (≤{tol_diag}). Got {np.diag(T)}.")
            if (T < -tol_diag).any():
                raise ValueError("T_mat must be nonnegative.")

            T_torch = torch.from_numpy(T).to(dtype=torch.float64)
            for cls in range(K):
                class_indices = (original_labels == cls).nonzero(as_tuple=True)[0]
                if len(class_indices) == 0:
                    continue
                num_to_flip = int(self.noise_rate * len(class_indices))
                if num_to_flip <= 0:
                    continue
                perm = torch.randperm(len(class_indices), generator=self.generator)
                idx_to_flip = class_indices[perm[:num_to_flip]]
                probs = T_torch[cls]
                sampled_targets = torch.multinomial(
                    probs, num_samples=num_to_flip, replacement=True, generator=self.generator
                ).to(dtype=torch.long)
                noisy_labels[idx_to_flip] = sampled_targets
                self.is_noisy_flags[idx_to_flip] = torch.tensor(1, dtype=torch.float32)

        self.noisy_labels = noisy_labels.long()
        
        
        
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD  = (0.26862954, 0.26130258, 0.27577711)

    # --- NEW: per-index, deterministic CPU generator ---
    def _per_sample_generator(self, idx: int) -> torch.Generator:
        """
        Create a torch.Generator seeded deterministically from (base seed, idx).
        Same idx -> same seed -> same noise, regardless of DataLoader batch/worker.
        """
        base = int(self.seed) if self.seed is not None else 0
        payload = f"input_noise|{base}|{idx}".encode()
        seed64 = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return torch.Generator(device="cpu").manual_seed(seed64)

    # --- CHANGED: now takes idx and (hard-)normalizes noise for float tensors ---
    def _uniform_noise_like_image(self, data, idx: int):
        """
        Full uniform noise matching type/shape of `data`.
        For float CHW tensors: draws U[0,1], then applies (x - mean)/std using the
        hard-coded OPENAI_* stats so the result is already in normalized space.
        """
        gen = self._per_sample_generator(idx)

        # Torch tensor path
        if isinstance(data, torch.Tensor):
            shape = data.shape

            if data.dtype == torch.uint8:
                # Raw 0..255 noise (no normalization applied here)
                return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)

            # Float tensor: assume CHW; generate in [0,1], then normalize with hard-coded stats
            noise = torch.rand(size=shape, dtype=torch.float32, generator=gen)

            if noise.ndim == 3 and shape[0] in (1, 3, 4):
                C = shape[0]
                # Use first C channels of the provided stats; broadcast to CHW
                mean = torch.tensor(self.OPENAI_DATASET_MEAN[:C], dtype=noise.dtype).view(C, 1, 1)
                std  = torch.tensor(self.OPENAI_DATASET_STD[:C],  dtype=noise.dtype).view(C, 1, 1)
                noise = (noise - mean) / std

            # Preserve original floating dtype
            return noise.to(dtype=data.dtype)

        # PIL image path (kept for completeness; typically your data is already a tensor)
        if isinstance(data, Image.Image):
            w, h = data.size
            bands = len(data.getbands())  # 1=L, 3=RGB, 4=RGBA
            if bands == 1:
                arr = torch.randint(0, 256, (h, w), dtype=torch.uint8, generator=gen).cpu().numpy()
            else:
                arr = torch.randint(0, 256, (h, w, bands), dtype=torch.uint8, generator=gen).cpu().numpy()
            return Image.fromarray(arr)

        raise TypeError(f"Unsupported data type for input noise: {type(data)}")


    def __getitem__(self, idx):
        data, _ = self.dataset[idx]
        if self.input_noise:
            data = self._uniform_noise_like_image(data, idx)

        if self.noisy_labels is None:
            return data, self.original_labels[idx], torch.tensor(0, dtype=torch.float32)

        if self.return_clean_labels:
            return data, self.original_labels[idx], torch.tensor(0, dtype=torch.float32)
        else:
            noisy_label = self.noisy_labels[idx]
            is_noisy = self.is_noisy_flags[idx]
            return data, noisy_label, is_noisy


class BinarizedClassificationDataset(Dataset):
    
    def __init__(self, dataset: Dataset, target_class:int):
        super().__init__()
        self.dataset = dataset
        self.target_class = target_class
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, tuple):
            data = list(data)
        y = data[1]
        if y == self.target_class:
            data[1] = torch.as_tensor(1.0, dtype=torch.long)
        else:
            data[1] = torch.as_tensor(0.0, dtype=torch.long)
        return data
    