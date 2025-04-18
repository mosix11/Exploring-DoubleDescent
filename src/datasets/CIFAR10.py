import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset


import os
import sys
from pathlib import Path


class CIFAR10:

    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        batch_size: int = 256,
        img_size: tuple = (32, 32),
        subsample_size: int = -1,
        class_subset: list = [],
        grayscale: bool = False,
        normalize_imgs: bool = False,
        flatten: bool = False,  # Whether to flatten images to vectors
        valset_ratio: float = 0.05,
        num_workers: int = 2,
        seed: int = None,
    ) -> None:

        super().__init__()

        if not data_dir.exists():
            raise RuntimeError("The data directory does not exist!")
        dataset_dir = data_dir.joinpath(Path("CIFAR10"))
        if not dataset_dir.exists():
            dataset_dir.mkdir()
        self.dataset_dir = dataset_dir

        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.subsample_size = subsample_size
        if class_subset != None and len(class_subset) >= 1:
            self.class_subset = class_subset
        self.grayscale = grayscale
        self.normalize_imgs = normalize_imgs
        self.flatten = flatten  # Store the flatten argument
        self.trainset_ration = 1 - valset_ratio
        self.valset_ratio = valset_ratio
        self.seed = seed

        transformations = [
            transforms.ToImage(),  # Convert PIL Image/NumPy to tensor
            transforms.ToDtype(
                torch.float32, scale=True
            ),  # Scale to [0.0, 1.0] and set dtype
        ]
        
        if self.grayscale:
            transformations.insert(0, transforms.Grayscale(num_output_channels=1))
        
        if self.img_size != (32, 32):
            transformations.insert(0, transforms.Resize(img_size))
            
        if self.normalize_imgs:
            mean, std = (
                (0.5,), (0.5,)
                if self.grayscale
                else ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Values Specific to CIFAR
            )
            transformations.append(transforms.Normalize(mean, std))

            
        if self.flatten:
            transformations.append(transforms.Lambda(lambda x: torch.flatten(x)))  # Use Lambda for flattening

        self.transformations = transforms.Compose(transformations)

        self._init_loaders()

    def get_train_dataloader(self):
        return self.train_loader

    def get_val_dataloader(self):
        return self.val_loader

    def get_test_dataloader(self):
        return self.test_loader

    def _init_loaders(self):
        train_dataset = datasets.CIFAR10(
            root=self.dataset_dir,
            train=True,
            transform=self.transformations,
            download=True,
        )
        test_dataset = datasets.CIFAR10(
            root=self.dataset_dir,
            train=False,
            transform=self.transformations,
            download=True,
        )
        
        # Subsample the dataset uniformly
        if self.subsample_size != -1:
            indices = torch.randperm(len(train_dataset))[:self.subsample_size]
            train_dataset = Subset(train_dataset, indices.tolist())
        gen = None
        if self.seed:
            gen = torch.Generator().manual_seed(self.seed)

        if self.valset_ratio == 0.0:
            trainset = train_dataset
            testset = test_dataset
        else:
            trainset, valset = random_split(
                train_dataset,
                [self.trainset_ration, self.valset_ratio],
                generator=gen,
            )
            testset = test_dataset
   
        self.train_loader = self._build_dataloader(trainset)
        self.val_loader = (
            self._build_dataloader(valset) if self.valset_ratio > 0 else None
        )
        self.test_loader = self._build_dataloader(testset)

    def _build_dataloader(self, dataset):
        gen = None
        if self.seed:
            gen = torch.Generator().manual_seed(self.seed)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=gen
        )
        return dataloader