import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
import os
import sys
from pathlib import Path


class MNIST:
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        batch_size: int = 256,
        img_size: tuple = (28, 28),
        subsample_size: int = -1,
        num_workers: int = 2,
        valset_ratio: float = 0.05,
        normalize_imgs: bool = False,
        flatten: bool = False,  # Whether to flatten images to vectors
        seed: int = None,
    ) -> None:
        super().__init__()

        data_dir.mkdir(exist_ok=True, parents=True)
        dataset_dir = data_dir.joinpath(Path("MNIST"))
        dataset_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_dir = dataset_dir

        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.subsample_size = subsample_size
        self.trainset_ration = 1 - valset_ratio
        self.valset_ratio = valset_ratio
        self.normalize_imgs = normalize_imgs
        self.flatten = flatten  # Whether to flatten images to vectors
        
        self.generator = None
        if seed:
            self.seed = seed
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

        transformations = [
            transforms.ToImage(),  # Convert PIL Image/NumPy to tensor
            transforms.ToDtype(
                torch.float32, scale=True
            ),  # Scale to [0.0, 1.0] and set dtype
        ]
        if self.img_size != (28, 28):
            transformations.insert(0, transforms.Resize(img_size))
        if self.normalize_imgs:
            transformations.append(transforms.Normalize((0.1307,), (0.3081,))) # Values Specific to MNIST
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
        train_dataset = datasets.MNIST(
            root=self.dataset_dir,
            train=True,
            transform=self.transformations,
            download=True,
        )
        test_dataset = datasets.MNIST(
            root=self.dataset_dir,
            train=False,
            transform=self.transformations,
            download=True,
        )
        
        # Subsample the dataset uniformly
        if self.subsample_size != -1:
            indices = torch.randperm(len(train_dataset), generator=self.generator)[:self.subsample_size]
            train_dataset = Subset(train_dataset, indices.tolist())

        if self.valset_ratio == 0.0:
            trainset = train_dataset
            valset = None
            testset = test_dataset
        else:
            trainset, valset = random_split(
                train_dataset,
                [self.trainset_ration, self.valset_ratio],
                generator=self.generator,
            )
            testset = test_dataset
   
        self.train_loader = self._build_dataloader(trainset)
        self.val_loader = (
            self._build_dataloader(valset) if self.valset_ratio > 0 else None
        )
        self.test_loader = self._build_dataloader(testset)

    def _build_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.generator
        )
        return dataloader