import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, FC2, FC3, CNN5
from src.trainers import TrainerEp, TrainerGS
import matplotlib.pyplot as plt
from src.utils import nn_utils
import torch
import torchmetrics
import torchvision.transforms.v2 as transformsv2
from functools import partial
from pathlib import Path
import pickle
import argparse
import os
import dotenv

from ray import train, tune
from ray.tune import TuneConfig, RunConfig, FailureConfig
import numpy as np


def train_fc_mog_parallel(outputs_dir: Path):
    max_epochs = 1000
    num_samples = 100000
    batch_size = 1024
    num_features = 512
    num_classes = 30
    label_noise = 0.2
    
    training_seed = 22
    dataset_seed = 22
    
    gpu_per_experiment:float = 1
    cpu_per_experiment:float = 10
    
    log_comet = True
    
    
    fc1_h_widths= np.array([
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        50,
        56,
        64,
        72,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        216,
        224,
        232,
        240,
        248,
        256,
        264,
        280,
        296,
        304,
        320,
        336,
        344,
        368,
        392,
        432,
        464,
        512,
        640,
        768,
        896,
        1024,
        2048,
        3072,
        4096,
        8192,
        16384
    ])
    
    
    fc2_h_width = np.array([
        [1, 1],
        [1, 18],
        [1, 52],
        [4, 19],
        [6, 5],
        [8, 6],
        [8, 34],
        [4, 112],
        [7, 77],
        [12, 21],
        [5, 140],
        [5, 155],
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        44,
        56,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        216,
        224,
        232,
        240,
        248,
        256,
        264,
        280,
        296,
        304,
        320,
        336,
        344,
        368,
        392,
        432,
        464,
        512,
        640,
        768,
        896,
        1024,
        2048,
        3072,
        4096,
        8192,
        16384
    ])
    
    fc1_param_counts = (num_features + 1) * fc1_h_widths + (fc1_h_widths + 1) * num_classes
    print(fc1_param_counts)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="The model to use for training.",
        type=str,
        choices=["fc", "cnn5", "resnet18k"],
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="The dataset used for trainig the model.",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "mog"],
        required=True,
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume training from the last checkpoint.",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        help="Whether to run experiments in parallel.",
        action="store_true"
    )
    args = parser.parse_args()
    
    
    dotenv.load_dotenv('.env')

    outputs_dir = Path("outputs/width_depth").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc" and args.dataset == "mnist":
        if args.parallel:
            # train_fc1_mnist_parallel(outputs_dir)
            pass
        else:
            # train_fc1_mnist(outputs_dir)
            pass
    elif args.model == "fc" and args.dataset == "cifar10":
        # train_fc1_cifar10(outputs_dir)
        pass
    elif args.model == "fc" and args.dataset == "mog":
        if args.parallel:
            # train_fc1_mog_parallel(outputs_dir)
            train_fc_mog_parallel(outputs_dir)
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        if args.parallel:
            # train_cnn5_cifar10_parallel(outputs_dir)
            pass
        else:
            # train_cnn5_cifar10(outputs_dir)
            pass