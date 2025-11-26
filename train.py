import os
PYTHON_HASH_SEED = 0
os.environ["PYTHONHASHSEED"] = str(PYTHON_HASH_SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
import comet_ml
import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 
torch.set_float32_matmul_precision("high")

from src.datasets import dataset_factory, dataset_wrappers
from src.models import model_factory
from src.trainers import StandardTrainer, utils as trainer_utils
from src.models import utils as model_utils

import matplotlib.pyplot as plt
import seaborn as sns


import torchvision.transforms.v2 as transformsv2
from torch.utils.data import Dataset, Subset, ConcatDataset
from functools import partial
from pathlib import Path
import pickle
import argparse
import dotenv
import yaml
import pickle
import copy
import random
import numpy as np
from torchmetrics import ConfusionMatrix
import json
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import re
import math



def parse_augmentations(dataset_cfg:dict):
    aug_list = dataset_cfg.pop('augmentations')
    augmentations = []
    if 'rnd_crop' in aug_list:
        augmentations.append(transformsv2.RandomCrop(dataset_cfg['img_size'], padding=4))
    if 'rnd_horizontal_flip' in aug_list:
        augmentations.append(transformsv2.RandomHorizontalFlip())
    
    dataset_cfg['augmentations'] = augmentations
    return dataset_cfg

def parse_model_configs(model_cfg:dict):
    model_cfg_list = []
    interpolation_ths = model_cfg.pop('interpolation_ths', None)
    interpolation_idx = None
    if model_cfg['type'] == 'fc1':
        params = model_cfg.pop('hidden_dims')
        for pidx, param in enumerate(params):
            mcfg = copy.deepcopy(model_cfg)
            mcfg['hidden_dim'] = param
            model_cfg_list.append(mcfg)
            if interpolation_ths and interpolation_ths == param:
                interpolation_idx = param
    elif model_cfg['type'] == 'fcN':
        params = model_cfg.pop('hidden_dims')
        for pidx, param in enumerate(params):
            mcfg = copy.deepcopy(model_cfg)
            mcfg['h_dims'] = param
            model_cfg_list.append(mcfg)
            if interpolation_ths and interpolation_ths == param:
                interpolation_idx = param
    elif model_cfg['type'] == 'cnn5':
        params = model_cfg.pop('num_channels')
        for pidx, param in enumerate(params):
            mcfg = copy.deepcopy(model_cfg)
            mcfg['num_channels'] = param
            model_cfg_list.append(mcfg)
            if interpolation_ths and interpolation_ths == param:
                interpolation_idx = param
    elif 'resnet' in model_cfg['type']:
        params = model_cfg.pop('init_channels')
        for pidx, param in enumerate(params):
            mcfg = copy.deepcopy(model_cfg)
            mcfg['init_channels'] = param
            model_cfg_list.append(mcfg)
            if interpolation_ths and interpolation_ths == param:
                interpolation_idx = param
    if interpolation_ths and not interpolation_idx:
        raise ValueError(f'The interpolation threshold {interpolation_ths} is not present in the specified model capacities.')
    return model_cfg_list, interpolation_idx



def train(outputs_dir:Path, cfg:dict, cfg_name:str):
    
    dataset_cfg = cfg["dataset"]
    dataset_cfg = parse_augmentations(dataset_cfg)
    model_cfg_list, interpolation_idx = parse_model_configs(model_cfg = cfg["model"])
    base_dataset, num_classes = dataset_factory.create_dataset(cfg['dataset'])
    
    if 'noise_config' in cfg:
        base_dataset.inject_noise(**cfg['noise_config'])
    
    for idx, model_cfg in enumerate(model_cfg_list):
        weight_init = None
        if cfg['expexperiment_typee'] == 'belkin_weigth_reuse':
            if idx == 0:
                weight_init = model_utils.init_xavier_uniform
            elif idx > interpolation_idx:
                weight_init = partial(model_utils.init_normal, mean=0.0, std=0.1)
            else:
                weight_init = partial(model_utils.init_normal, mean=0.0, std=0.1)

        model_cfg['weight_init'] = weight_init
        
        base_model = model_factory.create_model(model_cfg, num_classes)
        
        
        
    

def train_parallel(outputs_dir:Path, cfg:dict, cfg_name:str, cpe:float, gpe:float):
    pass



from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    ranks = trainer_utils.setup_distributed()


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    
    
    parser.add_argument(
        "-p",
        "--parallel",
        help="Whether to run multiple training instances on a single GPU. If set, please specify `cpe` and `gpe` arguments.",
        action="store_true",
    )
    
    parser.add_argument(
        "--cpe",
        help="If in parallel mode, this argument specifies cpu per experiment resources for Ray.",
        type=float,
    )
    
    parser.add_argument(
        "--gpe",
        help="If in parallel mode, this argument specifies gpu per experiment resources for Ray.",
        type=float,
    )
    
    
    args = parser.parse_args()

    if args.parallel and ((args.cpe and args.gpe) is None):
        raise ValueError('When in parallel mode, the values of the `cpe` and `gpe` arguments must be specified.')
    
    
    dotenv.load_dotenv(".env")
    

    
    cfg_path = Path("configs").absolute() / f"{args.config}.yaml"
    
    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)
        
    if cfg['experiment_type'] == 'weigth_reuse' and args.parallel:
        raise ValueError('Experiments which use `weight_reuse` cannot be run in parallel mode.')

    outputs_dir = Path("outputs/").absolute() / f"{args.config}"
    outputs_dir.mkdir(exist_ok=True, parents=True)
    # results_dir = Path("results/").absolute() / f"{args.config}"
    # results_dir.mkdir(exist_ok=True, parents=True)
    
    
    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])


    if args.parallel:
        train_parallel(outputs_dir, cfg, cfg_path.stem, args.cpe, args.gpe)
    else:
        train(outputs_dir, cfg, cfg_path.stem)
if __name__ == "__main__":
    main()