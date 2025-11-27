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

from ray import train, tune
from ray.tune import TuneConfig, RunConfig, FailureConfig



def parse_augmentations(dataset_cfg:dict):
    aug_list = dataset_cfg.pop('augmentations', [])
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



def train(outputs_dir:Path, cfg:dict):
    cfg['trainer']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    dataset_cfg = cfg["dataset"]
    dataset_cfg = parse_augmentations(dataset_cfg)
    model_cfg_list, interpolation_idx = parse_model_configs(model_cfg = cfg["model"])
    dataset, num_classes = dataset_factory.create_dataset(cfg['dataset'])
    # This is to save the best model in each capacity.
    dataset.set_valset(dataset.get_testset(), shuffle=False)
    
    
    if 'noise_config' in cfg:
        dataset.inject_noise(**cfg['noise_config'])
    
    last_capacity_expr_name = None
    for idx, model_cfg in enumerate(model_cfg_list):
        weight_init = None
        if cfg['weight_reuse']:
            if idx == 0:
                weight_init = model_utils.init_xavier_uniform
            elif idx > interpolation_idx:
                weight_init = partial(model_utils.init_normal, mean=0.0, std=0.1)
            else:
                weight_init = partial(model_utils.init_normal, mean=0.0, std=0.1)

        model_cfg['weight_init'] = weight_init
        
        model = model_factory.create_model(model_cfg, num_classes)
        
        experiment_name = model.get_identifier()
        
        if cfg['weight_reuse']:
            # In underparameterized regime, we reuse weights from the last trained model.
            # ** only for belkin epxeriments **
            if idx > 0 and idx <= interpolation_idx:
                last_capacity_weights_path = outputs_dir / Path(last_capacity_expr_name) / Path('weights/final_weights.pth')
                last_state = torch.load(last_capacity_weights_path)
                model.reuse_weights(last_state)
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        last_capacity_expr_name = experiment_name
        print(results)
    
        
    

def train_parallel(outputs_dir:Path, cfg:dict, cpe:float, gpe:float):
    cfg['trainer']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    dataset_cfg = cfg["dataset"]
    dataset_cfg = parse_augmentations(dataset_cfg)
    model_cfg_list, _ = parse_model_configs(model_cfg = cfg["model"])
    

    
        
        
    def experiment_trainable(config):
        dataset, num_classes = dataset_factory.create_dataset(config['dataset_cfg'])
        # This is to save the best model in each capacity.
        dataset.set_valset(dataset.get_testset(), shuffle=False)
        if 'noise_cfg' in config:
            dataset.inject_noise(**config['noise_cfg'])
        model = model_factory.create_model(config['model_cfg'], num_classes)

        experiment_name = model.get_identifier()
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **config['trainer_cfg'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        
        # print(results)
        
    configs = {
        'model_cfg': tune.grid_search(model_cfg_list),
        'dataset_cfg': copy.deepcopy(dataset_cfg),
        'trainer_cfg': copy.deepcopy(cfg['trainer']),
    }
    if 'noise_config' in cfg: configs['noise_cfg'] = cfg['noise_config']
        
    resources_per_expr = {"cpu": cpe, "gpu": gpe}
    trainable_with_gpu_resources = tune.with_resources(
        experiment_trainable,
        resources=resources_per_expr
    )
    
    ray_strg_dir = outputs_dir / Path('ray')
    ray_strg_dir.mkdir(exist_ok=True, parents=True)
    tuner = tune.Tuner(
        trainable_with_gpu_resources, # Your trainable function wrapped with resources
        param_space=configs,    # The hyperparameters to explore
        tune_config=TuneConfig(
            scheduler=None
        ),
        run_config=RunConfig(
            name=None, # Name for this experiment run
            storage_path=str(ray_strg_dir.absolute()), # Default location for results
            failure_config=FailureConfig(
                max_failures=-1 # -1: Continue running other trials if one fails
                                # 0 (Default): Stop entire run if one trial fails
            )
        )
    )
    results = tuner.fit()
    print(results)
    

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
        
    if cfg['weight_reuse'] and args.parallel:
        raise ValueError('Experiments which use `weight_reuse` cannot be run in parallel mode.')

    outputs_dir = Path("outputs/").absolute() / f"{args.config}"
    outputs_dir.mkdir(exist_ok=True, parents=True)
    # results_dir = Path("results/").absolute() / f"{args.config}"
    # results_dir.mkdir(exist_ok=True, parents=True)
    
    
    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])


    if args.parallel:
        train_parallel(outputs_dir, cfg, args.cpe, args.gpe)
    else:
        train(outputs_dir, cfg)
if __name__ == "__main__":
    main()