import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST
from src.models import FC1, CNN5
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



def train_fc1_mnist(outputs_dir: Path):
    
    max_epochs = 6000
    subsample_size = (4000,1000) # Train and Test
    batch_size = 256
    label_noise = 0.0
    seed = 11
    param_range = [
        3,
        4,
        7,
        9,
        10,
        20,
        30,
        40,
        45,
        47,
        49,
        50,
        51,
        53,
        55,
        60,
        70,
        80,
        90,
        100,
        110,
        128,
        150,
        170,
        196
    ]

    interpolation_ths = 50

    optim_cgf = {
        "type": "sgd",
        "lr": 1e-2,
        "momentum": 0.95
    }
    lr_schedule_cfg = {
        "type": "step",
        "milestones": list(range(500, 6000, 500)),
        "gamma": 0.9,
    }
    
    dataset = MNIST(
        batch_size=batch_size,
        subsample_size=subsample_size,
        num_workers=4,
        label_noise=label_noise,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=True,
        seed=seed,
    )

    loss_fn = torch.nn.MSELoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    
    experiment = dataset.get_identifier()
    experiment += "_" + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
    if lr_schedule_cfg: experiment += f"|{lr_schedule_cfg['type']}"
    
    outputs_dir = outputs_dir / Path(f"fc1_{experiment}")
    outputs_dir.mkdir(exist_ok=True, parents=True)
    
    last_experiment_name = None
    for idx, param in enumerate(param_range):

        if idx == 0:
            weight_init_method = nn_utils.init_xavier_uniform
        elif param > interpolation_ths:
            weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)
        else:
            weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)

        model = FC1(
            input_dim=784,
            hidden_dim=param,
            ouput_dim=10,
            weight_init=weight_init_method,
            loss_fn=loss_fn,
            metric=acc_metric,
        )
        
        experiment_name = model.get_identifier() + '_' + experiment
        experiment_tags = experiment_name.split('_')

        # In underparameterized regime, we reuse weights from the last trained model.
        if idx > 0 and param <= interpolation_ths:
            old_ckp_path = outputs_dir / Path(last_experiment_name) / Path('checkpoint/final_ckp.pth')
            old_state = torch.load(old_ckp_path)['model_state']
            model.reuse_weights(old_state)

        early_stopping = False if param > interpolation_ths else True
        
        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            max_epochs=max_epochs,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=early_stopping,
            validation_freq=1,
            save_best_model=True,
            run_on_gpu=False,
            use_amp=False,
            batch_prog=False,
            log_comet=False,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-modelwise',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=seed
        )

        results = trainer.fit(model, dataset, resume=False)
        print(f"Results for param {param}: {results}")

        last_experiment_name = experiment_name



def train_fc1_mnist_parallel(outputs_dir: Path):
    max_epochs = 6000
    subsample_size = (4000,1000) # Train and Test
    batch_size = 256
    label_noise = 0.0
    seed = 11
    param_range = [
        3,
        4,
        7,
        9,
        10,
        20,
        30,
        40,
        45,
        47,
        49,
        50,
        51,
        53,
        55,
        60,
        70,
        80,
        90,
        100,
        110,
        128,
        150,
        170,
        196
    ]

    optim_cgf = {
        "type": "sgd",
        "lr": 1e-2,
        "momentum": 0.95
    }
    lr_schedule_cfg = {
        "type": "step",
        "milestones": list(range(500, 6000, 500)),
        "gamma": 0.9,
    }
    
    loss_fn = torch.nn.MSELoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)
    
    experiment = f"FC1_MNIST(subsampe{subsample_size}+NoAug+{label_noise}Noise)_Parallel_Seed{seed}"
    
    outputs_dir = outputs_dir / Path(experiment)
    outputs_dir.mkdir(exist_ok=True, parents=True)
    

    
    def experiment_trainable(config):
        
        dataset = MNIST(
            batch_size=batch_size,
            subsample_size=subsample_size,
            num_workers=1,
            label_noise=label_noise,
            valset_ratio=0.0,
            normalize_imgs=False,
            flatten=True,
            seed=seed,
        )
        
        model = FC1(
            input_dim=784,
            hidden_dim=config['param'],
            ouput_dim=10,
            weight_init=weight_init_method,
            loss_fn=loss_fn,
            metric=acc_metric,
        )
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier()
        experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
        if lr_schedule_cfg: experiment_name += f"|{lr_schedule_cfg['type']}"
        experiment_tags = experiment_name.split('_')

        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            max_epochs=max_epochs,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=False,
            validation_freq=1,
            save_best_model=True,
            run_on_gpu=True,
            use_amp=False,
            batch_prog=False,
            log_comet=True,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-modelwise',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=seed
        )
        results = trainer.fit(model, dataset, resume=False)
        train.report(results)
        
    configs = {
        "param": tune.grid_search(param_range)
    }
    resources_per_expr = {"cpu": 1, "gpu": 0.1}
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


def train_fc1_cifar10(outputs_dir: Path):
    max_epochs = 6000
    subsample_size = (960, 500) # Train and Test
    batch_size = 512
    label_noise = 0.0
    seed = 11
    param_range = [
        2,
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        38,
        48,
        64,
        96,
        128,
        256,
        384,
        512,
        1024,
        2048,
    ]
    interpolation_ths = 30

    optim_cgf = {"type": "sgd", "lr": 1e-2, "momentum": 0.95}
    lr_schedule_cfg = {
        "type": "step_lr",
        "milestones": list(range(500, 6000, 500)),
        "gamma": 0.9,
    }
    dataset = CIFAR10(
        batch_size=batch_size,
        img_size=(8, 8),
        grayscale=True,
        subsample_size=subsample_size,
        class_subset=[3, 5],
        num_workers=8,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=True,
        seed=11,
    )

    loss_fn = torch.nn.MSELoss()

    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    experiment = dataset.get_identifier()
    experiment += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
    if lr_schedule_cfg: experiment += f"|{lr_schedule_cfg['type']}"
    
    outputs_dir = outputs_dir / Path(f"fc1_{experiment}")
    outputs_dir.mkdir(exist_ok=True, parents=True)

    last_experiment_name = None
    for idx, param in enumerate(param_range):

        if idx == 0:
            weight_init_method = nn_utils.init_xavier_uniform
        elif param > interpolation_ths:
            weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)
        else:
            weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)

        model = FC1(
            input_dim=64,
            hidden_dim=param,
            ouput_dim=2,
            weight_init=weight_init_method,
            loss_fn=loss_fn,
            metric=acc_metric,
        )

        experiment_name = model.get_identifier() + '_' + experiment
        experiment_tags = experiment_name.split('_')

        # In underparameterized regime, we reuse weights from the last trained model.
        if idx > 0 and param <= interpolation_ths:
            old_ckp_path = outputs_dir / Path(last_experiment_name) / Path('checkpoint/final_ckp.pth')
            old_state = torch.load(old_ckp_path)['model_state']
            model.reuse_weights(old_state)

        early_stopping = False if param > interpolation_ths else True
        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            max_epochs=max_epochs,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=early_stopping,
            validation_freq=1,
            save_best_model=True,
            run_on_gpu=True,
            use_amp=False,
            batch_prog=True,
            log_comet=True,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-modelwise',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=seed
        )

        results = trainer.fit(model, dataset, resume=False)
        print(f"Results for param {param}: {results}")

        last_experiment_name = experiment_name


def train_cnn5_cifar10(outputs_dir: Path):
    # 500k steps for SGD -> Each epoch with batch size 128 = 391 gradient steps -> 1279 epochs
    # 4k epochs for Adam
    
    max_gradient_steps = 500000
    subsample_size = (-1, -1) # Train and Test
    batch_size = 128
    label_noise = 0.0
    seed = 11
    param_range = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        24,
        28,
        32,
        38,
        48,
        64
    ]

    # optim_cgf = {
    #     'type': 'adamw',
    #     'lr': 1e-4,
    #     'betas': (0.9, 0.999)
    # }
    optim_cgf = {
        "type": "sgd",
        "lr": 1e-1,
        "momentum": 0.0
    }
    lr_schedule_cfg = {
        "type": "inv_sqr_root",
        "L": 512,
    }
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip()
    ]
    
    dataset = CIFAR10(
        batch_size=batch_size,
        grayscale=False,
        label_noise=label_noise,
        subsample_size=subsample_size,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=False,
        num_workers=4,
        seed=11,
    )
    
    experiment = dataset.get_identifier()
    experiment += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
    if lr_schedule_cfg: experiment += f"|{lr_schedule_cfg['type']}"
    
    outputs_dir = outputs_dir / Path(f"fc1_{experiment}")
    outputs_dir.mkdir(exist_ok=True, parents=True)
    

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    for idx, param in enumerate(param_range):

        model = CNN5(
            num_channels=param,
            num_classes=10,
            loss_fn=loss_fn,
            metric=acc_metric
        )
        
        experiment_name = model.get_identifier() + '_' + experiment
        experiment_tags = experiment_name.split('_')
        
        early_stopping = False 
        trainer = TrainerGS(
            outputs_dir=outputs_dir,
            max_gradient_steps=max_gradient_steps,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=early_stopping,
            validation_freq=1, # Epoch
            save_best_model=True,
            run_on_gpu=True,
            use_amp=True,
            log_comet=True,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-modelwise',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=seed
        )


        results = trainer.fit(model, dataset, resume=False)
        print(f"Results for param {param}: {results}")


def train_cnn5_cifar10_parallel(outputs_dir: Path):
        # 500k steps for SGD -> Each epoch with batch size 128 = 391 gradient steps -> 1279 epochs
    # 4k epochs for Adam
    
    max_gradient_steps = 500000
    subsample_size = (-1, -1) # Train and Test
    batch_size = 128
    label_noise = 0.0
    seed = 11
    param_range = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        24,
        28,
        32,
        38,
        48,
        64
    ]

    # optim_cgf = {
    #     'type': 'adamw',
    #     'lr': 1e-4,
    #     'betas': (0.9, 0.999)
    # }
    optim_cgf = {
        "type": "sgd",
        "lr": 1e-1,
        "momentum": 0.0
    }
    lr_schedule_cfg = {
        "type": "inv_sqr_root",
        "L": 512,
    }
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip()
    ]
    
    experiment = f"CNN5_CIFAR10+NoAug+{label_noise}Noise)_Parallel_Seed{seed}"
    
    outputs_dir = outputs_dir / Path(experiment)
    outputs_dir.mkdir(exist_ok=True, parents=True)
    
    
    def experiment_trainable(config):
        dataset = CIFAR10(
            batch_size=batch_size,
            grayscale=False,
            label_noise=label_noise,
            subsample_size=subsample_size,
            valset_ratio=0.0,
            normalize_imgs=False,
            flatten=False,
            num_workers=1,
            seed=seed,
        )
        
        model = CNN5(
            num_channels=config['param'],
            num_classes=10,
            loss_fn=loss_fn,
            metric=acc_metric
        )
        
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier()
        experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
        if lr_schedule_cfg: experiment_name += f"|{lr_schedule_cfg['type']}"
        experiment_tags = experiment_name.split('_')
        
        early_stopping = False 
        trainer = TrainerGS(
            outputs_dir=outputs_dir,
            max_gradient_steps=max_gradient_steps,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=early_stopping,
            validation_freq=1, # Epoch
            save_best_model=True,
            run_on_gpu=True,
            use_amp=True,
            log_comet=True,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-modelwise',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=seed
        )

        results = trainer.fit(model, dataset, resume=False)
        # print(f"Results for param {param}: {results}"")
        
        train.report(results)
        
    configs = {
        "param": tune.grid_search(param_range)
    }
    resources_per_expr = {"cpu": 1, "gpu": 0.1}
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="The model to use for training.",
        type=str,
        choices=["fc1", "cnn5", "resnet18k"],
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="The dataset used for trainig the model.",
        type=str,
        choices=["mnist", "cifar10", "cifar100"],
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

    outputs_dir = Path("outputs/modelwise").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc1" and args.dataset == "mnist":
        if args.parallel:
            train_fc1_mnist_parallel(outputs_dir)
        else:
            train_fc1_mnist(outputs_dir)
    elif args.model == "fc1" and args.dataset == "cifar10":
        train_fc1_cifar10(outputs_dir)
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        if args.parallel:
            train_cnn5_cifar10_parallel(outputs_dir)
        else:
            train_cnn5_cifar10(outputs_dir)
