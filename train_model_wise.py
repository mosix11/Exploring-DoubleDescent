import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
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
    training_seed = 11
    dataset_seed = 11
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
        seed=dataset_seed,
    )

    loss_fn = torch.nn.MSELoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    experiment = f"FC1_MNIST(subsampe{subsample_size}+NoAug+{label_noise}Noise)_WeightReuse_Seed{training_seed}"
    
    outputs_dir = outputs_dir / Path(f"{experiment}")
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
        
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{training_seed}" + 'Wreuse'
        experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
        if lr_schedule_cfg: experiment_name += f"|{lr_schedule_cfg['type']}"
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
            batch_prog=False,
            log_comet=True,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-modelwise',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=training_seed
        )

        results = trainer.fit(model, dataset, resume=False)
        print(f"Results for param {param}: {results}")

        last_experiment_name = experiment_name



def train_fc1_mnist_parallel(outputs_dir: Path):
    max_epochs = 2000
    subsample_size = (4000,1000) # Train and Test
    batch_size = 256
    label_noise = 0.0
    training_seed = 11
    dataset_seed = 11
    
    
    gpu_per_experiment:float = 0.1
    cpu_per_experiment:float = 1
    
    log_comet = True
    
    param_range = [2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
        26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
        52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
        78,  80,  82,  86,  90,  94,  98, 104, 110, 116, 122, 128,
        140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 
        300, 360, 420, 500, 600, 700, 800, 900, 1000, 1100,
        1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000,
        3200, 3400, 3600, 3800, 4000, 4400, 4800, 5200, 5600, 6000,
        7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000
        ]
    
    # param_range = [2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,
    #     26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
    #     52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
    #     78,  80,  82,  86,  90,  94,  98, 104, 110, 116, 122, 128,
    #     130, 135, 145, 155, 165, 175, 185, 195, 205,
    #     140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 
    #     210, 215, 225, 230, 235, 245, 250, 255, 265, 270, 275, 280, 285, 290, 295,
    #     300, 360, 420, 500, 600, 700, 800, 900, 1000, 1100,
    #     1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000,
    #     3200, 3400, 3600, 3800, 4000, 4400, 4800, 5200, 5600, 6000,
    #     7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000
    #     ]
    

    
    # optim_cgf = {
    #     "type": "sgd",
    #     "lr": 1e-2,
    #     "momentum": 0.95
    # }
    # lr_schedule_cfg = {
    #     "type": "step",
    #     "milestones": list(range(500, 6000, 500)),
    #     "gamma": 0.9,
    # }
    # optim_cgf = {
    #     "type": "sgd",
    #     "lr": 1e-2,
    #     "momentum": 0.95
    # }
    # lr_schedule_cfg = {
    #     "type": "step",
    #     "milestones": list(range(100, 2000, 100)),
    #     "gamma": 0.9,
    # }
    
    optim_cgf = {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999)
    }
    lr_schedule_cfg = None
    
    
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)
    
    experiment = f"FC1_MNIST(subsampe{subsample_size}+NoAug+{label_noise}Noise)_Parallel_Seed{training_seed}_ADAM+MSE"
    # experiment = f"FC1_MNIST(Full+NoAug+{label_noise}Noise)_Parallel_Seed{training_seed}"
    
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
            seed=dataset_seed,
        )
        
        model = FC1(
            input_dim=784,
            hidden_dim=config['param'],
            output_dim=10,
            weight_init=weight_init_method,
            loss_fn=loss_fn,
            metric=acc_metric,
        )
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{training_seed}"
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
            log_comet=log_comet,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='dd-modelwise-fc1-mnist-0noise-adam-mse-sub4000-1000',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=training_seed
        )
        results = trainer.fit(model, dataset, resume=False)
        tune.report(results)
        
    configs = {
        "param": tune.grid_search(param_range)
    }
    resources_per_expr = {"cpu": cpu_per_experiment, "gpu": gpu_per_experiment}
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
    training_seed = 11
    dataset_seed = 11
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
        seed=dataset_seed,
    )

    loss_fn = torch.nn.MSELoss()

    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    experiment = f"FC1_CIFAR10(subsampe{subsample_size}+NoAug+{label_noise}Noise)_WeightReuse_Seed{training_seed}"
    
    outputs_dir = outputs_dir / Path(experiment)
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

        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{training_seed}" +  'Wreuse'
        experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
        if lr_schedule_cfg: experiment_name += f"|{lr_schedule_cfg['type']}"
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
            seed=training_seed
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
    label_noise = 0.2
    use_amp = True
    log_comet = True
    training_seed = 11
    dataset_seed = 11
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
        "lr": 1e-2,
        "momentum": 0.0
    }
    lr_schedule_cfg = {
        "type": "isqrt",
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
        seed=dataset_seed,
    )
    
    experiment = f"CNN5_CIFAR10+NoAug+{label_noise}Noise)_Sequential_Seed{training_seed}"
    
    outputs_dir = outputs_dir / Path(experiment)
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
        
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{training_seed}"
        experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}"
        experiment_name += '|AMP' if use_amp else 'noAMP'
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
            use_amp=use_amp,
            log_comet=log_comet,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-test',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=training_seed
        )


        results = trainer.fit(model, dataset, resume=False)
        print(f"Results for param {param}: {results}")


def train_cnn5_cifar10_parallel(outputs_dir: Path):
        # 500k steps for SGD -> Each epoch with batch size 128 = 391 gradient steps -> 1279 epochs
    # 4k epochs for Adam
    
    max_gradient_steps = 500000
    subsample_size = (-1, -1) # Train and Test
    batch_size = 128
    label_noise = 0.2
    use_amp = True
    log_comet = True
    training_seed = 11
    dataset_seed = 11
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
        "lr": 1e-2,
        "momentum": 0.0
    }
    lr_schedule_cfg = {
        "type": "isqrt",
        "L": 512,
    }
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip()
    ]
    
    experiment = f"CNN5_CIFAR10+NoAug+{label_noise}Noise_Parallel_Seed{training_seed}"
    
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
            seed=dataset_seed,
        )
        
        model = CNN5(
            num_channels=config['param'],
            num_classes=10,
            loss_fn=loss_fn,
            metric=acc_metric
        )
        
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{training_seed}"
        experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}"
        experiment_name += '|AMP' if use_amp else 'noAMP'
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
            use_amp=use_amp,
            log_comet=log_comet,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-test',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=training_seed
        )

        results = trainer.fit(model, dataset, resume=False)
        # print(f"Results for param {param}: {results}"")
        
        tune.report(results)
        
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


def train_fc1_mog_parallel(outputs_dir: Path):
    
    
    
    max_epochs = 1000
    batch_size = 1024
    training_seed = 22
    dataset_seed = 22
    
    gpu_per_experiment:float = 0.1
    cpu_per_experiment:float = 1
    
    log_comet = True
    
    dataset_params = {
        'num_samples': 100000,
        'num_features': 512,         
        'num_classes': 30,          
        'label_noise': 0.2,         
        'clusters_per_class': 'random', 
        'base_cluster_std': 'random',   
        'covariance_type': 'full',   
        'class_sep': 1.0,           
        'intra_class_spread': 2.0,    
        
    }
    
    # dataset_params = {
    #     'num_samples': 20000,
    #     'num_features': 64,
    #     'num_classes': 40,
    #     'label_noise': 0.20,
    #     'clusters_per_class': 1, 
    #     'base_cluster_std': 1.0, 
    #     'covariance_type': 'full',
    #     'class_sep': 1.0,
    #     'intra_class_spread': 1.0,
    # }
    
    
    param_range = [16, 32, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
       4608, 4864, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 9216, 10240,
       11264, 12288, 13312, 14336, 15360, 16384, 18432, 22528, 26624, 32768, 64000]
    
    # param_range = [
    #     1,
    #     4,
    #     8,
    #     12,
    #     18,
    #     20,
    #     22,
    #     24,
    #     28,
    #     32,
    #     36,
    #     38,
    #     44,
    #     56,
    #     80,
    #     96,
    #     128,
    #     160,
    #     192,
    #     208,
    #     216,
    #     224,
    #     232,
    #     240,
    #     248,
    #     256,
    #     264,
    #     280,
    #     296,
    #     304,
    #     320,
    #     336,
    #     344,
    #     368,
    #     392,
    #     432,
    #     464,
    #     512,
    #     768,
    #     1024,
    #     2048,
    #     3072,
    #     4096,
    #     8192,
    #     16384,
    #     # 32768,
    #     # 65636,
    #     # 98504,
    #     # 131272,
    #     # 196908,
    #     # 262544,
        
    # ]
    
    # param_range = [
    #     1,
    #     2,
    #     4,
    #     6,
    #     8,
    #     12,
    #     16,
    #     18,
    #     20,
    #     22,
    #     24,
    #     28,
    #     32,
    #     36,
    #     38,
    #     40,
    #     42,
    #     44,
    #     46,
    #     48,
    #     50,
    #     52,
    #     54,
    #     56,
    #     58,
    #     60,
    #     62,
    #     64,
    #     68,
    #     72,
    #     80,
    #     96,
    #     128,
    #     160,
    #     192,
    #     200,
    #     208,
    #     216,
    #     224,
    #     232,
    #     240,
    #     248,
    #     256,
    #     264,
    #     272,
    #     280,
    #     288,
    #     296,
    #     304,
    #     312,
    #     320,
    #     328,
    #     336,
    #     344,
    #     352,
    #     360,
    #     368,
    #     376,
    #     384,
    #     392,
    #     400,
    #     416,
    #     432,
    #     448,
    #     464,
    #     480,
    #     512,
    #     768,
    #     1024,
    #     2048,
    #     3072,
    #     4096,
    #     8192,
    #     16384,
    #     32768,
    #     65636
    # ]

    optim_cgf = {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999)
    }
    lr_schedule_cfg = None
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=dataset_params['num_classes'])
    
    experiment = f"FC1_MoG(smpls{dataset_params['num_samples']}+ftrs{dataset_params['num_features']}+cls{dataset_params['num_classes']}+{dataset_params['label_noise']}Noise)_Parallel_B{batch_size}_Seed{training_seed}"
    
    outputs_dir = outputs_dir / Path(experiment)
    outputs_dir.mkdir(exist_ok=True, parents=True)
    

    
    def experiment_trainable(config):
        
        
        dataset = MoGSynthetic(
            batch_size=batch_size,
            num_samples=dataset_params['num_samples'],
            num_features=dataset_params['num_features'],
            num_classes=dataset_params['num_classes'],
            clusters_per_class=dataset_params['clusters_per_class'],
            base_cluster_std=dataset_params['base_cluster_std'],
            covariance_type=dataset_params['covariance_type'],
            class_sep=dataset_params['class_sep'],
            intra_class_spread=dataset_params['intra_class_spread'],
            label_noise=dataset_params['label_noise'],
            train_val_test_ratio=[0.7, 0.0, 0.3], 
            num_workers=1,                        
            seed=dataset_seed                     
        )
        
        model = FC1(
            input_dim=dataset_params['num_features'],
            hidden_dim=config['param'],
            output_dim=dataset_params['num_classes'],
            # weight_init=weight_init_method,
            loss_fn=loss_fn,
            metric=acc_metric,
        )
        
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{training_seed}"
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
            log_comet=log_comet,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='dd-modelwise-fc1-mog-new-setting',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=training_seed
        )
        results = trainer.fit(model, dataset, resume=False)
        tune.report(results)
        
    configs = {
        "param": tune.grid_search(param_range)
    }
    resources_per_expr = {"cpu": cpu_per_experiment, "gpu": gpu_per_experiment}
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
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 
    torch.set_float32_matmul_precision("high")

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

    outputs_dir = Path("outputs/modelwise").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc1" and args.dataset == "mnist":
        if args.parallel:
            train_fc1_mnist_parallel(outputs_dir)
        else:
            train_fc1_mnist(outputs_dir)
    elif args.model == "fc1" and args.dataset == "cifar10":
        train_fc1_cifar10(outputs_dir)
    elif args.model == "fc1" and args.dataset == "mog":
        if args.parallel:
            train_fc1_mog_parallel(outputs_dir)
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        if args.parallel:
            train_cnn5_cifar10_parallel(outputs_dir)
        else:
            train_cnn5_cifar10(outputs_dir)
