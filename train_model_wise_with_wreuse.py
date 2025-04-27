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

    experiment = f"FC1_MNIST(subsampe{subsample_size}+NoAug+{label_noise}Noise)_WeightReuse_Seed{seed}"
    
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
        
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{seed}" + 'Wreuse'
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
            seed=seed
        )

        results = trainer.fit(model, dataset, resume=False)
        print(f"Results for param {param}: {results}")

        last_experiment_name = experiment_name





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
    
    experiment = f"FC1_CIFAR10(subsampe{subsample_size}+NoAug+{label_noise}Noise)_WeightReuse_Seed{seed}"
    
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

        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{seed}" +  'Wreuse'
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
    label_noise = 0.2
    use_amp = True
    log_comet = True
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
        seed=seed,
    )
    
    experiment = f"CNN5_CIFAR10+NoAug+{label_noise}Noise)_Sequential_Seed{seed}"
    
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
        
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{seed}"
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
            seed=seed
        )


        results = trainer.fit(model, dataset, resume=False)
        print(f"Results for param {param}: {results}")



def train_fc1_mog(outputs_dir: Path):
    max_epochs = 1000
    num_samples = 100000
    batch_size = 1024
    num_features = 512
    num_classes = 30
    label_noise = 0.2
    seed = 22
    log_comet = True
    
    param_range = [
        1,
        4,
        8,
        12,
        18,
        20,
        22,
        24,
        28,
        32,
        36,
        38,
        44,
        56,
        80,
        96,
        128,
        160,
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
        768,
        1024,
        2048,
        3072,
        4096,
        8192,
        16384,
    ]

    optim_cgf = {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999),
    }
    lr_schedule_cfg = None
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    
    experiment = f"FC1_MoG(smpls{num_samples}+ftrs{num_features}+cls{num_classes}+{label_noise}Noise)_WeightReuse_Seed{seed}"
    
    outputs_dir = outputs_dir / Path(experiment)
    outputs_dir.mkdir(exist_ok=True, parents=True)
    

    dataset = MoGSynthetic(
        batch_size=batch_size,
        num_samples = num_samples,
        num_features=num_features,
        num_classes=num_classes,
        clusters_per_class='random',
        base_cluster_std='random',
        covariance_type='full',
        class_sep=1.0,
        intra_class_spread=2.0,
        label_noise=label_noise,
        train_val_test_ratio=[0.7, 0.0, 0.3],
        num_workers=4,
        seed=seed
    )

    prev_exp_name = None
    for idx, param in enumerate(param_range):
        
        model = FC1(
            input_dim=num_features,
            hidden_dim=param,
            ouput_dim=num_classes,
            # weight_init=weight_init_method,
            loss_fn=loss_fn,
            metric=acc_metric,
        )
    
        experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{seed}" + "_wr"
        experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
        if lr_schedule_cfg: experiment_name += f"|{lr_schedule_cfg['type']}"
        experiment_tags = experiment_name.split('_')
        
        if idx > 0:
            old_ckp_path = outputs_dir / Path(prev_exp_name) / Path('checkpoint/best_ckp.pth')
            old_state = torch.load(old_ckp_path)['model_state']
            model.reuse_weights(old_state)
        
        trainer = TrainerEp(
            outputs_dir=outputs_dir,
            max_epochs=max_epochs,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=False,
            validation_freq=1,
            save_best_model=True,
            checkpoint_freq=1,
            run_on_gpu=True,
            use_amp=False,
            batch_prog=False,
            log_comet=log_comet,
            comet_api_key=os.getenv('COMET_API_KEY'),
            comet_project_name='doubledescent-modelwise-fc-mog-weight-reuse',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            # model_log_call=True,
            seed=seed
        )
        results = trainer.fit(model, dataset, resume=False)

        prev_exp_name = experiment_name

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
        choices=["mnist", "cifar10", "cifar100", "mog"],
        required=True,
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume training from the last checkpoint.",
        action="store_true",
    )
    args = parser.parse_args()
    
    
    dotenv.load_dotenv('.env')

    outputs_dir = Path("outputs/modelwise").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc1" and args.dataset == "mnist":
        train_fc1_mnist(outputs_dir)
    elif args.model == "fc1" and args.dataset == "cifar10":
        train_fc1_cifar10(outputs_dir)
    elif args.model == "fc1" and args.dataset == "mog":
        train_fc1_mog(outputs_dir)
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        train_cnn5_cifar10(outputs_dir)
