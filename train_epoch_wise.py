import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, CNN5, PreActResNet, make_resnet18k
from src.trainers import TrainerEp, TrainerGS
import matplotlib.pyplot as plt
from src.utils import nn_utils
import torch
import torchvision
import torchmetrics
import torchvision.transforms.v2 as transformsv2
from functools import partial
from pathlib import Path
import pickle
import argparse
import os


def train_cnn5_mnist(outputs_dir: Path):
    max_epochs = 250
    batch_size = 1024
    label_noise = 0.75
    num_kernels_k = 128
    use_amp = True
    seed = 22
    max_gradient_steps=max_epochs * (60000 // batch_size)
    log_comet = True

    optim_cgf = {
        "type": "sgd",
        "lr": 1e-1,
        "momentum": 0.0
    }
    lr_schedule_cfg = {
        "type": "isqrt",
        "L": 512,
    }
    
    # augmentations = [
    #     transformsv2.RandomCrop(32, padding=4),
    # ]
    
    dataset = MNIST(
        batch_size=batch_size,
        img_size=(32, 32), # So it matches the configuration of the CNN5 network.
        num_workers=8,
        label_noise=label_noise,
        augmentations=[],
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=False,
        seed=seed,
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    model = CNN5(
        num_channels=num_kernels_k,
        num_classes=10,
        gray_scale=True,
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
        checkpoint_freq=1,
        save_best_model=True,
        run_on_gpu=True,
        use_amp=use_amp,
        log_comet=log_comet,
        comet_project_name='doubledescent-epochwise',
        exp_name=experiment_name,
        exp_tags=experiment_tags,
        seed=seed
    )

    results = trainer.fit(model, dataset, resume=False)
    print('results:', results)



def train_cnn5_cifar10(outputs_dir: Path):
    
    max_epochs = 2000
    batch_size = 1024
    label_noise = 0.0
    num_kernels_k = 128
    use_amp = True
    seed = 22
    max_gradient_steps=max_epochs * (50000 // batch_size)
    log_comet = True
    
    optim_cgf = {
        "type": "sgd",
        "lr": 1e-1,
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
        augmentations=augmentations,
        num_workers=8,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=False,
        seed=seed,
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    model = CNN5(
        num_channels=num_kernels_k,
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
        comet_project_name='doubledescent-epochwise',
        exp_name=experiment_name,
        exp_tags=experiment_tags,
        seed=seed
    )

    results = trainer.fit(model, dataset, resume=False)
    print('results:', results)
    # print(
    #     f"\n\n\nTraining experiment {experiment_name} finished with final results {results['final']}, and best results {results['best']}\n\n\n"
    # )
    
        

def train_resnet18k_cifar10(outputs_dir: Path):
    max_epochs = 30
    batch_size = 128
    label_noise = 0.2
    seed = 22
    optim_cgf = {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999)
    }
    
    lr_schedule_cfg = None
    
    augmentations = [
        transformsv2.RandomCrop(32, padding=4),
        transformsv2.RandomHorizontalFlip()
    ]
    
    dataset = CIFAR10(
        batch_size=batch_size,
        grayscale=False,
        label_noise=label_noise,
        augmentations=augmentations,
        num_workers=8,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=False,
        seed=seed,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    param = 64
    model = make_resnet18k(
        k=param,
        num_classes=10,
        loss_fn=loss_fn,
        metric=acc_metric
    )
    
    # experiment_tags = model.get_identifier().split('_').append(dataset.get_identifier().split('_'))
    
    experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{seed}"
    experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|AMP"
    if lr_schedule_cfg: experiment_name += f"|{lr_schedule_cfg['type']}"
    experiment_tags = experiment_name.split('_')
    
    early_stopping = False 
    trainer = TrainerEp(
        outputs_dir=outputs_dir,
        max_epochs=max_epochs,
        optimizer_cfg=optim_cgf,
        lr_schedule_cfg=lr_schedule_cfg,
        early_stopping=early_stopping,
        validation_freq=1,
        save_best_model=False,
        run_on_gpu=True,
        use_amp=True,
        batch_prog=False,
        log_comet=True,
        comet_project_name='doubledescent-epochwise',
        exp_name=experiment_name,
        exp_tags=experiment_tags,
        seed=seed
    )

    results = trainer.fit(model, dataset, resume=False)
    
    print(
        f"\n\n\nTraining experiment {experiment_name} finished with final results {results['final']}, and best results {results['best']}\n\n\n"
    )


def train_fc1_mog(outputs_dir: Path):
    max_epochs = 20000
    batch_size = 4096
    num_samples = 100000
    num_features = 512
    num_classes = 30
    hidden_size = 32818
    label_noise = 0.2

    max_gradient_steps = max_epochs * (100000 // batch_size)
    seed = 22
    log_comet = True

    optim_cgf = {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999)
    }
    lr_schedule_cfg = None
    
    # optim_cgf = {
    #     "type": "sgd",
    #     "lr": 1e-2,
    #     "momentum": 0.0
    # }
    # lr_schedule_cfg = {
    #     "type": "isqrt",
    #     "L": 64,
    # }
    
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    
    dataset = MoGSynthetic(
        batch_size=batch_size,
        num_samples=num_samples,
        num_features=num_features,
        num_classes=num_classes,
        clusters_per_class='random',
        base_cluster_std='random',
        covariance_type='full',
        class_sep=1.0,
        intra_class_spread=2.0,
        label_noise=label_noise,
        train_val_test_ratio=[0.7, 0.0, 0.3],
        num_workers=2,
        seed=seed
    )
    
    # weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)
    
    model = FC1(
        input_dim=num_features,
        hidden_dim=hidden_size,
        ouput_dim=num_classes,
        # weight_init=weight_init_method,
        loss_fn=loss_fn,
        metric=acc_metric,
    )

    experiment_name = model.get_identifier() + '_' + dataset.get_identifier() + f"_seed{seed}"
    experiment_name += '_' + f"{optim_cgf['type']}|lr{optim_cgf['lr']}|b{batch_size}|noAMP"
    if lr_schedule_cfg: experiment_name += f"|{lr_schedule_cfg['type']}"
    experiment_tags = experiment_name.split('_')
    
    
    early_stopping = False 
    trainer = TrainerEp(
        outputs_dir=outputs_dir,
        max_epochs=max_epochs,
        optimizer_cfg=optim_cgf,
        lr_schedule_cfg=lr_schedule_cfg,
        early_stopping=early_stopping,
        validation_freq=1,
        checkpoint_freq=1,
        save_best_model=True,
        run_on_gpu=True,
        use_amp=False,
        batch_prog=False,
        log_comet=log_comet,
        comet_project_name='doubledescent-epochwise',
        exp_name=experiment_name,
        exp_tags=experiment_tags,
        seed=seed
    )
    
    # trainer = TrainerGS(
    #     outputs_dir=outputs_dir,
    #     max_gradient_steps=max_gradient_steps,
    #     optimizer_cfg=optim_cgf,
    #     lr_schedule_cfg=lr_schedule_cfg,
    #     early_stopping=early_stopping,
    #     validation_freq=1, # Epoch
    #     save_best_model=False,
    #     run_on_gpu=True,
    #     use_amp=False,
    #     log_comet=log_comet,
    #     comet_project_name='doubledescent-epochwise',
    #     exp_name=experiment_name,
    #     exp_tags=experiment_tags,
    #     seed=seed
    # )
    
    results = trainer.fit(model, dataset, resume=False)
    
    # print(
    #     f"\n\n\nTraining experiment {experiment_name} finished with final results {results['final']}, and best results {results['best']}\n\n\n"
    # )

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
        choices=["fc1", "cnn5", "resnet18"],
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

    outputs_dir = Path("outputs/epwise")
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc1" and args.dataset == "mnist":
        ...
        # train_fc1_mnist(results_dir, checkpoints_dir, log_dir)
    elif args.model == "fc1" and args.dataset == "cifar10":
        ...
        # train_fc1_cifar10(results_dir, checkpoints_dir, log_dir)
    elif args.model == "fc1" and args.dataset == "mog":
        train_fc1_mog(outputs_dir)
    elif args.model == 'cnn5' and args.dataset == "mnist":
        train_cnn5_mnist(outputs_dir)
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        train_cnn5_cifar10(outputs_dir)
    elif args.model == 'resnet18' and args.dataset == 'cifar10':
        train_resnet18k_cifar10(outputs_dir)