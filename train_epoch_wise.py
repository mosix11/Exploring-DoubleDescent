import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST
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




def train_cnn5_cifar10(outputs_dir: Path):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    max_epochs = 100
    batch_size = 128
    label_noise = 0.0
    seed = 22
    max_gradient_steps=max_epochs * (50000 // 128)
    
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
        augmentations=augmentations,
        num_workers=8,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=False,
        seed=seed,
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    param = 128
    
    model = CNN5(
        num_channels=param,
        num_classes=10,
        loss_fn=loss_fn,
        metric=acc_metric
    )

    early_stopping = False 
    trainer = TrainerGS(
        max_gradient_steps=max_gradient_steps,
        optimizer_cfg=optim_cgf,
        lr_schedule_cfg=lr_schedule_cfg,
        early_stopping=early_stopping,
        validation_freq=1, # Epoch
        run_on_gpu=True,
        use_amp=True,
        log_tensorboard=True,
        log_dir=log_dir / Path(f"k{param}_sgd_invsqrt_aug_0nl_full_AMP"),
        seed=seed
    )

    results = trainer.fit(model, dataset, resume=False)
    
    print(
        f"\n\n\nTraining the model with hidden layer size {param} finished with test loss {results['test_loss']}, test acc {results['test_acc']}, train loss {results['train_loss']}, train acc {results['train_acc']}.\n\n\n"
    )
    
    torch.save(model.state_dict(), checkpoints_dir / Path(f"ckp_k{param}.pth"))
    result_path = results_dir / Path(f"res_k{param}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(results, f)
        

def train_resnet18k_cifar10(outputs_dir: Path):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    max_epochs = 30
    batch_size = 128
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
        label_noise=0.2,
        augmentations=augmentations,
        num_workers=8,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=False,
        seed=22,
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
    
    experiment_name = model.get_identifier() + '_' + dataset.get_identifier()
    experiment_name += '_' + f'{optim_cgf['type']}|lr{optim_cgf["lr"]}|b{batch_size}'
    if lr_schedule_cfg: experiment_name += f'|sch{lr_schedule_cfg['type']}'
    experiment_tags = experiment_name.split('_')
    
    early_stopping = False 
    trainer = TrainerEp(
        max_epochs=max_epochs,
        optimizer_cfg=optim_cgf,
        lr_schedule_cfg=lr_schedule_cfg,
        early_stopping=early_stopping,
        validation_freq=1,
        run_on_gpu=True,
        use_amp=True,
        batch_prog=False,
        log_comet=True,
        comet_project_name='doubledescent-epochwise',
        exp_name=experiment_name,
        exp_tags=experiment_tags,
        seed=22
    )

    results = trainer.fit(model, dataset, resume=False)
    
    print(
        f"\n\n\nTraining the model with hidden layer size {param} finished with test loss {results['test_loss']}, test acc {results['test_acc']}, train loss {results['train_loss']}, train acc {results['train_acc']}.\n\n\n"
    )
    
    # torch.save(model.state_dict(), checkpoints_dir / Path(f"ckp_k{param}.pth"))
    # result_path = results_dir / Path(f"res_k{param}.pkl")
    # with open(result_path, "wb") as f:
    #     pickle.dump(results, f)




if __name__ == "__main__":

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
        choices=["mnist", "cifar10", "cifar100"],
        required=True,
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume training from the last checkpoint.",
        action="store_true",
    )
    args = parser.parse_args()

    experiment_specifier = f"{args.model}_{args.dataset}"

    outputs_dir = Path("outputs/epwise")
    outputs_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc1" and args.dataset == "mnist":
        ...
        # train_fc1_mnist(results_dir, checkpoints_dir, log_dir)
    elif args.model == "fc1" and args.dataset == "cifar10":
        ...
        # train_fc1_cifar10(results_dir, checkpoints_dir, log_dir)
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        train_cnn5_cifar10(outputs_dir)
    elif args.model == 'resnet18' and args.dataset == 'cifar10':
        train_resnet18k_cifar10(outputs_dir)