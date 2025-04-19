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










def train_resnet18k_cifar10(results_dir: Path, checkpoints_dir: Path, log_dir: Path):
    max_epochs = 2000
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
        batch_size=128,
        grayscale=False,
        label_noise=0.2,
        augmentations=augmentations,
        num_workers=8,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=False,
        seed=11,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    param = 64
    model = make_resnet18k(
        k=64,
        num_classes=10,
        loss_fn=loss_fn,
        metric=acc_metric
    )
    
    early_stopping = False 
    trainer = TrainerEp(
        max_epochs=max_epochs,
        optimizer_cfg=optim_cgf,
        lr_schedule_cfg=lr_schedule_cfg,
        early_stopping=early_stopping,
        validation_freq=1,
        run_on_gpu=True,
        use_amp=False,
        batch_prog=False,
        log_tensorboard=True,
        log_dir=log_dir / Path(f"ep_wise_param{param}"),
    )

    results = trainer.fit(model, dataset, resume=False)
    
    print(
        f"\n\n\nTraining the model with hidden layer size {param} finished with test loss {results['test_loss']}, test acc {results['test_acc']}, train loss {results['train_loss']}, train acc {results['train_acc']}.\n\n\n"
    )
    
    torch.save(model.state_dict(), checkpoints_dir / Path(f"ckp_ep_wise_h{param}.pth"))
    result_path = results_dir / Path(f"ep_wise_param{param}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(results, f)




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
    args = parser.parse_args()

    experiment_specifier = f"{args.model}_{args.dataset}"

    outputs_dir = Path("outputs")
    results_dir = outputs_dir / Path(f"results/{experiment_specifier}")
    checkpoints_dir = outputs_dir / Path(f"checkpoints/{experiment_specifier}")
    log_dir = outputs_dir / Path(f"tensorboard/{experiment_specifier}")
    results_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc1" and args.dataset == "mnist":
        ...
        # train_fc1_mnist(results_dir, checkpoints_dir, log_dir)
    elif args.model == "fc1" and args.dataset == "cifar10":
        ...
        # train_fc1_cifar10(results_dir, checkpoints_dir, log_dir)
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        ...
        # train_cnn5_cifar10(results_dir, checkpoints_dir, log_dir)
    elif args.model == 'resnet18k' and args.dataset == 'cifar10':
        train_resnet18k_cifar10(results_dir, checkpoints_dir, log_dir)