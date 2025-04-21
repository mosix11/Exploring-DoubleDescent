
from src.datasets import MNIST, CIFAR10, FashionMNIST
from src.models import FC1, CNN5
from src.trainers import TrainerEp, TrainerGS
import matplotlib.pyplot as plt
from src.utils import nn_utils
import torch
import torchmetrics
from functools import partial
from pathlib import Path
import pickle
import argparse


def train_fc1_mnist(results_dir: Path, checkpoints_dir: Path, log_dir: Path):
    max_epochs = 6000
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
        200,
        256,
    ]

    interpolation_ths = 50

    optim_cgf = {"type": "sgd", "lr": 1e-2, "momentum": 0.95}
    lr_schedule_cfg = {
        "type": "step_lr",
        "milestones": list(range(500, 6000, 500)),
        "gamma": 0.9,
    }
    subsample_size = 4000
    dataset = MNIST(
        batch_size=256,
        subsample_size=subsample_size,
        num_workers=8,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=True,
        seed=11,
    )

    loss_fn = torch.nn.MSELoss()

    def accuracy_metric(preds: torch.Tensor, targets: torch.Tensor) -> float:
        _, predicted = preds.max(1)
        num_samples = targets.size(0)
        correct = predicted.eq(targets.argmax(1)).sum().item()
        return correct / num_samples

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
            metric=accuracy_metric,
        )

        # In underparameterized regime, we reuse weights from the last trained model.
        if idx > 0 and param <= interpolation_ths:
            last_trained_param = param_range[idx - 1]
            old_state_path = checkpoints_dir / Path(f"ckp_h{last_trained_param}.pth")
            old_state = torch.load(old_state_path)
            model.reuse_weights(old_state)

        early_stopping = False if param > interpolation_ths else True
        trainer = TrainerEp(
            max_epochs=max_epochs,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=early_stopping,
            run_on_gpu=True,
            use_amp=False,
            log_tensorboard=True,
            log_dir=log_dir / Path(f"h{param}"),
        )

        results = trainer.fit(model, dataset, resume=False)
        print(
            f"\n\n\nTraining the model with hidden layer size {param} finished with test loss of {results['test_loss']}, and test accuracy of {results['test_acc']}.\n\n\n"
        )

        torch.save(model.state_dict(), checkpoints_dir / Path(f"ckp_h{param}.pth"))
        result_path = results_dir / Path(f"res_h{param}.pkl")
        with open(result_path, "wb") as f:
            pickle.dump(results, f)


def train_fc1_cifar10(results_dir: Path, checkpoints_dir: Path, log_dir: Path):
    max_epochs = 6000
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
    subsample_size = 960
    dataset = CIFAR10(
        batch_size=512,
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

    def accuracy_metric(preds: torch.Tensor, targets: torch.Tensor) -> float:
        _, predicted = preds.max(1)
        num_samples = targets.size(0)
        correct = predicted.eq(targets.argmax(1)).sum().item()
        return correct / num_samples

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
            metric=accuracy_metric,
        )

        # In underparameterized regime, we reuse weights from the last trained model.
        if idx > 0 and param <= interpolation_ths:
            last_trained_param = param_range[idx - 1]
            old_state_path = checkpoints_dir / Path(f"ckp_h{last_trained_param}.pth")
            old_state = torch.load(old_state_path)
            model.reuse_weights(old_state)

        early_stopping = False if param > interpolation_ths else True
        trainer = TrainerEp(
            max_epochs=max_epochs,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=early_stopping,
            run_on_gpu=True,
            use_amp=False,
            log_tensorboard=False,
            log_dir=log_dir / Path(f"h{param}"),
        )

        results = trainer.fit(model, dataset, resume=False)
        print(
            f"\n\n\nTraining the model with hidden layer size {param} finished with test loss {results['test_loss']}, test acc {results['test_acc']}, train loss {results['train_loss']}, train acc {results['train_acc']}.\n\n\n"
        )

        torch.save(model.state_dict(), checkpoints_dir / Path(f"ckp_h{param}.pth"))
        result_path = results_dir / Path(f"res_h{param}.pkl")
        with open(result_path, "wb") as f:
            pickle.dump(results, f)


def train_cnn5_cifar10(results_dir: Path, checkpoints_dir: Path, log_dir: Path):
    # 500k steps for SGD -> Each epoch with batch size 128 = 391 gradient steps -> 1279 epochs
    # 4k epochs for Adam
    
    max_gradient_steps = 500000
    # max_gradient_steps = 200
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
    dataset = CIFAR10(
        batch_size=128,
        grayscale=False,
        label_noise=0.2,
        num_workers=4,
        # subsample_size=128,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=False,
        seed=11,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    for idx, param in enumerate(param_range):

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
            run_on_gpu=True,
            use_amp=True,
            log_tensorboard=True,
            log_dir=log_dir / Path(f"k{param}"),
        )

        results = trainer.fit(model, dataset, resume=False)
        print(
            f"\n\n\nTraining the model with hidden layer size {param} finished with test loss {results['test_loss']}, test acc {results['test_acc']}, train loss {results['train_loss']}, train acc {results['train_acc']}.\n\n\n"
        )

        torch.save(model.state_dict(), checkpoints_dir / Path(f"ckp_h{param}.pth"))
        result_path = results_dir / Path(f"res_k{param}.pkl")
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
    results_dir = outputs_dir / Path(f"results/modelwise/{experiment_specifier}")
    checkpoints_dir = outputs_dir / Path(f"checkpoints/modelwise/{experiment_specifier}")
    log_dir = outputs_dir / Path(f"tensorboard/modelwise/{experiment_specifier}")
    results_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    if args.model == "fc1" and args.dataset == "mnist":
        train_fc1_mnist(results_dir, checkpoints_dir, log_dir)
    elif args.model == "fc1" and args.dataset == "cifar10":
        train_fc1_cifar10(results_dir, checkpoints_dir, log_dir)
    elif args.model == 'cnn5' and args.dataset == 'cifar10':
        train_cnn5_cifar10(results_dir, checkpoints_dir, log_dir)
