import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, FC2, FC3, FCN, CNN5
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
    
    gpu_per_experiment:float = 0.1
    cpu_per_experiment:float = 1
    
    log_comet = False
    
    
    with open('fc_width_depth_confs.pkl', 'rb') as f:
        params_dict = pickle.load(f)
    
    fc_widths = [[], [], [], [], [], []]
    
    for h, confs in params_dict.items():
        fc_widths[0].append(h)
        fc_widths[1].append(confs[2]['balanced']['widths'])
        fc_widths[2].append(confs[3]['balanced']['widths'])
        fc_widths[3].append(confs[4]['balanced']['widths'])
        fc_widths[4].append(confs[5]['balanced']['widths'])
        fc_widths[5].append(confs[6]['balanced']['widths'])

    # print(len(fc_widths[0]))
    # print(len(fc_widths[1]))
    # print(len(fc_widths[2]))
    # print(len(fc_widths[3]))
    # print(len(fc_widths[4]))
    # print(len(fc_widths[5]))
        
    # print(fc_widths[0])
    # print('\n\n')
    # print(fc_widths[1])
    # print('\n\n')
    # print(fc_widths[2])
    # print('\n\n')
    # print(fc_widths[3])
    # print('\n\n')
    # print(fc_widths[4])
    # print('\n\n')
    # print(fc_widths[5])
    # return
    
    optim_cgf = {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999)
    }
    lr_schedule_cfg = None
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    
    experiment = f"FC_WD_MoG(smpls{num_samples}+ftrs{num_features}+cls{num_classes}+{label_noise}Noise)_Parallel_B{batch_size}_Seed{training_seed}"
    
    outputs_dir = outputs_dir / Path(experiment)
    outputs_dir.mkdir(exist_ok=True, parents=True)
    
    def experiment_trainable(config):
        
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
            num_workers=1,
            seed=dataset_seed
        )
        
        if config['model'] == 'fc1':
            model = FC1(
                input_dim=num_features,
                hidden_dim=config['param'],
                output_dim=num_classes,
                # weight_init=weight_init_method,
                loss_fn=loss_fn,
                metric=acc_metric,
            )
        elif config['model'] == 'fc2':
            model = FC2(
                input_dim=num_features,
                h_dims=config['param'],
                output_dim=num_classes,
                # weight_init=weight_init_method,
                loss_fn=loss_fn,
                metric=acc_metric,
            )
        elif config['model'] == 'fc3':
            model = FC3(
                input_dim=num_features,
                h_dims=config['param'],
                output_dim=num_classes,
                # weight_init=weight_init_method,
                loss_fn=loss_fn,
                metric=acc_metric,
            )
        elif config['model'] == 'fc4':
            model = FCN(
                input_dim=num_features,
                h_dims=config['param'],
                output_dim=num_classes,
                # weight_init=weight_init_method,
                loss_fn=loss_fn,
                metric=acc_metric,
            )
        elif config['model'] == 'fc5':
            model = FCN(
                input_dim=num_features,
                h_dims=config['param'],
                output_dim=num_classes,
                # weight_init=weight_init_method,
                loss_fn=loss_fn,
                metric=acc_metric,
            )
        elif config['model'] == 'fc6':
            model = FCN(
                input_dim=num_features,
                h_dims=config['param'],
                output_dim=num_classes,
                # weight_init=weight_init_method,
                loss_fn=loss_fn,
                metric=acc_metric,
            )
        else:
            raise ValueError('Unknown model config.')
        
        
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
            comet_project_name='doubledescent-modelwise-width_vs_depth-fc-mog-parallel-b1024',
            exp_name=experiment_name,
            exp_tags=experiment_tags,
            seed=training_seed
        )
        results = trainer.fit(model, dataset, resume=False)
        tune.report(results)
        
        
        
    for idx, fc_confs in enumerate(fc_widths):
        cfg = {
            'model': f"fc{idx+1}",
            "param": tune.grid_search(fc_confs)
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
            param_space=cfg,    # The hyperparameters to explore
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