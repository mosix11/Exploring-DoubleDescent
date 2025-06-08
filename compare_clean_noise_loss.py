import torch
import torchmetrics

from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, CNN5

import matplotlib.pyplot as plt
from src.utils import nn_utils, misc_utils

from functools import partial
from pathlib import Path

import pickle
import argparse
import os
import dotenv



if __name__ == "__main__":
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    base_dir = Path('outputs/modelwise/FC1_MoG(smpls100000+ftrs512+cls30+0.2Noise)_Parallel_B1024_Seed22')
    
    param_range = [
        1, 4, 8, 12, 18, 20, 22, 24, 28, 32, 36, 38, 44, 56, 80, 96, 128,
        160, 192, 208, 216, 224, 232, 240, 248, 256, 264, 280, 296, 304,
        320, 336, 344, 368, 392, 432, 464, 512, 768, 1024, 2048, 3072,
        4096, 8192, 16384, 32768, 65636, 98504, 131272, 196908, 262544
    ]
    
    num_samples = 100000
    batch_size = 1024
    num_features = 512
    num_classes = 30
    label_noise = 0.2
    dataset_seed = 22
    
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
        seed=dataset_seed
    )
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True) 
    torch.set_float32_matmul_precision("high")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    
    results = {}
    for h in param_range:
        prefix = f"fc1|h{h}_mog"
        subdirs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
        if not subdirs:
            continue
        
        last_model_path = os.path.join(base_dir, subdirs[0], "checkpoint", "final_ckp.pth")
        best_model_path = os.path.join(base_dir, subdirs[0], "checkpoint", "best_ckp.pth")
        
        last_model_ckp = torch.load(last_model_path)
        best_model_ckp = torch.load(best_model_path)
        
        model = FC1(
            input_dim=num_features,
            hidden_dim=h,
            output_dim=num_classes,
            # weight_init=weight_init_method,
            loss_fn=loss_fn,
            metric=acc_metric,
        )
        
        model.load_state_dict(last_model_ckp['model_state'])
        model.to(gpu)
        model.eval()
        
        clean_loss_met = misc_utils.AverageMeter()
        clean_acc_met = misc_utils.AverageMeter()
        noisy_loss_met = misc_utils.AverageMeter()
        noisy_acc_met = misc_utils.AverageMeter()
        

        train_dl = dataset.get_train_dataloader()
        
        # for batch in train_dl:
        #     batch = [tens.to(gpu) for tens in batch]
        #     x, y, noisy = batch
            

        #     # print("Number of clean samples:", x_clean.shape[0])
        #     # print("Number of noisy samples:", x_noisy.shape[0])
        #     loss, acc = model.validation_step(x, y)
            
        #     clean_loss_met.update(loss.detach().cpu().item(), n=x.shape[0])
        #     clean_acc_met.update(acc.detach().cpu().item(), n=x.shape[0])
            
            
        #     # total_count += x.shape[0]
        #     # noisy_count += torch.count_nonzero(noisy).item()
        
        # results[h] = {
        #     'Loss/Clean': clean_loss_met.avg,
        #     'ACC/Clean': clean_acc_met.avg
        # }
        for batch in train_dl:
            batch = [tens.to(gpu) for tens in batch]
            x, y, noisy = batch
            
            x_clean = x[~noisy]
            y_clean = y[~noisy]

            x_noisy = x[noisy]
            y_noisy = y[noisy]

            # print("Number of clean samples:", x_clean.shape[0])
            # print("Number of noisy samples:", x_noisy.shape[0])
            loss_clean, acc_clean = model.validation_step(x_clean, y_clean)
            loss_noisy, acc_noisy = model.validation_step(x_noisy, y_noisy)
            
            clean_loss_met.update(loss_clean.detach().cpu().item(), n=x_clean.shape[0])
            clean_acc_met.update(acc_clean.detach().cpu().item(), n=x_clean.shape[0])
            
            noisy_loss_met.update(loss_noisy.detach().cpu().item(), n=x_noisy.shape[0])
            noisy_acc_met.update(acc_noisy.detach().cpu().item(), n=x_noisy.shape[0])
            
            # total_count += x.shape[0]
            # noisy_count += torch.count_nonzero(noisy).item()
        
        results[h] = {
            'Loss/Clean': clean_loss_met.avg,
            'Loss/Noisy': noisy_loss_met.avg
        }
    
    
    x_values = param_range
    y_clean = [results[h]['Loss/Clean'] for h in param_range]
    y_noisy = [results[h]['Loss/Noisy'] for h in param_range]

        # Calculate the total loss
    y_total = [results[h]['Loss/Clean'] + results[h]['Loss/Noisy'] for h in param_range]

    # Create the plot
    plt.figure(figsize=(10, 6)) # Set a good figure size

    # Plot the 'Loss/Clean' curve
    plt.plot(x_values, y_clean, label='Loss/Clean', marker='o', linestyle='-', color='blue')

    # Plot the 'Loss/Noisy' curve
    plt.plot(x_values, y_noisy, label='Loss/Noisy', marker='x', linestyle='--', color='red')

    # Plot the 'Loss/Tot' curve
    # Using a different marker and linestyle for clarity
    plt.plot(x_values, y_total, label='Loss/Tot', marker='^', linestyle='-.', color='green')

    # Set log scale for both x and y axes
    plt.xscale('log')
    plt.yscale('log')

    # Add labels and title
    plt.xlabel('Parameter Range (Log Scale)')
    plt.ylabel('Loss Value (Log Scale)')
    plt.title('Loss Curves vs. Parameter Range')

    # Add a legend to distinguish the curves
    plt.legend()

    # Add a grid for better readability
    plt.grid(True, which="both", ls="-", alpha=0.6)

    # Display the plot
    plt.show()