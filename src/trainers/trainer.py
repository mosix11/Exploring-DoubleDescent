import torch
from torch.optim import AdamW, Adam, SGD
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import os
import socket
import datetime
from pathlib import Path
import time
from tqdm import tqdm

from typing import List, Tuple, Union

from ..utils import nn_utils, misc_utils


class Trainer:

    def __init__(
        self,
        max_epochs: int = 400,
        optimizer_type: str = "adamw",
        lr: float = 1e-4,
        # optim_beta1: float = 0.9,
        # optim_beta2: float = 0.95,
        lr_schedule_strategy: str = "plat",
        outputs_dir: Path = Path("./outputs"),
        early_stopping: bool = False,
        run_on_gpu: bool = True,
        use_amp: bool = True,
    ):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        self.cpu = nn_utils.get_cpu_device()
        self.gpu = nn_utils.get_gpu_device()
        if self.gpu == None and run_on_gpu:
            raise RuntimeError("""GPU device not found!""")
        self.run_on_gpu = run_on_gpu
        self.use_amp = use_amp

        outputs_dir.mkdir(exist_ok=True)
        self.outputs_dir = outputs_dir

        self.max_epochs = max_epochs
        self.optimizer_type = optimizer_type
        self.lr = lr
        # self.optim_betas = (optim_beta1, optim_beta2)
        self.lr_schedule_strategy = lr_schedule_strategy
        self.outputs_dir = outputs_dir
        self.early_stopping = early_stopping

    def setup_data_loaders(self, dataset):
        self.dataset = dataset
        self.train_dataloader = dataset.get_train_dataloader()
        self.val_dataloader = dataset.get_val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model, state_dict=None):
        if state_dict:
            model.load_state_dict(state_dict)
        if self.run_on_gpu:
            model.to(self.gpu)
        self.model = model

    def configure_optimizers(self, optim_state_dict=None, last_epoch=-1):
        if self.optimizer_type == "adamw":
            optim = AdamW(
                params=self.model.parameters(), lr=self.lr, betas=self.optim_betas
            )

        elif self.optimizer_type == "adam":
            optim = Adam(
                params=self.model.parameters(), lr=self.lr, betas=self.optim_betas
            )
        elif self.optimizer_type == "sgd":
            optim = SGD(params=self.model.parameters(), lr=self.lr)
        else:
            raise RuntimeError("Invalide optimizer type")
        if optim_state_dict:
            optim.load_state_dict(optim_state_dict)

        # if self.lr_schedule_strategy == 'cos':
        #     self.lr_scheduler = CosineAnnealingLR(optim, T_max=self.max_epochs, eta_min=1e-7)
        # elif self.lr_schedule_strategy == 'plat':
        #     self.lr_scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=3)
        # else:
        #     self.lr_scheduler = None

        # if self.early_stopping:
        #     self.early_stopping = nn_utils.EarlyStopping(patience=8, min_delta=0.001, mode='max', verbose=False)
        self.optim = optim

    def fit(self, model, dataset, resume=False):
        self.setup_data_loaders(dataset)

        if resume:
            ckp_path = self.checkpoints_dir.joinpath("model_ckp.pt")
            if not ckp_path.exists():
                raise RuntimeError(
                    "There is no checkpoint saved! Set the `resume` flag to False."
                )
            checkpoint = torch.load(ckp_path)
            self.prepare_model(model, checkpoint["model_state"])
            self.configure_optimizers(
                checkpoint["optim_state"], last_epoch=checkpoint["epoch"]
            )
            self.epoch = checkpoint["epoch"]
        else:
            self.prepare_model(model)
            self.configure_optimizers()
            self.epoch = 0

        self.grad_scaler = GradScaler("cuda", enabled=self.use_amp)

        self.early_stop = False
        for self.epoch in range(self.epoch, self.max_epochs):
            if self.early_stop:
                break
            self.fit_epoch()

        # if self.write_sum:
        #     self.writer.flush()

    def fit_epoch(self):

        # ******** Training Part ********
        self.model.train()

        epoch_start_time = time.time()
        epoch_train_loss = misc_utils.AverageMeter()
        epoch_train_ap = misc_utils.AverageMeter()

        for i, (input_batch, target_batch) in tqdm(
            enumerate(self.train_dataloader),
            total=self.num_train_batches,
            desc="Processing Training Epoch {}".format(self.epoch + 1),
        ):
            ...
            
            
            
        print( 
            f"Epoch {self.epoch + 1}/{self.max_epochs}, "
            f"Training Loss: {epoch_train_loss.avg}, "
            f"Average Precision: {epoch_train_ap.avg}, "
            f"Time taken: {int((time.time() - epoch_start_time)//60)}:"
            f"{int((time.time() - epoch_start_time)%60)} minutes"
        )
