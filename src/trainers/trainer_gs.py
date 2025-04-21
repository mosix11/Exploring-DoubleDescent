import torch
from torch.optim import AdamW, Adam, SGD
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from .custom_lr_schedulers import InverseSquareRootLR

import os
import socket
import datetime
from pathlib import Path
import time
from tqdm import tqdm
import random
import numpy as np

from typing import List, Tuple, Union

from ..utils import nn_utils, misc_utils



# Trainer based on gradient steps
class TrainerGS:

    def __init__(
        self,
        max_gradient_steps: int = 100000,
        optimizer_cfg: dict = {
                'type': 'adamw',
                'lr': 1e-4,
                'betas': (0.9, 0.999)
            },
        lr_schedule_cfg: dict = None,
        outputs_dir: Path = Path("./outputs"),
        validation_freq: int = None,
        early_stopping: bool = False,
        run_on_gpu: bool = True,
        use_amp: bool = True,
        log_tensorboard: bool = False,
        log_dir: Path = None,
        seed: int = None
    ):
        if seed:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True) 
        torch.set_float32_matmul_precision("high")

        self.cpu = nn_utils.get_cpu_device()
        self.gpu = nn_utils.get_gpu_device()
        if self.gpu == None and run_on_gpu:
            raise RuntimeError("""GPU device not found!""")
        self.run_on_gpu = run_on_gpu
        self.use_amp = use_amp

        outputs_dir.mkdir(exist_ok=True)
        self.outputs_dir = outputs_dir

        self.max_gradient_steps = max_gradient_steps
        self.optimizer_cfg = optimizer_cfg
        self.lr_schedule_cfg = lr_schedule_cfg
        self.outputs_dir = outputs_dir
        self.validation_freq = validation_freq
        self.early_stopping = early_stopping

        self.log_tensorboard = log_tensorboard

        if self.log_tensorboard:
            if log_dir:
                self.writer = SummaryWriter(str(log_dir.absolute()))
            else:
                self.writer = SummaryWriter(self.outputs_dir.joinpath('tensorboard/general').joinpath(socket.gethostname()+'-'+str(datetime.datetime.now())))


    def setup_data_loaders(self, dataset):
        self.dataset = dataset
        self.train_dataloader = dataset.get_train_dataloader()
        self.val_dataloader = dataset.get_val_dataloader()
        self.test_dataloader = dataset.get_test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )
        self.num_test_batches = len(self.test_dataloader)
        
    def prepare_batch(self, batch):
        if self.run_on_gpu:
            batch = [tens.to(self.gpu) for tens in batch]
            return batch
        else: return batch

    def prepare_model(self, model, state_dict=None):
        if state_dict:
            model.load_state_dict(state_dict)
        if self.run_on_gpu:
            model.to(self.gpu)
        self.model = model

    
    def configure_optimizers(self, optim_state_dict=None, last_epoch=-1):
        if self.optimizer_cfg['type'] == "adamw":
            optim = AdamW(
                params=self.model.parameters(),
                lr=self.optimizer_cfg['lr'],
                betas=self.optimizer_cfg['betas']
            )

        elif self.optimizer_cfg['type'] == "adam":
            optim = Adam(
                params=self.model.parameters(),
                lr=self.optimizer_cfg['lr'],
                betas=self.optimizer_cfg['betas']
            )
        elif self.optimizer_cfg['type'] == "sgd":
            optim = SGD(
                params=self.model.parameters(),
                lr=self.optimizer_cfg['lr'],
                momentum=self.optimizer_cfg['momentum']
            )
        else:
            raise RuntimeError("Invalide optimizer type")
        if optim_state_dict:
            optim.load_state_dict(optim_state_dict)
        
        # For schedulers based on gradient steps, `last_epoch` is works as `last_step`.
        if self.lr_schedule_cfg:
            if self.lr_schedule_cfg['type'] == 'step_lr':
                self.lr_scheduler = MultiStepLR(
                    optim,
                    milestones=self.lr_schedule_cfg['milestones'],
                    gamma=self.lr_schedule_cfg['gamma'],
                    last_epoch=last_epoch
                )
                
            elif self.lr_schedule_cfg['type'] == 'inv_sqr_root':
                self.lr_scheduler = InverseSquareRootLR(
                    optim,
                    L=self.lr_schedule_cfg['L'],
                    last_epoch=last_epoch
                )
            elif self.lr_schedule_cfg['type'] == 'red_on_plat':
                self.lr_scheduler = ReduceLROnPlateau(
                    optim,
                    mode=self.lr_schedule_cfg['mode'],
                    factor=self.lr_schedule_cfg['factor'],
                    patience=self.lr_schedule_cfg['patience'],
                )
            elif self.lr_schedule_cfg['type'] == 'cos_ann':
                self.lr_schedule_cfg = CosineAnnealingLR(
                    optim,
                    T_max=self.lr_schedule_cfg['T_max'],
                    eta_min=self.lr_schedule_cfg['eta_min']
                )

        # if self.early_stopping:
        #     self.early_stopping = nn_utils.EarlyStopping(patience=8, min_delta=0.001, mode='max', verbose=False)
        self.optim = optim

    def fit(self, model, dataset, resume=False):
        self.setup_data_loaders(dataset)

        if resume:
            ...
            # ckp_path = self.checkpoints_dir.joinpath("model_ckp.pt")
            # if not ckp_path.exists():
            #     raise RuntimeError(
            #         "There is no checkpoint saved! Set the `resume` flag to False."
            #     )
            # checkpoint = torch.load(ckp_path)
            # self.prepare_model(model, checkpoint["model_state"])
            # self.configure_optimizers(
            #     checkpoint["optim_state"], last_epoch=checkpoint["epoch"]
            # )
            # self.epoch = checkpoint["epoch"]
        else:
            self.prepare_model(model)
            self.configure_optimizers()
            self.g_step = 0
            self.epoch = 0

        self.grad_scaler = GradScaler("cuda", enabled=self.use_amp)

        self.fit_model()

        train_results = self.evaluate(set='train')
        test_results = self.evaluate(set='test')
        results = {
            'train_loss': train_results['loss'],
            'train_acc': train_results['acc'],
            'test_loss': test_results['loss'],
            'test_acc': test_results['acc']
        }
        if self.log_tensorboard:
            self.writer.flush()
        
        return results


    def fit_model(self):
        
        pbar = tqdm(range(self.g_step, self.max_gradient_steps), total=self.max_gradient_steps)
        
        # ******** Training Part ********
        self.model.train()
        

        while self.g_step < self.max_gradient_steps:
            
            epoch_train_loss = misc_utils.AverageMeter()
            epoch_train_acc = misc_utils.AverageMeter()
            for i, batch in enumerate(self.train_dataloader):
                input_batch, target_batch = self.prepare_batch(batch)

                self.optim.zero_grad()
                
                loss, metric = self.model.training_step(input_batch, target_batch, self.use_amp)
                
                if self.use_amp:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optim)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optim.step()
                    
                self.g_step += 1
                pbar.update()
                if self.lr_scheduler:
                    self.lr_scheduler.step()


                epoch_train_loss.update(loss.item(), n=input_batch.shape[0])
                epoch_train_acc.update(metric, input_batch.shape[0])
                
            
                
            if self.log_tensorboard:
                self.writer.add_scalar('Train/Loss', epoch_train_loss.avg, self.epoch)
                self.writer.add_scalar('Train/ACC', epoch_train_acc.avg, self.epoch)
                self.writer.add_scalar('Train/LR', self.lr_scheduler.get_last_lr()[0], self.epoch)
            
            if epoch_train_loss.avg == 0.0 or epoch_train_acc.avg == 1.0:
                if self.early_stopping: self.early_stop = True
            
            
            
            if self.validation_freq:
                if (self.epoch+1) % self.validation_freq == 0:
                    res = self.evaluate(set='test')
                    if self.log_tensorboard:
                        self.writer.add_scalar('Test/Loss', res['loss'], self.epoch)
                        self.writer.add_scalar('Test/ACC', res['acc'], self.epoch)
                    
            
            self.epoch += 1
            
            
        

    def evaluate(self, set='val'):
        self.model.eval()
        loss_met = misc_utils.AverageMeter()
        acc_met = misc_utils.AverageMeter()
        
        if set=='train':
            for i, batch in enumerate(self.train_dataloader):
                input_batch, target_batch = self.prepare_batch(batch)
                loss, metric = self.model.validation_step(input_batch, target_batch, self.use_amp)
                loss_met.update(loss.item(), n=input_batch.shape[0])
                acc_met.update(metric, input_batch.shape[0])
        elif set=='val':
            for i, batch in enumerate(self.val_dataloader):
                input_batch, target_batch = self.prepare_batch(batch)
                loss, metric = self.model.validation_step(input_batch, target_batch, self.use_amp)
                loss_met.update(loss.item(), n=input_batch.shape[0])
                acc_met.update(metric, input_batch.shape[0])
                
        elif set=='test':
            for i, batch in enumerate(self.test_dataloader):
                input_batch, target_batch = self.prepare_batch(batch)
                loss, metric = self.model.validation_step(input_batch, target_batch, self.use_amp)
                loss_met.update(loss.item(), n=input_batch.shape[0])
                acc_met.update(metric, input_batch.shape[0])
            
            
        results = {
            'loss': loss_met.avg,
            'acc': acc_met.avg
        }
        return results