import pdb

import pytorch_lightning as pl
import abc
import hydra
import torch.optim.lr_scheduler
import warnings
from model import small, medium
from model_proj import EncoderDecoder
from typing import Optional, Union, List, Dict, Any, Sequence
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

class BaseSystem(pl.LightningModule, abc.ABC):
    def __init__(self, cfg):
        super(BaseSystem, self).__init__()
        # when save  hyperparameters, the self.task will be ignored
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.train_cfg = cfg.train
        self.model_cfg = cfg.model
        self.model = self.build_model()
        self.loss_func = self.build_loss_func()

    def training_step(self, batch, **kwargs):
        optimizer = self.optimizers()
        param, traj, task = batch
        loss = self.forward(param, **kwargs)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        wandb.log({"train/loss": loss})
        return {'loss': loss}

    def build_model(self, **kwargs):
        n_embd = self.model_cfg.n_embd
        model = EncoderDecoder(n_embd)
        return model

    def build_loss_func(self):
        if 'loss_func' in self.train_cfg:
            loss_func = hydra.utils.instantiate(self.train_cfg.loss_func)
            return loss_func

    def configure_optimizers(self, **kwargs):
        parameters = self.model.parameters()
        self.optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, parameters)

        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        return self.optimizer

    def validation_step(self, batch, batch_idx, **kwargs):
        param, traj, task = batch
        embed = self.model.encode(param) 
        outputs = self.model.decode(embed)
        val_loss = self.loss_func(outputs, param) 
        wandb.log({"val/loss": val_loss.detach()})
        self.log('val_loss', val_loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': val_loss}
    
    def test_step(self, batch, batch_idx, **kwargs):
        pass


    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError
