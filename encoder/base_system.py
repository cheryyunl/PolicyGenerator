import pdb

import pytorch_lightning as pl
import abc
import hydra
import torch.optim.lr_scheduler
import warnings
from typing import Optional, Union, List, Dict, Any, Sequence
from omegaconf import DictConfig, OmegaConf

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
        params, sr = batch
        optimizer = self.optimizers()
        loss = self.forward(params, **kwargs)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

        return {'loss': loss}

    def build_model(self, **kwargs):
        model = hydra.utils.instantiate(self.model_cfg)
        return model

    def build_loss_func(self):
        if 'loss_func' in self.train_cfg:
            loss_func = hydra.utils.instantiate(self.train_cfg.loss_func)
            return loss_func

    def configure_optimizers(self, **kwargs):
        parameters = self.model.parameters()
        self.optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, parameters)

        if 'lr_scheduler' in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.optimizer

    def validation_step(self, batch, batch_idx, **kwargs):
        params, sr = batch
        outputs = self.model(params) 
        val_loss = self.loss_func(outputs, params) 
        return {'val_loss': val_loss}
    
    def test_step(self, batch, batch_idx, **kwargs):
        params, sr = batch
        outputs = self.model(params) 
        test_loss = self.loss_func(outputs, params)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True) 
        return {"test_loss": test_loss}


    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError