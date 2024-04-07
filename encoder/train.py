import os
import hydra
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from model import small
from dataset import Dataset
from system import Encoder
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger

def set_seed(seed):
    pl.seed_everything(seed)

def set_device(device_config):
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
    torch.cuda.set_device(device_config.cuda)
    torch.set_float32_matmul_precision('medium')

@hydra.main(config_path="config.yaml")  
def main(cfg):

    set_seed(cfg.seed)
    set_device(cfg.device)

    run_name = f"ae-{cfg.seed}-{cfg.data.batch_size}"
    wandb.init(project="policy_generator", name=run_name) 

    datamodule = Dataset(cfg.data) 
    system = Encoder(cfg) 
    trainer: Trainer = hydra.utils.instantiate(cfg.system.train.trainer)
    wandb_logger = WandbLogger()
    trainer.logger = wandb_logger

    # Train the model
    trainer.fit(system, datamodule=datamodule) 

if __name__ == "__main__":
    main()