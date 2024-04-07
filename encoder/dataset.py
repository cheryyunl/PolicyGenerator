import os
import torch
from torch.utils.data import Dataset, random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

class Dataset(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = getattr(cfg, 'batch_size', 64)
        self.num_workers = getattr(cfg, 'num_workers', 4)
        self.data_root = getattr(self.cfg, 'data_root', './data')

        # Check the root path
        assert os.path.exists(self.data_root), f'{self.data_root} not exists'

        if os.path.isfile(self.data_root):
            state = torch.load(self.data_root, map_location='cpu')
            self.params = state['param']
            # self.sr = state['success_rate']

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx]
        # sr = self.sr[idx]
        return params

    @property
    def dataset(self):
        train_size = int(0.85 * len(self))
        val_size = int(0.10 * len(self))
        test_size = len(self) - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self, [train_size, val_size, test_size]
        )


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)