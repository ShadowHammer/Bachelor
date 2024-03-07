# https://colab.research.google.com/drive/1-w8pJmMihuTkjNZ4INm_CFB7y8iWmHUC?usp=sharing#scrollTo=f65omx9A2Tbd

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10

import wandb

wandb.login()

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,batch_size,data_dir:str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.num_classes = 10
    
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            