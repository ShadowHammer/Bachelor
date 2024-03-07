import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl

class PotholeDataset(Dataset):
    """
    Custom dataset class for loading the pothole dataset.
    
    Args:
        data_dir (str): Path to the directory containing the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        """
        Returns the size of the dataset.
        
        Returns:
            int: Size of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: Tuple containing the sample and its corresponding label.
        """
        return self.data[idx]
    
class PotholeDetector(pl.LightningModule):
    """
    PyTorch Lightning module for the pothole detection model.
    """
    def __init__(self, input_shape=(3, 100, 100)):
        super(PotholeDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Optimizer for training.
        """
        return optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        
        Args:
            batch (tuple): Tuple containing the input batch and its corresponding labels.
            batch_idx (int): Index of the current batch.
            
        Returns:
            torch.Tensor: Loss value for the current step.
        """
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

if __name__ == "__main__":
    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, 'Pothole_yolo')
    batch_size = 32
    image_size = 100

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = PotholeDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_data = PotholeDataset(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    model = PotholeDetector()

    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, train_loader, val_loader)




