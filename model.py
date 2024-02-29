import lightning
from .model_store import SegmentationModelFactory
from typing import List
import torch
from conf.config import SegmentationModelConfig
import segmentation_models_pytorch as smp
from models.metrics import Metric


class SegmentationModel(lightning.LightningModule):
    """A lightning wrapper for the PyTorch segmentation model.

    """
    def __init__(self,
                 architecture: str = "unet",
                 metrics: List[Metric] = None,
                 learning_rate: float = 1e-3):
        super().__init__()

        self.save_hyperparameters()

        cfg = SegmentationModelConfig()
        print(architecture)
        self.model = SegmentationModelFactory.build_model(architecture, cfg)

        self.metrics = metrics
        self.loss_f = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.learning_rate = learning_rate

    def get_loss(self, batch):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.loss_f(y_hat, y)
        return loss

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.loss_f(y_hat, y)

        self.log('train/loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.loss_f(y_hat, y)

        self.log('val/loss', loss, on_step=False, on_epoch=True)

        for metric in self.metrics:
            self.log('val/' + metric.name, metric(y_hat, y), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.loss_f(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

        for metric in self.metrics:
            self.log('test/' + metric.name, metric(y_hat, y), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
