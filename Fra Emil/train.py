from data.coco_loader import COCOSegmentation
import torch.utils.data as data
import wandb
import lightning as L
from models.segmentation.model import SegmentationModel
from lightning.pytorch.loggers import WandbLogger
from data.transforms import get_transform
from models.metrics import MetricsFactory


def main():

    config = {
        "model": "unet",
        "dataset": "DemoData",
        "batch_size": 4,
        "num_workers": 4,
        "lr": 1e-3,
        "task": "binary",
        "metrics": ["f1", "precision", "recall"]
    }

    wandb.init(project="pavement-marking-segmentation",
               config=config)
    wandb_logger = WandbLogger(project="pavement-marking-segmentation")

    root = ('C:\\Users\\evr\\Eltronic Group A S\\Project SMART - General\\01 Design\\01 Software\\00 '
            'Datasets\\Lane detection\\DemoData\\Camera_Annotated')

    dataset = COCOSegmentation(root=root, transform=get_transform(train=True))
    # train_set, val_set = split_dataset(dataset)

    train_loader = data.DataLoader(dataset=dataset,
                                   batch_size=wandb.config.batch_size,
                                   num_workers=wandb.config.num_workers)

    metrics = [MetricsFactory.build_metric(m, wandb.config.task) for m in wandb.config.metrics]

    model = SegmentationModel(architecture=wandb.config.model,
                              learning_rate=wandb.config.lr,
                              metrics=metrics)

    trainer = L.Trainer(logger=wandb_logger)
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
