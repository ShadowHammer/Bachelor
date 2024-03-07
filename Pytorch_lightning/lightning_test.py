import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchvision
import torchmetrics
from torchmetrics import Metric
from torchvision import transforms as T
import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection._utils import Matcher
from torchvision.ops.boxes import box_iou


class PotholeCSVDataset(pl.LightningDataModule):
    def __init__(self, root_dir, train_data, test_data, valid_data, csv_file,  batch_size=64,):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])     
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.annotations.iloc[idx, 1]))
        bbox = torch.tensor([self.annotations.iloc[idx, 2], self.annotations.iloc[idx, 3], self.annotations.iloc[idx, 4], self.annotations.iloc[idx, 5]])
        if self.transform:
            image = self.transform(image)

        return image, label, bbox

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
        ])
        
        self.train_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.test_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.valid_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):        
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()
    
    def compute(self):
        return self.correct.float() / self.total.float()


class FasterRCNN(pl.LightningModule):
    def __init__(self,n_classes, lr):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.lr = lr

    def forward(self, imgs,targets=None):
      # Torchvision FasterRCNN returns the loss during training 
      # and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
      return optimizer

    def training_step(self, batch, batch_idx):

      imgs = batch[0]

      targets = []
      for boxes in batch[1]:
        target= {}
        target["boxes"] = boxes.cuda()
        target["labels"] = torch.ones(len(target["boxes"])).long().cuda()
        targets.append(target)

      # fasterrcnn takes both images and targets for training, returns
      loss_dict = self.detector(imgs, targets)
      loss = sum(loss for loss in loss_dict.values())
      return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        img, boxes = batch
        pred_boxes = self.forward(img)
        return {"val_loss": torch.mean(torch.stack([self.accuracy(b, pb["boxes"], iou_threshold=0.5) for b, pb in zip(boxes, pred_boxes)]))}

    def test_step(self, batch, batch_idx):
        img, boxes = batch
        pred_boxes = self.forward(img)
        return {"test_loss": torch.mean(torch.stack([self.accuracy(b, pb["boxes"], iou_threshold=0.5) for b, pb in zip(boxes, pred_boxes)])),
                "pred_boxes": pred_boxes}


    def accuracy(self, src_boxes,pred_boxes ,  iou_threshold = 1.):
      """
      The accuracy method is not the one used in the evaluator but very similar
      """
      total_gt = len(src_boxes)
      total_pred = len(pred_boxes)
      if total_gt > 0 and total_pred > 0:


        # Define the matcher and distance matrix based on iou
        matcher = Matcher(iou_threshold,iou_threshold,allow_low_quality_matches=False) 
        match_quality_matrix = box_iou(src_boxes,pred_boxes)

        results = matcher(match_quality_matrix)
        
        true_positive = torch.count_nonzero(results.unique() != -1)
        matched_elements = results[results > -1]
        
        #in Matcher, a pred element can be matched only twice 
        false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(matched_elements.unique()))
        false_negative = total_gt - true_positive

            
        return  true_positive / ( true_positive + false_positive + false_negative )

      elif total_gt == 0:
          if total_pred > 0:
              return torch.tensor(0.)#.cuda()
          else:
              return torch.tensor(1.)#.cuda()
      elif total_gt > 0 and total_pred == 0:
          return torch.tensor(0.)#.cuda()

class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 244)
        self.fc2 = nn.Linear(244, num_classes)
        self.loss_fn = nn.HingeEmbeddingLoss() # MSELoss, other Bianry loss functions
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        accuracy = self.my_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1': f1_score},
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'scores': scores, 'y': y}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
# Set device cuda for GPU if it's available otherwise run on the CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Hyperparameters
input_size = 244*244*3
num_classes = 2
lr = 0.001
batch_size = 2
epochs = 3
absolute_path = os.path.dirname(__file__)
root_dir = os.path.join(absolute_path, "Pothole_yolo")
train_path = os.path.join(absolute_path, "Pothole_yolo/train")
test_path = os.path.join(absolute_path, "Pothole_yolo/test")
valid_path = os.path.join(absolute_path, "Pothole_yolo/valid")
csv_file = os.path.join(absolute_path, "Pothole_yolo/train/_annotations.csv")

model = NN(input_size, num_classes)
#model = FasterRCNN(num_classes, lr)
dm = PotholeCSVDataset(root_dir=root_dir, train_data=train_path, test_data=test_path, valid_data=valid_path, csv_file=csv_file, batch_size=batch_size)

# Train Network
# trainer.tune() keeps track of the best learning rate and other hyperparameters

#trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=epochs, precision = 16) # peek trainer.py for more options
trainer = pl.Trainer(max_epochs=epochs, precision=16)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)

# Check accuracy on training & test to see how good our model
model.to(device)