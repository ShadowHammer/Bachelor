import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
import copy
import math
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import sys
#%matplotlib inline
print(torch.__version__)
print(torchvision.__version__)
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2


# Hyperparameters
batch_size = 16
num_epochs= 1
lr = 0.001
image_size = [600, 600]


def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


class PotholeDetectionClass(datasets.VisionDataset):
    def __init__(self, root, stage='/train', transform=None, target_transform=None, transforms=None, batch_size = batch_size):
        super().__init__(root, transforms, transform, target_transform)
        self.stage = stage #train, valid, test
        self.coco = COCO(root + stage + "/_annotations.coco.json") # annotations stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
        self.batch_size = batch_size

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name']
        path = "/" + path
        image = cv2.imread(self.root + self.stage + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))

        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed['image']
        boxes = transformed['bboxes']

        new_boxes = []
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['image_id'] = torch.tensor([t['image_id'] for t in target])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return image.div(255), targ # scale images
    def __len__(self):
        return len(self.ids)

"""
    def train_dataloader(self):
        train_dataset = self.root + "/train"
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        valid_dataset = self.root + "/valid"
        return DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def test_dataloader(self):
        test_dataset = self.root + "/test"
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
"""    


dataset_path = "/Pytorch_lightning/Pothole_coco"
dataset_path = os.getcwd() + dataset_path

coco = COCO(dataset_path + "/train" + "/_annotations.coco.json")
categories = coco.cats
n_classes = len(categories.keys())
categories

classes = [i[1]['name'] for i in categories.items()]

train_dataset = PotholeDetectionClass(root=dataset_path, transforms=get_transforms(True))
test_dataset = PotholeDetectionClass(root=dataset_path, stage='/test', transforms=get_transforms(False))

model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

def custom_collate(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=custom_collate)

device = torch.device("cuda") # use GPU to train
model = model.to(device)

# Now, and optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()

    all_losses = []
    all_losses_dict = []

    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)

model.eval()
torch.cuda.empty_cache()

for i in range(10,test_dataset.__len__()):
    img, _ = test_dataset[i]
    img_int = torch.tensor(img*255, dtype=torch.uint8)
    with torch.no_grad():
        prediction = model([img.to(device)])
        pred = prediction[0]
        fig = plt.figure(figsize=(14, 10))
    plt.imshow(draw_bounding_boxes(img_int,
        pred['boxes'][pred['scores'] > 0.8],
        [classes[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()], width=4
    ).permute(1, 2, 0))
    plt.show()

