import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from model import model
from dataset import train_data_loader

# the computation device
print(f'{torch.version.cuda}')
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
print(f'Using {device} device')

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

def train(train_dataloader):
    model.train()
    running_loss = 0
    for i, data in enumerate(train_dataloader):
        
        optimizer.zero_grad()
        images, targets, images_ids = data[0], data[1], data[2]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            print(f"Iteration #{i} loss: {loss}")
    train_loss = running_loss/len(train_dataloader.dataset)
    return train_loss

def save_model():
    working_dir = os.getcwd()
    faster = 'Faster_test\\'
    faster_dir = os.path.join(working_dir,faster)
    checkpoint = "checkpoints\\"
    checkpoint_dir = os.path.join(faster_dir , checkpoint)
    file = "fasterrcnn_resnet50_fpn.pth"
    file_dir = os.path.join(checkpoint_dir , file)
    torch.save(model.state_dict(),file_dir)

def visualize():
    """
    This function will only execute if the DEBUG flag in config.py is set to True
    """
    images, targets, image_ids = next(iter(train_data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]

    for i in range(1):
        boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
        sample = images[i].permute(1,2,0).cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(15,12))

        for box in boxes:
            cv2.rectangle(sample,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (220, 0, 0), 3)
            ax.set_axis_off()
            plt.imshow(sample)
            plt.show()
