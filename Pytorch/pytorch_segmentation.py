import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pandas as pd
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM

from customDataset import datasættet

import cv2

#----------- pip installs ----------------------------------------------------#

#pip install torch
#pip install torchvision
#pip install matplotlib
#pip install pandas
#pip install scikit-image
#pip3 install opencv-python 

#----------- Definitioner ----------------------------------------------------#
batch_size = 20
resize = 244

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



#----------- Funktioner ------------------------------------------------------#

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

# Set the working (writable) directory.
image_dir = '\\pothole_images\\'

working_dir = os.getcwd()

def save_model_checkpoint(model, cp_name):
    torch.save(model.state_dict(), os.path.join(working_dir, cp_name))

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")    


# Load model from saved checkpoint
def load_model_from_checkpoint(model, ckp_path):
    return model.load_state_dict(
        torch.load(
            ckp_path,
            map_location=get_device(),
        )
    )

# Send the Tensor or Model (input argument x) to the right device
# for this notebook. i.e. if GPU is enabled, then send to GPU/CUDA
# otherwise send to CPU.
print(f'{torch.version.cuda}')
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
print(f'Using {device} device')

def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")
# end if

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    # end while
# end def
        
def read_image(image_path):
    return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

# Validation: Check if CUDA is available
print(f"CUDA: {torch.cuda.is_available()}")
# ----------------------------------------------------------------------------------------------------------------------------------------#


from enum import IntEnum
class TrimapClasses(IntEnum):
    POTHOLE = 0
    BACKGROUND = 1
    BORDER = 2

# Convert a float trimap ({1, 2, 3} / 255.0) into a float tensor with
# pixel values in the range 0.0 to 1.0 so that the border pixels
# can be properly displayed.
def trimap2f(trimap):
    return (img2t(trimap) * 255.0 - 1) / 2

# Spot check a segmentation mask image after post-processing it
# via trimap2f().


# Simple torchvision compatible transform to send an input tensor
# to a pre-specified device.
class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={device})"
    
train_transforms = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

test_transforms = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])
valid_transforms = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])
    


root_dir = working_dir + image_dir
train_dir = root_dir + "training\\"
test_dir = root_dir + "testing\\"
valid_dir = root_dir + "validation\\"


if os.path.exists(train_dir) != True:
    print("The path does not exist: " + train_dir)

if os.path.exists(test_dir) != True:
    print("The path does not exist: " + test_dir)

if os.path.exists(valid_dir) != True:
    print("The path does not exist: " + valid_dir)
   



im = Image.open(train_dir + "image (1).png")
print(im)
I = np.array(im)

print(I)
im_seg=t2img(trimap2f(I))
plt.imshow(im_seg)
plt.show()

#lav en csv fil og indsæt navnet
#dataset = datasættet(csv_file = 'data',root_dir=root_dir,transform=transforms.ToTensor())

train_set = datasættet(csv_file=train_dir + 'training.csv',root_dir=train_dir+"/images/",transform=train_transforms)
test_set = datasættet(csv_file=test_dir + 'testing.csv',root_dir=test_dir+"/images/",transform=test_transforms)
valid_set = datasættet(csv_file=valid_dir + 'validation.csv',root_dir=valid_dir+"/images/",transform=valid_transforms)


train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)

# Check the size of the train, test and validation datasets
print(f"Train set: {len(train_set)}")
print(f"Test set: {len(test_set)}")
print(f"Validation set: {len(valid_set)}")

# Check the size of the train, test and validation dataloaders
print(f"Train loader: {len(train_loader)}")
print(f"Test loader: {len(test_loader)}")
print(f"Validation loader: {len(val_loader)}")
