import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence
from torchvision.transforms import functional as F
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM
from PIL import Image

import cv2

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

# Set the working (writable) directory.
image_dir = '/pothole_images/'

working_dir = os.getcwd()

dir_pothole600 = 'pothole600/images/'

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
train_dir_pothole600 =working_dir + image_dir + "training/" + dir_pothole600
#train_pothole_input = train_dir_pothole600[0]

#plt.imshow(read_image(train_dir_pothole600 + "0069.png"))
#im = cv2.imread(train_pothole_input[0])
#plt.show()
test_img = train_dir_pothole600 + "0069.png"


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
print(test_img)
im = Image.open(train_dir_pothole600 + "/0069.png")
I = np.array(im)

print(I)
im_seg=t2img(trimap2f(I))
plt.imshow(im_seg)
plt.show()