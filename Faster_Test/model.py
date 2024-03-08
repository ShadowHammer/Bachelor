"""
Python script to prepare FasterRCNN model.
"""

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import config

def model():
    # load the coco pre-trained model
    # keep image size at 800 for faster training
    # can increase min_size in config.py to 1024 for better results will be slower
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,min_size=config.MIN_SIZE)

    #one class is for potholes the other for background
    num_classes = 2
    #get the imput features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with the features head
    # the head layer will classify the images based on data  input features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model






