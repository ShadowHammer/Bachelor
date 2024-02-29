import os
import random
import math
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


#Laver bounding boxes

class BoundingBox:
    def __init__(self, path):
        self.path = path
        self.image_path = self.path
        self.anno_path = self.path

    def read_image(self,image_path):
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    def create_mask(self,bb, x):
        """Creates a mask for the bounding box of same shape as image"""
        rows,cols,*_ = x.shape
        Y = np.zeros((rows, cols))
        bb = bb.astype(np.int64)
        Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
        return Y

    def mask_to_bb(self,Y):
        """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
        cols, rows = np.nonzero(Y)
        if len(cols)==0: 
            return np.zeros(4, dtype=np.float32)
        top_row = np.min(rows)
        left_col = np.min(cols)
        bottom_row = np.max(rows)
        right_col = np.max(cols)
        return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32) 

    def create_bb_array(self, x):
        """Generates bounding box array from a train_df row"""
        return np.array([x[5],x[4],x[7],x[6]]) #xmin, ymin, xmax, ymax

    def resize_image_bb(self,read_path,write_path,bb,sz):
        """Resize an image and its bounding box and write image to new path"""
        im = self.read_image(read_path)
        im_resized = cv2.resize(im, (int(1.49*sz), sz))
        Y_resized = cv2.resize(self.create_mask(bb, im), (int(1.49*sz), sz))
        new_path = str(write_path/read_path.parts[-1]) 
        cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
        return new_path, self.mask_to_bb(Y_resized)

    def filelist(self, root, file_type):
        """Returns a fully-qualified list of filenames under root directory"""
        return [os.path.join(directory_path, f) for directory_path, directory_name, 
                files in os.walk(root) for f in files if f.endswith(file_type)]

    def generate_train_df (self,anno_path):
        annotations = self.filelist(anno_path, '.xml')
        anno_list = []
        for anno_path in annotations:
            root = ET.parse(anno_path).getroot()
            anno = {}
            anno['filename'] = Path(str(self.image_path) + '/'+ root.find("./filename").text)
            anno['width'] = root.find("./size/width").text
            anno['height'] = root.find("./size/height").text
            anno['class'] = root.find("./object/name").text
            anno['xmin'] = int(root.find("./object/bndbox/xmin").text)
            anno['ymin'] = int(root.find("./object/bndbox/ymin").text)
            anno['xmax'] = int(root.find("./object/bndbox/xmax").text)
            anno['ymax'] = int(root.find("./object/bndbox/ymax").text)
            anno_list.append(anno)
        return pd.DataFrame(anno_list)
    
    def create_corner_rect(self,bb, color='yellow'):
        print(bb)
        bb = np.array(bb, dtype=np.float32)
        return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color, fill=False, lw=3)

    def show_corner_bb(self,im, bb):
        print(bb)
        plt.imshow(im)
        plt.gca().add_patch(self.create_corner_rect(bb))
    
    
    def run(self):
        df_train = self.generate_train_df(self.anno_path)
        class_dict = {'background': 0, 'pothole': 1}
        df_train['class'] = df_train['class'].apply(lambda x:  class_dict[x])

        #Populating Training DF with new paths and bounding boxes
        #new_paths = []
        #new_bbs = []
        #train_path_resized = self.image_path
        #for index, row in df_train.iterrows():
        #    new_path,new_bb = self.resize_image_bb(row['filename'], train_path_resized, self.create_bb_array(row.values),300)
        #    new_paths.append(new_path)
        #    new_bbs.append(new_bb)
        #df_train['new_path'] = new_paths
        #df_train['new_bb'] = new_bbs

        im = cv2.imread(str(df_train.values[58][0]))
        bb = self.create_bb_array(df_train.values[58])
        print(im.shape)

        Y = self.create_mask(bb, im)
        self.mask_to_bb(Y)
        # loop to show all images
        """ for i in range(0, 1):
            im = cv2.imread(str(df_train.values[i][0]))
            bb = self.create_bb_array(df_train.values[i])
            Y = self.create_mask(bb, im)
            plt.imshow(im)
            plt.show()
            plt.imshow(Y, cmap='gray')
            plt.show() """
        for i in range(0, 10):
            im = cv2.imread(str(df_train.values[i][0]))
            bb = self.create_bb_array(df_train.values[i])
            self.show_corner_bb(im, bb)
            plt.show()
       


        