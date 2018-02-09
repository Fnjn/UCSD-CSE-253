#!/usr/bin/env python3

import torch
import numpy as np

import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def conv_no_pooling(in_channels, out_channels, kernel_size):
    return  nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size), nn.BatchNorm2d(out_channels), nn.ReLU())

def conv_pooling(in_channels, out_channels, kernel_size):
    return  nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size), nn.BatchNorm2d(out_channels), nn.MaxPool2d(2), nn.ReLU())




class model3(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = conv_no_pooling(3, 64, 3)

        # self.conv2 = conv_no_pooling(64, 128, 3)
        # self.conv3 = conv_no_pooling(128, 128, 3)
        # self.conv4 = conv_no_pooling(128, 128, 3)
        # self.conv5 = conv_no_pooling(128, 128, 3)
        # self.conv6 = conv_no_pooling(128, 128, 3)
        # self.conv7 = conv_no_pooling(128, 128, 3)
        # self.conv8 = conv_no_pooling(128, 128, 3)
        # self.conv9 = conv_no_pooling(128, 128, 3)
        # self.conv10 = conv_no_pooling(128, 128, 3)
        # self.conv11 = conv_no_pooling(128, 128, 1)
        # self.conv12 = conv_no_pooling(128, 128, 1)
        # self.conv13 = conv_no_pooling(128, 128, 3)
        # self.model =  nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, \
        #      self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13)

        
    # def forward(self, x):
    #     x = self.conv1(x)
    #     return x

        
    def forward(self, x):
        x = self.conv1(x)

        return x