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

def conv_no_pooling(in_channels, out_channels, kernel_size, padding=1):
    return  nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.BatchNorm2d(out_channels), nn.ReLU())

def conv_pooling(in_channels, out_channels, kernel_size, padding=1):
    return  nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.BatchNorm2d(out_channels),  nn.ReLU(), nn.MaxPool2d(2))




class model3(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = conv_no_pooling(3, 64, 3)
        nn.init.xavier_uniform(self.conv1.weight)
        self.conv1_0  = conv_no_pooling(64, 128, 3)
        nn.init.xavier_uniform(self.conv1_0.weight)
        self.conv2 = conv_no_pooling(128, 128, 3)
        nn.init.normal(self.conv2.weight, std=0.01)
        self.conv2_1 = conv_pooling(128, 128, 3)
        nn.init.normal(self.conv2_1.weight, std=0.01)
        self.conv2_2 = conv_no_pooling(128, 128, 3)
        nn.init.normal(self.conv2_2.weight, std=0.01)
        self.conv3 = conv_no_pooling(128, 128, 3)
        nn.init.xavier_uniform(self.conv3.weight)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        nn.init.xavier_uniform(self.conv4.weight)
        self.pool4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.conv4_1 = conv_no_pooling(256, 256, 3)
        nn.init.xavier_uniform(self.conv4_1.weight)
        self.conv4_2 = conv_no_pooling(256, 256, 3)
        nn.init.xavier_uniform(self.conv4_2.weight)
        self.pool4_2 = nn.MaxPool2d(2)
        self.conv4_0 = conv_no_pooling(256, 512, 3)
        nn.init.xavier_uniform(self.conv4_0.weight)
        self.cccp4 = nn.Conv2d(512, 2048, 1, padding=1)
        self.relu_cccp4 = nn.ReLU()
        self.cccp5 = nn.Conv2d(2048, 256, 1, padding=1)
        self.relu_cccp5 = nn.ReLU()
        self.pool_cccp5 = nn.MaxPool2d(2)
        self.cccp6 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu_cccp6 = nn.ReLU()
        self.pool_cccp6 = nn.MaxPool2d(2)
        self.linear = nn.Linear(2*2*256, 10)
        self.sm = nn.Softmax()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_0(x)
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4_2(x)
        x = self.conv4_0(x)

        x = self.cccp4(x)
        x = self.relu_cccp4(x)
        x = self.cccp5(x)
        
        x = self.relu_cccp5(x)
        x = self.pool_cccp5(x)
        x = self.cccp6(x)
        x = self.relu_cccp6(x)
        x = self.pool_cccp6(x)
        x = x.view(-1, 2*2*256)
        x = self.linear(x)
        x = self.sm(x)
        return x
