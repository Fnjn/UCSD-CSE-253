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

def conv_bn_relu(in_channels, out_channels, kernel_size, padding=1, init="xavier", pooling=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    if(init == "xavier"):
        nn.init.xavier_normal(conv.weight)
    if(init == "norm"):
        nn.init.normal(conv.weight, 0.01)
    if(pooling):
        return nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2))
    else:
        return  nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU())


class model1(nn.Module):
    # simple net, with 75% accuracy
    def __init__(self):
        super().__init__()
        self.conv1 = conv_bn_relu(3, 64, 3)
        self.conv2 = conv_bn_relu(64, 128, 3)
        self.conv3 = conv_bn_relu(128, 128, 3)
        self.conv4 = conv_bn_relu(128, 128, 3, pooling=True)
        self.conv5 = conv_bn_relu(128, 128, 3)
        self.conv6 = conv_bn_relu(128, 128, 3)
        self.conv7 = conv_bn_relu(128, 128, 3, pooling=True)
        self.conv8 = conv_bn_relu(128, 128, 3)
        self.conv9 = conv_bn_relu(128, 128, 3, pooling=True)
        self.conv10 = conv_bn_relu(128, 128, 3)
        self.conv11 = conv_bn_relu(128, 128, 1)
        self.conv12 = conv_bn_relu(128, 128, 1, pooling=True)
        self.conv13 = conv_bn_relu(128, 128, 3, pooling=True)
        self.fc = nn.Linear(128*2*2, 10)
        self.sm = nn.Softmax()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = x.view(-1,128*2*2)
        x = self.fc(x)
        x = self.sm(x)
        return x

class model2(nn.Module):
    # nagadomi/kaggle-cifar10-torch7
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 128, 5),
                                  nn.BatchNorm2d(128),
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             nn.Conv2d(128, 256, 5),
                             nn.BatchNorm2d(256),
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             nn.Conv2d(256, 512, 4, padding=1),
                             nn.BatchNorm2d(512),
                             nn.ReLU(),
                             nn.Conv2d(512, 1024, 2),
                             nn.BatchNorm2d(1024),
                             nn.ReLU(),
                             nn.Dropout(),
                             nn.Conv2d(1024, 10, 1),
                             nn.BatchNorm2d(10),
                             )
        self.fc = nn.Linear(10*3*3, 10)
        self.sm = nn.Softmax()
            
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 3*3*10)
        x = self.fc(x)
        x = self.sm(x)
        return x
    
class model3(nn.Module):
    # SimpleNet test model, 59%
    def __init__(self):
        super().__init__()
        print("new")
        self.conv1 = conv_bn_relu(3, 64, 3)
        self.conv1_0  = conv_bn_relu(64, 128, 3)
        self.conv2 = conv_bn_relu(128, 128, 3, init="normal")
        self.conv2_1 = conv_bn_relu(128, 128, 3, pooling=True, init="normal")
        self.conv2_2 = conv_bn_relu(128, 128, 3, init="normal")
        self.conv3 = conv_bn_relu(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        nn.init.xavier_normal(self.conv4.weight)
        self.pool4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.conv4_1 = conv_bn_relu(256, 256, 3)
        self.conv4_2 = conv_bn_relu(256, 256, 3)
        self.pool4_2 = nn.MaxPool2d(2)
        self.conv4_0 = conv_bn_relu(256, 512, 3)
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
