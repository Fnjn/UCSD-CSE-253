#!/usr/bin/env python3

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

def same_padding(input_size, kernel_size, stride_size):
    #assert input_size > 0, assert kernel_size > 0, assert stride_size > 0
    return input_size - 1 - (input_size - kernel_size // s)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, in_channels, 1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x_shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x.add_(x_shortcut)
        x = F.relu(x)

        return x

class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = Bottleneck(32, 16)

        self.conv2 = nn.Conv2d(32, 64, 7, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer2 = Bottleneck(64, 32)

        self.avgpool = nn.AvgPool2d(3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

        self.fc1 = nn.Linear(15*15*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(-1, 15*15*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
