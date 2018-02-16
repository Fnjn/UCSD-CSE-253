#!/usr/bin/env python3

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import time

def same_padding(input_size, kernel_size, stride_size):
    return (input_size - 1 - (input_size - kernel_size // stride_size)) // 2


class Bottleneck2(nn.Module):

    def __init__(self, input_size, in_channels, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=same_padding(input_size, 3, 1))
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, in_channels, 1)
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

class Bottleneck1(nn.Module):

    def __init__(self, input_size, in_channels, channels, out_channels, stride_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, 1, stride=stride_size)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=same_padding((input_size-1)//stride_size+1, 3, 1))
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, stride=stride_size)
        self.bn_sc = nn.BatchNorm2d(out_channels)

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

        x_shortcut = self.conv_sc(x_shortcut)
        x_shortcut = self.bn_sc(x_shortcut)

        x.add_(x_shortcut)
        x = F.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1A = nn.Conv2d(3, 64, 3, stride=1)
        self.bn1A = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2)

        self.layer2A = Bottleneck1(14, 64, 64, 256, 1)
        self.layer2B = Bottleneck2(14, 256, 64)
        self.layer2C = Bottleneck2(14, 256, 64)

        self.layer3A = Bottleneck1(14, 256, 128, 512, 2)
        self.layer3B = Bottleneck2(7, 512, 128)
        self.layer3C = Bottleneck2(7, 512, 128)
        self.layer3D = Bottleneck2(7, 512, 128)

        self.avgpool = nn.AvgPool2d(2, stride=2)

        self.fc = nn.Linear(3*3*512, 10)

    def forward(self, x):
        x = self.conv1A(x)
        x = self.bn1A(x)
        x = self.maxpool(x)

        x = self.layer2A(x)
        x = self.layer2B(x)
        x = self.layer2C(x)

        x = self.layer3A(x)
        x = self.layer3B(x)
        x = self.layer3C(x)
        x = self.layer3D(x)

        x = self.avgpool(x)
        x = x.view(-1, 3*3*512)
        x = self.fc(x)

        return x

