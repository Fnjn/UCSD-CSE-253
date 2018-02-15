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


class VGG_fe(nn.Module):

    def __init__(self, num_layers, num_features):
        super().__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        self.num_features = num_features

        for param in vgg16.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(vgg16.features.children())[:num_layers])
        self.fc = nn.Sequential(nn.Linear(in_features=num_features, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=256))


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_features)
        x = self.fc(x)
        return x

def train_model(model, dataset, criterion, optimizer, num_epochs, batch_size, scheduler=None):
    start_time = time.time()
    model.train(True)
    dataset_size = dataset.__len__()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        if scheduler is not None:
            scheduler.step()
        running_loss = 0.
        running_corrects = 0

        for data in dataloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size
        print('%d epoch loss: %f    accuracy: %f%%' % (epoch, epoch_loss, epoch_acc*100))

    model.train(False)
    time_elapsed = time.time() - start_time
    print('Training comple in %dm, %ds' % (time_elapsed//60, time_elapsed%60))
    return model

def test_model(model, dataset):
    testloader = DataLoader(dataset, batch_size=16)
    correct_cnt = 0

    for data in testloader:
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct_cnt += torch.sum(preds.data == labels.data)

    acc = correct_cnt / dataset.__len__()
    print('Test Set Accuracy: %f%%' % (acc*100))
    return acc

def each_class_accuracy(model, dataset, num_classes=256):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    testloader = DataLoader(dataset, batch_size=16)

    for data in testloader:
        images, labels = data
        outputs = model(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda())
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(num_classes):
        print('Accuracy of class %d : %2d %%' % (i, 100*class_correct[i]/class_total[i]))
