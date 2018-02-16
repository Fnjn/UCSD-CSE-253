
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import time

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

from caltech256_will import Caltech256
print(torch.cuda.is_available())


# In[2]:

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}


# In[3]:

# data_dir = 'data/256_ObjectCategories'
data_dir = '/datasets/Caltech256/256_ObjectCategories'
caltech256_train = Caltech256(data_dir, data_transforms['train'], train=True)
caltech256_test = Caltech256(data_dir, data_transforms['test'], train=False)


# In[16]:

# test data loader
dataloader = DataLoader(caltech256_train, batch_size=4)
dataiter = iter(dataloader)
image, label = dataiter.next()
print(image.size())
print(label.size())
print(label[0])


# In[22]:

def train_model(model, dataset, criterion, optimizer, scheduler, num_epochs, batch_size):
    start_time = time.time()
    model.train(True)
    dataset_size = dataset.__len__()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        scheduler.step()
        running_loss = 0.
        running_corrects = 0.
        batch_cnt = 0
        
        for data in dataloader:
            inputs, labels = data
            
            """TODO: DELETE THIS"""
            for lb in labels:
                if lb == 256:
                    print("WARNING:",labels)
            
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss
            running_corrects += torch.sum(preds == labels.data)
            
            batch_cnt += 1
            if batch_cnt % 150 == 0:
                print('Training completed [{}, {}]'.format(epoch, batch_cnt))
            
            
        epoch_loss = running_loss / float(dataset_size)
        epoch_acc = running_corrects / float(dataset_size)
        print('{} epoch loss: {} accuracy {}'.format(epoch, epoch_loss, epoch_acc))
        
    model.train(False)
    time_elapsed = time.time() - start_time
    print('Training comple in %dm, %ds' % (time_elapsed//60, time_elapsed%60))
    return model


# In[6]:

vgg16 = models.vgg16_bn(pretrained=True)
# print(vgg16)


# In[9]:

# freeze all layers
for param in vgg16.parameters():
    param.requires_grad = False

# modify last softmax layer output number
vgg16.classifier[6].out_features = 256

# set last layer's weights trainable
for param in vgg16.classifier[6].parameters():
    param.requires_grad = True

# check weights' training status
# for i in range(44):
#     print("vgg16 -> features [{}]".format(i))
#     for param in vgg16.features[i].parameters():
#         print(param.requires_grad)
# for i in range(7):
#     print("vgg16 -> classifier [{}]".format(i))
#     for param in vgg16.classifier[i].parameters():
#         print(param.requires_grad)


# In[10]:

print(vgg16)


# In[20]:

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier[6].parameters())
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30)

# vgg16 = nn.DataParallel(vgg16)
vgg16 = vgg16.cuda()


# In[ ]:

"""TODO: CHANGE THIS!"""
train_data = caltech256_train

model_tf = train_model(vgg16, train_data, criterion, optimizer, scheduler, num_epochs=5, batch_size=16)


# In[ ]:

test_dataloader = DataLoader(caltech256_test, batch_size=16)
correct_cnt = 0
cnt = 0
for data in test_dataloader:
    inputs, labels = data
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    outputs = vgg16(inputs)
    _, preds = torch.max(outputs, 1)
    correct_cnt += torch.sum(preds.data == labels.data)
    
acc = correct_cnt / caltech256_test.__len__()
print('Test Set Accuracy: %f' % (acc*100))


# In[ ]:



