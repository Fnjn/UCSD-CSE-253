
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

# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
import numpy as np
import time
import copy

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from caltech256_will import Caltech256
print(torch.cuda.is_available())


# In[2]:


data_transforms = {
    'train': transforms.Compose([
    	transforms.Scale(256),
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
# data_dir = '/datasets/Caltech256/256_ObjectCategories'
data_dir = '/home/ubuntu/datasets/256_ObjectCategories'

caltech256_train = Caltech256(data_dir, data_transforms['train'], train=True)
caltech256_test = Caltech256(data_dir, data_transforms['test'], train=False)


# In[16]:


# test data loader
dataloader = DataLoader(caltech256_train, batch_size=4)
dataiter = iter(dataloader)
image, label = dataiter.next()
# print(image.size())
# print(label.size())
# print(label[0])


# In[22]:


def train_model(model, dataset, criterion, optimizer, scheduler, num_epochs, batch_size):
    start_time = time.time()

    dataloader_train = DataLoader(dataset[0], batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset[1], batch_size=batch_size, shuffle=True)

    best_acc = 0.0

    acc_history = np.zeros((num_epochs, 2))
    loss_history = np.zeros((num_epochs, 2))
    
    for epoch in range(num_epochs):
        print('Epoch # {}'.format(epoch))

        scheduler.step()
        
        # gradient descent
        model.train(True)
        for batch_cnt, data in enumerate(dataloader_train):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            if batch_cnt % 150 == 0:
                print('Training completed [{}, {}]'.format(epoch, batch_cnt))

        # compute loss and accuracy
        model.train(False)
        for idx, dataloader in enumerate([dataloader_train, dataloader_test]):
            # compute loss and accuracy
            running_loss = 0.
            running_corrects = 0.
            for data in dataloader:
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)	
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            # save loss and accuracy
            dataset_size = dataset[idx].__len__()
            epoch_loss = running_loss / float(dataset_size)
            epoch_acc = running_corrects / float(dataset_size)
            print('loss: {} \naccuracy {}'.format(epoch_loss, epoch_acc))
            # print(type(epoch_loss), type(epoch_acc))
            
            acc_history[epoch, idx] = epoch_acc
            loss_history[epoch, idx] = epoch_loss

            # deep copy the model
            if idx == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
    
    # load best model weights
    print('Save Model')
    model.load_state_dict(best_model_wts)
    torch.save(model, "pytorch_vgg16_hw3.pth")

    time_elapsed = time.time() - start_time
    print('Training comple in {} minutes.'.format(time_elapsed//60))
    return model, acc_history, loss_history


# In[6]:

print('Build vgg16 model')
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
data = [caltech256_train, caltech256_test]

print('\n==============================')
print('Training in progress:')
md, acc_history, loss_history = train_model(vgg16, data, criterion, optimizer, scheduler, num_epochs=20, batch_size=32)
np.save('p2_acc_history.npy', acc_history)
np.save('p2_loss_history.npy', loss_history)


# In[ ]:

print('\n==============================')
print('Load Model')

model_tf = torch.load("pytorch_vgg16_hw3.pth")

print('\n==============================')
print('Test model')
test_dataloader = DataLoader(caltech256_test, batch_size=32)
correct_cnt = 0
cnt = 0
for data in test_dataloader:
    inputs, labels = data
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    outputs = model_tf(inputs)
    _, preds = torch.max(outputs, 1)
    correct_cnt += torch.sum(preds.data == labels.data)
    
acc = correct_cnt / float(caltech256_test.__len__())
print('Test Set Accuracy: {}'.format(acc*100))

