# Complement to "transfer_learning_step_by_step.py" file
# Execute this file after complementary file

from __future__ import print_function, division
from pickletools import optimize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

# --------------------------------------------------- #

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './data/hymenoptera_data'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------- #

### Debug "2. train_model" function and "3.1. Finetuning the convent" approach ###

# Define #train_model" function parameters
model = models.resnet18(pretrained= True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
num_epochs = 1 # set to one for debugging

since = time.time()
since

# Inspect pre-trained model
best_model_wts = copy.deepcopy(model.state_dict())

type(best_model_wts)
print(best_model_wts.keys())
print(best_model_wts['conv1.weight'][0][0]) # model weights at first convolutional neural network

best_acc = 0.0

# Print epochs
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

# Set the model to training mode
model.train()

running_loss = 0.0
running_corrects = 0


# Each epoch has a training and validation phase
for phase in ['train', 'val']:
    if phase == 'train':
        model.train()  # Set model to training mode
        print(phase)

    else:
        model.eval()   # Set model to evaluate mode
        print(phase)

for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)

inputs
labels

print(optimizer)
optimizer.zero_grad() # set gradients to cero, so they can start learning throug iterations
print(optimizer)


# track history if only in train
with torch.set_grad_enabled(phase == 'train'):
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, labels)

    # backward + optimize only if in training phase
    if phase == 'train':
        loss.backward()
        optimizer.step()

# statistics
running_loss += loss.item() * inputs.size(0)
running_loss

running_corrects += torch.sum(preds == labels.data)
running_corrects

if phase == 'train':
    scheduler.step()

epoch_loss = running_loss / dataset_sizes[phase]
epoch_loss

epoch_acc = running_corrects.double() / dataset_sizes[phase]
epoch_acc


print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


if phase == 'val' and epoch_acc > best_acc:
    best_acc = epoch_acc
    best_model_wts = copy.deepcopy(model.state_dict())

# --------------------------------------------------- #


### Debugg "3.2 ConvNet as fixed feature extractor" ###

model_conv = torchvision.models.resnet18(pretrained=True) 

model_conv
# Last layer is: " (fc): Linear(in_features=512, out_features=1000, bias=True) "
model_conv.parameters()

for param in model_conv.parameters():
    param.requires_grad = False
    print(param)


num_ftrs = model_conv.fc.in_features
num_ftrs

model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv.fc # " (fc): Linear(in_features=512, out_features=2, bias=True) "

model_conv
# Now las layer changed to: " (fc): Linear(in_features=512, out_features=2, bias=True) "

model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()