"""
Tutorial: https://gist.github.com/iyaja/f9fc63ef65b6cc74409dc635c2d80861
"""
# -*- coding: utf-8 -*-

# Import required libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torchvision.datasets as datasets

import os
import random
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import Linear

import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import seaborn as sns

import time

num_workers = 4
# Top level data directory. Here we assume the format of the directory conforms 
# to the ImageFolder structure
data_dir = "/home/larissa/Dev/TCC-Erik/5-fold/dataset/round_1/"


# Data Transforms
# Resize (224x224), Data augmentation, flips and normalization for training
# Just normalization for test
train_transforms = transforms.Compose([
                           transforms.Resize(size=[224,224]),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(size=[224,224]),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

train_dir = os.path.join(data_dir + 'train/')
valid_dir = os.path.join(data_dir + 'val/')
#test_dir = os.path.join(data_dir + 'test/')

# Applying the transformations (test transformation == validation transformation)
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms) 

# Load the data in parallel using multiprocessing workers through DataLoader iterator
train_dl = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=num_workers)

test_dl = torch.utils.data.DataLoader(valid_data, shuffle=True, num_workers=num_workers)

loss_func = F.cross_entropy

# Define function to check model accuracy
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# Guaranteeing results reproducible for COVID-19 problem on one specific platform and PyTorch version 1.4
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

GPUavailable = torch.cuda.is_available()
if GPUavailable:
    print('Treinamento em GPU!')
    device = torch.device("cuda:0")
else:
    print('Treinamento em CPU!')
    device = "cpu"

torch.backends.cudnn.deterministic = True


#model = models.squeezenet1_0(pretrained = True)
#model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))

#model = models.densenet121(pretrained = True)
#model.classifier = Linear(1024,4)

model = models.resnet50(pretrained = True)
model.classifier = Linear(2048,4)

# Move model to GPU for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set model to training mode
model.train()

def get_model_accuracy():
  tot_acc = 0
  avg_acc = 0
  
  # Set model to evaluation mode
  model.eval()

  for xbt, ybt in test_dl:

    pred = model(xbt.to(device))
    tot_acc += accuracy(pred,ybt.to(device))

  avg_acc = tot_acc / len(test_dl)
  
  return avg_acc.item()



def obj_func(lr, bs, mmt, epochs):
      
      # We need to round off bs and epochs because Gaussian processes cannot deal with discrete variables 
      bs = int(bs)
      epochs = int(epochs)
      #epochs = 30
      # Load the data in parallel using multiprocessing workers through DataLoader iterator
      train_dl = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=num_workers)

      #valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=bs, shuffle=True, num_workers=num_workers)

      test_dl = torch.utils.data.DataLoader(valid_data, batch_size=bs, shuffle=True, num_workers=num_workers)
      
      #optim = Adam(model.parameters(), lr = lr)
      optim = optm.SGD(model.parameters(), lr=lr, momentum=mmt)
      
      for epoch in range(epochs):
    
        for xb, yb in train_dl:
        
            # .to(device) moves torch.Tensor objects to the GPU for faster training
        
            preds = model(xb.to(device))
            loss = loss_func(preds, yb.to(device))
            acc = accuracy(preds,yb.to(device))
        
            loss.backward()
            optim.step()
            optim.zero_grad()
        
        print("Loss: " + str(loss.item()) + "\t \t Accuracy: " + str(100 * acc.item()))

      acc = get_model_accuracy()
      
      return acc


from bayes_opt import BayesianOptimization

# Bounded region of parameter space
#pbounds = {'lr': (1e-4, 1e-2), 'bs': (16, 32), 'epochs': (1,25)}
pbounds = {'lr': (1e-4, 1e-2), 'bs': (16, 32), 'mmt': (9e-1, 1), 'epochs': (1,30)}

optimizer = BayesianOptimization(
    f=obj_func,
    pbounds=pbounds,
    verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(init_points=2, n_iter=3,)

print(optimizer.max)