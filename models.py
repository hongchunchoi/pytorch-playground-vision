from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import dataload
from dataload import *

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

class vgg16_fcNet(nn.Module) :
    def __init__(self):
        super(vgg16_fcNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class resNet18_fcNet(nn.Module) :
    def __init__(self):
        super(resNet18_fcNet, self).__init__()
        self.fc = nn.Linear(in_features=512, out_features= 10, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class resNet18_deep_fcNet(nn.Module) :
    def __init__(self):
        super(resNet18_deep_fcNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x