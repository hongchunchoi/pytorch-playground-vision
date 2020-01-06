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

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''vgg16 model CNN pretrained 파트로 해체 '''
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = nn.Sequential()

pre_trained_net = vgg16
pre_trained_net.to(device)



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''컴퓨터에 파일 다운받고 이름별로 load해서 각각 subset구성하는 방식으로 다시 짜보는것 부터 12/28'''
def data_loader :




def convert_data(trainloader) :

    feature_data = []
    target_data = []
    #tensor_data = torch.tensor()
    '''cuda memory 초기화 '''
    torch.cuda.empty_cache()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        data = vgg16(inputs)
        feature_data.append(data.cpu().data)
        target_data.append(targets.cpu().data)
    print(feature_data.__len__(), feature_data[0].__len__())
    b = torch.Tensor(50000, 25088, 1)
    c = torch.LongTensor(50000, 1)
    torch.cat(feature_data, out=b)
    torch.cat(target_data, out=c)

    feature_dataset = TensorDataset(b, c)
    return feature_dataset


def get_metadata(data) :
    list0, list1, list2, list3, list4, list5, list6, list7, list8, list9 = []
    for i in range(len(data)) :
    trainloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)


    for i in range(10) :
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        label = data[0][1]
        parameters = [mean, std, label]

    return parameters

list = []

def genearate_train_data(number_data, parameters) :

    for i in range(parameters[0].shape[0]) :
        element = torch.empty(number_data).normal_(mean=parameters[0][i], std=parameters[1][i])
        list.append(element.cpu().data)

    fcn_input = torch.Tensor(50000, 25088, 1)
    torch.cat(list, out=fcn_input)
    print(fcn_input.shape)
    return fcn_input


feature_vectors = convert_data(trainloader)

#mean_std = get_metadata(feature_vectors)