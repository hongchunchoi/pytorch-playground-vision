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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''Transform(augumentation) '''
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_X, train_Y, test_X, test_Y = dataload._load_data()
data_list = dataload._split_by_label(10, train_X, train_Y)

for i in range(10):
    globals()['train_X_{}'.format(i)] = data_list[i][0]
    globals()['train_Y_{}'.format(i)] = data_list[i][1]
    globals()['train_Y_{}'.format(i)] = torch.as_tensor(globals()['train_Y_{}'.format(i)], dtype=torch.long)
    globals()['trainset_{}'.format(i)] = CustomDataset(globals()['train_X_{}'.format(i)], globals()['train_Y_{}'.format(i)], transform=transform_train)
    globals()['trainloader_{}'.format(i)] = torch.utils.data.DataLoader(globals()['trainset_{}'.format(i)],batch_size=100, shuffle=False)

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def convert_to_feature_data(trainloader, feature_dimension, model) :

    feature_data = []
    target_data = []
    model = model.to(device)
    '''cuda memory 초기화 '''
    torch.cuda.empty_cache()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        data = model(inputs)
        feature_data.append(data.cpu().data)
        target_data.append(targets.cpu().data)
        del inputs, targets

    number_per_class = feature_data.__len__(), feature_data[0].__len__()
    feature = torch.Tensor(5000, feature_dimension, 1)
    target = torch.LongTensor(5000, 1)
    torch.cat(feature_data, out=feature)
    torch.cat(target_data, out=target)
    del model
    return feature, target


def get_mean_std(data) :
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    label = data[0][1]
    parameters = [mean, std, label]

    return parameters

def get_meta_dataset(model) :
    list = []
    for i in range(10) :
        globals()['feature_data_{}'.format(i)],globals()['feature_label_{}'.format(i)] = convert_to_feature_data(globals()['trainloader_{}'.format(i)], 512, model)
        globals()['feature_mean_{}'.format(i)],globals()['feature_std_{}'.format(i)], globals()['labels_{}'.format(i)] = get_mean_std(data = globals()['feature_data_{}'.format(i)])
        mean = globals()['feature_mean_{}'.format(i)]
        std = globals()['feature_std_{}'.format(i)]
        label = globals()['labels_{}'.format(i)]
        m_s_l = [mean, std, label]
        list.append(m_s_l)

    '''delete variable for memory'''
    for j in range(10) :
        del globals()['feature_data_{}'.format(j)],globals()['feature_label_{}'.format(j)], globals()['feature_mean_{}'.format(j)],globals()['feature_std_{}'.format(j)], globals()['labels_{}'.format(j)]

    return list


def generate_train_data(number_data, feature_dimension, parameters, label_number) :
    temp = []
    for i in range(feature_dimension) :
        element = torch.empty(number_data).normal_(mean=parameters[label_number][0][i], std=parameters[label_number][1][i])
        temp.append(element)

    c = torch.stack(temp)
    c.transpose(1,0)
    fcn_input = c.reshape(number_data,feature_dimension,1)

    return fcn_input

def generate_final_trainloader(number_data_for_class, feature_dimension, meta_dataset) :
    for k in range(10):
        globals()["feature_trainset_{}".format(k)] = generate_train_data(number_data_for_class, feature_dimension = feature_dimension, parameters=meta_dataset, label_number=k)


    final_feature_dataset = torch.cat([feature_trainset_0, feature_trainset_1, feature_trainset_2, feature_trainset_3,
                                       feature_trainset_4, feature_trainset_5, feature_trainset_6, feature_trainset_7,
                                       feature_trainset_8, feature_trainset_9], dim=0)

    final_label_dataset = torch.cat([train_Y_0, train_Y_1, train_Y_2, train_Y_3,
                                     train_Y_4, train_Y_5, train_Y_6, train_Y_7, train_Y_8, train_Y_9], dim=0)

    numpy_feature_dataset = final_feature_dataset.numpy()
    numpy_label_datset = final_label_dataset.numpy()

    #save numpy file
    np.save("C:/Users/Hongjun/Desktop/dataset/vgg_feature_numpy/numpy_feature_dataset", numpy_feature_dataset)
    np.save("C:/Users/Hongjun/Desktop/dataset/vgg_feature_numpy/numpy_label_datset", numpy_label_datset)

    final_train_dataset = CustomDataset(final_feature_dataset, final_label_dataset)
    final_trainloader = torch.utils.data.DataLoader(final_train_dataset, batch_size=16, shuffle=True)

    '''del for memory'''
    for a in range(10):
        del globals()['feature_trainset_{}'.format(a)]
    del final_train_dataset

    return final_trainloader
