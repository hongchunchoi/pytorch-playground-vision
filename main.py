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
from feature_dataload import *
from models import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


global pretrained_model
global pretrained_cnn

def set_Train_Model(model) :
    if model == "ResNet18" :
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential()

    elif model == "vgg16" :
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential()

    else :
        print("There is no model")

    return model

def set_Infernce_Model(modelname, model) :
    inference_model = model
    if modelname == "ResNet18" :
        inference_model = models.resnet18(pretrained=True)
        inference_model.fc = model

    elif modelname == "vgg16" :
        inference_model = models.vgg16(pretrained=True)
        inference_model.classifier = model

    else :
        print("There is no model")

    return inference_model


def train(epoch, fcn_model, trainloader) :
    print('\nepoch : %d' % epoch)
    fcn_model.train()
    train_loss = 0
    correct = 0
    total = 0
    fcn_model = fcn_model.to(device)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs = torch.squeeze(inputs)
        outputs = fcn_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if epoch % 5 == 0 :
             print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        del inputs, targets

def test(epoch, model, testloader):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pre_trained_net = model
    pre_trained_net = pre_trained_net.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #mid_input = pre_trained_net(inputs)
            outputs = pre_trained_net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(targets)
            print(predicted)
            print(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            del inputs, targets
    del pre_trained_net
'''
for a in range(10) :
    del globals()['train_X_{}'.format(a)], globals()['train_Y_{}'.format(a)], globals()['trainset_{}'.format(a)], globals()['trainloader_{}'.format(a)]
'''

def generate_feature_vector(model) :
    train_X, train_Y, test_X, test_Y = dataload._load_data()
    data_list = dataload._split_by_label(10, train_X, train_Y)

    for i in range(10):
        globals()['train_X_{}'.format(i)] = data_list[i][0]
        globals()['train_Y_{}'.format(i)] = data_list[i][1]
        globals()['train_Y_{}'.format(i)] = torch.as_tensor(globals()['train_Y_{}'.format(i)], dtype=torch.long)
        globals()['trainset_{}'.format(i)] = CustomDataset(globals()['train_X_{}'.format(i)],
                                                           globals()['train_Y_{}'.format(i)], transform=transform_train)
        globals()['trainloader_{}'.format(i)] = torch.utils.data.DataLoader(globals()['trainset_{}'.format(i)],
                                                                            batch_size=100, shuffle=False)

    mean_std_label = get_meta_dataset(model)
    trainloader = generate_final_trainloader(5000, 512, mean_std_label)

    for a in range(10):
        del globals()['train_X_{}'.format(a)], globals()['train_Y_{}'.format(a)], globals()['trainset_{}'.format(a)], \
            globals()['trainloader_{}'.format(a)]



def main(model) :
    '''model setting'''
    global input_model
    global convert_model
    if model == "ResNet18":
        convert_model = models.resnet18(pretrained=True)
        convert_model.fc = nn.Sequential()
        #input_model = resNet18_fcNet()
        input_model = resNet18_deep_fcNet()

    elif model == "vgg16":
        convert_model = models.vgg16(pretrained=True)
        convert_model.classifier = nn.Sequential()
        input_model = vgg16_fcNet()

    '''loss function 정의'''
    global criterion
    criterion = nn.CrossEntropyLoss()
    global optimizer
    optimizer = optim.SGD(input_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    '''feature vector 생성 및 저장'''
    generate_feature_vector(convert_model)

    train_X, train_Y, test_X, test_Y = dataload._load_data()

    del train_X, train_Y

    train_X = np.load("C:/Users/Hongjun/Desktop/dataset/vgg_feature_numpy/numpy_feature_dataset.npy")
    train_Y = np.load("C:/Users/Hongjun/Desktop/dataset/vgg_feature_numpy/numpy_label_dataset.npy")

    train_Y = torch.tensor(train_Y)
    test_Y = torch.as_tensor(test_Y, dtype=torch.long)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainset = CustomDataset(train_X, train_Y, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

    test_Y = torch.as_tensor(test_Y, dtype=torch.long)
    testset = CustomDataset(test_X, test_Y, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False)

    del train_X, train_Y

    '''training'''
    for epoch in range(start_epoch, start_epoch + 20):
        train(epoch, input_model, trainloader)

    trained_model = set_Infernce_Model(model, input_model)


    torch.save(trained_model.state_dict(), "C:/Users/Hongjun/Desktop/dataset/vgg_feature_numpy/trained_{}.pt".format(str(model)))
    del trained_model, input_model

    '''Test'''
    #inference_model = set_Train_Model(model, )
    global inference_model

    if model == "ResNet18":
        inference_model = models.resnet18(pretrained=True)
        #inference_model.fc = resNet18_fcNet()
        inference_model.fc = resNet18_deep_fcNet()

    elif model == "vgg16":
        inference_model = models.vgg16(pretrained=True)
        inference_model.classifier = vgg16_fcNet()

    inference_model.load_state_dict = torch.load("C:/Users/Hongjun/Desktop/dataset/vgg_feature_numpy/trained_{}.pt".format(str(model)))

    inference_model = inference_model.to(device)

    test(0, inference_model, testloader=testloader)


main("ResNet18")