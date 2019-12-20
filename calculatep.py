import sys
import os
import time
import torch
from torch import optim
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from model import PixelCNN
from torchvision import *
from tqdm import tqdm
import numpy as np
import torchvision


path = 'data'
data_name = 'CIFAR'
batch_size = 64

layers = 10
kernel = 7
channels = 128
epochs = 25

save_path = 'models/model_23.pt'
no_images = 64
images_size = 32
images_channels = 3

normalize = transforms.Lambda(lambda image: np.array(image) / 255.0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def quantisize(image, levels):
    return np.digitize(image, np.arange(levels) / levels) - 1
discretize = transforms.Compose([
    transforms.Lambda(lambda image: quantisize(image, (channels - 1))),
    transforms.ToTensor()
])
cifar_transform = transforms.Compose([normalize, discretize])




net = PixelCNN(num_layers=layers, kernel_size=kernel, num_channels=channels).to(device)
net.load_state_dict(torch.load(save_path))
net.eval()



train = datasets.CIFAR10(root=path, train=True, download=True, transform = cifar_transform)
train = data.DataLoader(train, batch_size=1, shuffle=True, num_workers =0, pin_memory = True)
loglikes = []
counter = 0
for images, labels in tqdm(train, total=9):
    loglike = 0
    sample = torch.zeros(1, images_channels, images_size, images_size).to(device)
    #Generating images pixel by pixel
    for i in range(images_size):
        for j in range(images_size):
            for c in range(images_channels):
                out = net(sample)
                probs = torch.softmax(out[:, :, c, i, j], dim=1)[0].detach().cpu().numpy()
                # print(probs)
                loglike += np.log(np.max(probs))
                # print(np.max(probs))
                sample[:,c,i,j] = images[:, c, i, j]
    print(loglike)
    loglikes.append(loglike)
    counter += 1
    if counter == 10:
        break
print(np.mean(np.array(loglikes)))
train = datasets.CIFAR10(root=path, train=False, download=True, transform = cifar_transform)
train = data.DataLoader(train, batch_size=1, shuffle=True, num_workers =0, pin_memory = True)
loglikes = []
counter = 0
for images, labels in tqdm(train, total=9):
    loglike = 0
    sample = torch.zeros(1, images_channels, images_size, images_size).to(device)
    #Generating images pixel by pixel
    for i in range(images_size):
        for j in range(images_size):
            for c in range(images_channels):
                out = net(sample)
                probs = torch.softmax(out[:, :, c, i, j], dim=1)[0].detach().cpu().numpy()
                # print(probs)
                loglike += np.log(np.max(probs))
                # print(np.max(probs))
                sample[:,c,i,j] = images[:, c, i, j]
    print(loglike)
    loglikes.append(loglike)
    counter += 1
    if counter == 10:
        break
print(np.mean(np.array(loglikes)))

data_name = 'SVHN'
train = datasets.SVHN(root=path, split='train', download=True, transform = cifar_transform)
train = data.DataLoader(train, batch_size=1, shuffle=True, num_workers =0, pin_memory = True)
loglikesSVHN = []
counter = 0
for images, labels in tqdm(train, total=9):
    loglike = 0
    sample = torch.zeros(1, images_channels, images_size, images_size).to(device)
    #Generating images pixel by pixel
    for i in range(images_size):
        for j in range(images_size):
            for c in range(images_channels):
                out = net(sample)
                probs = torch.softmax(out[:, :, c, i, j], dim=1)[0].detach().cpu().numpy()
                # print(probs)
                loglike += np.log(np.max(probs))
                # print(np.max(probs))
                sample[:,c,i,j] = images[:, c, i, j]
    print(loglike)
    loglikesSVHN.append(loglike)
    counter += 1
    if counter == 10:
        break
print(np.mean(np.array(loglikesSVHN)))

data_name = 'SVHN'
train = datasets.SVHN(root=path, split='test', download=True, transform = cifar_transform)
train = data.DataLoader(train, batch_size=1, shuffle=True, num_workers =0, pin_memory = True)
loglikesSVHN = []
counter = 0
for images, labels in tqdm(train, total=9):
    loglike = 0
    sample = torch.zeros(1, images_channels, images_size, images_size).to(device)
    #Generating images pixel by pixel
    for i in range(images_size):
        for j in range(images_size):
            for c in range(images_channels):
                out = net(sample)
                probs = torch.softmax(out[:, :, c, i, j], dim=1)[0].detach().cpu().numpy()
                # print(probs)
                loglike += np.log(np.max(probs))
                # print(np.max(probs))
                sample[:,c,i,j] = images[:, c, i, j]
    print(loglike)
    loglikesSVHN.append(loglike)
    counter += 1
    if counter == 10:
        break
print(np.mean(np.array(loglikesSVHN)))