import sys
import os
import time
import torch
from torch import optim
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from modelmnist import PixelCNN
from torchvision import *
from tqdm import tqdm
import numpy as np
import torchvision


path = 'data'
data_name = 'MNIST'
batch_size = 1

layers = 7
kernel = 7
channels = 256
epochs = 25

save_path = 'models/model_9.pt'
no_images = 1
images_size = 28
images_channels = 1

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



train = datasets.MNIST(root=path, train=True, download=True, transform = transforms.ToTensor())
train = data.DataLoader(train, batch_size=1, shuffle=True, num_workers =0, pin_memory = True)
loglikes = []
counter = 0
for images, labels in tqdm(train, total=4):
    loglike = 0
    sample = torch.zeros(no_images, images_channels, images_size, images_size).to(device)
    # print('-------------------------------------SAMPLING!!!!!---------------------------------')
    for i in (range(images_size)):
        for j in range(images_size):
            out = net(sample)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            loglike += np.log(np.max(probs.cpu().numpy()))
            sample[:,:,i,j] = images[:, :, i, j]
    # print(loglike)
    loglikes.append(loglike)
    counter += 1
    if counter == 5:
        break
print(np.mean(np.array(loglikes)))

train = datasets.MNIST(root=path, train=False, download=True, transform = transforms.ToTensor())
train = data.DataLoader(train, batch_size=1, shuffle=True, num_workers =0, pin_memory = True)
loglikes = []
counter = 0
for images, labels in tqdm(train, total=4):
    loglike = 0
    sample = torch.zeros(no_images, images_channels, images_size, images_size).to(device)
    # print('-------------------------------------SAMPLING!!!!!---------------------------------')
    for i in (range(images_size)):
        for j in range(images_size):
            out = net(sample)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            loglike += np.log(np.max(probs.cpu().numpy()))
            sample[:,:,i,j] = images[:, :, i, j]
    # print(loglike)
    loglikes.append(loglike)
    counter += 1
    if counter == 5:
        break
print(np.mean(np.array(loglikes)))

train = datasets.CIFAR10(root=path, train=True, download=True, transform = transforms.Compose([transforms.Resize((28, 28)),transforms.Grayscale(), transforms.ToTensor()]))
train = data.DataLoader(train, batch_size=1, shuffle=True, num_workers =0, pin_memory = True)
loglikes = []
counter = 0
for images, labels in tqdm(train, total=4):
    loglike = 0
    sample = torch.zeros(no_images, images_channels, images_size, images_size).to(device)
    # print('-------------------------------------SAMPLING!!!!!---------------------------------')
    for i in range(images_size):
        for j in range(images_size):
            out = net(sample)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            loglike += np.log(np.max(probs.cpu().numpy()))
            sample[:,:,i,j] = images[:, :, i, j]
    # print(loglike)
    loglikes.append(loglike)
    counter += 1
    if counter == 5:
        break
print(np.mean(np.array(loglikes)))

train = datasets.CIFAR10(root=path, train=False, download=True, transform = transforms.Compose([transforms.Resize((28, 28)),transforms.Grayscale(), transforms.ToTensor()]))
train = data.DataLoader(train, batch_size=1, shuffle=True, num_workers =0, pin_memory = True)
loglikes = []
counter = 0
for images, labels in tqdm(train, total=4):
    loglike = 0
    sample = torch.zeros(no_images, images_channels, images_size, images_size).to(device)
    # print('-------------------------------------SAMPLING!!!!!---------------------------------')
    for i in range(images_size):
        for j in range(images_size):
            out = net(sample)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            loglike += np.log(np.max(probs.cpu().numpy()))
            sample[:,:,i,j] = images[:, :, i, j]
    # print(loglike)
    loglikes.append(loglike)
    counter += 1
    if counter == 5:
        break
print(np.mean(np.array(loglikes)))
