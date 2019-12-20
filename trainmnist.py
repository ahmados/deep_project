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
# torch.backends.cudnn.enabled = False

def sampling(net, epoch, channels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path = 'models/model_mnist_' + str(epoch) + '.pt'
    no_images = 144
    images_size = 28
    images_channels = 1
    net.eval()

    sample = torch.zeros(no_images, images_channels, images_size, images_size).to(device)
    print('-------------------------------------SAMPLING!!!!!---------------------------------')
    for i in tqdm(range(images_size)):
        for j in range(images_size):
            out = net(sample)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0

    torchvision.utils.save_image(sample, 'sample.png', nrow=12, padding=0)


def main():
    path = 'data'
    data_name = 'CIFAR'
    batch_size = 64

    layers = 7
    kernel = 7
    channels = 256
    epochs = 25
    save_path = 'models'

    train= datasets.MNIST(root=path, train=True, download=True, transform = transforms.ToTensor())
    test= datasets.MNIST(root=path, train=False, download=True, transform = transforms.ToTensor())
    
    train = data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers =0, pin_memory = True)
    test = data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers =0, pin_memory = True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = PixelCNN(num_layers=layers, kernel_size=kernel, num_channels=channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    loss_overall = []
    for i in range(epochs):
        if (i%3) == 1:
            sampling(net, i-1, channels)
        net.train(True)
        step = 0
        loss_= 0
        for images, labels in tqdm(train, desc='Epoch {}/{}'.format(i + 1, epochs)):
            images = images.to(device)
            optimizer.zero_grad()
            target = (images[:, 0, :, :] * 255).long()
            output = net(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_+=loss
            step+=1

        print('Epoch:'+str(i)+'       , '+ 'Average loss: ', loss_/step)
        with open("hst.txt", "a") as myfile:
            myfile.write(str(loss_/step) + '\n')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if(i==epochs-1):
            torch.save(net.state_dict(), save_path+'/model_mnist_'+'Last'+'.pt')
        else:
            torch.save(net.state_dict(), save_path+'/model_mnist_'+str(i)+'.pt')
        print('model saved')



if __name__=="__main__":
    main()