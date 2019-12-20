import sys
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from model import PixelCNN
from tqdm import tqdm

def main():
    save_path = 'models/model_23.pt'
    no_images = 64
    images_size = 32
    images_channels = 3
    

    #Define and load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PixelCNN().to(device)
    net.load_state_dict(torch.load(save_path))
    net.eval()

    sample = torch.zeros(no_images, images_channels, images_size, images_size).to(device)
    print('-------------------------------------SAMPLING!!!!!---------------------------------')

    for i in tqdm(range(images_size)):
        for j in range(images_size):
            for c in range(images_channels):
                out = net(sample)
                probs = torch.softmax(out[:, :, c, i, j], dim=1)
                # print(probs)
                sampled_levels = torch.multinomial(probs, 1).squeeze().float() / (63.0)
                sample[:,c,i,j] = sampled_levels


    torchvision.utils.save_image(sample, 'sample.png', nrow=12, padding=0)





if __name__=='__main__':
    main()