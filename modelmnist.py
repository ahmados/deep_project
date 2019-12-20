from masked import maskedConv
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(Block, self).__init__()
        self.Conv = maskedConv('B',num_channels,num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.BatchNorm = nn.BatchNorm2d(num_channels)
        self.ReLU= nn.ReLU(True)

    def forward(self, x):
        x = self.Conv(x)
        x = self.BatchNorm(x)
        x = self.ReLU(x)
        return x

class PixelCNN(nn.Module):
    def __init__(self, num_layers=10, kernel_size=7, num_channels=128):
        super(PixelCNN, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_channels = num_channels

        self.Conv2d_first = maskedConv('A',1,1*num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.BatchNorm2d_first = nn.BatchNorm2d(1*num_channels)
        self.ReLU_first= nn.ReLU(True)

        self.hidden_conv = nn.Sequential(
            *[Block(1*num_channels, kernel_size=3) for _ in range(num_layers)]
        )

        self.out = nn.Conv2d(1*num_channels, 1*num_channels, 1)

    def forward(self, x):
        x = self.Conv2d_first(x)
        x = self.BatchNorm2d_first(x)
        x = self.ReLU_first(x)

        x = self.hidden_conv(x)
        return self.out(x)