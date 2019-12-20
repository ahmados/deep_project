import torch

class maskedConv(torch.nn.Conv2d):

    def __init__(self, mask_type, *args, **kwargs):
        super(maskedConv, self).__init__(*args, **kwargs)
        self.mask_type = mask_type

        count, channels, height, width = self.weight.size()
        mask = torch.ones((count, channels, height, width))
        if mask_type =='A':
            mask = mask.view(count, channels, -1)
            mask[:, :, width*(height // 2) + width//2:] = 0
        else:
            mask = mask.view(count, channels, -1)
            mask[:, :, width*(height // 2) + width//2 + 1:] = 0
        mask = mask.view(count, channels, height, width)
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data*=self.mask
        return super(maskedConv, self).forward(x)