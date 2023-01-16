import torch.nn as nn
from ..base_parts import Encoder2D, Decoder2D


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.encoder = Encoder2D(in_channels)
        self.decoder = Decoder2D(n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x = self.encoder(x)
        return self.decoder(x, x1, x2, x3, x4)


class UNetWithoutConnection(nn.Module):
    """
    无跳跃连接的版本
    """
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.encoder = Encoder2D(in_channels)
        self.decoder = Decoder2D(n_classes, jump_num=0)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
