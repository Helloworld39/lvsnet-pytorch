"""
最基础的组件，网络结构可实现向前传递，
使用nn.Sequential组合
"""

import torch.nn as nn


class SimpleConvUnit2D(nn.Module):
    """
    单次2D卷积组件
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class SimpleConvUnit3D(nn.Module):
    """
    单次3D卷积组件
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm3d(out_channels),
                                  nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class DoubleConvUnit2D(nn.Module):
    """
    两次2D卷积组件
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(SimpleConvUnit2D(in_channels, mid_channels),
                                  SimpleConvUnit2D(mid_channels, out_channels))

    def forward(self, x):
        return self.conv(x)


class DoubleConvUnit3D(nn.Module):
    """
    两次3D卷积组件
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(SimpleConvUnit3D(in_channels, mid_channels),
                                  SimpleConvUnit3D(mid_channels, out_channels))

    def forward(self, x):
        return self.conv(x)
