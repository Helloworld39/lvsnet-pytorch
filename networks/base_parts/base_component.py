"""
基础组件设计
包括下采样、上采样、输入、输出等模块
"""

import torch
from .base_unit import *


class Input2D(nn.Module):
    """
    2D张量输入层
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvUnit2D(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)


class Output2D(nn.Module):
    """
    2D图像输出层
    """
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv = SimpleConvUnit2D(in_channels, n_classes)
        if n_classes == 1:
            self.out = nn.Sequential(nn.Conv2d(1, 1, 1),
                                     nn.Sigmoid())
        elif n_classes >= 2:
            self.out = nn.Sequential(nn.Conv2d(n_classes, n_classes, 1),
                                     nn.Softmax())
        else:
            print('n_classes:', n_classes, '错误参数')
            exit(0)

    def forward(self, x):
        x = self.conv(x)
        return self.out(x)


class DownSampling2D(nn.Module):
    """
    下采样2D卷积层
    """
    def __init__(self, in_channels, out_channels, pooling_type=0):
        super().__init__()
        if pooling_type == 0:
            self.pool = nn.MaxPool2d(2, 2)
        elif pooling_type == 1:
            self.pool = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False)
        elif pooling_type == 2:
            self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.MaxPool2d(2, 2)

        self.conv = DoubleConvUnit2D(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpSampling2D(nn.Module):
    """
    上采样2D卷积层
    """
    def __init__(self, in_channels, out_channels, mid_channels, upsample_type=0):
        super().__init__()
        if upsample_type == 0:
            self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                    SimpleConvUnit2D(in_channels, out_channels))
        elif upsample_type == 1:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        elif upsample_type == 2:
            self.up = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                    SimpleConvUnit2D(in_channels, out_channels))
        else:
            self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                    SimpleConvUnit2D(in_channels, out_channels))

        self.conv = SimpleConvUnit2D(mid_channels, out_channels)

    def forward(self, x, *args):
        """
        包含跳跃连接的上采样方法
        :param x: 需要上采样的特征图
        :param args: 跳跃连接的特征图
        :return:
        """
        x = self.up(x)
        x = torch.cat([x, *args], dim=1)
        return self.conv(x)


class DownSampling3D(nn.Module):
    """
    下采样3D卷积层
    """
    def __init__(self, in_channels, out_channels, pooling_type=0):
        super().__init__()
        if pooling_type == 0:
            self.pool = nn.MaxPool3d(2, 2)
        elif pooling_type == 1:
            self.pool = nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2, bias=False)
        elif pooling_type == 2:
            self.pool = nn.AvgPool3d(2, 2)
        else:
            self.pool = nn.MaxPool3d(2, 2)

        self.conv = DoubleConvUnit3D(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpSampling3D(nn.Module):
    """
    长采样3D卷积层
    """
    def __init__(self, in_channels, out_channels, mid_channels, upsample_type=0):
        super().__init__()
        if upsample_type == 0:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    SimpleConvUnit3D(in_channels, out_channels))
        elif upsample_type == 1:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        elif upsample_type == 2:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                    SimpleConvUnit3D(in_channels, out_channels))
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    SimpleConvUnit3D(in_channels, out_channels))

        self.conv = SimpleConvUnit3D(mid_channels, out_channels)

    def forward(self, x, *args):
        x = self.up(x)
        x = torch.cat([x, *args], dim=1)
        return self.conv(x)
