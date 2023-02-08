"""
encoder-decoder 组件设置
"""

from .base_component import *


class Encoder2D(nn.Module):
    """
    2D图像编码器，采用卷积+下采样的方式，每一层的特征图数量翻倍
    """
    def __init__(self, in_channels, save_feature=True):
        super().__init__()
        self.save_feature = save_feature
        self.input = Input2D(in_channels, 32)
        self.down1 = DownSampling2D(32, 64)
        self.down2 = DownSampling2D(64, 128)
        self.down3 = DownSampling2D(128, 256)
        self.down4 = DownSampling2D(256, 512)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        if self.save_feature:
            return x1, x2, x3, x4, x
        else:
            return x


class Decoder2D(nn.Module):
    """
    2D图像解码器，采用卷积+上采样的方式，可自定义是否自动连接
    """
    def __init__(self, n_classes, jump_num=1):
        super().__init__()
        self.jump_num = jump_num
        self.up4 = UpSampling2D(512, 256, 256*(jump_num+1))
        self.up3 = UpSampling2D(256, 128, 128*(jump_num+1))
        self.up2 = UpSampling2D(128, 64, 64*(jump_num+1))
        self.up1 = UpSampling2D(64, 32, 32*(jump_num+1))
        self.out = Output2D(32, n_classes)

    def forward(self, x, *args):
        """

        :param x: 网络输入张量
        :param args: 参加跳跃连接的张量，需要和jump_num参数共同使用，参数规定例如（layer1_jump1, layer1_jump2, layer2_jump1, layer2_jump2,...）
        :return:
        """
        if self.jump_num * 4 != len(args):
            print('跳跃连接数：', self.jump_num, '张量数：', len(args), '参数错误')
            exit(0)
        x = self.up4(x, *args[3*self.jump_num:])
        x = self.up3(x, *args[2*self.jump_num: 3*self.jump_num])
        x = self.up2(x, *args[self.jump_num: 2*self.jump_num])
        x = self.up1(x, *args[: self.jump_num])

        return self.out(x)


class Encoder3D(nn.Module):
    """
    3D编码器
    """
    def __init__(self, in_channels, save_feature=True):
        super().__init__()
        self.save_feature = save_feature
        self.input = Input3D(in_channels, 32)
        self.down1 = DownSampling3D(32, 64)
        self.down2 = DownSampling3D(64, 128)
        self.down3 = DownSampling3D(128, 256)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)

        if self.save_feature:
            return x1, x2, x3, x
        else:
            return x


class Decoder(nn.Module):
    """
    3D解码器
    """
    def __init__(self, n_classes, jump_num=1):
        super().__init__()
        self.jump_num = jump_num
        self.up3 = UpSampling3D(256, 128, 128*(jump_num+1))
        self.up2 = UpSampling3D(128, 64, 64*(jump_num+1))
        self.up1 = UpSampling3D(64, 32, 32*(jump_num+1))
        self.out = Output3D(32, n_classes)

    def forward(self, x, *args):
        if self.jump_num * 3 != len(args):
            print('跳跃连接数：', self.jump_num, '张量数：', len(args), '参数错误')
            exit(0)
        x = self.up3(x, *args[2*self.jump_num:])
        x = self.up2(x, *args[self.jump_num: 2*self.jump_num])
        x = self.up1(x, *args[:self.jump_num])
        return self.out(x)
