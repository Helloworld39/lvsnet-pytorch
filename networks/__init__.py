"""
更新日期：2023/01/11
本软件包为深度网络结构定义体
"""
# 模型载入
from .unet import *
# 评估方法
from .evaluate_function import dice_score

# pooling type config
POOLING_MAX = 0
POOLING_CONV = 1
POOLING_MEAN = 2

# upsample type config
UPSAMPLE_BILINEAR = 0
UPSAMPLE_TRANCONV = 1
UPSAMPLE_NEAREST = 2
