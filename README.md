# lvsnet-pytorch

基于Pytorch的分割模型框架
更新日期2023/01/17
版本V0.1.1

## 运行环境

Python 3.9
PyTorch 1.13
cuda 11.6
OpenCV 4.6
torchvision 0.14

## 文件结构
./data_utils 数据集处理工具包
./networks 网络结构定义
.main.py 程序入口，运行main即可

## 版本更新

### lvsnet-pytorch V0.1.1

2023/01/17

更新2D Encoder，2D Decoder，以及UNet模型。
创建4D张量数据集。
DICE评估方法
主函数程序入口
