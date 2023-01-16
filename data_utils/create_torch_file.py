import os
import cv2 as cv
import torch
import torchvision.transforms


def read_image_to_tensor(image_dir):
    img = cv.imread(image_dir, cv.IMREAD_GRAYSCALE)
    to_tensor = torchvision.transforms.ToTensor()
    tsr = to_tensor(img)
    return tsr


def create_4d_torch_file(data_name, data_dir, start, end, memory_threshold=500):
    img_list, tsr_list = [], []
    for i in range(start, end):
        print('Slice:', i, '/', end-1, end='\r')
        img_dir = os.path.join(data_dir, str(i)+'.png')
        img_list.append(read_image_to_tensor(img_dir))
        if (i % memory_threshold) == 0 or i == (end - 1):
            tsr_list.append(torch.stack(img_list))
            img_list = []

    torch.save(torch.cat(tsr_list, dim=0), data_name)
    print('已生成数据集', data_name)
