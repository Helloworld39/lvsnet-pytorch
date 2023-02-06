import os
import numpy as np
import SimpleITK as sitk


def read_nii(file_path):
    """
    读取nii或nii_gz格式的CT原文件，并转换为ndarray
    """
    if not (file_path[-4:] == '.nii' or file_path[-7:] == '.nii.gz'):
        print('文件后缀不是.nii或.nii.gz，无法通过read_nii()读取')
        exit(-1)

    mat = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    print('输入文件尺寸：', mat.shape)
    return mat


def read_dicom(dcm_dir, name_r='%04d.dcm'):
    """
    载入dcm文件夹，读取所有的CT源文件，并转换为ndarray
    """
    mat = []
    file_num = len(os.listdir(dcm_dir))
    for i in range(file_num):
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dcm_dir, name_r % i)))
        mat.append(img)
    mat = np.array(mat)
    print('输入文件尺寸：', mat.shape)
    return mat
