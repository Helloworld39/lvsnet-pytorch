import os
import torch
import data_utils as d
from networks import *

if not os.path.exists('./models'):
    os.makedirs('./models')

place = 'autodl'
dataset_name = d.DATASET_NAME_MSDC
data_dir_con = d.dir_manager(dataset_name, d.DATASET_PLACE_AUTODL)
slice_num_list = d.get_index(dataset_name)
create_torch_file = True

if create_torch_file:
    d.create_4d_torch_file('train_input.pth', data_dir_con['image'], slice_num_list[0], slice_num_list[101])
    d.create_4d_torch_file('train_gt.pth', data_dir_con['label'], slice_num_list[0], slice_num_list[101])
    d.create_4d_torch_file('valid_input.pth', data_dir_con['image'], slice_num_list[101], slice_num_list[103])
    d.create_4d_torch_file('valid_gt.pth', data_dir_con['label'], slice_num_list[101], slice_num_list[103])
    d.create_4d_torch_file('predict_input.pth', data_dir_con['image'], slice_num_list[298], slice_num_list[303])
    d.create_4d_torch_file('predict_gt.pth', data_dir_con['label'], slice_num_list[298], slice_num_list[303])

train_dataset = d.data_loader(torch.load(os.path.join(data_dir_con['root'], 'train_input.pth')),
                              torch.load(os.path.join(data_dir_con['root'], 'train_gt.pth')))
valid_dataset = d.data_loader(torch.load(os.path.join(data_dir_con['root'], 'valid_input.pth')),
                              torch.load(os.path.join(data_dir_con['root'], 'valid_gt.pth')))

train = TrainUNet(in_channels=1,
                  n_classes=1,
                  train_dataset=train_dataset,
                  valid_dataset=valid_dataset,
                  epochs=150,
                  checkpoint_dir=os.path.join(data_dir_con['root'], 'checkpoint', 'unet'),
                  model_dir=os.path.join('./models', 'unet-msdc-150.pth'))
train.train()
train.show_loss_arr()

predict_dataset = d.data_loader(torch.load(os.path.join(data_dir_con['root'], 'predict_input.pth')),
                                torch.load(os.path.join(data_dir_con['root'], 'predict_gt.pth')))
predict = PredictUNet(in_channels=1, n_classes=1,
                      predict_dataset=predict_dataset,
                      model_dir=os.path.join('./models', 'unet-msdc-150.pth'),
                      output_dir=os.path.join(data_dir_con['root'], 'output', 'unet-150'),
                      output_start_index=slice_num_list[298])
predict.predict()
