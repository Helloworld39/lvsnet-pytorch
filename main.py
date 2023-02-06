import os
import torch
import data_utils as d
from networks import *

if not os.path.exists('./models'):
    os.makedirs('./models')

dataset_name = d.DATASET_NAME_3D          # 换数据集是需要修改
data_dir_con = d.dir_manager(dataset_name, d.DATASET_PLACE_AUTODL)
slice_num_list = d.get_index(dataset_name)
data_separate = d.dataset_separate(dataset_name)
index_range = ((slice_num_list[data_separate[0][0]], slice_num_list[data_separate[0][1]]),
               (slice_num_list[data_separate[1][0]], slice_num_list[data_separate[1][1]]),
               (slice_num_list[data_separate[2][0]], slice_num_list[data_separate[2][1]]))
create_torch_file = True
epochs_num = 150
model_name = MODEL_UNET
target_name = TARGET_VESSEL

if create_torch_file:
    d.create_4d_torch_file(data_dir_con['root']+'/train_input.pth', data_dir_con['image'], *index_range[0])
    d.create_4d_torch_file(data_dir_con['root']+'/train_gt.pth', data_dir_con['vessel'], *index_range[0])
    d.create_4d_torch_file(data_dir_con['root']+'/valid_input.pth', data_dir_con['image'], *index_range[1])
    d.create_4d_torch_file(data_dir_con['root']+'/valid_gt.pth', data_dir_con['vessel'], *index_range[1])
    d.create_4d_torch_file(data_dir_con['root']+'/predict_input.pth', data_dir_con['image'], *index_range[2])
    d.create_4d_torch_file(data_dir_con['root']+'/predict_gt.pth', data_dir_con['vessel'], *index_range[2])

train_dataset = d.data_loader(torch.load(os.path.join(data_dir_con['root'], 'train_input.pth')),
                              torch.load(os.path.join(data_dir_con['root'], 'train_gt.pth')))
valid_dataset = d.data_loader(torch.load(os.path.join(data_dir_con['root'], 'valid_input.pth')),
                              torch.load(os.path.join(data_dir_con['root'], 'valid_gt.pth')))

train = TrainUNet(in_channels=1, n_classes=1,
                  train_dataset=train_dataset,
                  valid_dataset=valid_dataset,
                  epochs=epochs_num,
                  checkpoint_dir=os.path.join(data_dir_con['root'], 'checkpoint', model_name),
                  model_dir=os.path.join('./models', model_name+'-'+dataset_name+'-'+target_name+'-'+str(epochs_num)+'.pth'))
train.train()
train.show_loss_arr()

predict_dataset = d.data_loader(torch.load(os.path.join(data_dir_con['root'], 'predict_input.pth')),
                                torch.load(os.path.join(data_dir_con['root'], 'predict_gt.pth')))
predict = PredictUNet(in_channels=1, n_classes=1,
                      predict_dataset=predict_dataset,
                      model_dir=os.path.join('./models', model_name+'-'+dataset_name+'-'+target_name+'-'+str(epochs_num)+'.pth'),
                      output_dir=os.path.join(data_dir_con['root'], 'output', model_name+'-'+target_name+'-'+epochs_num),
                      output_start_index=data_separate[2][0])
predict.predict()
