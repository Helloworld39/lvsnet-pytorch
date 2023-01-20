import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import shutil
from .model import UNet
from .. import evaluate_function as ev


class TrainUNet:
    def __init__(self, in_channels, n_classes,
                 train_dataset, valid_dataset=None,
                 lr=1e-2, epochs=150,
                 checkpoint_dir='./checkpoint/unet', model_dir='./models/model.pth'):
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        self.model = UNet(in_channels, n_classes).to(self.device)
        self.criterion = nn.BCELoss().to(self.device) if n_classes == 1 else nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 50, 0.1)
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_dir = model_dir
        self.is_first_train = False if os.path.exists(self.model_dir) else True
        self.train_dataset = train_dataset(is_shuffled=True)
        self.valid_dataset = None if not valid_dataset else valid_dataset()

        self.loss_arr = {'train_loss': [], 'valid_loss': []}

    def train(self):
        print('device:', self.device)
        if not self.is_first_train:
            self.model.load_state_dict(torch.load(self.model_dir))
            print('已载入训练中的模型')

        min_loss = 1
        best_epoch = 1

        # 开始训练
        for epoch in range(self.epochs):
            self.model.train()  # 开启训练模式
            print('Epoch: %d/%d, lr: %.6f' % (epoch+1, self.epochs,
                                              self.optimizer.state_dict()['param_groups'][0]['lr']))
            epoch_loss = 0
            step = 0
            start_time = time.time()
            save_checkpoint = False

            # batch训练
            for _, (t_x, t_y) in enumerate(self.train_dataset):
                step += 1
                # 把数据按batch放入显存
                t_x = t_x.to(self.device)
                t_y = t_y.to(self.device)

                out = self.model(t_x)
                loss = self.criterion(out, t_y)
                epoch_loss = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            epoch_loss = epoch_loss / step
            self.loss_arr['train_loss'].append(epoch_loss)

            if self.valid_dataset:
                val_loss, val_dice = self.valid()
                print('time:', int(time.time()-start_time), ',train loss:', epoch_loss,
                      ',valid loss:', val_loss, 'valid dice', val_dice)
            else:
                print('time:', int(time.time()-start_time), ',train loss:', epoch_loss)

            # 如果不是最低loss值，不进行checkpoint
            if min_loss > epoch_loss:
                min_loss = epoch_loss
                save_checkpoint = True
            if save_checkpoint:
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, str(best_epoch)+'.pth'))

        print('Best Epoch:', best_epoch)
        shutil.copy(os.path.join(self.checkpoint_dir, str(best_epoch)+'.pth'), self.model_dir)

    def valid(self):
        step = 0
        valid_loss = 0
        dice_score = 0

        with torch.no_grad():
            self.model.eval()
            for _, (v_x, v_y) in enumerate(self.valid_dataset):
                step += 1

                v_x = v_x.to(self.device)
                v_y = v_y.to(self.device)
                out = self.model(v_x)
                loss = self.criterion(out, v_y)
                valid_loss += loss.item()
                dice_score += ev.dice_score(out, v_y)

            valid_loss = valid_loss / step
            dice_score = dice_score / step
            self.loss_arr['valid_loss'].append(valid_loss)

        torch.cuda.empty_cache()
        return valid_loss, dice_score

    def show_loss_arr(self):
        print('train loss:\n', self.loss_arr['train_loss'], '\nvalid loss:\n', self.loss_arr['valid_loss'])
