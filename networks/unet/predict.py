import os
import torch
import torch.nn as nn
import cv2 as cv
from .model import UNet


class PredictUNet:
    def __init__(self, in_channels, n_classes, predict_dataset,
                 model_dir, output_dir, output_start_index):
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        self.model = UNet(in_channels, n_classes).to(self.device)
        self.criterion = nn.BCELoss().to(self.device) if n_classes == 1 else nn.CrossEntropyLoss().to(self.device)
        if not os.path.exists(model_dir):
            print('模型加载失败，检查路径', model_dir, '是否存在')
            exit(-1)
        self.model_dir = model_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(output_dir)
        self.output_index = output_start_index
        self.predict_dataset = predict_dataset()

    def predict(self):
        print('device:', self.device)
        step = 0
        test_loss = 0
        self.model.load_state_dict(torch.load(self.model_dir))
        with torch.no_grad():
            self.model.eval()
            for _, (t_x, t_y) in enumerate(self.predict_dataset):
                step += 1

                t_x = t_x.to(self.device)
                t_y = t_y.to(self.device)

                out = self.model(t_x)

                loss = self.criterion(out, t_y)
                test_loss += loss.item()

                self.save_result(out)

        test_loss = test_loss / step
        print('Test Loss:', test_loss)

    def save_result(self, tsr, threshold=0.5):
        mat = tsr.cpu().detach().numpy()
        mat[mat < threshold] = 0.0
        mat[mat >= threshold] = 1.0
        mat = (mat*255).astype('uint8')
        for img in mat:
            cv.imwrite(os.path.join(self.output_dir, str(self.output_index)+'.png'), img[0])
            self.output_index += 1
