import cv2
import numpy as np
import pandas as pd
from torchvision.models.resnet import resnet50
import torch
from data_load_regression import Data
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision import models


class ModelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            models.resnet50()
        )

        self.out_put = nn.Linear(1000, 10)

    def forward(self, x):

        x = self.layers(x)
        x = self.out_put(x)

        return x

def model_load():
    # 使用自定义的网络结构和权重的加载
    net = resnet50(num_classes=10)  # ***
    # 载入预训练权重 model_weight_path：含权重名的模型路径 device：设备
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # 冻结所有参数 使其不更新
    model = net
    model.eval()
    return model
    # 替换全连接层

class ToPILImage(object):
    def __call__(self, tensor):
        # 将张量转换为NumPy数组
        img_np = tensor.numpy()[0].transpose(1, 2, 0)
        print(img_np.shape)

        img_np = img_np[:, :, ::-1] * 255
        # 将NumPy数组转换为PIL图像

        return img_np

# 定义逆转换组合


if __name__ == '__main__':

    inverse_transform = ToPILImage()
    device = 'cuda'
    model_weight_path = 'model/model_regression_best.pth'

    dataframe_valid = pd.read_csv('train.csv').head(20)
    dataset_test = Data(dataframe_valid)
    test_data = DataLoader(dataset_test, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    # net = model_load().to(device)

    # net.eval()

    net = ModelNet().to(device)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    with torch.no_grad():
        for i, (x, y) in enumerate(test_data):

            img = inverse_transform(x)
            img2 = np.array(img).copy()

            x = x.to(device)

            predict = net(x)

            predict_arr = predict.cpu().numpy()[0] * 224
            predict_arr = predict_arr.astype(np.int32)

            color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
            for m in range(5):
                x1, y1 = predict_arr[m], predict_arr[m+5]
                print(x1, y1)

                cv2.circle(img2, (x1, y1), 5, color_list[m], -1)

            cv2.imwrite(f'image2test/{i}.png', img2)







