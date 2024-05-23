import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from data_load_regression import Data
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from torchvision import models
# 使用自定义的网络结构和权重的加载
from torch.optim.lr_scheduler import CosineAnnealingLR



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



def valid(model, data_valid, device, iteration):
    model = model.eval()
    loss_list = []
    with torch.no_grad():
        for index, (img, labels) in enumerate(data_valid):
            img = img.to(device)
            labels = labels.to(device)

            predict = model(img)
            loss = iteration(predict, labels)

            loss_list.append(loss.item())

    loss_mean = np.mean(loss_list)

    return loss_mean



def train():
    dataframe = pd.read_csv('train.csv').head(500)
    dataframe_valid = pd.read_csv('test.csv').head(20)

    dataset = Data(dataframe)
    dataset_test = Data(dataframe_valid)

    train_data = DataLoader(dataset, shuffle=True, batch_size=4, num_workers=4, pin_memory=True)
    test_data = DataLoader(dataset_test, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    # 使用官方给定的api进行网络结构和权重的加载
    # net = resnet50(pretrained=True)
    # 冻结所有参数 使其不更新
    # for param in net.parameters():
    #     param.requires_grad = False
    # 替换全连接层
    # in_channel = net.fc.in_features  # 获得原模型全连接层的输入特征大小
    # net.fc = nn.Linear(in_channel, 10)  # num_classes代表分类器的类别


    device = "cuda"

    model = ModelNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    iteration = torch.nn.MSELoss(reduction='mean')

    train_loss = []
    valid_loss = []
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0.0001)  # T_max是总epoch数
    mini_value = 100
    for i in range(300):
        model = model.train()
        train_loss_epoch = []

        # if i == 0:
        #     for param in model.parameters():
        #         param.requires_grad = True

        for index, (img, labels) in enumerate(train_data):

            # print(torch.sum(labels))
            img = img.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predict = model(img)
            loss = iteration(predict, labels)

            loss.backward()
            optimizer.step()

            print(loss.item())
            train_loss_epoch.append(loss.item())
        scheduler.step()


        valid_mean = valid(model, test_data, device, iteration)
        train_loss_mean = np.mean(train_loss_epoch)

        # if i > 3:
        train_loss.append(train_loss_mean)
        valid_loss.append(valid_mean)

        if valid_mean < mini_value:
            mini_value = valid_mean

            torch.save(model.state_dict(), 'model/model_regression_best.pth')

        if i % 5 == 0:
            plt.plot(np.array(train_loss)[3:], color='blue', label='Training Loss')
            plt.plot(np.array(valid_loss)[3:], color='red', label='Validation Loss')
            plt.legend()  # 显示图例
            plt.savefig('loss_curve/loss_curve.png')
            plt.close()



        train_loss_arr = np.array(train_loss)
        valid_loss_arr = np.array(valid_loss)

        np.savetxt('loss_curve/train_loss.csv', train_loss_arr, fmt='%.5f', delimiter=',')
        np.savetxt('loss_curve/valid_loss.csv', valid_loss_arr, fmt='%.5f', delimiter=',')


        # 模型保留


if __name__ == '__main__':
    train()







