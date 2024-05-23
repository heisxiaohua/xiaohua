import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms
from PIL import Image
import cv2

import torch


class Data(Dataset):
    def __init__(self, dataframe):
        self.img_size = 224
        self.data = dataframe.values

        self.image_list, self.mean, self.std = self.read_image(self.data)
        # self.image_list = self.read_image(self.data)

        print(self.mean, self.std)
        self.transfer = transforms.Compose([
                            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色扭曲
                            transforms.ToTensor(),  # 将图像转换为Tensor
                            # transforms.Normalize(mean=0.4022927036590186, std=0.24144132860629525)  # 标准化
                        ])

    @staticmethod
    def read_image(data):
        img_list = []

        # 记录图像的数量
        num_images = 0

        mean_total = 0
        std_total = 0
        for img_data in data:
            img_path = img_data[0]  # 取得图像路径
            img = Image.open(os.path.join('data', img_path))
            # img = Image.open(os.path.join('data', img_path)).convert('L')

            img_array = np.array(img)/255
            img_list.append(img)

            mean_ = np.mean(img_array)
            std_ = np.std(img_array)
            mean_total += mean_
            std_total += std_

        mean_total /= data.shape[0]
        std_total /= data.shape[0]
        return img_list, mean_total, std_total

    @staticmethod
    def get_heatmap(x, y, h, w, sigma=1):
        y1 = np.linspace(1, w, w)
        x1 = np.linspace(1, h, h)

        [Y, X] = np.meshgrid(x1, y1)

        X = X - x
        Y = Y - y

        D2 = X * X + Y * Y
        E2 = 2 * sigma * sigma
        EX = D2 / E2
        heatmap = np.exp(-EX)
        return heatmap
    def point2label(self, site_point):

        label_list = []
        for i in range(0, len(site_point)//2):
            y, x = site_point[i], site_point[i+5]
            heat_map = self.get_heatmap(x, y, self.img_size, self.img_size, sigma=6)
            label_list.append(heat_map)

        return np.array(label_list)



    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):

        img = self.image_list[item]
        w, h = img.size
        img = img.resize((self.img_size, self.img_size))
        ratio_rows = h/self.img_size
        ratio_cols = w/self.img_size
        img = self.transfer(img).detach()

        site_data = np.array(self.data[item, 1:].copy().tolist())

        site_data[:5] = site_data[:5]/ratio_cols
        site_data[5:] = site_data[5:]/ratio_rows

        site_data /= self.img_size

        site = torch.from_numpy(site_data).float().detach()
        # print(site)

        return img, site



# class ToPILImage(object):
#     def __call__(self, tensor):
#         # 将张量转换为NumPy数组
#         img_np = tensor.numpy().transpose(1, 2, 0)
#         print(img_np.shape)
#
#         img_np = img_np[:, :, ::-1]
#         # 将NumPy数组转换为PIL图像
#
#         return img_np
#
# if __name__ == '__main__':
#     inverse_transform = ToPILImage()
#
#     data = pd.read_csv('test.csv').head(20)
#
#     data_set = Data(data)
#
#     img1, lables = data_set.__getitem__(12)
#
#     print(img1)
#     img = inverse_transform(img1) * 255
#     img2 = np.array(img).copy().astype(np.uint8)
#
#
#
#
#     predict_arr = lables.numpy()
#     predict_arr = predict_arr.astype(np.int32)
#
#     for m in range(5):
#         x1, y1 = predict_arr[m], predict_arr[m + 5]
#         print(x1, y1)
#
#         cv2.circle(img2, (x1, y1), 5, (0, 0, 255), -1)
#
#     cv2.imshow('img', img2)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     # print(lables)
# # #
