#coding=utf-8
'''
数据读取
'''
import os
import numpy as np
from numpy.lib.function_base import append
from torch.utils.data import DataLoader,Dataset
import cv2
import albumentations as albu
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

class Heart_Point(Dataset):
    def __init__(self,dir,set,classes,transform=None,augmentation=None,preprocessing=None):
        self.dir=dir
        self.samples = list()
        self.gts=list()
        self.transform=transform
        self.set=set+".txt"
        self.set_path=os.path.join(self.dir,self.set)
        self.classes=classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.load()

    def __len__(self):
        return len(self.samples)

    def load(self):
        with open(self.set_path, 'r') as f:
            datas = f.readlines()
            for info in datas:
                info_list = info.strip('\n').split(' ')
                # 基础信息
                img_path = info_list[0]
                top = info_list[1]
                # 坐标变换
                image = cv2.imread(img_path)
                H=image.shape[0]
                W=image.shape[1]
                top_x=int(float(info_list[2]))
                top_y=int(float(info_list[3]))
                top_x,top_y=self.resize_pos(top_x, top_y, (W,H), (160,160))
                top_loc = [top_x,top_y]
                bottom_x=int(float(info_list[5]))
                bottom_y=int(float(info_list[6]))
                bottom_x,bottom_y=self.resize_pos(bottom_x, bottom_y, (W,H), (160,160))
                bottom_loc = [bottom_x, bottom_y]
                image = cv2.medianBlur(image, 11)
                img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                img_gray=img_gray/255.
                img_gray=cv2.resize(img_gray, (160,160))
                # img=cv2.resize(image,(160,160))
                self.samples.append(img_gray)
                label = np.array([top_loc, bottom_loc])
                self.gts.append(label)
                
            self.gts=np.array(self.gts)
            print(self.gts.shape)

        f.close()
        return np.array(self.samples), np.array(self.gts)
    
    def __getitem__(self, item):
        H, W = 160, 160
        x = self.samples[item]
        gt = self.gts[item]
        heatmaps = self._putGaussianMaps(gt,H,W,stride=1,sigma=3)
        
        x = x.reshape((1,160,160)).astype(np.float32)
        heatmaps = heatmaps.astype(np.float32)
        
        return x, heatmaps, gt

    def resize_pos(self, x1, y1, src_size, tar_size):
        w1=src_size[0]
        h1=src_size[1]
        w2=tar_size[0]
        h2=tar_size[1]
        y2=(h2/h1)*y1
        x2=(w2/w1)*x1
        return x2, y2

    def _putGaussianMap(self, center, visible_flag, crop_size_y, crop_size_x, stride, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        if visible_flag == False:
            return np.zeros((grid_y,grid_x))
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(int(grid_y))]
        x_range = [i for i in range(int(grid_x))]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap
        
    def _putGaussianMaps(self, keypoints, crop_size_y, crop_size_x, stride, sigma):
        """

        :param keypoints: (15,2)
        :param crop_size_y: int
        :param crop_size_x: int
        :param stride: int
        :param sigma: float
        :return:
        """
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        
        for k in range(point_num):
            flag = ~np.isnan(all_keypoints[k,0])
            heatmap = self._putGaussianMap(all_keypoints[k], flag, crop_size_y, crop_size_x, stride, sigma)
            heatmap = heatmap[np.newaxis,...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img, axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
        return heatmaps_this_img
    
    # def img_augmentation(image):
    #     train_transform = [
    #     albu.HorizontalFlip(p=0.5),
    #     albu.VerticalFlip(p=0.5),
    #     albu.Rotate(p=0.5)
    # ]
    #     return albu.Compose(train_transform)

        

if __name__ == '__main__':
    dataset = Heart_Point(dir="/home/projectCodes/heart_point/",set="train",classes=["top","bottom"])
    dataLoader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
    for i, (x, y ,gt) in enumerate(dataLoader):
        print(x.size())
        print(y.size())
        print(gt.size())
    print("success")

