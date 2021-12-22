#coding=utf-8
"""
关键点
inference code
"""
import os
import cv2
import copy
import torch
import numpy as np
from time import time
from tqdm import tqdm
from src.segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from train import get_peak_points,get_mse
from src.dataset import Heart_Point

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]
_std = np.array(std).reshape((1,1,3))
_mean = np.array(mean).reshape((1,1,3))
classes=["top","bottom"]
batch_size=16
output_path="/home/projectCodes/heart_point/val_point"
txt_dir="/home/projectCodes/heart_point/test.txt"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
size=(320,320)
sigma=5
if not os.path.exists(output_path):
    os.mkdir(output_path)
try:
    print('----loading model----')
    model_file = '/home/projectCodes/heart_point/logs/resnet34/2021-12-15-best/avg mse:21.39_320_5.pth'
    model = torch.load(model_file, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    print('--load model sucessed--')
except Exception as e:
    print("--load model fail: {}--".format(e))

def resize_pos(x1,y1,src_size,tar_size):
    w1=src_size[0]
    h1=src_size[1]
    w2=tar_size[0]
    h2=tar_size[1]
    y2=(h2/h1)*y1
    x2=(w2/w1)*x1
    return x2,y2

def _putGaussianMaps(keypoints,crop_size_y, crop_size_x, stride, sigma):
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
        heatmap = _putGaussianMap(all_keypoints[k],flag,crop_size_y,crop_size_x,stride,sigma=sigma)
        heatmap = heatmap[np.newaxis,...]
        heatmaps_this_img.append(heatmap)
    heatmaps_this_img = np.concatenate(heatmaps_this_img,axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
    return heatmaps_this_img

def _putGaussianMap(center, visible_flag, crop_size_y, crop_size_x, stride, sigma):
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


def get_info(info):
    info_list = info.strip('\n').split(' ')
    # 基础信息
    img_path = info_list[0]
    print(img_path)
    top = info_list[1]
    #坐标变换 图像变换
    image = cv2.imread(img_path)
    H=image.shape[0]
    W=image.shape[1]
    top_x=int(float(info_list[2]))
    top_y=int(float(info_list[3]))
    top_x,top_y=resize_pos(top_x,top_y,(W,H),size)
    top_loc = [top_x,top_y]
    bottom_x=int(float(info_list[5]))
    bottom_y=int(float(info_list[6]))
    bottom_x,bottom_y=resize_pos(bottom_x,bottom_y,(W,H),size)
    bottom_loc = [bottom_x, bottom_y]
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_gray=cv2.resize(image_gray,size)
    img_gray=img_gray/255.
    gts = np.array([top_loc, bottom_loc])
    heatmaps=_putGaussianMaps(gts,H,W,stride=1,sigma=sigma)
    return image,heatmaps,gts,img_gray,img_path


def dist(prs,gts):
    dd=[]
    prs = Variable(torch.from_numpy(prs).float(),requires_grad=False).squeeze()
    gts = Variable(torch.from_numpy(gts).float(),requires_grad=False)
    
    for i in range(len(gts)):
        if i==0:
            dist_top=np.linalg.norm(prs[i]-gts[i])
        if i==1:
            dist_bottom=np.linalg.norm(prs[i]-gts[i])
    return dist_top,dist_bottom
        

def predict(txt_dir, out_dir):
    total_times = list()
    mse=[]
    with open(txt_dir,"r") as f:
        data=f.readlines()
        for info in data:
            or_image,or_heatmaps,re_gts,re_img,img_path=get_info(info)
            x_tensor = torch.from_numpy(re_img).to(device).float()
            x_tensor = torch.unsqueeze(x_tensor,0)
            x_tensor = torch.unsqueeze(x_tensor,0)
            start = time()
            pr_heatmap=model.predict(x_tensor)
            end = time()
            time_cost = end - start
            total_times.append(time_cost)
            gt_image=cv2.resize(or_image,size)
            pr_points=get_peak_points(pr_heatmap.cpu().data.numpy())
            loss = get_mse(pr_points, re_gts)
            dist_top,dist_bottom=dist(pr_points, re_gts)
            for pr_point in pr_points:
                for point in pr_point:
                    cv2.circle(gt_image, (int(point[0]),int(point[1])), 1, (0,0,255), 4)
            for re_gt in re_gts:
                cv2.circle(gt_image, (int(re_gt[0]),int(re_gt[1])), 1, (255,0,0), 4)
            filename=img_path[-17:]
            output=os.path.join(output_path,filename)
            cv2.imwrite(output,gt_image)
            mse.append(loss)
            mean_mse=np.mean(mse)
        print("avg mse:%.2f"%mean_mse)
        print("avg topdist:%.2f"%np.mean(dist_top))
        print("avg bottomdist:%.2f"%np.mean(dist_bottom))
        print('avg inference time:{:.2f} ms'.format(np.mean(total_times)*1000))
    f.close()

if __name__ == '__main__':
    # predict_ex()
    predict(txt_dir=txt_dir,out_dir=output_path)