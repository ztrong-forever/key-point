# coding: utf-8
# author: hxy
# 2021-12-21
"""
heart keypoint train codes
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import Heart_Point
import src.segmentation_models_pytorch as smp
from src.segmentation_models_pytorch.encoders import get_preprocessing_fn
# from src.dataset import get_training_augmentation, get_validation_augmentation, get_preprocessing
import datetime
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import *


def data_preprocess(data_dir, batch_size):
    train_dataset = Heart_Point(dir=data_dir, 
                                set="train",
                                classes=classes)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=False)
    
    val_dataset = Heart_Point(dir=data_dir,
                              set="val",
                              classes=classes)

    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False)

    num_samples=len(train_dataset)

    return train_loader, val_loader,num_samples


def calculate_mask(heatmaps_target):
    """
    :param heatmaps_target: Variable (N,15,96,96)
    :return: Variable (N,15,96,96)
    """
    N,C,_,_ = heatmaps_targets.size()
    N_idx = []
    C_idx = []
    for n in range(N):
        for c in range(C):
            max_v = heatmaps_targets[n,c,:,:].max().data[0]
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
    mask = Variable(torch.zeros(heatmaps_targets.size()))
    mask[N_idx,C_idx,:,:] = 1.
    mask = mask.float().cuda()
    return mask,[N_idx,C_idx]


def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    
    return all_peak_points


def get_mse(pred_points,gts,indices_valid=None):
    """
    :param pred_points: numpy (N,15,2)
    :param gts: numpy (N,15,2)
    :return:
    """
    #pred_points = pred_points[indices_valid[0],indices_valid[1],:]
    #gts = gts[indices_valid[0],indices_valid[1],:]
    pred_points = Variable(torch.from_numpy(pred_points).float(), requires_grad=False)
    gts = Variable(torch.from_numpy(gts).float(), requires_grad=False)
    criterion = nn.MSELoss()
    loss = criterion(pred_points,gts)
    return loss
  

if __name__ == '__main__':
    LR = 0.001
    MAX_EPOCHS = 120
    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'
    classes=["top","bottom"]

    preprocessing_fn = get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    model_dir="/home/projectCodes/heart_point/logs/crop"
    model_save_dir=os.path.join(model_dir,ENCODER,str(datetime.datetime.now()))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model = smp.Unet(encoder_name=ENCODER,
                            encoder_weights=ENCODER_WEIGHTS,
                            in_channels=1,
                            classes=2,
                            activation=ACTIVATION,
                            encoder_depth=5)                 

    loss_func = nn.MSELoss()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    metrics = [smp.utils.metrics.IoU(threshold=0.5)]

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LR)])
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[30,80],gamma = 0.9)
    
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=[4, 7])

    model.to(device)
    model.train()
    train_loader, val_loader,num_samples = data_preprocess(data_dir="/home/projectCodes/heart_point",
                                                            batch_size=16)

    for epoch in range(0, MAX_EPOCHS):
        scheduler.step()
        print('\nEpoch: {}'.format(epoch))
        print('\nLearing rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        for i, (inputs, heatmaps_targets, gts) in enumerate(train_loader):
            inputs = Variable(inputs).to(device)
            heatmaps_targets = Variable(heatmaps_targets).to(device)
            # mask,indices_valid = calculate_mask(heatmaps_targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            # outputs = outputs * mask
            # heatmaps_targets = heatmaps_targets * mask
            loss = loss_func(outputs, heatmaps_targets)
            
            loss.backward()
            optimizer.step()

            # 统计最大值与最小值
            v_max = torch.max(outputs)
            v_min = torch.min(outputs)

            # 评估
            all_peak_points = get_peak_points(outputs.cpu().data.numpy())
           
            loss_coor = get_mse(all_peak_points, gts.numpy())

            print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} loss_coor : {:15} max : {:10} min : {}'.format(
                epoch, i * 16,
                num_samples, loss.item(),loss_coor.item(),v_max.item(),v_min.item()))

        for i,(inputs, heatmaps_targets, gts) in enumerate(val_loader):
            inputs = Variable(inputs).to(device)
            heatmaps_targets = Variable(heatmaps_targets).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            val_loss = loss_func(outputs, heatmaps_targets)
        print('[ Epoch {:005d}] val_loss : {:15} '.format(epoch, loss.item()))

        if (epoch+1) % 10 == 0 or epoch == MAX_EPOCHS - 1:
            torch.save(model, os.path.join(model_save_dir, 'epoch_{}_model.pth'.format(epoch)))