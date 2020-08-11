# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--nclasses', default=751, type=int, help='num of classes')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###
# load the training config

str_ids = opt.gpu_ids.split(',')
which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



data_dir = test_dir

# 载入数据集
image_datasets = datasets.ImageFolder( os.path.join(data_dir, 'train_all') ,data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                         shuffle=False, num_workers=16)
use_gpu = torch.cuda.is_available()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,2048).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms: # 测试时的数据增强
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path): # 通过数据集图片命名，得到摄像头编码和行人标签
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

path = image_datasets.imgs

cam,label = get_id(path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model = ft_net(opt.nclasses)
# model.classifier.classifier = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()

# 提取特征
with torch.no_grad():
    feature = extract_feature(model,dataloaders)

# 保存特征、标签等信息
result = {'feature':feature.numpy(),'label':label,'cam':cam}
scipy.io.savemat('./tests/test-data/input/feat/with_pretrain_imagenet.mat',result)

