# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, transforms
import os
import scipy.io
from reid.resnet import resnet50

######################################################################
# Options

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')

# 训练调参
parser.add_argument('--conv_lr', default=0.01, type=float, help='conv_lr')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', default=40, type=int, help='stepsize')
parser.add_argument('--adapt', action='store_true', help='use adapt_lr' )
parser.add_argument('--w1', default=1.0, type=float, help='w1')
parser.add_argument('--w2', default=1.0, type=float, help='w2')

parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--epoch', default=60, type=int, help='epoch')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

# 数据集及模式选择
parser.add_argument('--dataset',default='market', type=str, help='the name of dataset ')

parser.add_argument('--stage1', action='store_true', help='use stage1' )
parser.add_argument('--stage2', action='store_true', help='use stage2' )
parser.add_argument('--ABN', action='store_true', help='use ABN+ResNet50' )
parser.add_argument('--att_w', default=1.0, type=float, help='w')

parser.add_argument('--all', action='store_true', help='use all before curri' )
parser.add_argument('--resume', action='store_true', help='resume before' ) # 接着先前结果继续
parser.add_argument('--test_more', action='store_true', help='use global max pooling' )

opt = parser.parse_args()
if opt.dataset == 'market' :
    data_dir = data_dir = '/home/brain-navigation/bishe_cjh/Market/pytorch'

if opt.stage1:
    stage = 'stage1'
    weight = 'none'
elif opt.stage2:
    stage = 'stage2'
    weight = 'w1:{},w2:{}'.format(opt.w1,opt.w2)
elif opt.all:
    stage = '751_all'
    weight = 'none'
else:
    stage = 'all'
    weight = 'none'

if opt.stage2:
    exp_dir = os.path.join('./exp', opt.dataset,
                           ('ABN_att_w:{}'.format(opt.att_w) if opt.ABN else '')+
                           '{}_'.format(stage) + 'bs:{}_'.format(opt.batchsize) + 'lr:{}_'.format(opt.lr) +
                           ('conv_lr:{}'.format(opt.conv_lr) ) +
                           'weight:{}_'.format(weight)
                           )
else:
    exp_dir = os.path.join('./exp', opt.dataset,
                           ('ABN_att_w:{}'.format(opt.att_w) if opt.ABN else '')+
                           '{}_'.format(stage) + 'bs:{}_'.format(opt.batchsize) + 'lr:{}_'.format(opt.lr) +
                           ('conv_lr:{}'.format(opt.conv_lr)) +
                           ('adapt_' if opt.adapt else '[no adapt]_') +
                           ('resume' if opt.resume else '')
                           )

if opt.all:
    num_class=751
else:
    num_class=718

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pth':
                file_based = file.split('.')[0]
                L.append(file_based[4:])
    return L

which_epoch = opt.which_epoch

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data

data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                         shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model

def load_network(network):
    save_path = os.path.join(exp_dir,'net_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Extract feature

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

        ff = torch.FloatTensor(n,2048).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            if opt.ABN:
                _, outputs, attention = model(input_img)
            else:
                outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
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

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
def extract_feat():
    if opt.resume:
        model_structure = resnet50(ABN=opt.ABN, num_classes=num_class)
    else:
        model_structure = resnet50(pretrained=True, ABN=opt.ABN, num_classes=num_class)

    model = load_network(model_structure)
    model.classifier = nn.Sequential()
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
    a = os.path.join(exp_dir,'epoch_%s'%which_epoch)
    scipy.io.savemat('pytorch_result.mat',result)
    scipy.io.savemat(a+'_result.mat',result)
    return query_feature,query_cam,query_label,gallery_feature,gallery_cam,gallery_label

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
def compute(query_feature,query_cam,query_label,gallery_feature,gallery_cam,gallery_label):
    query_feature = query_feature.cpu()
    query_cam = np.array(query_cam)
    query_label = np.array(query_label)
    gallery_feature = gallery_feature.cpu()
    gallery_cam = np.array(gallery_cam)
    gallery_label = np.array(gallery_label)

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    print(len(query_label))
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    context = '[epoch %s] Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(which_epoch,CMC[0],CMC[4],CMC[9],ap/len(query_label))
    print(context)

    file_path = os.path.join(exp_dir,'test.txt')
    with open(file_path, "a") as f:
        f.write(context+'\r\n')
        f.close()

if opt.test_more:
    epoches = file_name(exp_dir)
    for which_epoch in epoches:
        query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label=extract_feat()
        compute(query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label)
else:
    which_epoch = opt.which_epoch
    query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label = extract_feat()
    compute(query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label)

