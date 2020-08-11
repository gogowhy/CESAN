# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import os
import json
import time
from shutil import copyfile
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from reid.utils.random_erasing import RandomErasing
from reid.utils.dataloader import DataLoader,Com_DataLoader,Tri_DataLoader
from reid.resnet import resnet50

version =  torch.__version__

writer = SummaryWriter()

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')


parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='', type=str, help='output model name')

# 数据集及模式选择
parser.add_argument('--dataset',default='market', type=str, help='the name of dataset ')

parser.add_argument('--stage1', action='store_true', help='use stage1' )
parser.add_argument('--stage2', action='store_true', help='use stage2' )
parser.add_argument('--ABN', action='store_true', help='use ABN+ResNet50' )
parser.add_argument('--att_w', default=1.0, type=float, help='w')

parser.add_argument('--all', action='store_true', help='use all before curri' )
parser.add_argument('--resume', action='store_true', help='resume before' ) # 接着先前结果继续
parser.add_argument('--resume_path', type=str, default='', metavar='PATH')

# 训练调参
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--conv_lr', default=0.01, type=float, help='conv_lr')
parser.add_argument('--stepsize', default=40, type=int, help='stepsize')
parser.add_argument('--adapt', action='store_true', help='use adapt_lr' )
parser.add_argument('--w1', default=1.0, type=float, help='w1')
parser.add_argument('--w2', default=1.0, type=float, help='w2')

parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--epoch', default=60, type=int, help='epoch')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

opt = parser.parse_args()

# python train.py --gpu_ids 1 --batchsize 64 --lr 0.1 --conv_lr 0.01 --stage1
# python train.py --gpu_ids 1 --batchsize 32 --lr 0.001 --conv_lr 0.001 --stage2 --w1 0.3 --w2 0.7

if opt.dataset == 'market':
    data_dir = '/home/brain-navigation/bishe_cjh/Market/pytorch'

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

if opt.name == '':
    if opt.stage2:
        exp_dir = os.path.join('./exp', opt.dataset,
                               ('ABN_att_w:{}'.format(opt.att_w) if opt.ABN else '')+
                               '{}_'.format(stage) + 'bs:{}_'.format(opt.batchsize) + 'lr:{}_'.format(opt.lr) +
                               ('conv_lr:{}'.format(opt.conv_lr) ) +
                               ('adapt_' if opt.adapt else '[no adapt]_')+
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

else:
    exp_dir = os.path.join('./model',opt.name)


if opt.resume_path == '':
    if opt.resume:
        resume_path = os.path.join('./exp', opt.dataset,
                                  # ('ABN_att_w:{}'.format(opt.att_w) if opt.ABN else '') +
                                   'stage2_' + 'bs:32_' + 'lr:0.001_' +
                                   'conv_lr:0.001' +
                                   ('adapt_' if opt.adapt else '[no adapt]_') +
                                   'weight:w1:0.3,w2:0.7_' +
                                   'net_{}.pth'.format(opt.which_epoch)
                                   )

    elif opt.stage2:
        resume_path = os.path.join('./exp', opt.dataset,
                               ('ABN_att_w:{}'.format(opt.att_w) if opt.ABN else '')+
                               'stage1_' + 'bs:64_' + 'lr:0.1_' +
                                   'conv_lr:0.01'+
                                   ('adapt_' if opt.adapt else '[no adapt]_'),
                               'net_{}.pth'.format(opt.which_epoch)
                               )

    else:
        resume_path = ''

else:
    resume_path = opt.resume_path

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
transform_train_list = [
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]


data_transforms = {
    'train': transforms.Compose( transform_train_list ),
}


##curriculum
image_datasets = {}


if opt.stage1 or opt.stage2:
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, '11'),
                                              data_transforms['train'])
    if opt.stage2 :
        image_datasets['train_2'] = datasets.ImageFolder(os.path.join(data_dir, '22'),
                                                  data_transforms['train'])

elif opt.all:
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_all'),
                                              data_transforms['train'])
else:
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_curri'),
                                              data_transforms['train'])

#====================
id_sample=False

dataloaders={}
if opt.stage2:
    dataloaders['train'] = Com_DataLoader(image_datasets['train'],image_datasets['train_2'], batch_size=int(opt.batchsize),
                                          shuffle=True, num_workers=8,drop_last=True,id_sample=id_sample)                       ###opt.batchsize is the smallest of subset

elif opt.stage1:

    dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=opt.batchsize,
                                  shuffle=True, num_workers=8)
else:
    dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=opt.batchsize,
                                      shuffle=True, num_workers=8)

dataset_sizes = {}
dataset_sizes['train']=len(image_datasets['train'])
#============================
if opt.stage2:
    if opt.stage2:
        tmp_size = len(image_datasets['train_2'])

    dataset_sizes['train']=dataset_sizes['train']+tmp_size
#============================

class_names = image_datasets['train'].classes
use_gpu = torch.cuda.is_available()
since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)

######################################################################
y_loss_1 = {}
y_loss_2 = {}
y_loss_3 = {}
y_loss = {} # loss history
y_loss_1['train'] = []
y_loss_2['train'] = []
y_loss_3['train'] = []
y_loss['train'] = []

y_err_1 = {}
y_err_2 = {}
y_err_3 = {}
y_err = {}
y_err['train'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    bf_loss = 0.0
    bf_prec = 0.0
    count_loss = 0
    count_prec = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

############################### training
        train_val = ['train']
        for phase in train_val:
            if phase == 'train':
                if not opt.adapt:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            running_loss_3 = 0.0

            running_corrects = 0.0

            i = -1
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                i = i +1
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch  ######Maybe is not used
                    continue
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if opt.ABN:
                    att_outputs, outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)

                if opt.stage2:
                    length = opt.batchsize
                    if opt.ABN:
                        att_outputs_1 = att_outputs[:length]
                        outputs_1 = outputs[:length]
                        labels_1 = labels[:length]
                        att_outputs_2 = att_outputs[length:]
                        outputs_2 = outputs[length:]
                        labels_2 = labels[length:]

                        att_loss_1 = criterion(att_outputs_1, labels_1)
                        per_loss_1 = criterion(outputs_1, labels_1)
                        att_loss_2 = criterion(att_outputs_2, labels_2)
                        per_loss_2 = criterion(outputs_2, labels_2)

                        loss_1 = att_loss_1+per_loss_1
                        loss_2 = att_loss_2+per_loss_2
                    else:
                        outputs_1 = outputs[:length]
                        labels_1 = labels[:length]
                        outputs_2 = outputs[length:]
                        labels_2 = labels[length:]
                        loss_1 = criterion(outputs_1, labels_1)
                        loss_2 = criterion(outputs_2, labels_2)
                    if phase == 'train':
                        loss = opt.w1*loss_1 + opt.w2*loss_2
                        writer.add_scalar('data/loss_1',loss_1,epoch)
                        writer.add_scalar('data/loss_2',loss_2,epoch)
                        writer.add_scalar('data/loss',loss,epoch)
                        writer.add_scalars('data/loss_group',{'loss_1':loss_1,'loss_2':loss_2,'loss':loss},epoch)


                else:
                    if opt.ABN:
                        att_loss = criterion(att_outputs, labels)
                        per_loss = criterion(outputs, labels)
                        loss = opt.att_w*att_loss + per_loss
                    else:
                        loss = criterion(outputs, labels)
                    writer.add_scalar('data/loss',loss,epoch)
                    writer.add_scalar('data/loss_group',loss,epoch)

                _, preds = torch.max(outputs.data, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if opt.stage2:
                    running_loss_1 += loss_1.item() * now_batch_size
                    running_loss_2 += loss_2.item() * now_batch_size

                running_loss += loss.item() * now_batch_size

                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                now_loss = float('%.4f' % epoch_loss)
                if now_loss == bf_loss:
                    count_loss = count_loss + 1
                else:
                    count_loss = 0
                bf_loss = now_loss

            if phase == 'train':
                if opt.adapt:
                    scheduler.step(epoch_loss)

            if opt.stage2:
                epoch_loss_1 = running_loss_1 / dataset_sizes[phase]
                epoch_loss_2 = running_loss_2 / dataset_sizes[phase]


            writer.add_scalar('data/epoch_loss', epoch_loss, epoch)

            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == 'train':
                now_prec = float('%.4f' % epoch_acc)
                if now_prec == bf_prec:
                    count_prec = count_prec + 1
                else:
                    count_prec = 0
                bf_prec = now_prec
            if opt.stage2:
                print('{} Loss: {:.4f} Loss_1: {:.4f} Loss_2: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_loss_1, epoch_loss_2, epoch_acc))
            else:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            if opt.stage2:
                y_loss_1[phase].append(epoch_loss_1)
                y_loss_2[phase].append(epoch_loss_2)

            y_err[phase].append(1.0 - epoch_acc)

            last_model_wts = model.state_dict()
            if count_loss>=4 or count_prec>=4:
                save_network(model, epoch)
            if epoch >= 40:
                if (epoch + 1) % 3 == 0:
                    save_network(model, epoch)
            draw_curve(epoch, stage2=opt.stage2)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch,stage2=False):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')

    if stage2:
        ax0.plot(x_epoch, y_loss_1['train'], 'yo-', label='train1')
        ax0.plot(x_epoch, y_loss_2['train'], 'go-', label='train2')


    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(exp_dir,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(exp_dir,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

num_class = len(class_names)

if opt.resume:
    model = resnet50(pretrained=False, ABN=opt.ABN, num_classes=num_class)
else:
    model = resnet50(pretrained=True, ABN=opt.ABN, num_classes=num_class)
######################################################################
# Loaded model
#---------------------------
import os.path as osp
import sys
from reid.utils.logging import Logger
sys.stdout = Logger(osp.join(exp_dir,'log.txt'))

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath.split('/'[-1])))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
if resume_path != '':
    checkpoint = load_checkpoint(resume_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)




################################################
######################################################################
# adjust learning
# ---------------------------
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in opt.schedule:
        state['lr'] *= opt.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

######################################################################


if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()


if opt.ABN:
    base_params = list(map(id, model.conv1.parameters()))
    base_params += (list(map(id, model.bn1.parameters() ))
                       +list(map(id, model.layer1.parameters() ))
                       +list(map(id, model.layer2.parameters() ))
                     +list(map(id, model.layer3.parameters() ))
                      )
    ignored_params = filter(lambda p: id(p) not in base_params, model.parameters())
    param_groups = [
        {'params': model.conv1.parameters(), 'lr': opt.conv_lr},
        {'params': model.bn1.parameters(), 'lr': opt.conv_lr},
        {'params': model.layer1.parameters(), 'lr': opt.conv_lr},
        {'params': model.layer2.parameters(), 'lr': opt.conv_lr},
        {'params': model.layer3.parameters(), 'lr': opt.conv_lr},
        {'params': ignored_params, 'lr': opt.lr}]
    optimizer_ft = torch.optim.SGD(param_groups,
                            weight_decay=5e-4, momentum=0.9,
                            nesterov=True)

else:
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': opt.conv_lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

if opt.adapt:
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
else:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=opt.stepsize, gamma=0.1)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)
copyfile('./train.py', exp_dir+'/train.py')
if opt.ABN:
    copyfile('./reid/resnet.py', exp_dir + '/resnet.py')
else:
    copyfile('./reid/model.py', exp_dir+'/model.py')
# save opts
with open('%s/opts.json'%exp_dir,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=opt.epoch)
