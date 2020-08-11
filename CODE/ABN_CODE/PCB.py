#coding: utf-8
from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn

from reid import datasets_pcb as datasets
from reid import models
from reid.trainers_partloss import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint, save_checkpoint

from reid.utils.dataloader_pcb import DataLoader,Com_DataLoader,Tri_DataLoader
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reid.utils.logging import Logger
from reid.utils.random_erasing import RandomErasing



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_data(name, data_dir, height, width, batch_size, workers):
    root = data_dir
    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    if args.all:
        num_classes = dataset.num_trainall_ids
    else:
        num_classes = dataset.num_train_ids

    train_list = [
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ]
    if args.random_erasing:
        train_list.append(RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]))

    train_transformer = T.Compose(train_list)

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])


    if args.val:
        val_loader = DataLoader(
            Preprocessor(dataset.val, root=osp.join(dataset.images_dir,dataset.val_path),
                        transform=train_transformer),
            batch_size=int(batch_size/2), num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)

    if args.all:
        train_loader = DataLoader(
            Preprocessor(dataset.trainall, root=osp.join(dataset.images_dir, dataset.trainall_path),
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=osp.join(dataset.images_dir, dataset.train_path),
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query, root=osp.join(dataset.images_dir,dataset.query_path),
                     transform=test_transformer),
        batch_size=16, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir,dataset.gallery_path),
                     transform=test_transformer),
        batch_size=16, num_workers=workers,
        shuffle=False, pin_memory=True)

    if args.val:
        return dataset, num_classes, train_loader, val_loader, query_loader, gallery_loader
    else:
        return dataset, num_classes, train_loader, query_loader, gallery_loader


def  main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if args.evaluate:
        sys.stdout = Logger(osp.join(exp_dir, 'test{}.txt'.format(args.which_epoch)))
    else:
        sys.stdout = Logger(osp.join(exp_dir, 'log.txt'))


    # Create data loaders
    if args.val:
        dataset, num_classes, train_loader, val_loader, query_loader, gallery_loader = \
            get_data(args.dataset, data_dir, args.height,
                     args.width, args.batchsize, args.workers,
                     )
    else:
        dataset, num_classes, train_loader, query_loader, gallery_loader = \
            get_data(args.dataset,  data_dir, args.height,
                     args.width, args.batchsize, args.workers,
                     )

    dataloaders = {}
    dataloaders['train'] = train_loader
    if args.val:
        dataloaders['val'] = val_loader


    # Create model
    model = models.create('resnet50_PCB', num_features=args.features,
                      dropout=args.dropout, num_classes=num_classes, ABN=args.ABN, justglobal=args.justglobal, s2=args.s2)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if resume_path != '' or args.evaluate:
        if args.evaluate:
            checkpoint = load_checkpoint(exp_dir+'/epoch_{}.pth.tar'.format(args.which_epoch))
        else:
            checkpoint = load_checkpoint(resume_path)
        model_dict = model.state_dict()
        if args.curri and not args.evaluate:
            checkpoint_load = {k: v for k, v in (checkpoint).items() if k in model_dict}
        else:
            checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
#        model.load_state_dict(checkpoint['state_dict'])
        if args.curri and not args.evaluate:
            start_epoch =0
            best_top1=0
        else:
            start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    model = nn.DataParallel(model).cuda()


    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader,  dataset.query, dataset.gallery)
        trainer = Trainer(model, 'here', ABN=args.ABN, justglobal=args.justglobal)
        trainer.test(query_loader, gallery_loader, dataset.query, dataset.gallery, re_rank=args.re_rank) # another way to test
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    base_param_ids = list(map(id, model.module.conv1.parameters()))

    if args.ABN:
        base_param_ids += (list(map(id, model.module.bn1.parameters() ))
                           +list(map(id, model.module.layer1.parameters() ))
                           +list(map(id, model.module.layer2.parameters() ))
                         +list(map(id, model.module.layer3.parameters() ))
                          )
    else:           ####PCB_resnet
        base_param_ids += (list(map(id, model.module.bn1.parameters() ))
                           +list(map(id, model.module.layer1.parameters() ))
                           +list(map(id, model.module.layer2.parameters() ))
                         +list(map(id, model.module.layer3.parameters() ))
                         +list(map(id, model.module.layer4.parameters() ))
                          )
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    if args.ABN:
        param_groups = [
            {'params': model.module.conv1.parameters(), 'lr': args.conv_lr},
            {'params': model.module.bn1.parameters(), 'lr': args.conv_lr},
            {'params': model.module.layer1.parameters(), 'lr': args.conv_lr},
            {'params': model.module.layer2.parameters(), 'lr': args.conv_lr},
            {'params': model.module.layer3.parameters(), 'lr': args.conv_lr},
            {'params': new_params, 'lr': args.lr}]

    else:
        param_groups = [
            {'params': model.module.conv1.parameters(), 'lr': args.conv_lr},
            {'params': model.module.bn1.parameters(), 'lr': args.conv_lr},
            {'params': model.module.layer1.parameters(), 'lr': args.conv_lr},
            {'params': model.module.layer2.parameters(), 'lr': args.conv_lr},
            {'params': model.module.layer3.parameters(), 'lr': args.conv_lr},
            {'params': model.module.layer4.parameters(), 'lr': args.conv_lr},
            {'params': new_params, 'lr': args.lr}]

    optimizer = torch.optim.SGD(param_groups,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    print("====> Start Training ====<")
    # Trainer
    trainer = Trainer(model, criterion, ABN=args.ABN, justglobal=args.justglobal)
    if args.adapt:
        exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                                      verbose=True,
                                                                      threshold=0.0001, threshold_mode='rel',
                                                                      cooldown=0, min_lr=0,
                                                                      eps=1e-08)
    else:
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=0.1)

    # Start training
    bf_loss = 0.0
    bf_prec = 0.0
    count_loss = 0
    count_prec = 0
    n=0
    for epoch in range(start_epoch, args.epoch):
        n+=1
        if args.val:
            train_val = ['train', 'val']
        else:
            train_val = ['train']
        for phase in train_val:
            if phase == 'train':
                if not args.adapt:
                    exp_lr_scheduler.step()
                model.train(True)
            else:
                model.train(False)
            if args.ABN or args.justglobal:
                loss, loss_g, loss_p, prec = trainer.train(epoch, train_loader, optimizer)
            else:
                loss, prec = trainer.train(epoch, train_loader, optimizer)

            if args.adapt:
                exp_lr_scheduler.step(loss)
            y_loss[phase].append(loss)
            if args.ABN:
                y_loss_g[phase].append(loss_g)
                y_loss_p[phase].append(loss_p)
            y_prec[phase].append(prec)
            if phase == 'train':
                is_best = True
                save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(exp_dir, 'epoch_last.pth.tar'))
                now_loss = float('%.3f' % loss)
                if now_loss == bf_loss:
                    count_loss = count_loss + 1
                else:
                    count_loss = 0
                bf_loss = now_loss

                now_prec = float('%.3f' % prec)
                if now_prec == bf_prec:
                    count_prec = count_prec + 1
                else:
                    count_prec = 0
                bf_prec = now_prec

                if count_loss >= 8 or count_prec >=8:
                    filename = 'epoch_%s.pth.tar' % (epoch + 1)
                    save_checkpoint({
                        'state_dict': model.module.state_dict(),
                        'epoch': epoch + 1,
                        'best_top1': best_top1,
                    }, is_best, fpath=osp.join(exp_dir, filename))
                if epoch >= int(args.epoch*3/4) :
                    if (epoch + 1) % 5 == 0:
                        filename = 'epoch_%s.pth.tar' % (epoch + 1)
                        save_checkpoint({
                            'state_dict': model.module.state_dict(),
                            'epoch': epoch + 1,
                            'best_top1': best_top1,
                        }, is_best, fpath=osp.join(exp_dir, filename))
            if args.val:
                if phase == 'val':
                    draw_curve(epoch)
            else:
                draw_curve(epoch)

    from shutil import copyfile
    import json
    if not args.evaluate:
        copyfile('./PCB.py', exp_dir + '/PCB.py')
        copyfile('./reid/models/resnet2.py', exp_dir + '/resnet2.py')
        # save argss
        with open('%s/opts.json' % exp_dir, 'w') as fp:
            json.dump(vars(args), fp, indent=1)

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(exp_dir+'/epoch_last.pth.tar')
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
    trainer.test(query_loader, gallery_loader, dataset.query, dataset.gallery, re_rank=args.re_rank)


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    label = 'loss'
    fig = plt.figure()
    plt.title(label)
    plt.plot(x_epoch,y_loss['train'], label='all_loss')
    if args.ABN:
        plt.plot(x_epoch, y_loss_g['train'], label='global_loss')
        plt.plot(x_epoch, y_loss_p['train'], label='part_loss')

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    fig.savefig(os.path.join(exp_dir, 'loss.jpg'))
    plt.close(fig)

    label = 'acc'
    fig = plt.figure()
    plt.title(label)
    plt.plot(x_epoch,y_prec['train'], label='train_acc')

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    fig.savefig(os.path.join(exp_dir, 'acc.jpg'))
    plt.close(fig)
######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")

    # 不变参数
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='',type=str, metavar='PATH')

    # 数据集及模式选择
    parser.add_argument('-d', '--dataset', type=str, default='market', choices=datasets.names())

    parser.add_argument('--ABN', action='store_true', help='use ABN+ResNet50')
    parser.add_argument('--justglobal', action='store_true', help='use ABN+ResNet50')
    parser.add_argument('--curri', action='store_true', help='use curriculum learning')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--re_rank', action='store_true', help='use re_rank')

    parser.add_argument('--val', action='store_true', help='validation')
    parser.add_argument('--all', action='store_true', help='use all dataset')
    parser.add_argument('--resume', action='store_true', help='test')
    parser.add_argument('--resume_path', type=str, default='', metavar='PATH')

    # 训练调参
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--conv_lr', default=0.1, type=float, help='conv_lr')
    parser.add_argument('--stepsize', default=40, type=int, help='stepsize')
    parser.add_argument('--adapt', action='store_true', help='use adapt_lr')
    parser.add_argument('--s2', action='store_true', help='layer4 with stride 2')


    parser.add_argument('--random_erasing', action='store_true', help='Random Erasing probability, 0.5')
    parser.add_argument('--epoch', default=60, type=int, help='epoch')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

    parser.add_argument('--height', type=int, default=384,help="input height, default: 384 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128, help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--features', type=int, default=256,help='in PCB')
    parser.add_argument('--dropout', type=float, default=0.5,help='in PCB')

    working_dir = osp.dirname(osp.abspath(__file__))

    args = parser.parse_args()

    if args.dataset == 'market':
        data_dir = '/home/brain-navigation/bishe_cjh/Market'

    if args.name == '':
        exp_dir = os.path.join('./exp', args.dataset,
                               ('ABN_' if args.ABN else '') +
                               ('curri_' if args.curri else '') +
                               'bs:{}_'.format(args.batchsize) + 'lr:{}_'.format(args.lr) +
                               ('conv_lr:{}'.format(args.conv_lr)) +
                               ('adapt' if args.adapt else '[no adapt]') +
                               ('_resume' if args.resume else '')
                               + ('_RE' if args.random_erasing else '')
                               + ('_all' if args.all else '')
                               + ('_stride2' if args.s2 else '')
                               )

    else:
        exp_dir = os.path.join('./model', args.name)


    if args.resume_path == '':
        if args.resume:
            resume_path = os.path.join('./exp', args.dataset,
                               ('ABN_' if args.ABN else '') +
                               ('curri_' if args.curri else '') +
                               'bs:{}_'.format(args.batchsize) + 'lr:{}_'.format(args.lr) +
                               ('conv_lr:{}'.format(args.conv_lr)) +
                               ('adapt' if args.adapt else '[no adapt]') +
                               # ('resume' if args.resume else '')
                                ('_justglobal' if args.justglobal else '') +
                                ('_RE' if args.random_erasing else ''),
                               'epoch_last.pth.tar'
                               )

        elif args.curri:
            resume_path = os.path.join('./exp', args.dataset,
                                       'stage2_' + 'bs:32_' + 'lr:0.001_' +
                                       'conv_lr:0.001' +
                                       ('adapt_' if args.adapt else '[no adapt]_') +
                                       'weight:w1:0.3,w2:0.7_',
                                       'net_{}.pth'.format(args.which_epoch)
                                       )
        else:
            resume_path = ''


    else:
        resume_path = args.resume_path

    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    y_loss = {}
    y_loss_g = {}
    y_loss_p = {}

    y_loss['train'] = []
    y_loss_g['train'] = []
    y_loss_p['train'] = []

    y_prec = {}
    y_prec['train'] = []
    if args.val:
        y_loss['val'] = []
        y_prec['val'] = []
    start_epoch=0
    main()

# CUDA_VISIBLE_DEVICES=0,1 python PCB.py --batchsize 64  --lr 0.1 --conv_lr 0.01 --ABN --curri
