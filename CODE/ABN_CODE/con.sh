#!/usr/bin/env bash
# Example:
# .vison_operation.sh
#
#OR
#
# .vison_operation.sh name
# 


python train.py --gpu_ids 1 --batchsize 64 --lr 0.1 --conv_lr 0.01 --stage1
python test.py --gpu_ids 1 --batchsize 64 --lr 0.1 --conv_lr 0.01 --stage1
python train.py --gpu_ids 1 --batchsize 32 --lr 0.001 --conv_lr 0.001 --stage2 --w1 0.3 --w2 0.7
python test.py --gpu_ids 1 --batchsize 32 --lr 0.001 --conv_lr 0.001 --stage2 --w1 0.3 --w2 0.7
CUDA_VISIBLE_DEVICES=0,1 python PCB.py --batchsize 64  --lr 0.1 --conv_lr 0.01 --ABN --curri
