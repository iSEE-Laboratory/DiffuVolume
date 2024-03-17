#!/usr/bin/env bash
set -x
DATAPATH="/home/zhengdian/dataset/KITTI/2012/"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti12_train.txt --testlist ./filenames/kitti12_val.txt \
    --epochs 300  --lr 0.001 --batch_size 4 --lrepochs "200:10" \
    --model pcw_ddim --logdir ./checkpoints/kitti12/test \
    --test_batch_size 12