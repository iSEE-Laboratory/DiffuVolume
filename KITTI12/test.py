from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage.io

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
parser = argparse.ArgumentParser(description='PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching')
parser.add_argument('--model', default='pwc_ddimgc', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--test_batchsize', type=int, default=1)
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/mnt/Datasets/KITTI/2012/", help='data path')
parser.add_argument('--testlist', default="./filenames/kitti12_all.txt", help='testing list')
parser.add_argument('--loadckpt', default="./checkpoints/our_best.ckpt",
                    help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batchsize, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

model_origin = __models__['gwcnet-gc'](args.maxdisp)
model_origin = nn.DataParallel(model_origin)
model_origin.cuda()
state_dict = torch.load("./checkpoints/origin.ckpt")
model_origin.load_state_dict(state_dict['model'])


def test():
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs = test_sample(sample, compute_metrics=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx,
                                                                    len(TestImgLoader), loss,
                                                                    time.time() - start_time))
    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars", avg_test_scalars)
    gc.collect()


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    model_origin.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    # disp_ests, qwe = model_origin(imgL, imgR)
    
    disp_, _ = model_origin(imgL, imgR)
    disp_ = disp_[-1]
    disp_net = torch.clamp(disp_, 0, args.maxdisp - 1).unsqueeze(1)
    b, c, h, w = disp_net.shape
    disp_net = F.interpolate(disp_net, size=(h // 4, w // 4), mode='bilinear') / 4

    disp_ests, pred3 = model(imgL, imgR, disp_, disp_net, None)

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    #image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    #scalar_outputs["D1_pred3"] = [D1_metric(pred, disp_gt, mask) for pred in pred3]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)#, image_outputs


if __name__ == '__main__':
    test()
