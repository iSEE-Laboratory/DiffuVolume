# from __future__ import print_function, division
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
# from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
from utils import *
from models.submodule import *
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import gc
import matplotlib.pyplot as plt
# from apex import amp
import cv2
from thop import profile
from thop import clever_format

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='acvnet_ddim', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/mnt/Datasets/Sceneflow/", help='data path')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--loadckpt', default='checkpoints/checkpoint_000046.ckpt')

# parse arguments, set seeds
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, False, False)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter our: %.2fM" % (total/1e6))
model = nn.DataParallel(model)
model.cuda()

model_origin = __models__['acvnet'](args.maxdisp, False, False)
total = sum([param.nelement() for param in model_origin.parameters()])
print("Number of parameter origin: %.2fM" % (total/1e6))

model_origin = nn.DataParallel(model_origin)
model_origin.cuda()

# # load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


state_dict = torch.load('pretrained_model/sceneflow.ckpt')
model_origin.load_state_dict(state_dict['model'])

def test():
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):    
        start_time = time.time()
        loss, scalar_outputs = test_sample(sample)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx,
                                                                    len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        
    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars", avg_test_scalars)


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    model_origin.eval()
    imgL, imgR, disp_gt, filename = sample['left'], sample['right'], sample['disparity'], sample['left_filename']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    mask_gt = (disp_gt < args.maxdisp) & (disp_gt > 0)

    #disp_ests = model_origin(imgL, imgR)
    disp_ = model_origin(imgL, imgR)[-1]

    disp_net = torch.clamp(disp_, 0, args.maxdisp - 1).unsqueeze(1)

    b, c, h, w = disp_net.shape
    disp_net = F.interpolate(disp_net, size=(h // 4, w // 4), mode='bilinear') / 4

    disp_ests = model(imgL, imgR, disp_, disp_net, None)

    disp_gts = [disp_gt]
    loss = model_loss_test(disp_ests, disp_gt, mask_gt)
    scalar_outputs = {"loss": loss}
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask_gt) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask_gt) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_gt, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_gt, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_gt, 3.0) for disp_est in disp_ests]

    # if scalar_outputs["EPE"][0] > 1:
    #     print(filename)
    #     raise
    return tensor2float(loss), tensor2float(scalar_outputs)

if __name__ == '__main__':
    test()
