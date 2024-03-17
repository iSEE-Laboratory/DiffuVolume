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
# from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import matplotlib.pyplot as plt
import skimage
import skimage.io
import cv2

# cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser(
    description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='pwc_ddimgc', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/home/zhengdian/dataset/KITTI/2012/", help='data path')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--testlist', default='./filenames/test_temp.txt', help='testing list')
parser.add_argument('--loadckpt', default='./checkpoints/kitti12/test_all/checkpoint_000244.ckpt')
# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

model_origin = __models__['gwcnet-gc'](args.maxdisp)
model_origin = nn.DataParallel(model_origin)
model_origin.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

state_dict = torch.load('./PCWNet_kitti12_best.ckpt')
model_origin.load_state_dict(state_dict['model'])

save_dir = './speed_test/'


def test():
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.synchronize()
        start_time = time.time()
        # disp_est_ = test_sample(sample)
        # for i in range(len(disp_est_)):
        #     disp_est_np = tensor2numpy(disp_est_[i]).squeeze(0)
        #     torch.cuda.synchronize()
        #     print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
        #                                             time.time() - start_time))
        #     left_filenames = sample["left_filename"]
        #     top_pad_np = tensor2numpy(sample["top_pad"])
        #     right_pad_np = tensor2numpy(sample["right_pad"])
        #
        #     for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
        #         assert len(disp_est.shape) == 2
        #         disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
        #         # disp_est = np.array(disp_est, dtype=np.float32)
        #         fn = os.path.join(save_dir, fn.split('/')[-1])
        #         print("saving to", fn, disp_est.shape)
        #         disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
        #         # skimage.io.imsave(fn, disp_est_uint)
        #         plt.imsave(str(i)+'.png', disp_est_uint, cmap='jet')
        disp_est_np = tensor2numpy(test_sample(sample))
        torch.cuda.synchronize()
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        left_filenames = sample["left_filename"]
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            #disp_est = np.array(disp_est, dtype=np.float32)
            fn = os.path.join(save_dir, fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            #skimage.io.imsave(fn, disp_est_uint)
            plt.imsave('a.png', disp_est_uint, cmap='jet')
            #cv2.imwrite(fn, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01), cv2.COLORMAP_JET))


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    model_origin.eval()
    imgL, imgR, filename = sample['left'], sample['right'], sample['left_filename']
    imgL = imgL.cuda()
    imgR = imgR.cuda()

    # disp_ests, qwe = model_origin(imgL, imgR)
    disp_, qwe = model_origin(imgL, imgR)
    disp_ = disp_[-1]
    disp_net = torch.clamp(disp_, 0, args.maxdisp - 1).unsqueeze(1)

    b, c, h, w = disp_net.shape
    disp_net = F.interpolate(disp_net, size=(h // 4, w // 4), mode='bilinear') / 4

    disp_ests, qwe = model(imgL, imgR, disp_, disp_net, None)


    return disp_ests[-1]


if __name__ == '__main__':
    test()
