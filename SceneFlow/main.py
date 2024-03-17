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
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='acvnet_ddim', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/mnt/Datasets/Sceneflow/", help='data path')
parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=23, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lrepochs',default="16,24,32,40,48:2", type=str,  help='the epochs to decay lr: the downscale rate')
parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')
parser.add_argument('--freeze_attention_weights', default=False, type=str,  help='freeze attention weights parameters')
parser.add_argument('--logdir',default='./checkpoints/', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='./pretrained_model/sceneflow.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
# logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, args.attention_weights_only, args.freeze_attention_weights)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict) 
    model.load_state_dict(model_dict)
   
print("start at epoch {}".format(start_epoch))


def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        all_loss = 0
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=False)
            all_loss += loss
            # if do_summary:
            #     save_scalars(logger, 'train', scalar_outputs, global_step)
            #     save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        print('Epoch {}/{}, train loss = {:.3f}'.format(epoch_idx, args.epochs, all_loss))
        # saving checkpoints

        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            #id_epoch = (epoch_idx + 1) % 100
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_net = torch.clamp(disp_gt, 0, args.maxdisp-1).unsqueeze(1)
    b, c, h, w = disp_net.shape
    disp_net = F.interpolate(disp_net, size=(h//4, w//4), mode='bilinear') / 4
    optimizer.zero_grad()
    disp_ests = model(imgL, imgR, None, disp_net, None)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    if args.attention_weights_only:
        loss = model_loss_train_attn_only(disp_ests, disp_gt, mask)
    elif args.freeze_attention_weights:
        loss = model_loss_train_freeze_attn(disp_ests, disp_gt, mask)
    else:
        loss = model_loss_train(disp_ests, disp_gt, mask)
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

if __name__ == '__main__':
    train()
