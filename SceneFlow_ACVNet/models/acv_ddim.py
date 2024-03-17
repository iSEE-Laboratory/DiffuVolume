from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
from .head import DynamicHead
import math
import gc
import time


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        return {"gwc_feature": gwc_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ACVNet_DDIM(nn.Module):
    def __init__(self, maxdisp, attn_weights_only, freeze_attn_weights):
        super(ACVNet_DDIM, self).__init__()
        self.maxdisp = maxdisp
        self.attn_weights_only = attn_weights_only
        self.freeze_attn_weights = freeze_attn_weights
        self.num_groups = 40
        self.concat_channels = 32
        # build diffusion
        self.scale = 1.0
        timesteps = 1000
        sampling_timesteps = 5
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1
        self.renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


        self.feature_extraction = feature_extraction()
        self.concatconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,
                                                  bias=False))

        self.patch = nn.Conv3d(40, 40, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=40, padding=(0, 1, 1),
                               bias=False)
        self.patch_l1 = nn.Conv3d(8, 8, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=8, padding=(0, 1, 1),
                                  bias=False)
        self.patch_l2 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=2, groups=16, padding=(0, 2, 2),
                                  bias=False)
        self.patch_l3 = nn.Conv3d(16, 16, kernel_size=(1, 3, 3), stride=1, dilation=3, groups=16, padding=(0, 3, 3),
                                  bias=False)

        self.dres1_att_ = nn.Sequential(convbn_3d(40, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn_3d(32, 32, 3, 1, 1))

        self.dres2_att_ = hourglass(32)
        self.classif_att_ = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.time_embedding = DynamicHead(d_model=48)

        self.dres0 = nn.Sequential(convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, volume, noise, t):
        b, c, d, h, w = volume.shape
        noise = self.time_embedding(noise, t)
        noise = torch.clamp(noise, min=-1 * self.scale, max=self.scale)
        noise = ((noise / self.scale) + 1) / 2

        volume = volume * noise.unsqueeze(1).float()
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)

        cost_v = self.classif2(out2)
        cost2 = F.upsample(cost_v, [self.maxdisp, h*4, w*4], mode='trilinear')
        cost2 = torch.squeeze(cost2, 1)
        pred_volume2 = F.softmax(cost2, dim=1)
        pred = disparity_regression(pred_volume2, self.maxdisp)

        disp_net = torch.clamp(pred, 0, self.maxdisp - 1).unsqueeze(1)
        b, c, h, w = disp_net.shape
        disp_net = F.interpolate(disp_net, size=(h // 4, w // 4), mode='bilinear') / 4
        disp_net = disp_net.unsqueeze(2)

        b, c, _, h, w = disp_net.shape
        disp_volume = torch.zeros([b, 48, h, w], dtype=torch.float32).cuda()
        real = torch.floor(disp_net).long()
        mask_num = real == 47
        # print(real[:, 0, 63, 127])
        coff = real - disp_net + 1

        disp_volume = disp_volume.view(b, 48, -1).scatter_(1, real.view(b, 1, -1), coff.view(b, 1, -1)).reshape(
            b, 48, h, w)
        disp_volume = disp_volume.view(b, 48, -1).scatter_(1, torch.clamp(real + 1, 0, 47).view(b, 1, -1),
                                                           (1 - coff).view(b, 1, -1)).reshape(b, 48, h, w)
        fuzhi = torch.zeros([b, 48, h, w], dtype=torch.float32).cuda()
        fuzhi[:, -1, :, :] = 1
        x_start = torch.where(mask_num.cuda().squeeze(1) == True, fuzhi, disp_volume)
        x_start = self.scale * (x_start * 2 - 1.)
        x_start = torch.clamp(x_start, min=-self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(noise, t, x_start)

        return pred_noise, x_start, pred, pred_volume2

    @torch.no_grad()
    def ddim_sample(self, volume, used, asd):
        batch, channel, depth, h, w = volume.shape

        shape = (batch, 48, h, w)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img_noise = torch.randn(shape, device=volume.device)
        img = asd
        final = []
        final.append(used.unsqueeze(0))
        mask = torch.zeros([img.shape[0], h, w], dtype=torch.float32)
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=volume.device, dtype=torch.long)
            pred_noise, x_start, disp, pred_volume = self.model_predictions(volume, img, time_cond)
            final.append(disp.unsqueeze(0))

            if self.renewal:  # filter

                dif = torch.abs(disp - used)
                mask_temp1 = torch.where(dif < 1, 1, 0)
                
                disp_values = torch.arange(0, 192, dtype=disp.dtype, device=disp.device)
                disp_values = disp_values.view(1, 192, 1, 1)
                difference = torch.abs(disp.unsqueeze(1) - disp_values)
                score = difference * pred_volume
                uncertainty = torch.sum(score, dim=1)
                mask_temp2 = torch.where(uncertainty < 3, 1, 0)
                mask_temp = mask_temp2 * mask_temp1

                mask_temp = F.interpolate(mask_temp.unsqueeze(1).float(), size=(h, w), mode='bilinear').squeeze(1)

                #mask = mask_temp.cuda()
                #mask_temp = torch.where(mask_temp>0.5, 1, 0)
                mask = mask.cuda() + mask_temp.cuda()
                mask = torch.clamp(mask, 0, 1)
                #print(len(torch.nonzero(mask[0])))
                # prob = torch.rand_like(mask)
                # prob_mask = torch.where(prob <= 0.1, 0, 1)
                # mask = mask * prob_mask

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            
            t = torch.randint(time, time+1, (1,), device=img.device).long()
            asdd = self.q_sample(asd, t)
            no = torch.rand_like(asdd)

            img = torch.where(mask.unsqueeze(1)==0, no, img)

        results = disp
        if self.use_ensemble:
            final = torch.cat(final, dim=0)
            cof = torch.tensor([0.5, 0., 0., 0., 0.2, 0.3], device=final.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            final_prediction = torch.sum(final*cof, dim=0, keepdim=False)
            return final_prediction, final
        return results

    def forward(self, left, right, used, disp, mask_gt=None):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)
        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"],
                                      self.maxdisp // 4, self.num_groups)
        gwc_volume = self.patch(gwc_volume)
        patch_l1 = self.patch_l1(gwc_volume[:, :8])
        patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
        patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
        patch_volume = torch.cat((patch_l1, patch_l2, patch_l3), dim=1)
        cost_attention = self.dres1_att_(patch_volume)
        cost_attention = self.dres2_att_(cost_attention)
        att_weights = self.classif_att_(cost_attention)

        concat_feature_left = self.concatconv(features_left["gwc_feature"])
        concat_feature_right = self.concatconv(features_right["gwc_feature"])
        concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.maxdisp // 4)

        ac_volume = F.softmax(att_weights, dim=2) * concat_volume
        
        cost0 = self.dres0(ac_volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)

        cost2 = self.classif2(out2)
        cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost2 = torch.squeeze(cost2, 1)
        pred2 = F.softmax(cost2, dim=1)
        pred2 = disparity_regression(pred2, self.maxdisp)
        if not self.training:
            b, c, h, w = disp.shape
            disp_volume = torch.zeros([b, 48, h, w], dtype=torch.float32).cuda()
            real = torch.floor(disp).long()
            mask = real == 47
            coff = real - disp + 1

            disp_volume = disp_volume.view(b, 48, -1).scatter_(1, real.view(b, 1, -1), coff.view(b, 1, -1)).reshape(b, 48, h, w)
            disp_volume = disp_volume.view(b, 48, -1).scatter_(1, torch.clamp(real + 1, 0, 47).view(b, 1, -1),
                                                               (1 - coff).view(b, 1, -1)).reshape(b, 48, h, w)
            fuzhi = torch.zeros([b, 48, h, w], dtype=torch.float32).cuda()
            fuzhi[:, -1, :, :] = 1
            disp_volume_final = torch.where(mask.cuda() == True, fuzhi, disp_volume)
            if mask_gt is not None:
                fuzhi_allone = torch.ones([b, 48, h, w], dtype=torch.float32).cuda() / 48
                disp_volume_final = torch.where(mask_gt==0, fuzhi_allone, disp_volume_final)

            disp_volume_final = (disp_volume_final * 2 - 1) * self.scale
            pred, pred_all = self.ddim_sample(ac_volume, used, disp_volume_final)

            return [pred]

        else:
            b, c, h, w = disp.shape
            disp_volume = torch.zeros([b, 48, h, w], dtype=torch.float32).cuda()
            real = torch.floor(disp).long()
            mask = real == 47
            coff = real - disp + 1

            disp_volume = disp_volume.view(b, 48, -1).scatter_(1, real.view(b, 1, -1), coff.view(b, 1, -1)).reshape(b, 48, h, w)
            disp_volume = disp_volume.view(b, 48, -1).scatter_(1, torch.clamp(real + 1, 0, 47).view(b, 1, -1),
                                                               (1 - coff).view(b, 1, -1)).reshape(b, 48, h, w)
            fuzhi = torch.zeros([b, 48, h, w], dtype=torch.float32).cuda()
            fuzhi[:, -1, :, :] = 1
            disp_volume_final = torch.where(mask.cuda() == True, fuzhi, disp_volume)
            if mask_gt is not None:
                fuzhi_allone = torch.ones([b, 1, 48, h, w], dtype=torch.float32).cuda() / 48
                disp_volume_final = torch.where(mask_gt.unsqueeze(1)==0, fuzhi_allone, disp_volume_final.unsqueeze(1)).squeeze(1)
            disp_volume_final = (disp_volume_final * 2 - 1) * self.scale
            t = torch.randint(0, self.num_timesteps, (1,), device=disp_volume_final.device).long()
            noisy = self.q_sample(disp_volume_final, t)

            noisy = self.time_embedding(noisy, t)

            noisy = torch.clamp(noisy, min=-1 * self.scale, max=self.scale)
            noisy = ((noisy / self.scale) + 1) / 2.
            noisy = noisy.unsqueeze(1)
            noisy = torch.tensor(noisy, dtype=torch.float32)

            ac_volume = ac_volume * noisy
            cost0 = self.dres0(ac_volume)
            cost0 = self.dres1(cost0) + cost0
            out1 = self.dres2(cost0)
            out2 = self.dres3(out1)

            cost_attention = F.upsample(att_weights, [self.maxdisp, left.size()[2], left.size()[3]],
                                        mode='trilinear')
            cost_attention = torch.squeeze(cost_attention, 1)
            pred_attention = F.softmax(cost_attention, dim=1)
            pred_attention = disparity_regression(pred_attention, self.maxdisp)

            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            return [pred_attention, pred0, pred1, pred2]

def acv(d):
    return ACVNet(d)
