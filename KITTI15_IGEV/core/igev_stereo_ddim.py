import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.head import DynamicHead
from core.extractor import MultiBasicEncoder, Feature
from core.geometry_ddim import Combined_Geo_Encoding_Volume
from core.submodule import *
import time
import math


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

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

class IGEVStereo_ddim(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.scale = 1.
        timesteps = 1000
        sampling_timesteps = 2
        print(sampling_timesteps)
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
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        self.time_embedding = DynamicHead(d_model=48)
        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp

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
    
    def model_predictions(self, coords0, coords1, flow_init, iters, net_list, inp_list, corr_fn, noise, t):

        noise = self.time_embedding(noise, t)
        # noise = noise + t.unsqueeze(1).unsqueeze(1).unsqueeze(1) / self.num_timesteps
        noise = torch.clamp(noise, min=-1 * self.scale, max=self.scale)
        noise = ((noise / self.scale) + 1) / 2
      
        if flow_init is not None:
            coords1 = coords1 + flow_init

        for itr in range(iters):
            coords1 = coords1.detach()   # 2,2,80,180,第二个2代表坐标索引XY
            flow = coords1 - coords0
            corr = corr_fn(coords1, noise)    # index correlation volume, 2,36,80,180

            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0   #2,2,80,180,stereo matching do not need Y difference, set0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:   #2,144,80,180
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]
        pred = flow_up
        b, c, h, w = pred.shape
        disp_net = torch.clamp(pred, -w + 1, 0)
 
        disp_net = F.interpolate(disp_net, size=(h // 4, w // 4), mode='bilinear') / 4
        true_coords1 = coords0[:, :1] + disp_net
        b, c, h, w = true_coords1.shape
        true_coords1 = torch.clamp(true_coords1, 0, w-1)
        disp_volume = torch.zeros([b, w, h, w], dtype=torch.float32, device=true_coords1.device)
        real = torch.floor(true_coords1).long()
        mask = real == w - 1
        coff = real - true_coords1 + 1

        disp_volume = disp_volume.view(b, w, -1).scatter_(1, real.view(b, 1, -1), coff.view(b, 1, -1)).reshape(b, w, h, w)

        disp_volume = disp_volume.view(b, w, -1).scatter_(1, torch.clamp(real + 1, 0, w - 1).view(b, 1, -1),
                                                           (1 - coff).view(b, 1, -1)).reshape(b, w, h, w)
        fuzhi = torch.zeros([b, w, h, w], dtype=torch.float32, device=true_coords1.device)
        fuzhi[:, -1, :, :] = 1
        x_start = torch.where(mask.cuda() == True, fuzhi, disp_volume)

        x_start = self.scale * (x_start * 2 - 1.)
        x_start = torch.clamp(x_start, min=-self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(noise, t, x_start)

        return pred_noise, x_start, pred, coords1

    @torch.no_grad()
    def ddim_sample(self, coords0, coords1, flow_init, iters, net_list, inp_list, corr_fn, used, asd):
        batch, d, h, w = asd.shape
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn_like(asd, device=asd.device)
        # img = asd
        final = []
        final.append(used.squeeze(1).unsqueeze(0))
        mask = torch.zeros([batch, h, w], dtype=torch.float32)
        # time_cond = torch.full((batch,), times[0], device=volume.device, dtype=torch.long)
        # pred_noise, x_start, disp, pred_volume = self.model_predictions(volume, img, time_cond)
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=asd.device, dtype=torch.long)
            #img = self.q_sample(img, time_cond)
            pred_noise, x_start, disp, coords1 = self.model_predictions(coords0, coords1, flow_init, iters, net_list, inp_list, corr_fn, img, time_cond)

            if self.renewal:  # filter
                dif = torch.abs(disp - used)
                mask_temp = torch.where(dif < 5, 1, 0)

                mask_temp = F.interpolate(mask_temp.float(), size=(h, w), mode='bilinear').squeeze(1)

                mask = mask.cuda() + mask_temp.cuda()
                mask = torch.clamp(mask, 0, 1)
                
            dif = torch.abs(disp - used)
            mask_temp1 = torch.where(dif < 3, 1, 0)
            disp = torch.where(mask_temp1.unsqueeze(1)==0, used, disp).squeeze(1).squeeze(0)

            final.append(disp.squeeze(1).unsqueeze(0))               
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

            img = torch.where(mask.unsqueeze(1)==0, asdd, img)
        # dif = torch.abs(disp - used)
        # mask_temp = torch.where(dif < 0.8, 1, 0)
        # disp = torch.where(mask_temp.unsqueeze(1)==0, used, disp).squeeze(1).squeeze(0)
        # alpha = 0.5
        # print(alpha)
        # results = alpha * disp + (1-alpha) * used.squeeze(0)
        if self.use_ensemble:
            final = torch.cat(final, dim=0)
            cof = torch.tensor([0.6, 0.1, 0.3], device=final.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)

            final_prediction = torch.sum(final*cof, dim=0, keepdim=False)
            return final_prediction
        return results
    
    def forward(self, image1, image2, flow_full, flow_gt, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.args.max_disp//4)
            
            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]


        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        true_coords1 = flow_gt
        true_coords1 = torch.clamp(true_coords1, 0, 48-1)
        disp_volume = torch.zeros([b, 48, h, w], dtype=torch.float32, device=true_coords1.device)
        real = torch.floor(true_coords1).long()
        mask = real == 47
        # print(real[:, 0, 63, 127])
        coff = real - true_coords1 + 1

        disp_volume = disp_volume.view(b, 48, -1).scatter_(1, real.view(b, 1, -1), coff.view(b, 1, -1)).reshape(b, 48, h, w)
        disp_volume = disp_volume.view(b, 48, -1).scatter_(1, torch.clamp(real + 1, 0, 47).view(b, 1, -1),
                                                               (1 - coff).view(b, 1, -1)).reshape(b, 48, h, w)
        fuzhi = torch.zeros([b, 48, h, w], dtype=torch.float32).cuda()
        fuzhi[:, -1, :, :] = 1
        disp_volume_final = torch.where(mask.cuda() == True, fuzhi, disp_volume)

        disp_volume_final = (disp_volume_final * 2 - 1) * self.scale
        
        if not self.training:
            pred = self.ddim_sample(disp , flow_init, iters, net_list, inp_list, geo_fn, flow_full, disp_volume_final)
            
            return pred, pred
            
            
        t = torch.randint(0, self.num_timesteps, (1,), device=disp_volume_final.device).long()
        noisy = self.q_sample(disp_volume_final, t)
        noisy = self.time_embedding(noisy, t)
        noisy = noisy + t.unsqueeze(1).unsqueeze(1).unsqueeze(1) / self.num_timesteps
        noisy = torch.clamp(noisy, min=-1 * self.scale, max=self.scale)
        noisy = ((noisy / self.scale) + 1) / 2.
        noisy = torch.tensor(noisy, dtype=torch.float32)
        
        disp = init_disp
        disp_preds = []
        
        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, noisy)
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res ConvGRU and mid-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds
