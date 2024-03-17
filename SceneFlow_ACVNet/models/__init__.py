from models.acv import ACVNet
from models.pwcnet import PWCNet_G, PWCNet_GC
from models.acv_ddpm import ACVNet_DDPM
from models.acv_ddim import ACVNet_DDIM
from models.acv_ddim_lowD import ACVNet_DDIM_D1
from models.loss import model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test

__models__ = {
    "acvnet": ACVNet,
    "acvnet_ddpm": ACVNet_DDPM,
    "acvnet_ddim": ACVNet_DDIM,
    "acvnet_ddim_d1": ACVNet_DDIM_D1,
    "gwcnet-g": PWCNet_G,
    "gwcnet-gc": PWCNet_GC
}
