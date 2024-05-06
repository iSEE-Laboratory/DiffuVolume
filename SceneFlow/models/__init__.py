from models.acv import ACVNet
from models.acv_ddim import ACVNet_DDIM
from models.loss import model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test

__models__ = {
    "acvnet": ACVNet,
    "acvnet_ddim": ACVNet_DDIM,
}
