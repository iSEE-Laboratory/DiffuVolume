from models.pwcnet import PWCNet_G, PWCNet_GC
from models.pwcnet_ddim import PWCNet_ddimgc
from models.loss import model_loss

__models__ = {
    "gwcnet-g": PWCNet_G,
    "gwcnet-gc": PWCNet_GC,
    "pwc_ddimgc": PWCNet_ddimgc
}
