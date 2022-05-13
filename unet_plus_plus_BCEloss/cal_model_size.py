from src.networks.network_rcf import RCF
from src.networks.network_unet import UNET
from src.networks.network_ende import ENDE
from src.networks.network_ende_plus import ENDEPlus
from src.networks.network_pidinet import PiDiNet
from src.networks.network_dexined import DexiNed
from src.networks.network_unetplusplus import NestedUNet
from src.networks.network_fined import FINED
from src.networks.network_cenet import CENet
from src.networks.network_msunet import MSUNet
from src.networks.network_stdc1 import BiSeNetSTDC1
from src.networks.network_stdc2 import BiSeNetSTDC2
from src.networks.network_stdc1_plus import BiSeNetSTDC1Plus
from src.networks.network_stdc2_plus import BiSeNetSTDC2Plus
from src.config import Config
import numpy as np
from thop import profile
import torch

config = Config("config.yml.example")
edge_detect = ENDE(config, in_channels=7)

input = torch.randn(1, 7, 224, 224)
flops, params = profile(edge_detect, inputs=(input, ))
print('params: ', params/(1e6), 'flops: ', flops/(1e9))
