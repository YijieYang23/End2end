from fvcore.nn import FlopCountAnalysis, parameter_count_table
import os

import torch

import time
import shutil
import pytorch_warmup
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from core.config.cfg import cfg, cfg_name
from core.net.end2end import Model
from core.data.ntuannot import NTUAnnot
from core.data.feeder import Feeder
from core.functions.loss import *
from core.functions.utils import *
from core.functions.function import *

model = Model(cfg, is_train=False)
from thop import profile


input = torch.randn(1, 32, 3, 192,320)
flops, params = profile(model, inputs=(input,))
print(flops)
print(params)

# tensor = torch.rand((1,32, 3, 128, 128))
#
# flops = FlopCountAnalysis(model, tensor)
# print("FLOPs: ", flops.total())