# encoding: utf-8

# from .BNNeck import Baseline
# from .LBNNeck import Baseline
# from .Cuhk import Baseline
# from .Sugger_base import Baseline
# from .new_channel_6 import Baseline
# from .new_channel_gcb_res import Baseline
# from .new_channel_gcb_ch import Baseline
# from .new_channel_gcb_all import Baseline
# from .new_channel_4_gcb_ch import Baseline
# from .new_channel_4_gcb_all import Baseline
# from .new_channel_6_gcb_ch import Baseline
# from .base_resnet import Baseline
# from .new_channel_NOCS import Baseline
# from .new_channel_bnn import Baseline

# from .Sugger_base_modify import Baseline
# from .new_channel import Baseline

from .Sugger_base_lbn import Baseline
# from .Sugger_base_AAPooling import Baseline
# from .Sugger_base import Baseline
# from .Sugger_base_varify_aappoling import Baseline
# from .Sugger_base_modify import Baseline



# from .new_channel import Baseline

import torch


def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet50':
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE)
    return model

