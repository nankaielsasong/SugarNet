# -*-encoding: utf-8-*-
"""
@Dongshichao
@904281665@qq.com
@Nku_Tianjin
@2019-10-18
"""


import random
import copy as cp
from .losses.face import ArcMarginProduct

import torch
from torch import nn
import torchvision.models as models
from torch.nn.parameter import Parameter
from .backbones.resnet import ResNet
# from .backbones.resnet_dong import ResNet
from .backbones.GCPooling import *
from .layers.SE_Resnet import SEResnet
from .layers.SE_module import SELayer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class LBNNeck_v2(nn.Module):

    def __init__(self, in_planes, num_classes):
        super(LBNNeck_v2, self).__init__()
        self.bn = nn.BatchNorm1d(in_planes)
        self.alpha = Parameter(torch.ones(1))
        self.BN = nn.BatchNorm1d(in_planes)
        self.BN.bias.requires_grad_(False)
        self.fc1 = nn.Linear(in_planes, in_planes, bias=False)
        self.fc = ArcMarginProduct(in_planes, num_classes, s=20, m=0.2)
        self.BN.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)
        self.fc1.apply(weights_init_classifier)


    def forward(self, x, label):
        x = x.view(x.shape[0], -1)
        x = self.bn(x)
        x = self.fc1(x)
        triplet_feat = nn.functional.normalize(x) * self.alpha  # L2 normal
        test_feat = self.BN(triplet_feat)
        if self.training:
            cl_feat = self.fc(test_feat, label)
            return cl_feat, triplet_feat, self.alpha
        # return triplet_feat
        return test_feat


class AAPooling(nn.Module):

    def __init__(self, inplanes):
        super(AAPooling, self).__init__()
        self.inplanes = inplanes
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.attention_pool = GCPooling(self.inplanes)

    def forward(self, x):
        max_x = self.max_pool(x)
        out = self.attention_pool(x, max_x)
        return out


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride):
        super(Baseline, self).__init__()

        self.base = ResNet(last_stride)

        # load pre_trained model
        res = models.resnet50(pretrained=True)
        model_dict = self.base.state_dict()
        res_pretrained_dict = res.state_dict()
        res_pretrained_dict = {k: v for k, v in res_pretrained_dict.items() if k in model_dict}
        model_dict.update(res_pretrained_dict)
        self.base.load_state_dict(model_dict)

        self.pooling = AAPooling(2048)
        
        self.num_classes = num_classes

        self.lbn = LBNNeck_v2(self.in_planes, self.num_classes)

    def forward(self, x, label=None):
        reid_feat, feat1 = self.base(x)

        feat1 = self.pooling(feat1)

        if self.training:
            cl_1, triplet_1, alpha = self.lbn(feat1, label)
            return [cl_1], [triplet_1], alpha
        
        return self.lbn(feat1, label)






