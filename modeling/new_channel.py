# -*-encoding: utf-8-*-
"""
@Dongshichao
@904281665@qq.com
@Nku_Tianjin
@2019-10-18
"""

import random
import copy as cp
import torch
from torch import nn
import torchvision.models as models
from torch.nn.parameter import Parameter
from .backbones.resnet import ResNet
from .backbones.resnet import Bottleneck
from .backbones.resnet import DeepthConv
import torch.nn.functional as F
from .backbones.GCPooling import *

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
        try:
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
        except:
            pass

class Linear(nn.Module):

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = 1.2
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))

    def forward(self, input, label):
        cosine = F.linear(input, F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output


class ExtraConv(nn.Module):

    def __init__(self):
        super(ExtraConv, self).__init__()
        self.extra_layer = nn.Sequential(
            Bottleneck(1024, 512, gcb=True, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512, gcb=True),
            DeepthConv(2048, 512, gcb=True, se=False),
        )

        res = models.resnet50(pretrained=True)
        model_dict = self.extra_layer.state_dict()
        res_pretrained_dict = res.state_dict()
        res_pretrained_dict = {k: v for k, v in res_pretrained_dict.items() if k in model_dict}
        model_dict.update(res_pretrained_dict)
        self.extra_layer.load_state_dict(model_dict)

    def forward(self, x):
        x1, x2 = self.extra_layer(x)
        return x1, x2


class LBNNeck(nn.Module):

    def __init__(self, in_planes, num_classes):
        super(LBNNeck, self).__init__()
        self.BN = nn.BatchNorm1d(in_planes)
        self.BN.bias.requires_grad_(False)
        self.fc = nn.Linear(in_planes, num_classes) # Linear to nn.Linear
        self.BN.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def forward(self, x, target):
        x = x.view(x.shape[0], -1)
        triplet_feat = nn.functional.normalize(x) # parameter 10
        test_feat = self.BN(triplet_feat) # multiply 10 TODO
        if self.training:
            cl_feat = self.fc(test_feat)
            return cl_feat, triplet_feat
        # return triplet_feat
        return test_feat


class AAPooling(nn.Module):

    def __init__(self, inplanes, poo_size=1):
        super(AAPooling, self).__init__()
        self.inplanes = inplanes
        self.max_pool = nn.AdaptiveMaxPool2d(poo_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(poo_size)
        self.attention_pool = GCPooling(self.inplanes)

    def forward(self, x):
        max_x = self.max_pool(x)
        avg_x = self.avg_pool(x)
        out = self.attention_pool(x, max_x, avg_x)
        return out


class BNNeck(nn.Module):

    def __init__(self, in_planes, num_classes):
        super(BNNeck, self).__init__()
        self.BN = nn.BatchNorm1d(in_planes)
        self.BN.bias.requires_grad_(False)
        self.fc = nn.Linear(in_planes, num_classes)
        self.BN.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def forward(self, global_feat):
        global_feat = global_feat.view(global_feat.shape[0], -1)
        test_feat = self.BN(global_feat)
        if self.training:
            cl_feat = self.fc(test_feat)
            return cl_feat, global_feat
        # return triplet_feat
        return test_feat


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride):
        super(Baseline, self).__init__()

        self.base = ResNet(last_stride)
        self.channel_extra = ExtraConv()

        # load pre_trained model
        res = models.resnet50(pretrained=True)
        model_dict = self.base.state_dict()
        res_pretrained_dict = res.state_dict()
        res_pretrained_dict = {k: v for k, v in res_pretrained_dict.items() if k in model_dict}
        model_dict.update(res_pretrained_dict)
        self.base.load_state_dict(model_dict)

        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.chan_1_max_pooling = nn.AdaptiveMaxPool2d(1)
        self.chan_1_avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.chan_2_max_pooling = nn.AdaptiveMaxPool2d(1)
        self.chan_2_avg_pooling = nn.AdaptiveAvgPool2d(1)


        self.num_classes = num_classes

        self.bn = LBNNeck(self.in_planes, self.num_classes) # uncomment this line
        self.channel_bn_1 = LBNNeck(self.in_planes, self.num_classes)
        self.channel_bn_2 = LBNNeck(self.in_planes, self.num_classes)

    def forward(self, x, label=None):
        reid_feat, feat1 = self.base(x)
        feat1 = self.global_max_pooling(feat1) + self.global_avg_pooling(feat1) # uncomment this line 

        chan_1, chan_2 = self.channel_extra(reid_feat)
        chan_1 = self.chan_1_max_pooling(chan_1) + self.chan_1_avg_pooling(chan_1)
        chan_2 = self.chan_2_max_pooling(chan_2) + self.chan_2_avg_pooling(chan_2)

        if self.training:

            cl_1, triplet_1 = self.bn(feat1, label) # uncomment this line 

            c1_chan_1, triplet_chan_1 = self.channel_bn_1(chan_1, label)

            c1_chan_2, triplet_chan_2 = self.channel_bn_2(chan_2, label)

            return [cl_1, c1_chan_1, c1_chan_2], [triplet_1, triplet_chan_1, triplet_chan_2]

        return torch.cat([self.bn(feat1, label), self.channel_bn_1(chan_1, label), self.channel_bn_2(chan_2, label)], dim=1)

