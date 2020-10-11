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
from .backbones.resnet import ResNet
from .backbones.resnet import Bottleneck
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


class LBNNeck(nn.Module):

    def __init__(self, in_planes, num_classes):
        super(LBNNeck, self).__init__()
        self.BN = nn.BatchNorm1d(in_planes)
        self.BN.bias.requires_grad_(False)
        self.fc = nn.Linear(in_planes, num_classes, bias=False)
        self.BN.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        triplet_feat = nn.functional.normalize(x)   # L2 normal
        test_feat = self.BN(triplet_feat)
        if self.training:
            cl_feat = self.fc(test_feat)
            return cl_feat, triplet_feat
        # return triplet_feat
        return test_feat


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

        self.branch1_layer4 = nn.Sequential(
            Bottleneck(1024, 512, gcb=True, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512, gcb=True),
            Bottleneck(2048, 512, gcb=True),
        )

        model_dict = self.branch1_layer4.state_dict()
        res_pretrained_dict = res.state_dict()
        res_pretrained_dict = {k: v for k, v in res_pretrained_dict.items() if k in model_dict}
        model_dict.update(res_pretrained_dict)
        self.branch1_layer4.load_state_dict(model_dict)

        self.max_gap_global = nn.AdaptiveMaxPool2d(1)
        self.avg_gap_global = nn.AdaptiveAvgPool2d(1)
        self.branc1_avg_2 = nn.AdaptiveAvgPool2d((2, 1))
        self.branc1_max_2 = nn.AdaptiveMaxPool2d((2, 1))

        self.num_classes = num_classes

        self.lbn = LBNNeck(self.in_planes, self.num_classes)

        self.lbn_branch1_1_1 = LBNNeck(int(self.in_planes/2), self.num_classes)
        self.lbn_branch1_1_2 = LBNNeck(int(self.in_planes/2), self.num_classes)
        self.lbn_branch1_2_1 = LBNNeck(int(self.in_planes/2), self.num_classes)
        self.lbn_branch1_2_2 = LBNNeck(int(self.in_planes/2), self.num_classes)

    def forward(self, x):
        reid_feat, feat1 = self.base(x)

        feat1 = self.max_gap_global(feat1) + self.avg_gap_global(feat1)

        feat2 = self.branch1_layer4(reid_feat)
        feat2 = self.branc1_max_2(feat2) + self.branc1_avg_2(feat2)
        
        feat2_1 = torch.squeeze(feat2[:, :, 0])
        feat2_1_1 = feat2_1[:, :1024]
        feat2_1_2 = feat2_1[:, 1024:]
        feat2_2 = torch.squeeze(feat2[:, :, 1])
        feat2_2_1 = feat2_2[:, :1024]
        feat2_2_2 = feat2_2[:, 1024:]

        if self.training:
            cl_1, triplet_1 = self.lbn(feat1)
            cl_2_1_1, triplet_2_1_1 = self.lbn_branch1_1_1(feat2_1_1)
            cl_2_1_2, triplet_2_1_2 = self.lbn_branch1_1_2(feat2_1_2)
            cl_2_2_1, triplet_2_2_1 = self.lbn_branch1_2_1(feat2_2_1)
            cl_2_2_2, triplet_2_2_2 = self.lbn_branch1_2_2(feat2_2_2)
            return [cl_1, cl_2_1_1, cl_2_1_2, cl_2_2_1, cl_2_2_2], \
                   [triplet_1, triplet_2_1_1, triplet_2_1_2, triplet_2_2_1, triplet_2_2_2]
            
        return torch.cat([self.lbn(feat1), self.lbn_branch1_1_1(feat2_1_1), self.lbn_branch1_1_2(feat2_1_2),
                          self.lbn_branch1_2_1(feat2_2_1), self.lbn_branch1_2_2(feat2_2_2)], dim=1)

