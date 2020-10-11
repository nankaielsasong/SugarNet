# -*-encoding: utf-8-*-
"""
@Dongshichao
@904281665@qq.com
@Nku_Tianjin
@2020-2-13
"""


import copy as cp
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parameter import Parameter
from .backbones.resnet import ResNet


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
        self.m = 0.03
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))

    def forward(self, input, label):
        # cosine = F.linear(input, F.normalize(self.weight))
        cosine = F.linear(input, self.weight)
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output


class LBNNeck(nn.Module):

    def __init__(self, in_planes, num_classes):
        super(LBNNeck, self).__init__()
        self.BN = nn.BatchNorm1d(in_planes)
        self.BN.bias.requires_grad_(False)
        self.fc = Linear(in_planes, num_classes)
        self.BN.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def forward(self, x, target):
        x = x.view(x.shape[0], -1)
        triplet_feat = nn.functional.normalize(x)   # L2 normal
        test_feat = self.BN(triplet_feat)
        if self.training:
            cl_feat = self.fc(test_feat, target)
            return cl_feat, triplet_feat
        return triplet_feat
        # return test_feat


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

        self.avg_gap_global = nn.AdaptiveAvgPool2d(1)
        
        self.num_classes = num_classes

        self.lbn = LBNNeck(self.in_planes, self.num_classes)

    def forward(self, x, target=None):
        reid_feat, feat1 = self.base(x)

        feat1 = self.avg_gap_global(feat1)

        if self.training:
            cl_1, triplet_1 = self.lbn(feat1, target)
            return [cl_1], [triplet_1]
        
        return self.lbn(feat1, target)






