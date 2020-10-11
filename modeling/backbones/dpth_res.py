# -*-encoding: utf-8-*-
"""
@Dongshichao
@904281665@qq.com
@Nku_Tianjin
@2019-10-18
"""

import math
from .Context import *
import torch
from torch import nn


class DeepthConv(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, gcb, stride=1):
        super(DeepthConv, self).__init__()
        self.with_gcb = gcb
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.dep_conv1 = nn.Conv2d(planes//2, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.dep_conv2 = nn.Conv2d(planes//2, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2_1 = nn.BatchNorm2d(planes)
        self.bn2_2 = nn.BatchNorm2d(planes)
        self.conv3_1 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(planes * 4)
        self.bn3_2 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.LeakyReLU(0.1)
        self.stride = stride
        # GC_Net
        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block_1 = ContextBlock(inplanes=gcb_inplanes)
            self.context_block_2 = ContextBlock(inplanes=gcb_inplanes)

    def forward(self, x):
        residual_1 = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_1 = out[:, :256]
        out_2 = out[:, 256:]

        out_1 = self.dep_conv1(out_1)
        out_1 = self.bn2_1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.conv3_1(out_1)
        out_1 = self.bn3_1(out_1)

        out_2 = self.dep_conv2(out_2)
        out_2 = self.bn2_2(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv3_2(out_2)
        out_2 = self.bn3_2(out_2)

        if self.with_gcb:
            out_1 = self.context_block_1(out_1)
            out_2 = self.context_block_2(out_2)

        # out_1 += residual
        out_1 = self.relu(out_1)

        # out_2 += residual
        out_2 = self.relu(out_2)

        return out_1, out_2


def conv_dw(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, gcb, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.with_gcb = gcb
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.conv2 = conv_dw(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride
        # GC_Net
        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.with_gcb:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], gcb=None)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, gcb=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, gcb=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, gcb=True)

    def _make_layer(self, block, planes, blocks, gcb, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, gcb, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, gcb))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        a = self.layer3(x)
        x = self.layer4(a)

        return a, x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




