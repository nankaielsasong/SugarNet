import torch
from torch import nn


__all__ = ['ContextBlock']


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, val=0)
        if hasattr(m[-1], 'bias') and m[-1].bias is not None:
            nn.init.constant_(m[-1].bias, 0)
    else:
        nn.init.constant_(m.weight, val=0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SELayer(nn.Module):
    def __init__(self, channel, ratio=1./16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, int(channel * ratio), kernel_size=1),
            nn.LayerNorm([int(channel * ratio), 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(channel * ratio), channel, kernel_size=1),
        )
        last_zero_init(self.fc)

    def forward(self, x):
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        y = self.avg_pool(x)
        # y = self.fc(y).view(b, c, 1, 1)
        y = self.fc(y)
        return x * torch.sigmoid(y)


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.max(x, 1)[0].unsqueeze(1) + torch.mean(x, 1).unsqueeze(1)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio=1./16.,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
            self.channel_pool = ChannelPool()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = SELayer(self.inplanes)
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(self.conv_mask, 'bias') and self.conv_mask.bias is not None:
                nn.init.constant_(self.conv_mask.bias, 0)
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            pass
            # last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            # context_mask = self.conv_mask(x) + self.channel_pool(x)
            context_mask = self.channel_pool(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        x = self.channel_add_conv(x)

        context = self.spatial_pool(x)

        out = x + context

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            pass
            # [N, C, 1, 1]
            # channel_add_term = self.channel_add_conv(context)
            # out = out + channel_add_term

        return out

