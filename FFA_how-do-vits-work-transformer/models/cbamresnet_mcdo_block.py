import torch
import torch.nn as nn
import torch.nn.functional as F

import models.layers as layers
import models.gates as gates


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, rate=0.3, sd=0.0,
                 reduction=16, **block_kwargs):
        super(BasicBlock, self).__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(channels * (width_per_group / 64.)) * groups

        self.rate = rate

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(in_channels, channels * self.expansion, stride=stride))
            self.shortcut.append(layers.bn(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = nn.Sequential(
            layers.conv3x3(in_channels, width, stride=stride),
            layers.bn(width),
            layers.relu()
        )
        self.conv2 = nn.Sequential(
            layers.conv3x3(width, channels * self.expansion),
            layers.bn(channels * self.expansion),
        )

        self.sd = layers.DropPath(sd) if sd > 0.0 else nn.Identity()
        self.relu = layers.relu()
        self.gate = nn.Sequential(
            gates.ChannelGate(channels * self.expansion, reduction, max_pool=True),
            gates.SpatialGate(kernel_size=7, max_pool=True)
        )

    def forward(self, x):
        skip = self.shortcut(x)

        x = self.conv1(x)
        x = F.dropout(x, p=self.rate)
        x = self.conv2(x)
        x = self.gate(x)

        x = self.sd(x) + skip
        x = self.relu(x)

        return x

    def extra_repr(self):
        return "rate=%.3e" % self.rate


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, rate=0.3, sd=0.0,
                 reduction=16, **block_kwargs):
        super(Bottleneck, self).__init__()

        width = int(channels * (width_per_group / 64.)) * groups

        self.rate = rate

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(
                in_channels, channels * self.expansion, stride=stride))
            self.shortcut.append(layers.bn(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = nn.Sequential(
            layers.conv1x1(in_channels, width),
            layers.bn(width),
            layers.relu(),
        )
        self.conv2 = nn.Sequential(
            layers.conv3x3(width, width, stride=stride, groups=groups),
            layers.bn(width),
            layers.relu(),
        )
        self.conv3 = nn.Sequential(
            layers.conv1x1(width, channels * self.expansion),
            layers.bn(channels * self.expansion)
        )

        self.sd = layers.DropPath(sd) if sd > 0.0 else nn.Identity()
        self.relu = layers.relu()
        self.gate = nn.Sequential(
            gates.ChannelGate(channels * self.expansion, reduction, max_pool=True),
            gates.SpatialGate(kernel_size=7, max_pool=True)
        )


    def forward(self, x):
        skip = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.dropout(x, p=self.rate)
        x = self.conv3(x)
        x = self.gate(x)

        x = self.sd(x) + skip
        x = self.relu(x)

        return x

    def extra_repr(self):
        return "rate=%.3e" % self.rate
