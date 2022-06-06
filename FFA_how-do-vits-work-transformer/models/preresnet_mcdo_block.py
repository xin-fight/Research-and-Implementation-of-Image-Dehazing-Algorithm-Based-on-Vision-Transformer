import torch
import torch.nn as nn
import models.layers as layers
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, rate=0.3, sd=0.0,
                 **block_kwargs):
        super(BasicBlock, self).__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(channels * (width_per_group / 64.)) * groups

        self.rate = rate

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(in_channels, channels * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = layers.bn(in_channels)
        self.relu = layers.relu()

        self.conv1 = layers.conv3x3(in_channels, width, stride=stride)
        self.conv2 = nn.Sequential(
            layers.bn(width),
            layers.relu(),
            layers.conv3x3(width, channels * self.expansion),
        )

        self.sd = layers.DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = F.dropout(x, p=self.rate)
        x = self.conv2(x)

        x = self.sd(x) + skip

        return x

    def extra_repr(self):
        return "rate=%.3e" % self.rate


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, rate=0.3, sd=0.0,
                 **block_kwargs):
        super(Bottleneck, self).__init__()

        width = int(channels * (width_per_group / 64.)) * groups

        self.rate = rate

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(in_channels, channels * self.expansion, stride=stride))
            self.shortcut.append(layers.bn(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = layers.bn(in_channels)
        self.relu = layers.relu()

        self.conv1 = layers.conv1x1(in_channels, width)
        self.conv2 = nn.Sequential(
            layers.bn(width),
            layers.relu(),
            layers.conv3x3(width, width, stride=stride, groups=groups),
        )
        self.conv3 = nn.Sequential(
            layers.bn(width),
            layers.relu(),
            layers.conv1x1(width, channels * self.expansion),
        )

        self.sd = layers.DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.dropout(x, p=self.rate)
        x = self.conv3(x)

        x = self.sd(x) + skip

        return x

    def extra_repr(self):
        return "rate=%.3e" % self.rate
