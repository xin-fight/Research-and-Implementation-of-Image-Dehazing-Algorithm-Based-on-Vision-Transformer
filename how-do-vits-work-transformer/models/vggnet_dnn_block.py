import torch
import torch.nn as nn
import models.layers as layers


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **block_kwargs):
        super(BasicBlock, self).__init__()

        self.conv = layers.conv3x3(in_channels, out_channels)
        self.bn = layers.bn(out_channels)
        self.relu = layers.relu()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
