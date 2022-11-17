from typing import List, Optional

import torch
from torch import nn

class ShortcutProjection(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        # Performs the linear projection of Wx
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        # Perform batch normalization
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        # Convolution and batch normalization
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act3 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1))
        x = self.act2(self.bn2(self.conv2))
        x = self.bn3(self.conv3)
        return self.act3(x + shortcut)


class ResNetBase(nn.Module):

    def __init__(self, cfg):

        n_blocks = cfg['n_blocks']
        n_channels = cfg['n_channels']
        bottlenecks = cfg['bottlenecks']
        img_channels = cfg['img_channels']
        first_kernel_size = cfg['first_kernel_size']

        super().__init__()

        assert len(n_blocks) == len(n_channels)
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        self.conv = nn.Conv2d(img_channels, n_channels[0], kernel_size=first_kernel_size, stride=2,
                              padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels[0])

        blocks = []

        prev_channels = n_channels[0]

        for i, channels in enumerate(n_channels):
            stride = 2 if len(blocks) == 0 else 1
            if bottlenecks is None:
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleneckResidualBlock(prev_channels, bottlenecks[i], channels, stride=stride))

            prev_channels = channels

            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResidualBlock(channels, channels, stride=1))
                else:
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        x = self.bn(self.conv(x))
        x = self.blocks(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        return x.mean(dim=-1)
