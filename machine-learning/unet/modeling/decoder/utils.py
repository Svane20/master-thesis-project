import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class Basic_Conv3x3(nn.Module):
    """
    Basic 3x3 conv followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, padding: int = 1) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Convolution stride. Default: 2.
            padding (int): Convolution padding. Default: 1.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvStream(nn.Module):
    """
    Extracts multiscale detail features from the input RGB image.
    """

    def __init__(self, in_channels: int = 4, out_channels: List[int] = [48, 96, 192, 384]) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (List[int]): List of output channels for each convolution. Default: [48, 96, 192, 384].
        """
        super().__init__()

        self.convs = nn.ModuleList()
        self.conv_chans = out_channels.copy()
        self.conv_chans.insert(0, in_channels)
        for i in range(len(self.conv_chans) - 1):
            self.convs.append(Basic_Conv3x3(self.conv_chans[i], self.conv_chans[i + 1]))

    def forward(self, x: torch.Tensor) -> dict:
        out_dict = {'D0': x}
        for i, conv in enumerate(self.convs):
            x = conv(x)
            out_dict[f'D{i + 1}'] = x
        return out_dict


class Fusion_Block(nn.Module):
    """
    Upsamples deep features and fuses them with corresponding detail features.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        self.conv = Basic_Conv3x3(in_channels, out_channels, stride=1, padding=1)

    def forward(self, x: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([up_x, detail], dim=1)
        return self.conv(out)


class Matting_Head(nn.Module):
    """
    Final head that produces a 1-channel alpha matte.
    """

    def __init__(self, in_channels: int = 16, mid_channels: int = 16):
        """
        Args:
            in_channels (int): Number of input channels. Default: 16.
            mid_channels (int): Number of intermediate channels. Default: 16.
        """
        super().__init__()

        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.matting_convs(x)