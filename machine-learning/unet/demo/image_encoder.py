import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from typing import Tuple, List

from utils import DoubleConv, DownSample


class ImageEncoder(nn.Module):
    """
    Image Encoder with pretrained VGG16 for semantic segmentation.
    """

    def __init__(self, pretrained: bool = True, freeze_pretrained: bool = False) -> None:
        """
        Args:
            pretrained (bool): Use pretrained weights. Default is True.
            freeze_pretrained (bool): Freeze pretrained weights. Default is False.
        """
        super().__init__()

        encoder = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT if pretrained else None).features
        self.down1 = nn.Sequential(*encoder[:6])  # 3 -> 64
        self.down2 = nn.Sequential(*encoder[6:13])  # 64 -> 128
        self.down3 = nn.Sequential(*encoder[13:20])  # 128 -> 256
        self.down4 = nn.Sequential(*encoder[20:27])  # 256 -> 512
        self.down5 = nn.Sequential(*encoder[27:34])  # 512 -> 512

        self.bottle_neck = nn.Sequential(
            *encoder[34:],  # 512 -> 512
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Freeze pretrained layers if required
        if pretrained and freeze_pretrained:
            self._freeze_layers()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the Image Encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Tuple of bottleneck tensor and list of skip connections.
        """
        down1 = self.down1(x)  # 3 -> 64
        down2 = self.down2(down1)  # 64 -> 128
        down3 = self.down3(down2)  # 128 -> 256
        down4 = self.down4(down3)  # 256 -> 512
        down5 = self.down5(down4)  # 512 -> 512
        bottleneck = self.bottle_neck(down5)  # 512 -> 1024

        return bottleneck, [down5, down4, down3, down2, down1]

    def _freeze_layers(self) -> None:
        """
        Freeze the pretrained layers.
        """
        for param in self.parameters():
            param.requires_grad = False


class ImageEncoderV0(nn.Module):
    """
    Image Encoder for semantic segmentation.
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.0) -> None:
        """
        Args:
            in_channels (int): Number of input channels. Default is 3.
            dropout (float): Dropout probability. Default is 0.5.
        """
        super().__init__()

        # Define the encoder (contracting path)
        self.inc = DoubleConv(in_channels, 64, dropout=dropout)
        self.down1 = DownSample(64, 128, dropout=dropout)
        self.down2 = DownSample(128, 256, dropout=dropout)
        self.down3 = DownSample(256, 512, dropout=dropout)

        # Define the bottleneck
        self.bottle_neck = DownSample(512, 1024, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the Image Encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Tuple of bottleneck tensor and list of skip connections.
        """
        down1 = self.inc(x)
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)

        bottleneck = self.bottle_neck(down4)

        return bottleneck, [down4, down3, down2, down1]
