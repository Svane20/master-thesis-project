import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from typing import Tuple, List


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained (bool): Use pretrained weights. Default is True.
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
