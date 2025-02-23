import torch
import torch.nn as nn

from typing import List

from unet.modeling.utils import UpSample


class MaskDecoder(nn.Module):
    """
    Mask Decoder to convert the features to the mask.
    """

    def __init__(self, out_channels: int = 1, dropout: float = 0.0):
        """
        Args:
            out_channels (int): Number of output channels. Default is 1.
            dropout (float): Dropout probability. Default is 0.0.
        """
        super().__init__()

        self.up1 = UpSample(in_channels=1024, skip_channels=512, out_channels=512, dropout=dropout)  # 1024 -> 512
        self.up2 = UpSample(in_channels=512, skip_channels=512, out_channels=256, dropout=dropout)  # 512 -> 256
        self.up3 = UpSample(in_channels=256, skip_channels=256, out_channels=128, dropout=dropout)  # 256 -> 128
        self.up4 = UpSample(in_channels=128, skip_channels=128, out_channels=64, dropout=dropout)  # 128 -> 64
        self.up5 = UpSample(in_channels=64, skip_channels=64, out_channels=32, dropout=dropout)  # 64 -> 32

        self.head = nn.Conv2d(32, out_channels, kernel_size=1, bias=False)

    def forward(self, bottle_neck: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        down5, down4, down3, down2, down1 = skip_connections

        up1 = self.up1(bottle_neck, down5)  # 1024 + 512 -> 512
        up2 = self.up2(up1, down4)  # 512 + 512 -> 256
        up3 = self.up3(up2, down3)  # 256 + 256 -> 128
        up4 = self.up4(up3, down2)  # 128 + 128 -> 64
        up5 = self.up5(up4, down1)  # 64 + 64 -> 32

        return torch.sigmoid(self.head(up5))  # 32 -> out_channels


class MaskDecoderV0(nn.Module):
    """
    Mask Decoder to convert the features to the mask.
    """

    def __init__(self, out_channels: int = 1, dropout: float = 0.0):
        """
        Args:
            out_channels (int): Number of output channels. Default is 1.
            dropout (float): Dropout probability. Default is 0.0.
        """
        super().__init__()

        self.up1 = UpSample(in_channels=1024, skip_channels=512, out_channels=512, dropout=dropout)
        self.up2 = UpSample(in_channels=512, skip_channels=256, out_channels=256, dropout=dropout)
        self.up3 = UpSample(in_channels=256, skip_channels=128, out_channels=128, dropout=dropout)
        self.up4 = UpSample(in_channels=128, skip_channels=64, out_channels=64, dropout=dropout)

        self.head = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

    def forward(self, bottle_neck: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        down4, down3, down2, down1 = skip_connections

        up1 = self.up1(bottle_neck, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)

        return self.head(up4)
