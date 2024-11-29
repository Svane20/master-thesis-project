import torch
import torch.nn as nn

from typing import Optional

class AttentionBlock(nn.Module):
    """
    Attention block to learn the importance of each feature map

    Args:
        in_channels (int): Number of input channels
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.conv(x))


class DecoderBlock(nn.Module):
    """
    Decoder block for the UNET-R architecture

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        skip_channels (int): Number of skip channels to concatenate. If None, no skip connection is used
    """

    def __init__(self, in_channels: int, out_channels: int, skip_channels: Optional[int] = None):
        super().__init__()
        # Upsampling using interpolation
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_upsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # If skip_channels is provided, reduce channels if they don't match
        if skip_channels:
            self.reduce_channels = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
            conv_in_channels = out_channels * 2
        else:
            self.reduce_channels = None
            conv_in_channels = out_channels

        self.conv1 = nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        # Attention Block
        self.attention = AttentionBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upsample the input tensor
        x = self.upsample(x)
        x = self.conv_upsample(x)

        if skip is not None:
            # Reduce skip channels if necessary
            if self.reduce_channels:
                skip = self.reduce_channels(skip)

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension

        # Convolutional block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Convolutional block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        # Apply attention
        return self.attention(x)