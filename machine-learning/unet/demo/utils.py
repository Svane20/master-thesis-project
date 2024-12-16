import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution Block with Batch Normalization and ReLU Activation
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float): Dropout probability. Default is 0.0.
        """
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        ])

        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownSample(nn.Module):
    """
    Down Sampling Layer with Max Pooling and Double Convolution
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float): Dropout probability. Default is 0.0.
        """
        super().__init__()

        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),  # Handle arbitrary input sizes
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool_conv(x)


class UpSample(nn.Module):
    """
    Up Sampling Layer with Double Convolution and Transposed Convolution
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        """
        Args:
            in_channels (int): Number of input channels.
            skip_channels (int): Number of input channels from the skip connection.
            out_channels (int): Number of output channels.
            dropout (float): Dropout probability. Default is 0.0.
        """
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Ensure x1 and x2 have the same spatial dimensions (input is NCHW)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
