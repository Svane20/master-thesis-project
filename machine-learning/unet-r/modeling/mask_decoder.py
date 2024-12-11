import torch
import torch.nn as nn


class MaskDecoder(nn.Module):
    """
    Mask decoder that upsamples the encoder features back to the original image size
    """

    def __init__(
            self,
            in_channels: int = 256,
            num_intermediate_channels: int = 256,
            final_size: int = 1024,
            patch_size: int = 16
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels (matches encoder output channels).
            num_intermediate_channels (int): Number of channels to use in intermediate steps.
            final_size (int): Target spatial resolution of the output (e.g., 1024).
            patch_size (int): Patch size used in encoder.
        """
        super().__init__()

        self.stages = nn.ModuleList()
        current_channels = in_channels
        current_size = final_size // patch_size  # = 64

        # Stage 1: 64 -> 128
        self.stages.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, num_intermediate_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        )
        current_channels = num_intermediate_channels
        current_size *= 2  # 128

        # Stage 2: 128 -> 256
        self.stages.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, num_intermediate_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        )
        current_channels = num_intermediate_channels // 2
        current_size *= 2  # 256

        # Stage 3: 256 -> 512
        self.stages.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, num_intermediate_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        )
        current_channels = num_intermediate_channels // 4
        current_size *= 2  # 512

        # Stage 4: 512 -> 1024
        self.stages.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, num_intermediate_channels // 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        )
        current_channels = num_intermediate_channels // 8
        current_size *= 2  # 1024

        # Final conv to single channel
        self.final_conv = nn.Conv2d(current_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)

        return torch.sigmoid(self.final_conv(x))
