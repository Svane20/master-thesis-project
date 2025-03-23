import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AttentionFusionBlock(nn.Module):
    """
    Fusion block that uses convolution followed by an attention mechanism to
    refine the concatenated features from the upsampled decoder and encoder skip.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Simple channel attention
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        attn_weight = self.attn(feat)
        return feat * attn_weight


class UNetDecoder(nn.Module):
    """
    SOTA decoder for image matting using a ResNet-34 encoder.

    Expects encoder outputs:
      - bottleneck: The deepest feature map (e.g., shape: [B, 512, H/32, W/32])
      - skip_connections: List of 4 feature maps from the encoder in order
                          [e0, e1, e2, e3] from highest to lowest resolution.
    """

    def __init__(
            self,
            in_channels: int = 512,
            encoder_channels: List[int] = [64, 64, 128, 256, 512],
            decoder_channels: List[int] = [256, 128, 64, 64],
            final_channels: int = 64
    ):
        super().__init__()
        self.in_channels = in_channels

        # Upsampling and fusion blocks.
        self.up4 = nn.ConvTranspose2d(encoder_channels[-1], decoder_channels[0], kernel_size=2, stride=2)
        self.fuse4 = AttentionFusionBlock(decoder_channels[0] + encoder_channels[-2], decoder_channels[0])

        self.up3 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2)
        self.fuse3 = AttentionFusionBlock(decoder_channels[1] + encoder_channels[-3], decoder_channels[1])

        self.up2 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        self.fuse2 = AttentionFusionBlock(decoder_channels[2] + encoder_channels[-4], decoder_channels[2])

        self.up1 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        self.fuse1 = AttentionFusionBlock(decoder_channels[3] + encoder_channels[0], final_channels)

        self.up0 = nn.ConvTranspose2d(final_channels, final_channels, kernel_size=2, stride=2)
        self.fuse0 = AttentionFusionBlock(final_channels, final_channels)

        self.final_conv = nn.Conv2d(final_channels, 1, kernel_size=1)

        # Detail branch: additional branch to refine high-resolution details.
        self.detail_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Fusion layer for detail branch and high-resolution decoder output.
        self.detail_fuse = nn.Conv2d(final_channels + 32, final_channels, kernel_size=3, padding=1)

    def forward(
            self,
            bottleneck: torch.Tensor,
            skip_connections: List[torch.Tensor],
            image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            bottleneck: Encoder bottleneck feature map.
            skip_connections: List of encoder skip feature maps.
                              Expected order: [e0, e1, e2, e3] from highest to lowest resolution.
            image: The original input image (for the optional detail branch).
        """
        # Stage 1: Upsample bottleneck and fuse with the deepest skip (e3)
        d4 = self.up4(bottleneck)  # Upsampled from 512 channels to decoder_channels[0]
        d4 = torch.cat([d4, skip_connections[3]], dim=1)  # skip_connections[3]: e3 (e.g., 256 channels)
        d4 = self.fuse4(d4)

        # Stage 2: Fuse with e2
        d3 = self.up3(d4)
        d3 = torch.cat([d3, skip_connections[2]], dim=1)  # skip_connections[2]: e2 (e.g., 128 channels)
        d3 = self.fuse3(d3)

        # Stage 3: Fuse with e1
        d2 = self.up2(d3)
        d2 = torch.cat([d2, skip_connections[1]], dim=1)  # skip_connections[1]: e1 (e.g., 64 channels)
        d2 = self.fuse2(d2)

        # Stage 4: Fuse with e0 (highest resolution skip)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, skip_connections[0]], dim=1)  # skip_connections[0]: e0 (e.g., 64 channels)
        d1 = self.fuse1(d1)

        # 5) Final upsampling from H/2 -> H/1
        d0 = self.up0(d1)
        d0 = self.fuse0(d0)

        # Detail Branch: refine high-resolution features with the original image.
        detail = self.detail_branch(image)
        if detail.shape[-2:] != d0.shape[-2:]:
            detail = F.interpolate(detail, size=d0.shape[-2:], mode='bilinear', align_corners=False)

        # Fuse the detail branch output with final decoder feature
        d0 = torch.cat([d0, detail], dim=1)
        d0 = self.detail_fuse(d0)

        return torch.sigmoid(self.final_conv(d0))


if __name__ == "__main__":
    image_size = 512
    feature_size = 16  # Size of the bottleneck feature map.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    bottleneck = torch.rand(1, image_size, feature_size, feature_size).to(device)
    skip_connections = [
        torch.rand(1, 64, 256, 256).to(device),
        torch.rand(1, 64, 128, 128).to(device),
        torch.rand(1, 128, 64, 64).to(device),
        torch.rand(1, 256, 32, 32).to(device),
    ]

    model = UNetDecoder(
        in_channels=image_size,
        encoder_channels=[64, 64, 128, 256, 512],
        decoder_channels=[256, 128, 64, 64],
        final_channels=64
    ).to(device)
    print(model)
    output = model(bottleneck, skip_connections, dummy_input)
    print(f"Output shape: {output.shape}")  # Expected shape: e.g., torch.Size([1, 1, 512, 512])
