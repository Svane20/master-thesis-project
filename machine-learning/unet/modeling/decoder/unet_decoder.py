import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from typing import List


class AttentionFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        attn_weight = self.attn(feat)
        return feat * attn_weight


class RefinementBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class UNetDecoder(nn.Module):
    """
    SOTA decoder for image matting using a ResNet-34 encoder.

    Expects encoder outputs:
      - bottleneck: The deepest feature map (e.g., shape: [B, 512, H/32, W/32])
      - skip_connections: List of 4 feature maps from the encoder in order
                          [e0, e1, e2, e3] from highest to lowest resolution.
    """

    def __init__(self, encoder_channels: List[int], decoder_channels: List[int], final_channels: int = 64):
        super().__init__()

        self.up4 = nn.ConvTranspose2d(encoder_channels[-1], decoder_channels[0], kernel_size=2, stride=2)
        self.fuse4 = AttentionFusionBlock(decoder_channels[0] + encoder_channels[-2], decoder_channels[0])

        self.up3 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2)
        self.fuse3 = AttentionFusionBlock(decoder_channels[1] + encoder_channels[-3], decoder_channels[1])

        self.up2 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        self.fuse2 = AttentionFusionBlock(decoder_channels[2] + encoder_channels[-4], decoder_channels[2])

        self.up1 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        self.fuse1 = AttentionFusionBlock(decoder_channels[3] + encoder_channels[0], final_channels)

        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse0 = AttentionFusionBlock(final_channels, final_channels)

        self.detail_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.detail_fuse = nn.Conv2d(final_channels + 32, final_channels, kernel_size=3, padding=1)

        self.refinement = RefinementBlock(final_channels)
        self.final_conv = nn.Conv2d(final_channels, 1, kernel_size=1)

    def forward(
            self,
            bottleneck: torch.Tensor,
            skip_connections: List[torch.Tensor],
            image: torch.Tensor
    ) -> torch.Tensor:
        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, skip_connections[3]], dim=1)
        d4 = self.fuse4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, skip_connections[2]], dim=1)
        d3 = self.fuse3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, skip_connections[1]], dim=1)
        d2 = self.fuse2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, skip_connections[0]], dim=1)
        d1 = self.fuse1(d1)

        d0 = self.up0(d1)
        d0 = self.fuse0(d0)

        detail = self.detail_branch(image)
        if detail.shape[-2:] != d0.shape[-2:]:
            detail = F.interpolate(detail, size=d0.shape[-2:], mode='bilinear', align_corners=False)

        d0 = torch.cat([d0, detail], dim=1)
        d0 = self.detail_fuse(d0)
        d0 = self.refinement(d0)

        return torch.sigmoid(self.final_conv(d0))


if __name__ == "__main__":
    image_size = 512
    feature_size = 16  # Size of the bottleneck feature map.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    bottleneck = torch.rand(1, 2048, feature_size, feature_size).to(device)
    skip_connections = [
        torch.rand(1, 64, 256, 256).to(device),
        torch.rand(1, 256, 128, 128).to(device),
        torch.rand(1, 512, 64, 64).to(device),
        torch.rand(1, 1024, 32, 32).to(device),
    ]

    model = UNetDecoder(
        encoder_channels=[64, 256, 512, 1024, 2048],
        decoder_channels=[512, 256, 128, 64],
        final_channels=64
    ).to(device)
    summary(model, input_size=(3, image_size, image_size))

    with torch.no_grad():
        output = model(bottleneck, skip_connections, dummy_input)
    print(f"Output shape: {output.shape}")  # Expected shape: e.g., torch.Size([1, 1, 512, 512])
