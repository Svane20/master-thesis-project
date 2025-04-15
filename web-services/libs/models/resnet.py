import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import List, Tuple, Any, Dict


class ResNet50Matte(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        encoder_config = config['encoder']
        decoder_config = config['decoder']

        self.encoder = ResNet50(
            pretrained=encoder_config['pretrained'],
        )
        self.decoder = UNetDecoder(
            encoder_channels=decoder_config['encoder_channels'],
            decoder_channels=decoder_config['decoder_channels'],
            final_channels=decoder_config['final_channels'],
        )

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)
        return self.decoder(bottleneck, skip_connections, x)


class ResNet50(nn.Module):
    """
    ResNet-50 model modified for UNet.
    """

    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained (bool): If True, returns a model pretrained on ImageNet
        """
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottleneck = self.enc5(e4)

        return bottleneck, [e1, e2, e3, e4]


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
