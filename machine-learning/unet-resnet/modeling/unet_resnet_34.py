import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ----------------------------
# Encoder: ResNet-34 Based
# ----------------------------
class ResNetEncoder(nn.Module):
    """
    Encoder using a pretrained ResNet-34. Extracts multi-scale features and returns the bottleneck
    and skip connections.
    """

    def __init__(self, pretrained: bool = True, freeze_pretrained: bool = False):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # Output: 64 channels, H/2
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # Output: 64 channels, H/4
        self.encoder2 = resnet.layer2  # Output: 128 channels, H/8
        self.encoder3 = resnet.layer3  # Output: 256 channels, H/16
        self.encoder4 = resnet.layer4  # Output: 512 channels, H/32

        # Freeze pretrained layers if required
        if pretrained and freeze_pretrained:
            self._freeze_layers()

    def forward(self, x: torch.Tensor):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Return bottleneck (e4) and skip connections in order: [e3, e2, e1, e0]
        return e4, [e3, e2, e1, e0]

    def _freeze_layers(self) -> None:
        """
        Freeze the pretrained layers.
        """
        for param in self.parameters():
            param.requires_grad = False


# ----------------------------
# Decoder: Upsampling with Skip Connections
# ----------------------------
class ResNetDecoder(nn.Module):
    """
    Decoder that upsamples the bottleneck features and fuses them with skip connections
    to produce the final mask.
    """

    def __init__(self):
        super().__init__()
        # Upsampling layers: using ConvTranspose2d followed by a double-conv block
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)  # Concatenation: 256 (upsampled) + 256 (e3)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)  # 128 + 128 (e2)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)  # 64 + 64 (e1)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)  # 64 + 64 (e0)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, bottleneck: torch.Tensor, skip_connections: list, target_size: torch.Size) -> torch.Tensor:
        # skip_connections order: [e3, e2, e1, e0]
        d4 = self.up4(bottleneck)  # Upsample from e4 to H/16
        d4 = torch.cat([d4, skip_connections[0]], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)  # Upsample to H/8
        d3 = torch.cat([d3, skip_connections[1]], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # Upsample to H/4
        d2 = torch.cat([d2, skip_connections[2]], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # Upsample to H/2
        d1 = torch.cat([d1, skip_connections[3]], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


# ----------------------------
# UNet: Combines Encoder and Decoder
# ----------------------------
class UNetResNet34(nn.Module):
    """
    UNet model that combines the ResNet-based encoder and the decoder.
    """

    def __init__(self, pretrained: bool = True, freeze_pretrained: bool = False):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained, freeze_pretrained=freeze_pretrained)
        self.decoder = ResNetDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder(x)
        target_size = x.shape[2:]
        return self.decoder(bottleneck, skips, target_size)


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetResNet34(pretrained=True).to(device)

    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    output = model(dummy_input)
    print("Output shape:", output.shape)
