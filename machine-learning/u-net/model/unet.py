import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from model.parts import DoubleConv, UpSample


class UNet(nn.Module):
    """
    U-Net architecture with VGG-16 backbone for encoder
    """

    def __init__(self):
        super().__init__()

        # Load VGG16 with weights
        vgg16 = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        features = list(vgg16.features.children())

        # Use VGG-16 layers for down-sampling
        self.enc1 = nn.Sequential(*features[:6])  # First block of VGG16
        self.enc2 = nn.Sequential(*features[6:13])  # Second block
        self.enc3 = nn.Sequential(*features[13:20])  # Third block
        self.enc4 = nn.Sequential(*features[20:27])  # Fourth block

        # Bottleneck
        self.bottle_neck = DoubleConv(512, 1024)

        # Up-sampling
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        # Output layer
        self.classifier = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down1 = self.enc1(x)
        down2 = self.enc2(down1)
        down3 = self.enc3(down2)
        down4 = self.enc4(down3)

        bottle_neck = self.bottle_neck(down4)

        up1 = self.up1(bottle_neck, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)

        return torch.sigmoid(self.classifier(up4))


if __name__ == '__main__':
    input_image = torch.rand((1, 3, 512, 512))
    model = UNet()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    outputs = model(input_image)
    print(f"Output shape: {outputs.shape}")
