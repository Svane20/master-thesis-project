import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from model.model_builder import DoubleConv, UpSample, DownSample


class UNetV1VGG(nn.Module):
    """
    UNet model for semantic segmentation with arbitrary input sizes.

    Args:
        out_channels (int): Number of output channels. Default is 1.
        pretrained (bool): Use pretrained VGG16 with batch normalization. Default is True.
    """

    def __init__(self, out_channels: int = 1, pretrained: bool = True):
        super().__init__()

        # Encoder with VGG16
        encoder = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT if pretrained else None).features
        self.down1 = nn.Sequential(*encoder[:6])  # 3 -> 64
        self.down2 = nn.Sequential(*encoder[6:13])  # 64 -> 128
        self.down3 = nn.Sequential(*encoder[13:20])  # 128 -> 256
        self.down4 = nn.Sequential(*encoder[20:27])  # 256 -> 512
        self.down5 = nn.Sequential(*encoder[27:34])  # 512 -> 512

        # Bottleneck
        self.bottle_neck = nn.Sequential(
            *encoder[34:],  # 512 -> 512
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Decoder with skip connections
        self.up1 = UpSample(in_channels=1024, skip_channels=512, out_channels=512)  # 1024 -> 512
        self.up2 = UpSample(in_channels=512, skip_channels=512, out_channels=256)  # 512 -> 256
        self.up3 = UpSample(in_channels=256, skip_channels=256, out_channels=128)  # 256 -> 128
        self.up4 = UpSample(in_channels=128, skip_channels=128, out_channels=64)  # 128 -> 64
        self.up5 = UpSample(in_channels=64, skip_channels=64, out_channels=32)  # 64 -> 32

        # Classifier
        self.classifier = nn.Conv2d(32, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        down1 = self.down1(x)  # 3 -> 64
        down2 = self.down2(down1)  # 64 -> 128
        down3 = self.down3(down2)  # 128 -> 256
        down4 = self.down4(down3)  # 256 -> 512
        down5 = self.down5(down4)  # 512 -> 512

        # Bottleneck
        bottle_neck = self.bottle_neck(down5)  # 512 -> 1024

        # Decoder with skip connections
        up1 = self.up1(bottle_neck, down5)  # 1024 + 512 -> 512
        up2 = self.up2(up1, down4)  # 512 + 512 -> 256
        up3 = self.up3(up2, down3)  # 256 + 256 -> 128
        up4 = self.up4(up3, down2)  # 128 + 128 -> 64
        up5 = self.up5(up4, down1)  # 64 + 64 -> 32


        # Classifier
        return self.classifier(up5)  # 32 -> out_channels


class UNetV0(nn.Module):
    """
    UNet model for semantic segmentation with arbitrary input sizes.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        out_channels (int): Number of output channels. Default is 1.
        dropout (float): Dropout probability. Default is 0.5.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, dropout: float = 0.5) -> None:
        super().__init__()

        # Define the encoder (contracting path)
        self.inc = DoubleConv(in_channels, 64, dropout=dropout)
        self.down1 = DownSample(64, 128, dropout=dropout)
        self.down2 = DownSample(128, 256, dropout=dropout)
        self.down3 = DownSample(256, 512, dropout=dropout)

        # Define the bottleneck
        self.bottle_neck = DownSample(512, 1024, dropout=dropout)

        # Define the decoder (expansive path)
        self.up1 = UpSample(in_channels=1024, skip_channels=512, out_channels=512, dropout=dropout)
        self.up2 = UpSample(in_channels=512, skip_channels=256, out_channels=256, dropout=dropout)
        self.up3 = UpSample(in_channels=256, skip_channels=128, out_channels=128, dropout=dropout)
        self.up4 = UpSample(in_channels=128, skip_channels=64, out_channels=64, dropout=dropout)

        # Define the classifier
        self.classifier = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        down1 = self.inc(x)
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)

        # Bottleneck
        bottle_neck = self.bottle_neck(down4)

        # Decoder with skip connections
        up1 = self.up1(bottle_neck, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)

        return self.classifier(up4)


if __name__ == "__main__":
    baseline = UNetV0(in_channels=3, out_channels=1)
    baseline_total_params = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline Model Total Params: {baseline_total_params:,}".replace(',', '.'))

    vgg = UNetV1VGG(out_channels=1)
    vgg_total_params = sum(p.numel() for p in vgg.parameters())
    print(f"VGG Model Total Params: {vgg_total_params:,}".replace(',', '.'))
    print('\n')


    for resolution in [(128, 128), (256, 256), (512, 512)]:  # Test different input sizes
        dummy_input = torch.randn(1, 3, *resolution)

        # Test the baseline model
        baseline = UNetV0(in_channels=3, out_channels=1)
        output = baseline(dummy_input)
        print(f"Baseline Input shape: {dummy_input.shape}")
        print(f"Baseline Output shape: {output.shape}")

        # Test the VGG model
        vgg = UNetV1VGG(out_channels=1)
        output = vgg(dummy_input)
        print(f"VGG Input shape: {dummy_input.shape}")
        print(f"VGG Output shape: {output.shape}\n")

