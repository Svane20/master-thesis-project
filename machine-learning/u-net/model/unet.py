import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

from model.model_builder import DoubleConv, UpSample, DownSample


class UNetV1VGG(nn.Module):
    """
    UNet model for semantic segmentation with arbitrary input sizes.

    Args:
        out_channels (int): Number of output channels. Default is 1.
        dropout (float): Dropout probability. Default is 0.5.
        pretrained (bool): Use pretrained VGG16 with batch normalization. Default is True.
    """
    def __init__(self, out_channels: int = 1, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        self.encoder = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT if pretrained else None).features
        self.down1 = nn.Sequential(*self.encoder[:6])
        self.down2 = nn.Sequential(*self.encoder[6:13])
        self.down3 = nn.Sequential(*self.encoder[13:20])
        self.down4 = nn.Sequential(*self.encoder[20:27])
        self.down5 = nn.Sequential(*self.encoder[27:34])

        self.bottle_neck = DoubleConv(512, 1024, dropout=dropout)

        self.up1 = UpSample(1024, 512, dropout=dropout)
        self.up2 = UpSample(512, 256, dropout=dropout)
        self.up3 = UpSample(256, 128, dropout=dropout)
        self.up4 = UpSample(128, 64, dropout=dropout)

        self.classifier = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        bottle_neck = self.bottle_neck(down5)

        up1 = self.up1(bottle_neck, down5)
        up2 = self.up2(up1, down4)
        up3 = self.up3(up2, down3)
        up4 = self.up4(up3, down2)

        return self.classifier(up4)


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
        self.up1 = UpSample(1024, 512, dropout=dropout)
        self.up2 = UpSample(512, 256, dropout=dropout)
        self.up3 = UpSample(256, 128, dropout=dropout)
        self.up4 = UpSample(128, 64, dropout=dropout)

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
    # Create a dummy input tensor with batch size 1 and image size 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    model = UNetV0()
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    assert output.shape == torch.Size([1, 1, 224, 224]), "Output shape is incorrect"
