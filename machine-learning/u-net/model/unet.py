import torch
import torch.nn as nn

from model.model_builder import DoubleConv, UpSample, DownSample


class UNet(nn.Module):
    """
    UNet model for semantic segmentation with arbitrary input sizes.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        out_channels (int): Number of output channels. Default is 1.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()

        # Define the encoder (contracting path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)

        # Define the bottleneck
        self.bottle_neck = DownSample(512, 1024)

        # Define the decoder (expansive path)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

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


if __name__ == '__main__':
    input_image = torch.rand((1, 3, 512, 512))
    model = UNet(in_channels=3, out_channels=1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") # Total parameters: 31,036,672

    outputs = model(input_image)
    print(f"Output shape: {outputs.shape}") # Output shape: torch.Size([1, 1, 512, 512])
