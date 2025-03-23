import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34V0(nn.Module):
    """
    ResNet-34 model modified for UNet that now accepts a 4-channel input (RGB + alpha).
    """

    def __init__(self, image_channels: int = 3, pretrained: bool = True):
        """
        Args:
            image_channels (int): number of channels in the input image.
            pretrained (bool): If True, returns a model pretrained on ImageNet
        """
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        # Replace the original conv1 (3 channels) with one that accepts 4 channels.
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            if image_channels == 3:
                # If input is RGB only, copy weights directly.
                self.conv1.weight.data = resnet.conv1.weight.data.clone()
            elif image_channels == 4:
                # Copy pretrained weights for the first three channels.
                self.conv1.weight.data[:, :3] = resnet.conv1.weight.data.clone()
                # Initialize the alpha channel (e.g., by copying the weights from the first channel).
                self.conv1.weight.data[:, 3] = resnet.conv1.weight.data[:, 0].clone()
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.enc1 = nn.Sequential(self.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        return self.enc5(e4)


if __name__ == "__main__":
    image_channels = 3
    image_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, image_channels, image_size, image_size).to(device)

    model = ResNet34V0(pretrained=True, image_channels=image_channels).to(device)
    print(model)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected shape: torch.Size([1, 512, 16, 16])
