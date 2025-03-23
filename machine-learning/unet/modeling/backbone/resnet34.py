import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34(nn.Module):
    """
    ResNet-34 model modified for the UNet architecture.
    """
    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained (bool): If True, use the pretrained weights.
        """
        super().__init__()

        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
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
    image_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    model = ResNet34().to(device)
    print(model)

    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected shape: torch.Size([1, 512, 16, 16])
