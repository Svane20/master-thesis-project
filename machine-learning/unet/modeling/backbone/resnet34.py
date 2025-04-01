import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torchsummary import summary
from typing import List, Tuple


class ResNet34(nn.Module):
    """
    ResNet-34 model modified for UNet
    """

    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained (bool): If True, returns a model pretrained on ImageNet
        """
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

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


if __name__ == "__main__":
    image_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    model = ResNet34(pretrained=True).to(device)
    summary(model, input_size=(3, image_size, image_size))

    with torch.no_grad():
        bottleneck, skip_connections = model(dummy_input)

    for index, skip_connection in enumerate(skip_connections):
        print(f"Skip connection {index + 1} shape: {skip_connection.shape}")
        """
        torch.Size([1, 64, 112, 112])
        torch.Size([1, 64, 56, 56])
        torch.Size([1, 128, 28, 28])
        torch.Size([1, 256, 14, 14])
        """
    print(f"Bottleneck shape: {bottleneck.shape}")  # Excepted shape: torch.Size([1, 512, 7, 7])
