import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetModel, ResNetConfig

from typing import List, Dict, Any


# ----------------------------
# Encoder: Hugging Face ResNet-50 Based
# ----------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        model_name = config["model"]["pretrained_model_name_or_path"]
        self.config = ResNetConfig.from_pretrained(
            **config["model"]
        )

        pretrained = config["pretrained"]
        if pretrained:
            self.model = ResNetModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = ResNetModel(self.config)

        if pretrained and config["freeze_pretrained"]:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        outputs = self.model(pixel_values=x)
        hidden_states = outputs.hidden_states  # Tuple of 5 tensors

        # Use the last hidden state as bottleneck and the rest as skip connections:
        bottleneck = hidden_states[4]  # 2048 channels from Stage 3

        skip_connections = [hidden_states[3],  # 1024 channels from Stage 2
                            hidden_states[2],  # 512 channels from Stage 1
                            hidden_states[1],  # 256 channels from Stage 0
                            hidden_states[0]]  # 64 channels from the embedder

        return bottleneck, skip_connections


# ----------------------------
# Decoder: Adjusted for ResNet-50 Channel Dimensions
# ----------------------------
class ResNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # For ResNet-50, channels are: e4=2048, e3=1024, e2=512, e1=256, e0=64.
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024 + 1024, 1024)  # 1024 (upsampled) + 1024 (e3)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512 + 512, 512)  # 512 (upsampled) + 512 (e2)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256 + 256, 256)  # 256 (upsampled) + 256 (e1)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64 + 64, 64)  # 64 (upsampled) + 64 (e0)

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

    def forward(self, bottleneck: torch.Tensor, skip_connections: List[torch.Tensor],
                target_size: List[int]) -> torch.Tensor:
        d4 = self.up4(bottleneck)  # Upsample from 2048 -> 1024 channels
        d4 = torch.cat([d4, skip_connections[0]], dim=1)  # Concatenate with e3 (1024)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)  # Upsample from 1024 -> 512 channels
        d3 = torch.cat([d3, skip_connections[1]], dim=1)  # Concatenate with e2 (512)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # Upsample from 512 -> 256 channels
        d2 = torch.cat([d2, skip_connections[2]], dim=1)  # Concatenate with e1 (256)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # Upsample from 256 -> 64 channels
        # Interpolate skip_connections[3] from 128x128 to match d1's 256x256
        skip0_resized = F.interpolate(skip_connections[3], size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, skip0_resized], dim=1)  # Concatenate with resized e0 (64)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


# ----------------------------
# UNet: Combining the Encoder and Decoder
# ----------------------------
class UNet(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.encoder = ResNetEncoder(config["encoder"])
        self.decoder = ResNetDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder(x)
        target_size = x.shape[2:]  # (height, width)
        return self.decoder(bottleneck, skips, target_size)


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    config = {
        "encoder": {
            "model": {
                "pretrained_model_name_or_path": "microsoft/resnet-50",
                "output_hidden_states": True,
                "out_indices": [0, 1, 2, 3, 4]
            },
            "pretrained": True,
            "freeze_pretrained": False
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(config=config).to(device)

    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    output = model(dummy_input)
    print("Output shape:", output.shape)
