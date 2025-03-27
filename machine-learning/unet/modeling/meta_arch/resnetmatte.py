import torch
import torch.nn as nn
from typing import Any, Dict
from torchsummary import summary

from unet.modeling.backbone.resnet50 import ResNet50
from unet.modeling.decoder.unet_decoder import UNetDecoder


class ResNetMatteV0(nn.Module):
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


if __name__ == '__main__':
    image_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    config = {
        "encoder": {
            "pretrained": True,
        },
        "decoder": {
            "encoder_channels": [64, 256, 512, 1024, 2048],
            "decoder_channels": [512, 256, 128, 64],
            "final_channels": 64,
        }
    }

    # Print model summary
    model = ResNetMatteV0(config).to(device)
    summary(model, input_size=(3, image_size, image_size))

    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected shape: torch.Size([1, 1, 512, 512])
