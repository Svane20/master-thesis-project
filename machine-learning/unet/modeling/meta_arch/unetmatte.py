import torch
import torch.nn as nn

from typing import Any, Dict

from unet.modeling.backbone.resnet34 import ResNet34
from unet.modeling.decoder.capture_details import Detail_Capture


class UnetMatte(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        encoder_config = config['encoder']
        decoder_config = config['decoder']

        self.encoder = ResNet34(pretrained=encoder_config['pretrained'])
        self.decoder = Detail_Capture(
            in_channels=decoder_config['in_channels'],
            convstream_out=decoder_config['convstream_out'],
            fusion_out=decoder_config['fusion_out'],
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        return self.decoder(bottleneck, x)


if __name__ == '__main__':
    image_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    config = {
        "encoder": {
            "pretrained": True,
        },
        "decoder": {
            "in_channels": 512,
            "convstream_out": [48, 96, 192, 384],
            "fusion_out": [256, 128, 64, 32, 16],
        }
    }

    model = UnetMatte(config).to(device)
    output = model(dummy_input)
    print("Output shape:", output.shape) # Expected shape: torch.Size([1, 1, 512, 512])
