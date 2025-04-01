import torch
import torch.nn as nn
from torchsummary import summary
from typing import Any, Dict

from libs.utils.mem_utils import estimate_max_batch_size
from unet.modeling.backbone.resnet34 import ResNet34
from unet.modeling.decoder.unet_decoder import UNetDecoder


class ResNet34Matte(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        encoder_config = config['encoder']
        decoder_config = config['decoder']

        self.encoder = ResNet34(
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
    image_size = 224
    input_size = (3, image_size, image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    config = {
        "encoder": {
            "pretrained": True,
        },
        "decoder": {
            "encoder_channels": [64, 64, 128, 256, 512],
            "decoder_channels": [256, 128, 64, 64],
            "final_channels": 64,
        }
    }

    # Print model summary
    model = ResNet34Matte(config).to(device)
    summary(model, input_size=input_size)

    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected shape: torch.Size([1, 1, 224, 224])


    def generate_resnetmatte_inputs(batch_size, input_size, device):
        return (torch.randn(batch_size, *input_size, device=device),)


    # Input resolutions to test
    for res in [224, 512, 1024]:
        input_size = (3, res, res)
        print(f"\nTesting input size: {input_size}...")

        # Estimate max batch size for this resolution
        max_batch_size = estimate_max_batch_size(
            model,
            input_size=input_size,
            max_memory_gb=10.0,
            safety_factor=0.9,
            input_generator_fn=generate_resnetmatte_inputs
        )

        print(f"Estimated max batch size for {res}x{res}: {max_batch_size}")
