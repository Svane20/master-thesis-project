import torch
from torch import nn
from torchsummary import summary
from transformers import DPTImageProcessor
import numpy as np
from typing import Any, Dict

from dpt.modeling.backbone.dpt_swinv2_tiny_256 import DPTSwinV2Tiny256Encoder
from dpt.modeling.decoder.capture_details import Detail_Capture
from libs.utils.mem_utils import estimate_max_batch_size


class DPTSwinV2Tiny256Matte(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        encoder_config = config['encoder']
        decoder_config = config['decoder']

        self.encoder = DPTSwinV2Tiny256Encoder(model_name=encoder_config['model_name'])
        self.decoder = Detail_Capture(
            in_channels=decoder_config['in_channels'],
            convstream_out=decoder_config['convstream_out'],
            fusion_out=decoder_config['fusion_out'],
        )

        self.processor = DPTImageProcessor.from_pretrained(encoder_config['model_name'], do_rescale=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encoder(x)
        return self.decoder(bottleneck, x)


if __name__ == "__main__":
    image_size = 256
    input_size = (3, image_size, image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "encoder": {
            "model_name": "Intel/dpt-swinv2-tiny-256",
        },
        "decoder": {
            "in_channels": 768,
            "convstream_out": [48, 96, 192, 384, 512],
            "fusion_out": [48, 96, 192, 384, 512],
        }
    }

    model = DPTSwinV2Tiny256Matte(config).to(device)
    summary(model, input_size=input_size)

    # Dummy input image in [0, 1] range (normalized RGB)
    image_np = np.random.rand(image_size, image_size, 3).astype(np.float32)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        output = model(image_tensor)
    print("Output shape:", output.shape)  # Expected: [1, 1, 256, 256]


    def generate_dptmatte_inputs(batch_size, input_size, device):
        return (torch.randn(batch_size, *input_size, device=device),)


    # Input resolutions to test
    for res in [256, 512, 1024]:
        input_size = (3, res, res)
        print(f"\nTesting input size: {input_size}...")

        # Estimate max batch size for this resolution
        max_batch_size = estimate_max_batch_size(
            model,
            input_size=input_size,
            max_memory_gb=10.0,
            safety_factor=0.9,
            input_generator_fn=generate_dptmatte_inputs
        )

        print(f"Estimated max batch size for {res}x{res}: {max_batch_size}")
