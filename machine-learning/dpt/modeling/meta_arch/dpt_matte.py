import torch
from torch import nn
from torchsummary import summary
from transformers import DPTImageProcessor
import numpy as np
from typing import Any, Dict

from dpt.modeling.backbone.dpt_swinv2_tiny_256 import DPTEncoder
from dpt.modeling.decoder.capture_details import Detail_Capture
from libs.utils.mem_utils import estimate_max_batch_size


class DPTMatte(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        encoder_config = config['encoder']
        decoder_config = config['decoder']

        self.encoder = DPTEncoder(model_name=encoder_config['model_name'])
        self.decoder = Detail_Capture(
            in_channels=decoder_config['in_channels'],
            convstream_out=decoder_config['convstream_out'],
            fusion_out=decoder_config['fusion_out'],
        )

    def forward(self, pixel_values: torch.Tensor, image_rgb: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encoder(pixel_values)
        return self.decoder(bottleneck, image_rgb)


if __name__ == "__main__":
    image_size = 512
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

    model = DPTMatte(config).to(device)
    summary(model, input_size=input_size)
    processor = DPTImageProcessor.from_pretrained(config['encoder']['model_name'])

    image_np = np.random.rand(image_size, image_size, 3).astype(np.float32)  # dummy input
    inputs = processor(images=image_np, return_tensors="pt", do_rescale=False)
    pixel_values = inputs["pixel_values"].to(device)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    output = model(pixel_values, image_tensor)
    print("Output shape:", output.shape)  # Expected: [1, 1, 512, 512]


    def generate_dptmatte_inputs(batch_size, input_size, device):
        C, H, W = input_size
        pixel_values = torch.randn(batch_size, C, H, W, device=device)
        image_rgb = torch.randn(batch_size, C, H, W, device=device)
        return pixel_values, image_rgb


    max_batch_size = estimate_max_batch_size(
        model,
        input_size=input_size,
        max_memory_gb=10.0,
        safety_factor=0.9,
        input_generator_fn=generate_dptmatte_inputs,
    )
    print("Estimated max batch size:", max_batch_size)
