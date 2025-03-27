import torch
from torch import nn
from transformers import DPTImageProcessor
import numpy as np

from dpt.modeling.backbone.dpt_swinv2_tiny_256 import DPTEncoder
from dpt.modeling.decoder.capture_details import Detail_Capture

class DPTMatteModel(nn.Module):
    def __init__(self, encoder_name="Intel/dpt-swinv2-tiny-256"):
        super().__init__()
        self.encoder = DPTEncoder(encoder_name)
        self.decoder = Detail_Capture(
            in_channels=768,  # SwinV2 output channels
            convstream_out=[48, 96, 192, 384, 512]
        )

    def forward(self, pixel_values: torch.Tensor, image_rgb: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encoder(pixel_values)
        return self.decoder(bottleneck, image_rgb)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DPTMatteModel().to(device)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-swinv2-tiny-256")

    image_np = np.random.rand(512, 512, 3).astype(np.float32)  # dummy input
    inputs = processor(images=image_np, return_tensors="pt", do_rescale=False)
    pixel_values = inputs["pixel_values"].to(device)

    # RGB input (you’ll want normalized RGB in real case)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        alpha = model(pixel_values, image_tensor)
    print("Alpha matte output:", alpha.shape)  # Expected: [1, 1, 512, 512]