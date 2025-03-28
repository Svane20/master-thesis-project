import torch
import torch.nn as nn
from transformers import DPTImageProcessor, DPTForDepthEstimation
from torchsummary import summary
from typing import Tuple, List
import numpy as np


class DPTEncoder(nn.Module):
    def __init__(self, model_name: str = "Intel/dpt-swinv2-tiny-256"):
        super().__init__()

        self.model = DPTForDepthEstimation.from_pretrained(model_name)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output = self.model(pixel_values=pixel_values, output_hidden_states=True)

        # SwinV2 hidden_states are already shaped [B, C, H, W]
        hidden_states = output.hidden_states[-4:]
        features = [self.reshape_tokens(h) for h in hidden_states]
        bottleneck = features[-1]
        return bottleneck, features

    def reshape_tokens(self, x: torch.Tensor):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        return x.permute(0, 2, 1).reshape(B, C, H, W)


if __name__ == "__main__":
    image_size = 512
    image_channels = 3
    input_size = (image_channels, image_size, image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "Intel/dpt-swinv2-tiny-256"
    model = DPTEncoder(model_name=model_name).to(device)
    summary(model, input_size=input_size)

    # Create image in range [0, 1]
    dummy_image = np.random.rand(image_size, image_size, image_channels).astype(np.float32)

    # Use updated image processor (was DPTFeatureExtractor)
    processor = DPTImageProcessor.from_pretrained(model_name)
    inputs = processor(images=dummy_image, return_tensors="pt", do_rescale=False)

    # Move input tensors to the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    bottleneck, features = model(**inputs)
    features = features[::-1]
    bottleneck = features[0]
    skip_connections = features[1:]
    for index, skip_connection in enumerate(skip_connections):
        print(f"Skip connection {index + 1} shape: {skip_connection.shape}")
        """
        torch.Size([1, 384, 16, 16])
        torch.Size([1, 192, 32, 32])
        torch.Size([1, 96, 64, 64])
        """
    print(f"Bottleneck shape: {bottleneck.shape}")  # Excepted shape: torch.Size([1, 768, 8, 8])
