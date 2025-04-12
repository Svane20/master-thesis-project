import torch
import torch.nn as nn
from transformers import AutoBackbone
from torchsummary import summary

from libs.utils.mem_utils import estimate_max_batch_size


class SwinEncoder(nn.Module):
    def __init__(self, model_name="microsoft/swin-small-patch4-window7-224"):
        super().__init__()
        # Load pre-trained Swin-Small backbone, set it to output multi-scale feature maps
        self.backbone = AutoBackbone.from_pretrained(model_name, out_indices=(1, 2, 3, 4))
        # out_indices=(1,2,3,4) yields feature maps from stage1, stage2, stage3, stage4

    def forward(self, x):
        """
        Forward pass for the encoder.
        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W] (expected 512x512).
        Returns:
            List[torch.Tensor]: Feature maps at 1/4, 1/8, 1/16, 1/32 resolutions.
        """
        # HuggingFace backbone expects input key "pixel_values"
        outputs = self.backbone(pixel_values=x)
        features = outputs.feature_maps  # tuple of feature maps at the specified indices
        # Ensure output is a list for easier handling
        features = list(features)
        return features

if __name__ == "__main__":
    image_size = 512
    image_channels = 3
    input_size = (image_channels, image_size, image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "microsoft/swin-small-patch4-window7-224"
    model = SwinEncoder(model_name=model_name).to(device)
    summary(model, input_size=input_size)

    # Dummy input image in [0, 1] range (normalized RGB)
    image = torch.randn(image_channels, image_size, image_size).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        features = model(image)

    for index, feature in enumerate(features):
        print(f"Feature {index + 1} shape: {feature.shape}")
        """
        torch.Size([1, 384, 16, 16])
        torch.Size([1, 192, 32, 32])
        torch.Size([1, 96, 64, 64])
        """

    def generate_inputs(batch_size, input_size, device):
        return (torch.randn(batch_size, *input_size, device=device),)


    # Input resolutions to test
    for res in [224, 512, 1024]:
        input_size = (image_channels, res, res)
        print(f"\nTesting input size: {input_size}...")

        # Estimate max batch size for this resolution
        max_batch_size = estimate_max_batch_size(
            model,
            input_size=input_size,
            max_memory_gb=12.0,
            safety_factor=0.9,
            input_generator_fn=generate_inputs
        )

        print(f"Estimated max batch size for {res}x{res}: {max_batch_size}")
