import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from typing import List

from dpt.modeling.decoder.utils import ConvStream, Fusion_Block, Matting_Head, RefinementBlock


class Detail_Capture(nn.Module):
    """
    Detail Capture Decoder that fuses deep encoder features with multiscale detail features.
    Automatically aligns channels for 5-level progressive upsampling and matting.
    """

    def __init__(
            self,
            in_channels: int,
            convstream_out: List[int],
            fusion_out: List[int]
    ):
        super().__init__()

        # Extract multiscale features from input image
        self.convstream = ConvStream(in_channels=3, out_channels=convstream_out)
        self.detail_chans = convstream_out[::-1]

        # Fusion output channels (e.g., halve gradually until 16)
        self.fusion_out = fusion_out

        # First channel is encoder bottleneck, then intermediate fusion outputs
        self.fusion_in = [in_channels] + self.fusion_out[:-1]

        # Build fusion blocks with upsampling + concat(detail) + conv
        self.fusion_blks = nn.ModuleList([
            Fusion_Block(
                in_channels=self.fusion_in[i] + self.detail_chans[i],
                out_channels=self.fusion_out[i]
            )
            for i in range(len(self.fusion_out))
        ])

        # Final upsample + refinement
        self.upsample_final = nn.Sequential(
            nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False),
            nn.Conv2d(self.fusion_out[-1], self.fusion_out[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.fusion_out[-1]),
            nn.ReLU(inplace=True),
        )

        self.refine = RefinementBlock(self.fusion_out[-1])
        self.matting_head = Matting_Head(in_channels=self.fusion_out[-1])

    def forward(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        # Extract detail features: D0 (input) to D5 (most downsampled)
        detail_features = self.convstream(images)

        # Reverse order: from D5 to D1
        detail_keys = [f'D{i}' for i in range(len(self.fusion_out), 0, -1)]

        for i, blk in enumerate(self.fusion_blks):
            detail = detail_features[detail_keys[i]]

            # Ensure features match resolution
            if features.shape[-2:] != detail.shape[-2:]:
                features = F.interpolate(features, size=detail.shape[-2:], mode='bilinear', align_corners=False)

            features = blk(features, detail)

        features = self.upsample_final(features)
        features = self.refine(features)
        return torch.sigmoid(self.matting_head(features))


if __name__ == "__main__":
    image_channels = 3
    image_size = 512
    feature_size = 8  # From DPT encoder (bottleneck)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_input = torch.randn(1, image_channels, image_size, image_size).to(device)
    bottleneck = torch.randn(1, 768, feature_size, feature_size).to(device)

    model = Detail_Capture(
        in_channels=768,
        convstream_out=[48, 96, 192, 384, 512]
    ).to(device)
    summary(model, input_size=(3, image_size, image_size))

    output = model(bottleneck, dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: [1, 1, 512, 512]
