import torch
import torch.nn as nn

from typing import List

from unet.modeling.decoder.utils import ConvStream, Fusion_Block, Matting_Head


class Detail_Capture(nn.Module):
    """
    Detail Capture Decoder that fuses deep encoder features with multiscale detail features.
    """

    def __init__(
            self,
            in_channels=512,  # Encoder bottleneck channels
            image_channels: int = 4,
            convstream_out: List[int] = [48, 96, 192, 384],
            fusion_out: List[int] = [256, 128, 64, 32, 16]
    ):
        super().__init__()

        self.convstream = ConvStream(in_channels=image_channels, out_channels=convstream_out)
        self.conv_chans = self.convstream.conv_chans
        self.fus_chans = fusion_out.copy()
        self.fus_chans.insert(0, in_channels)

        self.fusion_blks = nn.ModuleList()
        for i in range(len(self.fus_chans) - 1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_channels=self.fus_chans[i] + self.conv_chans[-(i + 1)],
                    out_channels=self.fus_chans[i + 1]
                )
            )

        self.matting_head = Matting_Head(in_channels=fusion_out[-1])

    def forward(self, features, images):
        # Extract detail features from the input image.
        detail_features = self.convstream(images)
        # Fuse from coarse to fine.
        for i in range(len(self.fusion_blks)):
            # Compute key: for 5 blocks, keys: D4, D3, D2, D1, D0.
            d_key = 'D' + str(len(self.fusion_blks) - i - 1)
            features = self.fusion_blks[i](features, detail_features[d_key])

        return torch.sigmoid(self.matting_head(features))


if __name__ == "__main__":
    image_channels = 3
    image_size = 512
    feature_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, image_channels, image_size, image_size).to(device)
    features = torch.rand(1, image_size, feature_size, feature_size).to(device)

    model = Detail_Capture(
        in_channels=image_size,
        image_channels=image_channels,
        convstream_out=[48, 96, 192, 384],
        fusion_out=[256, 128, 64, 32, 16],
    ).to(device)
    print(model)
    output = model(features, dummy_input)
    print(f"Output shape: {output.shape}")  # Expected shape: torch.Size([1, 1, 512, 512])
