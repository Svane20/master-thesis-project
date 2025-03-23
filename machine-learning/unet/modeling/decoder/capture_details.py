import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic_Conv3x3(nn.Module):
    """
    Basic 3x3 conv followed by BatchNorm and ReLU.
    """

    def __init__(self, in_chans, out_chans, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvStream(nn.Module):
    """
    Extracts multi-scale detail features from the input RGB image.
    Now produces outputs D0..D4.
    """

    def __init__(self, in_chans=3, out_chans: list = [48, 96, 192, 384]):
        super().__init__()
        self.convs = nn.ModuleList()
        # Channel progression: [in_chans, 48, 96, 192, 384]
        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)
        for i in range(len(self.conv_chans) - 1):
            self.convs.append(Basic_Conv3x3(self.conv_chans[i], self.conv_chans[i + 1]))

    def forward(self, x: torch.Tensor) -> dict:
        out_dict = {'D0': x}
        for i, conv in enumerate(self.convs):
            x = conv(x)
            out_dict[f'D{i + 1}'] = x

        return out_dict


class Fusion_Block(nn.Module):
    """
    Upsamples deep features and fuses them with corresponding detail features.
    """

    def __init__(self, in_chans, out_chans):
        super().__init__()
        # Use Basic_Conv3x3 with stride=1 to keep spatial dimensions.
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([up_x, detail], dim=1)
        return self.conv(out)


class Matting_Head(nn.Module):
    """
    Final head that produces a 1-channel alpha matte.
    """

    def __init__(self, in_chans=16, mid_chans=16):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chans, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.matting_convs(x)


class Detail_Capture(nn.Module):
    """
    Detail Capture Decoder that fuses deep encoder features with multi-scale detail features.
    """

    def __init__(self,
                 in_channels=512,  # Encoder bottleneck channels
                 convstream_out: list = [48, 96, 192, 384],
                 fusion_out: list = [256, 128, 64, 32, 16]):
        super().__init__()
        # Ensure fusion_out has one more element than convstream_out.
        assert len(fusion_out) == len(convstream_out) + 1, "fusion_out must have one more element than convstream_out."
        self.convstream = ConvStream(in_chans=3, out_chans=convstream_out)
        self.conv_chans = self.convstream.conv_chans  # e.g. [3, 48, 96, 192, 384]
        self.fus_chans = fusion_out.copy()
        self.fus_chans.insert(0, in_channels)  # e.g. [512, 256, 128, 64, 32, 16]
        self.fusion_blks = nn.ModuleList()
        # Create as many fusion blocks as len(fus_chans)-1 (here, 5 blocks)
        for i in range(len(self.fus_chans) - 1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans=self.fus_chans[i] + self.conv_chans[-(i + 1)],
                    out_chans=self.fus_chans[i + 1]
                )
            )
        self.matting_head = Matting_Head(in_chans=fusion_out[-1])

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
    image_size = 512
    feature_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    features = torch.rand(1, image_size, feature_size, feature_size).to(device)

    model = Detail_Capture(
        in_channels=image_size,
        convstream_out=[48, 96, 192, 384],
        fusion_out=[256, 128, 64, 32, 16],
    ).to(device)
    print(model)

    output = model(features, dummy_input)
    print(f"Output shape: {output.shape}") # Expected shape: torch.Size([1, 1, 512, 512])
