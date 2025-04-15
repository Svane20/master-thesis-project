import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DPTForDepthEstimation
from typing import Tuple, List, Any, Dict


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encoder(x)
        return self.decoder(bottleneck, x)


class DPTSwinV2Tiny256Encoder(nn.Module):
    def __init__(self, model_name: str = "Intel/dpt-swinv2-tiny-256"):
        super().__init__()

        self.model = DPTForDepthEstimation.from_pretrained(model_name)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output = self.model(pixel_values=x, output_hidden_states=True)

        # SwinV2 hidden_states are already shaped [B, C, H, W]
        hidden_states = output.hidden_states[-4:]
        features = [self.reshape_tokens(h) for h in hidden_states]
        bottleneck = features[-1]
        return bottleneck, features

    def reshape_tokens(self, x: torch.Tensor):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        return x.permute(0, 2, 1).reshape(B, C, H, W)


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


class Basic_Conv3x3(nn.Module):
    """
    Basic 3x3 conv followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, padding: int = 1) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Convolution stride. Default: 2.
            padding (int): Convolution padding. Default: 1.
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvStream(nn.Module):
    """
    Extracts multiscale detail features from the input RGB image.
    """

    def __init__(self, in_channels: int = 4, out_channels: List[int] = [48, 96, 192, 384]) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (List[int]): List of output channels for each convolution. Default: [48, 96, 192, 384].
        """
        super().__init__()

        self.convs = nn.ModuleList()
        self.conv_chans = out_channels.copy()
        self.conv_chans.insert(0, in_channels)
        for i in range(len(self.conv_chans) - 1):
            self.convs.append(Basic_Conv3x3(self.conv_chans[i], self.conv_chans[i + 1]))

    def forward(self, x: torch.Tensor) -> dict:
        out_dict = {'D0': x}
        for i, conv in enumerate(self.convs):
            x = conv(x)
            out_dict[f'D{i + 1}'] = x
        return out_dict


class RefinementBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Fusion_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = Basic_Conv3x3(in_channels, out_channels, stride=1, padding=1)

    def forward(self, x: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # 🔧 Match spatial size of detail to up_x before concatenation
        if up_x.shape[-2:] != detail.shape[-2:]:
            detail = F.interpolate(detail, size=up_x.shape[-2:], mode='bilinear', align_corners=False)

        out = torch.cat([up_x, detail], dim=1)
        return self.conv(out)


class Matting_Head(nn.Module):
    """
    Final head that produces a 1-channel alpha matte.
    """

    def __init__(self, in_channels: int = 16, mid_channels: int = 16):
        """
        Args:
            in_channels (int): Number of input channels. Default: 16.
            mid_channels (int): Number of intermediate channels. Default: 16.
        """
        super().__init__()

        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.matting_convs(x)
