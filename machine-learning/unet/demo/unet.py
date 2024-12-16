import torch
import torch.nn as nn

from image_encoder import ImageEncoder
from mask_decoder import MaskDecoder


class UNet(nn.Module):
    """
    UNet model for Image Matting.
    """

    def __init__(
            self,
            image_encoder: ImageEncoder,
            mask_decoder: MaskDecoder
    ):
        """
        Args:
            image_encoder (ImageEncoder): Image encoder
            mask_decoder (MaskDecoder): Mask decoder
        """
        super().__init__()

        self.encoder = image_encoder
        self.decoder = mask_decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skip_connections = self.encoder(x)

        return self.decoder(bottleneck, skip_connections)
