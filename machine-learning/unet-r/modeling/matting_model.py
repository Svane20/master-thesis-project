import torch
import torch.nn as nn

from functools import partial
from typing import Tuple

from modeling.image_encoder import ImageEncoderViT
from modeling.mask_decoder import MaskDecoder


class ImageMattingModel(nn.Module):
    def __init__(
            self,
            image_size: int = 1024,
            patch_size: int = 16,
            encoder_out_channels=256,
            encoder_embedding_dimension=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_mlp_ratio=4.0,
            encoder_qkv_bias=True,
            encoder_window_size=14,
            global_attention_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            image_size (int): Size of the input image. Default is 1024.
            patch_size (int): Size of the image patch. Default is 16.
            encoder_out_channels (int): Number of output channels from the encoder. Default is 256.
            encoder_embedding_dimension (int): Dimension of the encoder embedding. Default is 768.
            encoder_depth (int): Number of layers in the encoder. Default is 12.
            encoder_num_heads (int): Number of attention heads in the encoder. Default is 12.
            encoder_mlp_ratio (float): Multiplier for the hidden layer in the encoder. Default is 4.0.
            encoder_qkv_bias (bool): If True, use bias in the query, key, value projection. Default is True.
            encoder_window_size (int): Window size for the encoder. Default is 14.
            global_attention_indexes (Tuple[int, ...]): Indexes of the global attention layers. Default is ().
        """
        super().__init__()

        self.encoder = ImageEncoderViT(
            image_size=image_size,
            patch_size=patch_size,
            out_channels=encoder_out_channels,
            embedding_dimension=encoder_embedding_dimension,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            qkv_bias=encoder_qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_size=encoder_window_size,
            global_attention_indexes=global_attention_indexes
        )

        self.decoder = MaskDecoder(
            in_channels=encoder_out_channels,
            num_intermediate_channels=256,
            final_size=image_size,
            patch_size=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        alpha_matte = self.decoder(features)

        return alpha_matte


if __name__ == "__main__":
    image_size = 1024

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random image tensor
    image = torch.randn(1, 3, image_size, image_size).to(device)

    # Initialize the image encoder
    model = ImageMattingModel(
        image_size=image_size,
        patch_size=16,
        encoder_embedding_dimension=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_mlp_ratio=4,
        encoder_qkv_bias=True,
        encoder_window_size=14,
        global_attention_indexes=[2, 5, 8, 11],
    ).to(device)

    # Count number of parameters with commas
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Inference
    with torch.inference_mode():
        output = model(image)

    # Print the output shape
    print(f"Output shape: {output.shape}")  # Output shape: torch.Size([1, 1, 1024, 1024])
