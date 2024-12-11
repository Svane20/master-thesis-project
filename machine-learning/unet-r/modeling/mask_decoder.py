import torch
from torch import nn
from torch.nn import functional as F
from typing import Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
            self,
            transformer_dim: int,
            transformer: nn.Module,
            activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        A transformer-based decoder for producing a single alpha mask.

        Arguments:
          transformer_dim (int): The channel dimension of the transformer.
          transformer (nn.Module): The transformer used to refine mask embeddings.
          activation (nn.Module): The activation function used during upsampling.
        """
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # A single query embedding to represent the mask
        self.mask_query = nn.Embedding(num_embeddings=1, embedding_dim=transformer_dim)

        # Upsampling layers to go from transformer resolution to full resolution
        # Adjust depending on how much you need to upscale
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
            nn.Conv2d(transformer_dim // 8, 1, kernel_size=3, padding=1)
        )

        # A small MLP to map the transformer output query to mask weights
        self.mask_mlp = MLP(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,
            output_dim=transformer_dim // 8,
            num_layers=3
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predicts a single-channel alpha mask.

        Arguments:
          image_embeddings (torch.Tensor): The [B, C, H, W] embeddings from the encoder.
          image_pe (torch.Tensor): Positional encoding matching image_embeddings, if used.

        Returns:
          torch.Tensor: The predicted alpha mask of shape [B, 1, H_out, W_out].
        """
        B, C, H, W = image_embeddings.shape

        # Prepare queries
        # mask_query: [1, C], expand to [B, 1, C]
        mask_query = self.mask_query.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, C]

        # Flatten spatial dimensions for the transformer: [B, C, H, W] -> [B, HW, C]
        # Transformer often expects: src as [B, N, C], here N=H*W
        src = image_embeddings.flatten(2).transpose(1, 2)  # [B, HW, C]

        # If using positional embeddings:
        pos_embed = image_pe.flatten(2).transpose(1, 2) if image_pe is not None else None

        # Run the transformer:
        # Define your transformer interface similarly to original code:
        # hs shape: [B, num_queries, C], src: [B, HW, C] after refinement
        hs, src_refined = self.transformer(src, pos=pos_embed, query=mask_query)

        # hs: [B, 1, C]
        mask_token_out = hs[:, 0, :]  # Extract the single query vector

        # Project the mask token through the MLP
        mask_embedding = self.mask_mlp(mask_token_out)  # [B, C//8]

        # Reshape src_refined to spatial shape [B, C, H, W]
        c_new = src_refined.shape[-1]
        src_refined = src_refined.transpose(1, 2).view(B, c_new, H, W)

        # Upscale the refined features
        upscaled_embedding = self.output_upscaling(src_refined)  # [B, 1, H_out, W_out]

        # Apply sigmoid to ensure [0, 1] range for alpha
        alpha_mask = torch.sigmoid(upscaled_embedding)

        return alpha_mask


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.sigmoid_output:
            x = torch.sigmoid(x)

        return x
