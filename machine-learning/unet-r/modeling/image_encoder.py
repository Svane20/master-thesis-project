import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Type
from functools import partial

from .common import LayerNorm2d, MLPBlock


class ImageEncoderViT(nn.Module):
    """
    Vision Transformer (ViT) image encoder.
    """

    def __init__(
            self,
            image_size: int = 1024,
            patch_size: int = 16,
            in_channels: int = 3,
            out_channels: int = 256,
            embedding_dimension: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            activation_layer: Type[nn.Module] = nn.GELU,
            window_size: int = 0,
            global_attention_indexes: Tuple[int, ...] = (),
    ):
        """
        Args:
            image_size (int): Image size. Default is 1024.
            patch_size (int): Patch size. Default is 16.
            in_channels (int): Number of input channels. Default is 3.
            out_channels (int): Number of output channels. Default is 256.
            embedding_dimension (int): Patch embedding dimension. Default is 768.
            depth (int): Number of transformer blocks. Default is 12.
            num_heads (int): Number of attention heads. Default is 12.
            mlp_ratio (float): Multi-layer perceptron (MLP) ratio. Default is 4.0.
            qkv_bias (bool): If True, add bias to the query, key, value projection. Default is True.
            norm_layer (nn.Module): Normalization layer. Default is nn.LayerNorm.
            activation_layer (nn.Module): Activation layer. Default is nn.GELU.
            window_size (int): Window size for local attention. Default is 0.
            global_attention_indexes (Tuple[int]): Indexes of the blocks to use global attention. Default is ().
        """
        super().__init__()

        self.image_size = image_size

        self.patch_embedding = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_channels=in_channels,
            embedding_dimension=embedding_dimension
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(size=(
                1,  # Batch size
                image_size // patch_size,  # Number of patches (height)
                image_size // patch_size,  # Number of patches (width)
                embedding_dimension  # Embedding dimension
            ))
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dimension=embedding_dimension,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                window_size=window_size if i not in global_attention_indexes else 0,
                input_size=(image_size // patch_size, image_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dimension,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            LayerNorm2d(num_channels=out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            LayerNorm2d(num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert image to patch embeddings
        x = self.patch_embedding(x)

        # Add positional embedding
        x += self.positional_embedding

        # Apply blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply neck
        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """
    Transformer blocks with support of window attention and residual propagation blocks
    """

    def __init__(
            self,
            dimension: int,
            num_heads: int,
            input_size: Tuple[int, int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            activation_layer: Type[nn.Module] = nn.GELU,
            window_size: int = 0,
    ):
        """
        Args:
            dimension (int): Input dimension.
            num_heads (int): Number of attention heads.
            input_size (Tuple[int, int]): Input size.
            mlp_ratio (float): Multi-layer perceptron (MLP) ratio. Default is 4.0.
            qkv_bias (bool): If True, add bias to the query, key, value projection. Default is True.
            norm_layer (nn.Module): Normalization layer. Default is nn.LayerNorm.
            activation_layer (nn.Module): Activation layer. Default is nn.GELU.
            window_size (int): Window size for local attention. Default is 0.
        """
        super().__init__()

        self.norm1 = norm_layer(dimension)
        self.attention = Attention(
            dimension=dimension,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dimension)
        self.mlp = MLPBlock(
            embedding_dimension=dimension,
            mlp_dimension=int(dimension * mlp_ratio),
            activation=activation_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, self.window_size)

        x = self.attention(x)

        # Reverse window partition
        if self.window_size > 0:
            x = _window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """
    Multi-head Attention block with relative positional embeddings.
    """

    def __init__(
            self,
            dimension: int,
            num_heads: int,
            input_size: Tuple[int, int],
            qkv_bias: bool = True,
    ):
        """
        Args:
            dimension (int): Input dimension.
            num_heads (int): Number of attention heads.
            input_size (Tuple[int, int]): Input size.
            qkv_bias (bool): If True, add bias to the query, key, value projection. Default is True.
        """
        super().__init__()

        self.num_heads = num_heads
        head_dimension = dimension // num_heads
        self.scale = head_dimension ** -0.5

        self.qkv = nn.Linear(in_features=dimension, out_features=dimension * 3, bias=qkv_bias)
        self.projection = nn.Linear(in_features=dimension, out_features=dimension)

        # Initialize relative positional embeddings
        self.relative_positional_height = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dimension))
        self.relative_positional_width = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        # Linear transformation for query, key, value
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # Split the query, key, value
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # Compute the attention scores
        attention = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative positional embeddings
        attention = _add_decomposed_relative_positional_embeddings(
            attention=attention,
            q=q,
            relative_positional_height=self.relative_positional_height,
            relative_positional_width=self.relative_positional_width,
            q_size=(H, W),
            k_size=(H, W),
        )

        # Apply softmax
        attention = attention.softmax(dim=-1)

        # Apply attention to value
        x = (attention @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        # Project the output
        x = self.projection(x)

        return x


class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_channels: int = 3,
            embedding_dimension: int = 768
    ):
        """
        Args:
            kernel_size (Tuple[int, int]): Kernel size of the projection layer. Default is (16, 16).
            stride (Tuple[int, int]): Stride of the projection layer. Default is (16, 16).
            padding (Tuple[int, int]): Padding of the projection layer. Default is (0, 0).
            in_channels (int): Number of input channels. Default is 3.
            embedding_dimension (int): Patch embedding dimension. Default is 768.
        """
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dimension,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)

        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)

        return x


def _window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.

    Args:
        x (Tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows, (Hp, Wp)


def _window_unpartition(
        windows: torch.Tensor,
        window_size: int,
        pad_hw: Tuple[int, int],
        hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (Tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x (Tensor): unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw

    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    return x


def _add_decomposed_relative_positional_embeddings(
        attention: torch.Tensor,
        q: torch.Tensor,
        relative_positional_height: torch.Tensor,
        relative_positional_width: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950

    Args:
        attention (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        relative_positional_height (Tensor): relative position embeddings (Lh, C) for height axis.
        relative_positional_width (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attention (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = _get_relative_positional_embeddings(q_h, k_h, relative_positional_height)
    Rw = _get_relative_positional_embeddings(q_w, k_w, relative_positional_width)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    # Add relative positional embeddings.
    attention = (
            attention.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attention


def _get_relative_positional_embeddings(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of query and key sizes.

    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        relative_positional_embeddings (Tensor): Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            input=rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )

        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]
