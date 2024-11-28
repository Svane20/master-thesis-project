import torch
from torch import nn, Tensor

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, hidden_dimensions: int = 768):
        super().__init__()

        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, hidden_dimensions, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dimensions)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.projection(x).flatten(2).transpose(1, 2)) # B, N, hidden_dimensions


if __name__ == "__main__":
    dummy_input = torch.rand(1, 3, 256, 256)

    # Convert the input image into embedded patches
    embedding = PatchEmbedding()
    embedded_input = embedding(dummy_input)
    assert embedded_input.shape == (1, 256, 768), f"Expected {(1, 256, 768)} but got {embedded_input.shape}"
