from torch import nn, Tensor


class UnetR(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            image_size: int = 256,
            patch_size: int = 16,
            hidden_dimensions: int = 768,

    ):
        super().__init__()

        self.patch_embed = nn.Linear(
            in_features=patch_size ** 2 * in_channels,
            out_features=hidden_dimensions
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.patch_embed(x)
