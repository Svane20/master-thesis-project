import torch
from torch import nn

from model.model_builder import DoubleConvBlock


class FashionMnistModelV0(nn.Module):
    """
    Model for Fashion MNIST dataset.

    Args:
        input_shape (int): Input shape.
        hidden_units (int): Number of hidden units.
        output_shape (int): Output shape.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.block_1 = nn.Sequential(
            DoubleConvBlock(in_channels=input_shape, out_channels=hidden_units),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block_2 = nn.Sequential(
            DoubleConvBlock(in_channels=hidden_units, out_channels=hidden_units),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 4 * 4,
                out_features=output_shape
            )
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.block_2(self.block_1(x)))
