import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block to distinguish data that is not linearly separable.
    """

    def __init__(
            self,
            embedding_dimension: int,
            mlp_dimension: int,
            activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            embedding_dimension (int): Dimension of the input tensor.
            mlp_dimension (int): Dimension of the hidden layer.
            activation (nn.Module): Activation function to use. Default is GELU.
        """
        super().__init__()

        self.linear1 = nn.Linear(in_features=embedding_dimension, out_features=mlp_dimension)
        self.linear2 = nn.Linear(in_features=mlp_dimension, out_features=embedding_dimension)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class LayerNorm2d(nn.Module):
    """
    Layer normalization for 2D data.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """
        Args:
            num_channels (int): Number of channels in the input tensor.
            eps (float): Epsilon value to prevent division by zero. Default is 1e-6.
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the mean
        u = x.mean(1, keepdim=True)

        # Calculate the variance
        s = (x - u).pow(2).mean(1, keepdim=True)

        # Normalize the tensor
        x = (x - u) / torch.sqrt(s + self.eps)

        # Scale and shift the tensor
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        return x
