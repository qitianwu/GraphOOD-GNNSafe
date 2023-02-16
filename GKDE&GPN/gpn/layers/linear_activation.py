from typing import Optional
import torch.nn as nn
from torch import Tensor
from .linear_spectral import SpectralLinear


class LinearActivation(nn.Module):
    """layer combining a (spectral) linear layer, activation, and dropout depending on specifications"""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            dropout_prob: Optional[float] = None,
            k_lipschitz: Optional[float] = None,
            activation: nn.Module = nn.ReLU(),
            bias: bool = True):

        super().__init__()

        self.dropout = nn.Identity()
        # dropout on input
        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)

        self.linear = SpectralLinear(input_dim, output_dim, k_lipschitz, bias=bias)

        # nonlinearity
        self.act = nn.Identity()
        if activation is not None:
            self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.linear(self.dropout(x)))

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.dropout.reset_parameters()
