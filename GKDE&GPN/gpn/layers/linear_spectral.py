from typing import Optional
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import spectral_norm


class SpectralLinear(nn.Module):
    """linear layer with option to use it as a spectral linear layer with lipschitz-norm of k"""

    def __init__(self, input_dim: int, output_dim: int, 
                 k_lipschitz: Optional[float] = 1.0, bias: bool = True):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        linear = nn.Linear(input_dim, output_dim, bias=bias)

        if self.k_lipschitz is not None:
            self.linear = spectral_norm(linear)
        else:
            self.linear = linear

    def forward(self, x: Tensor) -> Tensor:
        if self.k_lipschitz is None:
            y = self.linear(x)

        else:
            y = self.k_lipschitz * self.linear(x)

        return y

    def reset_parameters(self): 
        self.linear.reset_parameters()
