from typing import Optional
import numpy as np
import torch.nn as nn
from .linear_spectral import SpectralLinear


def LinearSequentialLayer(
        input_dims: int, hidden_dims: int, output_dim: int,
        dropout_prob: Optional[float] = None,
        batch_norm: bool = False,
        k_lipschitz: Optional[float] = None,
        num_layers: Optional[int] = None,
        activation_in_all_layers=False,
        **_) -> nn.Module:
    """creates a chain of combined linear and activation layers depending on specifications"""

    if isinstance(hidden_dims, int):
        if num_layers is not None:
            hidden_dims = [hidden_dims] * (num_layers - 1)
        else:
            hidden_dims = [hidden_dims]

    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []

    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz=k_lipschitz ** (1./num_layers))
        else:
            l = nn.Linear(dims[i], dims[i + 1])

        layers.append(l)

        if activation_in_all_layers or (i < num_layers - 1):
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())

            if dropout_prob is not None:
                layers.append(nn.Dropout(p=dropout_prob))

    return nn.Sequential(*layers)
