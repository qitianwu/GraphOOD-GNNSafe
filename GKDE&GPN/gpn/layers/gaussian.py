from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

from . import LinearActivation
from . import GCNPropagate


class GaussianTransformation(nn.Module):
    """wrapper class providing linear embeddings for mu and var"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout_prob: float, activation: bool = True):
        super().__init__()

        if activation:
            self.mu = LinearActivation(input_dim, output_dim, dropout_prob, activation=nn.ELU(), bias=False)
            self.var = LinearActivation(input_dim, output_dim, dropout_prob, activation=nn.ReLU(), bias=False)

        else:
            self.mu = LinearActivation(input_dim, output_dim, dropout_prob, activation=None, bias=False)
            self.var = LinearActivation(input_dim, output_dim, dropout_prob, activation=None, bias=False)

    def forward(self, mu: Tensor, var: Tensor) -> Tuple[Tensor, Tensor]:
        return self.mu(mu), self.var(var)


class GaussianPropagation:
    """transform mu, var and propagate"""
    
    def __init__(self, gamma: float = 1):
        self.gamma = gamma

        self.mu_propagation = GCNPropagate(
            improved=False,
            cached=False,
            add_self_loops=True,
            normalization='sym')

        self.var_propagation = GCNPropagate(
            improved=False,
            cached=False,
            add_self_loops=True,
            normalization='sym-var'
        )

        self.mu_activation = nn.ELU()
        self.var_activation = nn.ReLU()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, mu: Tensor, var, edge_index):
        # variance weights
        alpha_mu = torch.exp(-self.gamma * var)
        alpha_var = alpha_mu * alpha_mu

        # propagate
        mu = self.mu_activation(self.mu_propagation(mu * alpha_mu, edge_index))
        var = self.var_activation(self.var_propagation(var * alpha_var, edge_index))

        return mu, var
