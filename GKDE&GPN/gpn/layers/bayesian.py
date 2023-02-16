from typing import Tuple
import math
import torch
import torch.nn as nn
from torch_geometric.typing import Adj
from .gcn_conv import GCNPropagate


def get_sigma(rho: torch.Tensor) -> torch.Tensor:
    """transforms rho into variance sigma"""
    return torch.log(1.0 + torch.exp(rho))


def sample_weight(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """samples NN-weights based on mean mu and rho"""

    sigma = get_sigma(rho)
    eps = torch.zeros_like(mu).normal_()
    return mu + eps * sigma


def gaussian_log_prob(w: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """calculates (log) probability of weights w based on mean mu and variance sigma of a gaussian"""

    log_prob = -0.5 * math.log(2 * math.pi) \
        - torch.log(sigma + 1.0e-10) \
        - (w - mu) ** 2 / (2 * sigma ** 2 + 1.0e-10)

    return log_prob


def gaussian_posterior_log_prob(w: torch.Tensor, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """calculates the (log) posterior probability of weights w based on mu and rho"""

    sigma = get_sigma(rho)
    return gaussian_log_prob(w, mu, sigma).mean()


def gaussian_mixture_log_prob(w: torch.Tensor, pi: float, sigma_1: float, sigma_2: float) -> torch.Tensor:
    """calculates (log) probability of a mixture of two gaussian with variances sigma_1 and sigma_2 and pi being the mixture coefficient"""

    sigma_1 = torch.FloatTensor([sigma_1]).to(w.device)
    sigma_2 = torch.FloatTensor([sigma_2]).to(w.device)

    prob_1 = gaussian_log_prob(w, 0.0, sigma_1).exp()
    prob_2 = gaussian_log_prob(w, 0.0, sigma_2).exp()

    return torch.log(pi * prob_1 + (1 - pi) * prob_2 + 1.0e-10).mean()


class BayesianLinear(nn.Module):
    """linear transformation layer for a Bayesian GCN"""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            pi: float = 0.5,
            sigma_1: float = 1,
            sigma_2: float = 1e-6):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.pi = pi
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

        self.init_rho = -3

        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim))

        self.b_mu = nn.Parameter(torch.Tensor(output_dim))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim))

        self.log_prior = None
        self.log_q = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.log_prior = None
        self.log_q = None

        # normal weights
        nn.init.normal_(self.w_mu, std=0.1)
        # small constant variance
        nn.init.constant_(self.w_rho, self.init_rho)
        nn.init.constant_(self.b_rho, self.init_rho)

        # zeros for biass
        nn.init.zeros_(self.b_mu)

    def forward_impl(self, sample: bool = False,
                     calculate_log_probs: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training or sample:
            weight = sample_weight(self.w_mu, self.w_rho)
            bias = sample_weight(self.b_mu, self.b_rho)

        else:
            weight = self.w_mu
            bias = self.b_mu

        if self.training or calculate_log_probs:
            # prior probabilities
            self.log_prior = gaussian_mixture_log_prob(
                weight, self.pi, self.sigma_1, self.sigma_2)
            self.log_prior += gaussian_mixture_log_prob(
                bias, self.pi, self.sigma_1, self.sigma_2)

            # posterior probabilities
            self.log_q = gaussian_posterior_log_prob(
                weight, self.w_mu, self.w_rho
            )
            self.log_q += gaussian_posterior_log_prob(
                bias, self.b_mu, self.b_rho
            )

        return weight, bias

    def forward(self, x: torch.Tensor, sample: bool = False, calculate_log_probs: bool = False) -> torch.Tensor:
        weight, bias = self.forward_impl(sample, calculate_log_probs)
        return torch.mm(x, weight) + bias


class BayesianGCNConv(BayesianLinear):
    """convolutional layer for a Bayesian GCN"""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            pi: float = 0.5,
            sigma_1: float = 1,
            sigma_2: float = 1.0e-6):

        super().__init__(
            input_dim, output_dim,
            pi=pi,
            sigma_1=sigma_1,
            sigma_2=sigma_2
        )

        self.propagation = GCNPropagate(
            improved=False,
            cached=False,
            add_self_loops=True,
            normalization='sym')

    def forward(self, x: torch.Tensor, edge_index: Adj,
                sample: bool = False, calculate_log_probs: bool = False) -> torch.Tensor:

        weight, bias = self.forward_impl(
            sample=sample, calculate_log_probs=calculate_log_probs)

        x = torch.mm(x, weight)
        x = self.propagation(x, edge_index)

        return x + bias
