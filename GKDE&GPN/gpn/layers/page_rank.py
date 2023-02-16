import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import Adj

from .gcn_conv import GCNPropagate


class PageRank(nn.Module):
    """layer computing page rank scores"""

    def __init__(
            self,
            add_self_loops: bool = True,
            normalization: str = 'sym',
            alpha: float = 0.1,
            eps_thresh=1.0e-5,
            **kwargs):

        super().__init__()

        self.propagation = GCNPropagate(
            improved=False, cached=True,
            add_self_loops=add_self_loops,
            normalization=normalization,
            **kwargs)

        self.alpha = alpha
        self.eps_thresh = eps_thresh

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # reset cache befor start of iterations
        # but keep cache during iterations
        self.propagation.reset_parameters()

        N = x.size(0)

        pi = torch.rand((N, 1), device=x.device)
        pi = pi / torch.norm(pi, p=1, dim=0)
        pi_prev = pi
        eps = 1.0e10

        while eps > self.eps_thresh:
            pi = (1.0 - self.alpha) * self.propagation(pi, edge_index) + self.alpha * 1.0 / N
            eps = torch.norm(pi - pi_prev, p=2, dim=0)
            pi_prev = pi

        return pi


class PageRankDiffusion(PageRank):
    """diffusion of node features based on PageRank scores"""

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        pi = super().forward(x, edge_index)
        x = pi * x.sum(dim=0, keepdim=True)

        return x
