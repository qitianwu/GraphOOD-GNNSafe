from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Data
from .appnp_propagation import APPNPPropagation
from .page_rank import PageRankDiffusion


class CertaintyDiffusion(nn.Module):
    """simple parameterless baseline for node classification with a representation of uncertainty"""
    
    def __init__(
            self, num_classes: int,
            K: int = 10,
            alpha_teleport: float = 0.1,
            add_self_loops: bool = True,
            cached: bool = False,
            normalization: str = 'sym',
            w: float = 0.9,
            **_):

        super().__init__()

        self.personalized_page_rank = APPNPPropagation(
            K=K,
            alpha=alpha_teleport,
            cached=cached,
            add_self_loops=add_self_loops,
            normalization=normalization)

        self.page_rank = PageRankDiffusion(
            add_self_loops=add_self_loops,
            normalization=normalization)

        self.num_classes = num_classes
        self.w = w

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # init
        n_nodes = data.x.size(0)
        p_uc = torch.zeros((n_nodes, self.num_classes), device=data.y.device)
        l_c = torch.zeros(self.num_classes, device=p_uc.device)
        y_train = data.y[data.train_mask]

        # calculate class_counts L(c)
        for c in range(self.num_classes):
            class_count = (y_train == c).int().sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        # prepare p_uc with 1-hot values
        # for density 1-hot times 1.0 / l_c
        p_uc_train = p_uc[data.train_mask]
        p_uc_train.scatter_(1, y_train.unsqueeze(dim=-1), 1)
        p_uc[data.train_mask] = p_uc_train
        p_uc = p_uc * 1.0 / l_c

        # diffuse scores
        # personalized PageRank
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        p_uc_ppr = self.personalized_page_rank(p_uc, edge_index)
        p_uc_pr = self.page_rank(p_uc, edge_index)

        p_uc = self.w * p_uc_ppr + (1.0 -  self.w) * p_uc_pr
        p_uc = p_uc * p_c
        p_u = p_uc.sum(-1)

        return p_uc, p_u, p_c
