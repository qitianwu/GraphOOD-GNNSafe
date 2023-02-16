from torch import Tensor
from torch_geometric.data import Data

import torch
import torch_geometric.utils as tu
from torch_scatter import scatter_add


def degree(edge_index: Tensor, direction='out', num_nodes=None, edge_weight=None):
    """calulcates the degree of each node in the graph

    Args:
        edge_index (Tensor): tensor edge_index encoding the graph structure 
        direction (str, optional): either calculate 'in'-degree or 'out'-degree. Defaults to 'out'.
        num_nodes (int, optional): number of nodes. Defaults to None.
        edge_weight (Tensor, optional): weight of edges. Defaults to None.

    Raises:
        AssertionError: raised if unsupported direction is passed

    Returns:
        Tensor: node degree
    """
    row, col = edge_index[0], edge_index[1]

    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1), ),
            device=edge_index.device)

    if direction == 'out':
        return scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    elif direction == 'in':
        return scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    else:
        raise AssertionError


def get_k_hop_diversity(data: Data, k=1, kind='diversity'):
    """returns k-hop-diversity of each node in the grap

    Args:
        data (Data): pytorch-geometric data object representing graph
        k (int, optional): k specifying k-hop neighborhood. Defaults to 1.
        kind (str, optional): either return 'purity' or 'diversity'. Defaults to 'diversity'.

    Raises:
        AssertionError: raised if unsurported kind is passed

    Returns:
        Tensor: divsierty of purity
    """
    n_nodes = data.y.size(0)
    diversity = torch.zeros_like(data.y)

    if kind == 'purity':
        diversity = diversity.float()

    for n in range(n_nodes):
        k_hop_nodes, _, _, _ = tu.k_hop_subgraph(n, k, data.edge_index)
        if kind == 'diversity':
            div = len(data.y[k_hop_nodes].unique())
        elif kind == 'purity':
            y_center = data.y[n]
            y_hop = data.y[k_hop_nodes]
            div = (y_hop == y_center.item()).float().mean()

        else:
            raise AssertionError

        diversity[n] = div

    return diversity
