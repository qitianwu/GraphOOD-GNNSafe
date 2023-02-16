from typing import Optional, Tuple

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from .utils import mat_norm


class APPNPPropagation(MessagePassing):
    """APPNP-like propagation (approximate personalized page-rank)
    code taken from the torch_geometric repository on GitHub (https://github.com/rusty1s/pytorch_geometric)
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalization: str = 'sym', **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        assert normalization in ('sym', 'rw', 'in-degree', 'out-degree', None)
        self.normalization = normalization

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalization is not None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = mat_norm(
                        edge_index, edge_weight, x.size(self.node_dim), improved=False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype,
                        normalization=self.normalization
                    )

                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)

                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:

                    edge_index = mat_norm(
                        edge_index, edge_weight, x.size(self.node_dim), improved=False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype,
                        normalization=self.normalization
                    )

                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x

        for _ in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)

                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)

            x = x * (1 - self.alpha)
            x += self.alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)
