import torch
import torch.nn as nn
import torch_geometric.nn as tnn
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from .utils import mat_norm, propagation_wrapper


class GCNPropagate(tnn.MessagePassing):
    """propagation layer from original graph convolutional layer

    code taken from the torch_geometric repository on GitHub (https://github.com/rusty1s/pytorch_geometric)
    """

    def __init__(
            self,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: bool = True,
            normalization: str = 'sym',
            **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalization = normalization

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            edge_weight: OptTensor = None) -> Tensor:

        if self.normalization is not None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = mat_norm(
                        edge_index, edge_weight, x.size(self.node_dim), improved=self.improved,
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
                        edge_index, edge_weight, x.size(self.node_dim), improved=self.improved,
                        add_self_loops=self.add_self_loops, dtype=x.dtype,
                        normalization=self.normalization
                    )

                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class GCNConv(nn.Module):
    """graph convolutional layer from original GCN with separate layers for linear transformations and propagation

    code taken from the torch_geometric repository on GitHub (https://github.com/rusty1s/pytorch_geometric)
    """

    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: bool = True,
            normalization: str = 'sym',
            bias: bool = True,
            **kwargs):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.propagation = GCNPropagate(improved, cached, add_self_loops, normalization, **kwargs)

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.propagation.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None,
                unc_node_weight: OptTensor = None,
                unc_edge_weight: OptTensor = None,
                node_normalization: str = 'none',
                return_normalizer=False) -> Tensor:
        """
        combined transformation and propagation step
        """

        x = self.transform(x=x)
        x = self.propagate(
            x=x, edge_index=edge_index,
            edge_weight=edge_weight,
            unc_node_weight=unc_node_weight,
            unc_edge_weight=unc_edge_weight,
            node_normalization=node_normalization,
            return_normalizer=return_normalizer)

        return x

    def transform(self, x: Tensor) -> Tensor:
        """
        transform nodes' features
        """
        return torch.matmul(x, self.weight)

    def propagate(self, x: Tensor, edge_index: Adj,
                  edge_weight: OptTensor = None,
                  unc_node_weight: OptTensor = None,
                  unc_edge_weight: OptTensor = None,
                  node_normalization: str = 'none',
                  return_normalizer=False) -> Tensor:
        """
        propagate and apply bias
        """

        out = propagation_wrapper(
            self.propagation, x=x, edge_index=edge_index,
            node_normalization=node_normalization,
            unc_node_weight=unc_node_weight,
            unc_edge_weight=unc_edge_weight,
            edge_weight=edge_weight,
            return_normalizer=return_normalizer)

        if self.bias is not None:
            if return_normalizer:
                out, norm = out
                out += self.bias
                return out, norm

            return out + self.bias

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'
