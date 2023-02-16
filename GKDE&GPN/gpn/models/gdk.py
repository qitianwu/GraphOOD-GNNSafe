import math
import torch
from torch import Tensor
import networkx as nx
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from gpn.utils import Prediction, ModelConfiguration
from .model import Model


class GDK(Model):
    """simple parameterless baseline for node classification based on the Graph Dirichlet Kernel"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        self.cached_alpha = None

    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        if self.cached_alpha is None:
            distance_evidence = compute_kde(data, self.params.num_classes, sigma=1.0)
            alpha = 1.0 + distance_evidence
            self.cached_alpha = alpha

        else:
            alpha = self.cached_alpha
            distance_evidence = alpha - 1.0

        soft = alpha / alpha.sum(-1, keepdim=True)
        max_soft, hard = soft.max(-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # prediction and intermediary scores
            soft=soft,
            hard=hard,
            alpha=alpha,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=distance_evidence[[torch.arange(hard.size(0)), hard]],

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=None,
            sample_confidence_structure=distance_evidence.sum(-1),
        )
        # ---------------------------------------------------------------------------------

        return pred

    def expects_training(self) -> bool:
        return False

    def save_to_file(self, model_path: str) -> None:
        assert self.cached_alpha is not None
        torch.save(self.cached_alpha, model_path)

    def load_from_file(self, model_path: str) -> None:
        if not torch.cuda.is_available():
            alpha = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            alpha = torch.load(model_path)
        self.cached_alpha = alpha


def kernel_distance(x: Tensor, sigma: float = 1.0) -> Tensor:
    sigma_scale = 1.0 / (sigma * math.sqrt(2 * math.pi))
    k_dis = torch.exp(-torch.square(x)/ (2 * sigma * sigma))
    return sigma_scale * k_dis


def compute_kde(data: Data, num_classes: int, sigma: float = 1.0) -> Tensor:
    transform = T.AddSelfLoops()
    data = transform(data)
    n_nodes = data.y.size(0)

    idx_train = torch.nonzero(data.splits['train'], as_tuple=False).squeeze().tolist()
    evidence = torch.zeros((n_nodes, num_classes), device=data.y.device)
    G = to_networkx(data, to_undirected=True)

    for idx_t in idx_train:
        distances = single_source_shortest_path_length(G, source=idx_t, cutoff=10)
        distances = torch.Tensor(
            [distances[n] if n in distances else 1e10 for n in range(n_nodes)]).to(data.y.device)
        evidence[:, data.y[idx_t]] += kernel_distance(distances, sigma=sigma).unsqueeze(1)

    return evidence
