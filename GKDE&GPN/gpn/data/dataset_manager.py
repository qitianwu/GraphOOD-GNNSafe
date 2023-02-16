from typing import Optional, Union
import os
import torch_geometric.datasets as D
import torch_geometric.transforms as T
import ogb.nodeproppred as ogbn
from torch_geometric.transforms.to_undirected import to_undirected
from .split import get_idx_split, get_idx_split_arxiv


class BinarizeFeatures:
    """BinarizeFeatures Transformation for data objects in torch-geometric 
    
    When instantiated transformation object is called, features (data.x) are binarized, i.e. non-zero elements are set to 1.
    """

    def __call__(self, data):
        nz = data.x.bool()
        data.x[nz] = 1.0
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ToUndirected(object):
    """ToUndirected Transformation for data objects in torch-geometric
    
    When instantiated transfomation object is called, the underlying graph in the data  object is converted to an undirected graph, 
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in \mathcal{E}`.
    Depending on the representation of the data object, either data.edge_index or data.adj_t is modified.
    """
    
    def __call__(self, data):
        if 'edge_index' in data:
            data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        if 'adj_t' in data:
            data.adj_t = data.adj_t.to_symmetric()
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def DatasetManager(
        dataset: str,
        root: str,
        split: str = 'public',
        train_samples_per_class: Optional[Union[float, int]] = None,
        val_samples_per_class: Optional[Union[float, int]] = None,
        test_samples_per_class: Optional[Union[float, int]] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        **_):
    """DatasetManager
    
    Method acting as DatasetManager for loading the desired dataset and split when calling with corresponding specifications.
    If the dataset already exists in the root-directory, it is loaded from disc. Otherwise it is downloaded and stored in the specified root-directory.

    Args:
        dataset (str): Name of the dataset to load. Supported datasets are 'CoauthorCS', 'CoauthorPhysics', 'AmazonComputers', 'AmazonPhotos', 'CoraFull', 'CoraML', 'PubMedFull', 'CiteSeerFull', 'Cora', 'PubMed', 'CiteSeer', 'ogbn-arxiv'.
        root (str): Path of data root-directory for either saving or loading dataset.
        split (str, optional): Desired dataset split ('random', or 'public'). Defaults to 'public'.
        train_samples_per_class (Optional[Union[float, int]], optional): number or fraction of training samples per class. Defaults to None.
        val_samples_per_class (Optional[Union[float, int]], optional): number or fraction of validation samples per class. Defaults to None.
        test_samples_per_class (Optional[Union[float, int]], optional): number or fraction of test samples per class. Defaults to None.
        train_size (Optional[int], optional): size of the training set. Defaults to None.
        val_size (Optional[int], optional): size of the validation set. Defaults to None.
        test_size (Optional[int], optional): size of the test set. Defaults to None.

    Raises:
        ValueError: raised if unsupported dataset passed

    Returns:
        dataset: pytorch-geometric dataset as specified
    """

    supported_datasets = {
        'CoauthorCS',
        'CoauthorPhysics',
        'AmazonComputers',
        'AmazonPhotos',
        'CoraFull',
        'CoraML',
        'PubMedFull',
        'CiteSeerFull',
        'Cora',
        'PubMed',
        'CiteSeer',
        'ogbn-arxiv',
    }

    default_transform = T.Compose([
        T.NormalizeFeatures(),
        ToUndirected(),
    ])

    if dataset == 'CoauthorCS':
        assert split == 'random'
        root = os.path.join(root, 'CoauthorCS')
        data = D.Coauthor(root, 'CS', default_transform, None)

    elif dataset == 'CoauthorPhysics':
        assert split == 'random'
        root = os.path.join(root, 'CoauthorPhysics')
        data = D.Coauthor(root, 'Physics', default_transform, None)

    elif dataset == 'AmazonComputers':
        assert split == 'random'
        root = os.path.join(root, 'AmazonComputers')
        data = D.Amazon(root, 'Computers', default_transform, None)

    elif dataset == 'AmazonPhotos':
        assert split == 'random'
        root = os.path.join(root, 'AmazonPhotos')
        data = D.Amazon(root, 'Photo', default_transform, None)

    elif dataset == 'CoraFull':
        assert split == 'random'
        data = D.CitationFull(root, 'Cora', default_transform, None)

    elif dataset == 'CoraML':
        assert split == 'random'
        data = D.CitationFull(root, 'Cora_ML', default_transform, None)

    elif dataset == 'PubMedFull':
        assert split == 'random'
        data = D.CitationFull(root, 'PubMed', default_transform, None)

    elif dataset == 'CiteSeerFull':
        assert split == 'random'
        data = D.CitationFull(root, 'CiteSeer', default_transform, None)

    elif dataset == 'Cora':
        data = D.Planetoid(
            root, 'Cora',
            pre_transform=None,
            transform=default_transform,
            split='public', # always load public split
            num_train_per_class=train_samples_per_class,
            num_test=test_size, num_val=val_size)

    elif dataset == 'PubMed':
        # PubMed contains non-binary node features
        # here, only binary bag-of-word features are used
        transform = T.Compose([
            BinarizeFeatures(),
            T.NormalizeFeatures(),
            ToUndirected(),
        ])

        data = D.Planetoid(
            root, 'PubMed',
            pre_transform=None,
            transform=transform,
            split='public', # always load public split
            num_train_per_class=train_samples_per_class,
            num_test=test_size, num_val=val_size)

    elif dataset == 'CiteSeer':
        data = D.Planetoid(
            root, 'CiteSeer',
            pre_transform=None,
            transform=default_transform,
            split='public', # always load public split
            num_train_per_class=train_samples_per_class,
            num_test=test_size, num_val=val_size)

    elif dataset == 'ogbn-arxiv':
        assert split == 'public'
        transform = T.Compose([ToUndirected()])
        data = ogbn.PygNodePropPredDataset(name='ogbn-arxiv', root='./data', transform=transform)
        data = get_idx_split_arxiv(data)
        data.data.y = data.data.y.squeeze()
        return data

    else:
        raise ValueError(f'{dataset} not in set of supported datasets {supported_datasets}!')

    # default split
    data = get_idx_split(
        data,
        split=split,
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        test_samples_per_class=test_samples_per_class,
        train_size=train_size, val_size=val_size, test_size=test_size)

    return data
