from typing import Union, Optional, List
from torch import Tensor
from torch_geometric.data import Dataset

import torch
import numpy as np

from torch_geometric.io.planetoid import index_to_mask


def get_idx_split_arxiv(dataset: Dataset) -> Dataset:
    """create dataset split for the ogbn-arxiv dataset compatible with our pipeline

    Args:
        dataset (Dataset): dataset object for which the split should be created

    Returns:
        Dataset: modified dataset
    """

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    num_nodes = dataset.data.y.size(0)
    dataset.data.train_mask = index_to_mask(train_idx, num_nodes)
    dataset.data.val_mask = index_to_mask(valid_idx, num_nodes)
    dataset.data.test_mask = index_to_mask(test_idx, num_nodes)

    return dataset


def get_idx_split(
        dataset: Dataset,
        split: str = 'random',
        train_samples_per_class: Union[int, float] = None,
        val_samples_per_class: Union[int, float] = None,
        test_samples_per_class: Union[int, float] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None) -> Dataset:
    """utility function for creating train/test/val split for a dataset.

    The split is either created by specifying the number or fraction of samples per class or the overall size
    the training/validation/test set. If the fraction of samples per class is chosen, the fraction is relative
    to the number of labeled data points for each class separately. 

    code taken partially from (https://github.com/shchur/gnn-benchmark)

    Args:
        dataset (Dataset): dataset object
        split (str, optional): selected split ('random' or 'public'). Defaults to 'random'.
        train_samples_per_class (Union[int, float], optional): number of fraction of samples per class in the training set. Defaults to None.
        val_samples_per_class (Union[int, float], optional): number or fraction of samples per class in the validation set. Defaults to None.
        test_samples_per_class (Union[int, float], optional): number or fraction of samples per cleass in the test set. Defaults to None.
        train_size (Optional[int], optional): size of the training set. Defaults to None.
        val_size (Optional[int], optional): size of the validation set. Defaults to None.
        test_size (Optional[int], optional): size of the test set. Defaults to None.

    Returns:
        Dataset: modified dataset object containing the dataset split
    """

    data = dataset.data

    assert (train_size is None) ^ (train_samples_per_class is None)
    assert (val_size is None) ^ (val_samples_per_class is None)
    assert (test_size is None) ^ (test_samples_per_class is None)

    if split == 'public':
        assert hasattr(data, 'train_mask') and hasattr(data, 'test_mask') \
            and hasattr(data, 'test_mask')
        return dataset

    labels = data.y
    num_nodes = labels.size(0)
    num_classes = max(labels) + 1
    classes = range(num_classes)
    remaining_indices = list(range(num_nodes))

    # drop classes which do not have enough samples
    _train_samples_per_class = 0
    _val_samples_per_class = 0
    _test_samples_per_class = 0
    if isinstance(train_samples_per_class, int):
        _train_samples_per_class = train_samples_per_class
    if isinstance(val_samples_per_class, int):
        _val_samples_per_class = val_samples_per_class
    if isinstance(test_samples_per_class, int):
        _test_samples_per_class = test_samples_per_class

    forbidden_indices = None
    min_samples_per_class = _train_samples_per_class + _val_samples_per_class + _test_samples_per_class

    if min_samples_per_class > 0:
        dropped_classes = []
        forbidden_indices = []
        for c in classes:
            class_indices = (labels == c).nonzero(as_tuple=False).squeeze()
            if (labels == c).sum() < min_samples_per_class:
                dropped_classes.append(c)
                forbidden_indices.append(class_indices.numpy())

        if len(forbidden_indices) > 0:
            forbidden_indices = np.concatenate(forbidden_indices)

    else:
        dropped_classes = []
        forbidden_indices = []

    classes = [c for c in classes if c not in dropped_classes]

    # train indices
    if train_samples_per_class is not None:
        train_indices = sample_per_class(labels, num_nodes, classes,
                                         train_samples_per_class, forbidden_indices=forbidden_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        train_indices = np.random.choice(remaining_indices, train_size, replace=False)

    # validation indices (exclude train indices)
    forbidden_indices = np.concatenate((forbidden_indices, train_indices))
    if val_samples_per_class is not None:
        val_indices = sample_per_class(labels, num_nodes, classes,
                                       val_samples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        val_indices = np.random.choice(remaining_indices, val_size, replace=False)

    # test indices (exclude test indices)
    forbidden_indices = np.concatenate((forbidden_indices, val_indices))
    if test_samples_per_class is not None:
        test_indices = sample_per_class(labels, num_nodes, classes,
                                        test_samples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))

    data.train_mask = index_to_mask(train_indices, num_nodes)
    data.test_mask = index_to_mask(test_indices, num_nodes)
    data.val_mask = index_to_mask(val_indices, num_nodes)

    if len(dropped_classes) > 0:
        data.dropped_classes = torch.LongTensor(dropped_classes)

    return dataset


def sample_per_class(labels: Tensor, num_nodes: int, classes: List[int],
                     samples_per_class: Union[int, float],
                     forbidden_indices: np.array = None) -> np.array:
    """samples a subset of indices based on specified number of samples per class

    Args:
        labels (Tensor): tensor of ground-truth labels
        num_nodes (int): number nof nodes
        classes (List[int]): classes (labels) for which the subset is sampled
        samples_per_class (Union[int, float]): number or fraction of samples per class
        forbidden_indices (np.array, optional): indices to ignore for sampling. Defaults to None.

    Returns:
        np.array: sampled indices
    """

    sample_indices_per_class = {index: [] for index in classes}
    num_samples_per_class = {index: None for index in classes}

    # get indices sorted by class
    for class_index in classes:
        for sample_index in range(num_nodes):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    for class_index in classes:
        if isinstance(samples_per_class, float):
            class_labels = sample_indices_per_class[class_index]
            num_samples_per_class[class_index] = int(samples_per_class * len(class_labels))
        else:
            num_samples_per_class[class_index] = samples_per_class

    # get specified number of indices for each class
    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_samples_per_class[class_index], replace=False)
         for class_index in classes
        ])
