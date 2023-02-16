import os
from typing import Union, Any, Tuple
from sacred.observers import SlackObserver
import random
import collections.abc
import torch
from torch import Tensor
from torch_geometric.data import Data
import numpy as np
from .prediction import Prediction


def map_tensor(tensor: Tensor, mapping: dict):
    """map elements of a tensor according to a specified mapping

    Args:
        tensor (Tensor): input tensor
        mapping (dict): dictionary specifying the mapping

    Returns:
        Tensor: mapped tensor
    """

    tensor = tensor.clone()

    if tensor.dim() == 1:
        for i in range(tensor.size(0)):
            tensor[i] = mapping[tensor[i].item()]

    else:
        for i in range(tensor.size(0)):
            tensor[i, :] = map_tensor(tensor[i, :], mapping)

    return tensor



def __apply(v: Tensor, m: Tensor) -> Tensor:
    """internal function to apply a mask to a tensor or value"""

    if v.dim() == 0:
        return v

    if v.size(0) == m.size(0):
        return v[m]

    return v


def strip_prefix(string: str, prefix: str) -> str:
    """strips prefix from a string

    Args:
        string (str): input string
        prefix (str): prefix to strip

    Returns:
        str: stripped string
    """

    if string.startswith(prefix):
        return string[len(prefix):]

    return string


def _apply_mask(y_hat: Union[dict, Tensor, Prediction], mask: Tensor) -> Union[dict, Tensor, Prediction]:
    """applies a mask to a representation of a model's predictions

    Args:
        y_hat (Union[dict, Tensor, Prediction]): model's predictions
        mask (Tensor): mask, e.g. mask for a validation split

    Raises:
        AssertionError: raised if predictions are of an unsupported type

    Returns:
        Union[dict, Tensor, Prediction]: returns predictions selected by mask
    """

    if isinstance(y_hat, dict):
        _y_hat = {k: __apply(v, mask) for k, v in y_hat.items()}

    elif isinstance(y_hat, torch.Tensor):
        _y_hat = __apply(y_hat, mask)

    elif isinstance(y_hat, Prediction):
        y_hat_dict = _apply_mask(y_hat.to_dict(), mask)
        _y_hat = Prediction(**y_hat_dict)

    else:
        raise AssertionError

    return _y_hat


def apply_mask(data: Data, y_hat: Union[dict, Tensor, Prediction], split: str,
               return_target: bool = True) -> Union[Union[dict, Tensor, Prediction], Tuple[Union[dict, Tensor, Prediction], Tensor]]:
    """applies a specified split/mask to model's predictions

    Args:
        data (Data): data representation
        y_hat (Union[dict, Tensor, Prediction]): model's predictions
        split (str): specified split
        return_target (bool, optional): whether or whether not to return ground-truth labels of desired split in addition to masked predictions. Defaults to True.

    Raises:
        NotImplementedError: raised if passed split is not supported

    Returns:
        Union[Union[dict, Tensor, Prediction], Tuple[Union[dict, Tensor, Prediction], Tensor]]: predictions (and ground-truth labels) after applying mask
    """

    if split == 'train':
        mask = data.train_mask

    elif split == 'val':
        mask = data.val_mask

    elif split == 'test':
        mask = data.test_mask

    elif split == 'ood':
        mask = data.ood_mask

    elif split == 'id':
        mask = data.id_mask

    elif split == 'ood_val':
        mask = data.ood_val_mask

    elif split == 'ood_test':
        mask = data.ood_test_mask

    elif split == 'ood_train':
        # not intended as mask
        # for not breaking pipeline: empty mask
        mask = torch.zeros_like(data.y, dtype=bool)

    elif split == 'id_val':
        mask = data.id_val_mask

    elif split == 'id_test':
        mask = data.id_test_mask

    elif split == 'id_train':
        # not intended as mask
        # for not breaking pipeline: empty mask
        mask = torch.zeros_like(data.y, dtype=bool)

    else:
        raise NotImplementedError(f'split {split} is not implemented!')

    _y_hat = _apply_mask(y_hat, mask)

    if return_target:
        return _y_hat, data.y[mask]
    return _y_hat


def to_one_hot(targets: Tensor, num_classes: int) -> Tensor:
    """maps hard-coded ground-truth targets to one-hot representation of those

    Args:
        targets (Tensor): ground-truth labels
        num_classes (int): number of classes

    Returns:
        Tensor: one-hot encoding
    """

    if len(targets.shape) == 1:
        targets = targets.unsqueeze(dim=-1)

    soft_output = torch.zeros((targets.size(0), num_classes), device=targets.device)
    soft_output.scatter_(1, targets, 1)

    return soft_output


def recursive_update(d: dict, u: dict):
    """recursively update a dictionary d with might contain nested sub-dictionarieswith values from u"""
    
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d[k], v)

        else:
            d[k] = v

    return d


def recursive_delete(d: dict, k: dict) -> dict:
    """delete a key k from a dict d which might be contained in nested sub-dictionaries"""

    for key, v in d.items():
        if key == k:
            del d[k]
            return d

        if isinstance(v, collections.abc.Mapping):
            d[key] = recursive_delete(d.get(key, {}), k)

    return d


def recursive_clean(d: dict) -> Union[dict, None]:
    """recursively clean a dictionary d which might contain nested sub-dictionaries, i.e. remove None-entries"""

    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = recursive_clean(v)
        if v is not None:
            new_dict[k] = v
    return new_dict or None


def recursive_get(d: dict, k: Any) -> Any:
    """recursively get a value specified by a key k from a dictionary d which might contain nested sub-dictionaries"""

    for key, v in d.items():
        if key == k:
            return v

        if isinstance(v, collections.abc.Mapping):
            _v = recursive_get(d.get(key, {}), k)
            if _v is not None:
                return _v

    return None


def set_seed(seed: int) -> None:
    """set seeds for controlled randomness"""

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
