from typing import Optional
import math
import os
import random
import torch
import numpy as np
from torch_geometric.data import Data
from collections import Counter
import torch_geometric.utils as tu
from gpn.utils import map_tensor


def swap_features(data: Data, i: int, j: int):
    """swaps feature vectors of two nodes

    Args:
        data (Data): pytorch-geometric data object
        i (int): node index
        j (int): node index of swapping partner

    Returns:
        Data: modified data object
    """

    tmp = data.x[i]
    data.x[i] = data.x[j]
    data.x[j] = tmp

    return data


def swap_labels(data: Data, i: int, j: int):
    """swaps labels of two nodes

    Args:
        data (int): pytorch-geometric data object
        i (int): node index
        j (int): node index of swapping partner

    Returns:
        Data: modified data object
    """
    tmp = data.y[i]
    data.y[i] = data.y[j]
    data.y[j] = tmp

    return data


def get_ood_split(
        data,
        ood_frac_left_out_classes: float = 0.45,
        ood_num_left_out_classes: Optional[int] = None,
        ood_leave_out_last_classes: Optional[bool] = False,
        ood_left_out_classes: Optional[list] = None,
        **_):
    """splits data in ID and OOD data for Leave-Out-Classes experiment

    The split can be either specified by specifying the fraction of left-out classes, the number of left-out-classes, or by passing a list of class
    indices to leave out. In the first two cases, the flag ood_leave_out_last_classes can be set to leave the last class indices out. Otherwise, 
    the left-out classes are simply sampled randomly. 

    Args:
        data (torch_geometric.data.Data): data object representing graph data
        ood_frac_left_out_classes (float, optional): fraction nof left-out classes. Defaults to 0.45.
        ood_num_left_out_classes (Optional[int], optional): number of left-out classes. Defaults to None.
        ood_leave_out_last_classes (Optional[bool], optional): whether or whether not to leave the last class indices out (assuming c in [1, ... C]). Defaults to False.
        ood_left_out_classes (Optional[list], optional): optional list of class indices to leave out. Defaults to None.

    Returns:
        Tuple[torch_geometric.data.Data, int]: tuple of data object and number of classes in the ID case
    """

    # creates masks / data copies for ood dataset (left out classes) and
    # default dataset (without the classes being left out)
    data = data.clone()

    assert hasattr(data, 'train_mask') and hasattr(data, 'val_mask') \
            and hasattr(data, 'test_mask')

    num_classes = data.y.max().item() + 1
    classes = np.arange(num_classes)

    if ood_left_out_classes is None:
        # which classes are left out
        if ood_num_left_out_classes is None:
            ood_num_left_out_classes = math.floor(num_classes * ood_frac_left_out_classes)

        if not ood_leave_out_last_classes:
            # create random perturbation of classes to leave out
            # classes in the end
            # if not specified: leave out classes which are originally
            # at the end of the array of sorted classes
            np.random.shuffle(classes)

        left_out_classes = classes[num_classes - ood_num_left_out_classes: num_classes]

    else:
        ood_num_left_out_classes = len(ood_left_out_classes)
        left_out_classes = np.array(ood_left_out_classes)
        # reorder c in classes, such that left-out-classes
        # are at the end of classes-array
        tmp = [c for c in classes if c not in left_out_classes]
        tmp = tmp + [c for c in classes if c in left_out_classes]
        classes = np.array(tmp)

    class_mapping = {c:i for i, c in enumerate(classes)}

    left_out = torch.zeros_like(data.y, dtype=bool)
    for c in left_out_classes:
        left_out = left_out | (data.y == c)

    left_out_val = left_out & data.val_mask
    left_out_test = left_out & data.test_mask

    data.ood_mask = left_out
    data.id_mask = ~left_out

    if hasattr(data, 'train_mask'):
        data.train_mask[left_out] = False
        data.test_mask[left_out] = False
        data.val_mask[left_out] = False

        data.ood_val_mask = left_out_val
        data.ood_test_mask = left_out_test

        data.id_val_mask = data.val_mask
        data.id_test_mask = data.test_mask

    num_classes = num_classes - ood_num_left_out_classes

    # finally apply positional mapping of classes from above to ensure that
    # classes are ordered properly (i.e. create dataset with labels being in range 0 --- new_num_classes - 1)
    data.y = torch.LongTensor([class_mapping[y.item()] for y in data.y], device=data.y.device)

    return data, num_classes


def get_ood_split_evasion(data, **ood_kwargs):
    """split data in ID and OOD data for LOC experiments in an evasion setting

    In this case, the left-out classes are removed from the graph for training completely (id_data), but added back again for inference.
    In the default case, they classes are kept in the graph but simply not trained on.

    Args:
        data (torch_geometric.data.Data): data object representing the graph
        ood_kwargs: kwargs for specifying split (see get_ood_split)

    Returns:
        Tuple[torch_geometric.data.Data, torch_geometric.data.Data, int]: tuple of id_data, ood_data, and number of classes in the ID case 
    """

    ood_data, num_classes = get_ood_split(data, **ood_kwargs)
    id_nodes = ood_data.id_mask
    id_edge_index, _ = tu.subgraph(
        id_nodes, ood_data.edge_index,
        relabel_nodes=True)
    id_data = ood_data.clone()

    for k, v in id_data.__dict__.items():
        if v is None:
            continue

        if k == 'edge_index':
            id_data.edge_index = id_edge_index

        elif isinstance(v, torch.Tensor):
            id_data[k] = v[id_nodes]

    id_data.ood_mask = None
    id_data.id_mask = None

    if hasattr(id_data, 'train_mask'):
        id_data.ood_test_mask = None
        id_data.ood_val_mask = None
        id_data.id_test_mask = None
        id_data.id_val_mask = None

    return id_data, ood_data, num_classes


def get_perturbed_indices(data, ood_budget_per_graph, perturb_train_indices, **_):
    """get node indices for global graph perturbation of features 

    Args:
        data (torch_geometric.data.Data): data object representing the graph
        ood_budget_per_graph (float): fraction of perturbed nodes (budget)
        perturb_train_indices (bool): whether or whether not to perturb nodes from the training set

    Returns:
        np.array: indices of perturbed nodes
    """

    # select random indices from validation split, test split,
    # and (if existing) from remaining nodes
    # training nodes are assumed to be not-perturbed if specified

    if hasattr(data, 'train_mask'):
        remaining_indices = (~data.train_mask) & (~data.val_mask) & (~data.test_mask)
        remaining_indices = remaining_indices.nonzero().squeeze().tolist()
        val_indices = data.val_mask.nonzero().squeeze().tolist()
        test_indices = data.test_mask.nonzero().squeeze().tolist()
        train_indices = data.train_mask.nonzero().squeeze().tolist()

        sample_indices = [val_indices, test_indices, remaining_indices]
        if perturb_train_indices:
            sample_indices.append(train_indices)

        ind_perturbed = []
        for indices in sample_indices:
            n_perturbed = int(len(indices) * ood_budget_per_graph)
            _ind_perturbed = np.random.choice(indices, n_perturbed, replace=False)
            ind_perturbed.extend(_ind_perturbed)        

    else:
        indices = range(0, data.y.size(0))
        n_perturbed = int(len(indices) * ood_budget_per_graph)
        ind_perturbed = np.random.choice(indices, n_perturbed, replace=False).tolist()

    return ind_perturbed


def perturb_features(
        data, perturb_train_indices=False,
        ood_budget_per_graph=0.1, ood_noise_scale=1.0, ood_perturbation_type='bernoulli_0.5',
        ind_perturbed=None, **_):
    """perturb features in the graph

    Supported perturbation types are
        - reference
        - dense
        - bernoulli_most_common
        - bernoulli_least_common
        - bernoulli_0.5
        - bernoulli_0.05
        - bernoulli_0.95
        - not_normalized
        - uniform
        - scaled
        - normal

    Args:
        data (torch_geometric.data.Data): data object representing the graph
        perturb_train_indices (bool, optional): whether or whether not to perturb nodes in the training set. Defaults to False.
        ood_budget_per_graph (float, optional): fraction of perturbed nodes in the graph. Defaults to 0.1.
        ood_noise_scale (float, optional): optional scale-factor for feature perturbations. Defaults to 1.0.
        ood_perturbation_type (str, optional): type of perturbations. Defaults to 'bernoulli_0.5'.
        ind_perturbed (np.array, optional): optional array of perturbed indices. Defaults to None.

    Raises:
        ValueError: raised if unsupported perturbation type is passed

    Returns:
        Tuple[torch_geometric.data.Data, np.array]: tuple of data objects and perturbed indices
    """

    #perturbs the features of nodes with a random noise
    #where the budget corresponds to fraction of perturbed features

    dim_features = data.x.size(1)
    data = data.clone()

    if ind_perturbed is None:
        ind_perturbed = get_perturbed_indices(data, ood_budget_per_graph, perturb_train_indices)

    n_perturbed = len(ind_perturbed)

    noise = torch.zeros((n_perturbed, dim_features))
    eps = 1e-10

    if ood_perturbation_type == 'reference':
        assert ood_noise_scale == 1.0
        noise = data.x[ind_perturbed]

    elif ood_perturbation_type == 'dense':
        # normalized uniform noise
        noise = noise.uniform_()
        noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)

    elif ood_perturbation_type == 'bernoulli_most_common':
        # pick among 100 most-common words 20 at random
        words = list(range(data.x.size(1)))
        word_counts = Counter()

        for w in words:
            word_counts[w] = data.x[:, w].nonzero().size(0)

        most_common = [w[0] for w in word_counts.most_common(100)]
        for i in range(n_perturbed):
            # sample 20 most-common words
            size = np.random.choice([5, 10, 20], size=1)[0]
            idx = np.random.choice(len(most_common), size=size, replace=False)
            # create noise from least-common words
            noise[i, [most_common[i] for i in idx]] = 1.0

        noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)

    elif ood_perturbation_type == 'bernoulli_least_common':
        # pick among 100 least-common words 20 at random
        words = list(range(data.x.size(1)))
        word_counts = Counter()

        for w in words:
            word_counts[w] = data.x[:, w].nonzero().size(0)

        least_common = [w[0] for w in word_counts.most_common()[-100:-1]]
        for i in range(n_perturbed):
            # sample 5,10, or 20 least-common words
            size = np.random.choice([5, 10, 20], size=1)[0]
            idx = np.random.choice(len(least_common), size=size, replace=False)
            # create noise from least-common words
            noise[i, [least_common[i] for i in idx]] = 1.0

        noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)

    elif ood_perturbation_type == 'bernoulli_0.5':
        prob = 0.5
        noise = noise.bernoulli(prob)
        noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)

    elif ood_perturbation_type == 'bernoulli_0.05':
        prob = 0.05
        noise = noise.bernoulli(prob)
        noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)

    elif ood_perturbation_type == 'bernoulli_0.95':
        prob = 0.95
        noise = noise.bernoulli(prob)
        noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)

    elif ood_perturbation_type == 'not_normalized':
        noise = data.x[ind_perturbed]
        nz_ind = noise.bool()
        noise[nz_ind] = 1.0
        noise[~nz_ind] = 0.0

    elif ood_perturbation_type == 'uniform':
        noise = noise.uniform_()

    elif ood_perturbation_type == 'scaled':
        # simply select original values of perturbed features
        # later scale according to noise_scale
        assert ood_noise_scale > 1.0
        noise = data.x[ind_perturbed]

    elif ood_perturbation_type == 'normal':
        noise = noise.normal_()
        noise = ood_noise_scale * noise

    else:
        raise ValueError(f'perturbation {ood_perturbation_type} is not supported!')

    data.x[ind_perturbed] = ood_noise_scale * noise

    ood_mask = torch.zeros_like(data.y, dtype=bool)
    ood_mask[ind_perturbed] = True
    id_mask = ~ood_mask

    data.ood_mask = ood_mask
    data.id_mask = id_mask

    condition = hasattr(data, 'ood_val_mask')
    condition = condition & hasattr(data, 'ood_test_mask')
    condition = condition & hasattr(data, 'id_val_mask')
    condition = condition & hasattr(data, 'id_test_mask')

    if condition:
        data.ood_val_mask = data.ood_val_mask | (ood_mask & data.val_mask)
        data.id_val_mask = data.id_val_mask | (id_mask & data.val_mask)
        data.ood_test_mask = data.ood_test_mask | (ood_mask & data.test_mask)
        data.id_test_mask = data.id_test_mask | (id_mask & data.test_mask)

    elif hasattr(data, 'train_mask'):
        data.ood_val_mask = ood_mask & data.val_mask
        data.id_val_mask = id_mask & data.val_mask
        data.ood_test_mask = ood_mask & data.test_mask
        data.id_test_mask = id_mask & data.test_mask

    return data, ind_perturbed


def random_attack_targeted(
        data,
        ood_budget_per_graph=None,
        perturb_train_indices=True,
        ood_budget_per_node=0.25,
        ind_perturbed=None,
        **_):
    """perturb graph edges based on a random targeted attack

    Args:
        data (torch_geometric.data.Data): data object representing the graph
        ood_budget_per_graph (float, optional): fraction of perturbed nodes in the graph. Defaults to None.
        perturb_train_indices (bool, optional): whether or whether not to perturb training nodes. Defaults to True.
        ood_budget_per_node (float, optional): fraction of perturbed egdes per node. Defaults to 0.25.
        ind_perturbed (np.array, optional): optional array of perturbed indices. Defaults to None.

    Returns:
        Tuple[torch_geometric.data.Data, np.array]: tuple of perturbed graph data and array of perturbed indices
    """

    #performs a random attack on the graph structure
    #intended for a targeted setting (i.e. one target node under consideration)
    #for each targeted node v:
    #    - remove delta = budget[%] x degree of its original edges
    #    - add delta cross-community edges vu, s.t. c_v != c_u
    #

    data = data.clone()

    if ind_perturbed is None:
        ind_perturbed = get_perturbed_indices(data, ood_budget_per_graph, perturb_train_indices)

    # find nodes for each of the classes
    num_classes = data.y.max() + 1
    nodes_per_class = [None] * num_classes
    for c in range(num_classes):
        nodes_per_class[c] = (data.y == c).nonzero()

    # initialize
    target_nodes = torch.Tensor([])
    neighbors_to_delete = torch.Tensor([]).long()
    neighbors_to_insert = torch.Tensor([]).long()

    for i in ind_perturbed:
        c_i = data.y[i]

        # out-degree
        degree = (data.edge_index[0] == i).sum()
        # out-going neighbors
        neighbors = data.edge_index[1, data.edge_index[0] == i]

        budget = int(ood_budget_per_node * degree)

        # find all nodes from other communities
        cross_community_nodes = []
        for c in range(num_classes):
            if c == c_i:
                continue

            cross_community_nodes.extend(nodes_per_class[c])

        target = torch.Tensor([]).long()
        nodes_to_delete = torch.Tensor([]).long()
        nodes_to_insert = torch.Tensor([]).long()

        if budget > 0:
            # starting node
            target = torch.Tensor([i] * budget)

            # sample edges to delete
            weight = torch.zeros_like(data.y).float()
            weight[neighbors] = 1.0
            nodes_to_delete = torch.multinomial(weight, budget, replacement=False).long()

            # sample edges to insert
            weight = torch.zeros_like(data.y).float()
            weight[cross_community_nodes] = 1.0
            nodes_to_insert = torch.multinomial(weight, budget, replacement=False).long()

        target_nodes = torch.cat([target_nodes, target])
        neighbors_to_delete = torch.cat([neighbors_to_delete, nodes_to_delete])
        neighbors_to_insert = torch.cat([neighbors_to_insert, nodes_to_insert])

    # delete edges
    edge_indices_deleted = []
    for i, t in enumerate(target_nodes):
        neighbor = neighbors_to_delete[i]

        # outgoing edges
        start = data.edge_index[0] == t
        end = data.edge_index[1] == neighbor
        edge_index = (start & end).nonzero().item()
        edge_indices_deleted.append(edge_index)

        # incoming edges
        start = data.edge_index[0] == neighbor
        end = data.edge_index[1] == t
        edge_index = (start & end).nonzero().item()
        edge_indices_deleted.append(edge_index)

    # insert edges
    edges_inserted = []
    for i, t in enumerate(target_nodes):
        neighbor = neighbors_to_insert[i]

        # outgoing edges
        start = data.edge_index[0] == t
        end = data.edge_index[1] == neighbor

        # if edge to be "added" already contained: delete
        # and insert again to not break the pipeline
        if (start & end).sum() > 0:
            edge_index = (start & end).nonzero().item()
            edge_indices_deleted.append(edge_index)

        # define new edges
        edge = torch.Tensor([t, neighbor]).long()
        edges_inserted.append(edge)

        # incoming edges
        start = data.edge_index[0] == neighbor
        end = data.edge_index[1] == t
        if (start & end).sum() > 0:
            edge_index = (start & end).nonzero().item()
            edge_indices_deleted.append(edge_index)

        # define new edge
        edge = torch.Tensor([neighbor, t]).long()
        edges_inserted.append(edge)

    # finally delete edges from edge_index
    # i.e. collect all edges not being deleted
    edge_index = data.edge_index[:, [i for i in range(data.edge_index.size(1)) if i not in set(edge_indices_deleted)]]
    # current edges are ID edges

    if len(edges_inserted) > 0:
        edges_inserted = torch.stack(edges_inserted, dim=1)

        # finally insert edges
        data.edge_index = torch.cat([edge_index, edges_inserted], dim=1)

    else:
        data.edge_index = edge_index

    # define ID / OOD masks for nodes
    ood_mask = torch.zeros_like(data.y, dtype=bool)
    ood_mask[ind_perturbed] = True
    id_mask = ~ood_mask
    data.ood_mask = ood_mask
    data.id_mask = id_mask

    condition = hasattr(data, 'ood_val_mask')
    condition = condition & hasattr(data, 'ood_test_mask')
    condition = condition & hasattr(data, 'id_val_mask')
    condition = condition & hasattr(data, 'id_test_mask')

    if condition:
        data.ood_val_mask = data.ood_val_mask | (ood_mask & data.val_mask)
        data.id_val_mask = data.id_val_mask | (id_mask & data.val_mask)
        data.ood_test_mask = data.ood_test_mask | (ood_mask & data.test_mask)
        data.id_test_mask = data.id_test_mask | (id_mask & data.test_mask)

    elif hasattr(data, 'train_mask'):
        data.ood_val_mask = ood_mask & data.val_mask
        data.id_val_mask = id_mask & data.val_mask
        data.ood_test_mask = ood_mask & data.test_mask
        data.id_test_mask = id_mask & data.test_mask

    # define masks for edges
    num_kept_edges = edge_index.size(1)

    id_edges = torch.zeros(data.edge_index.size(1), device=data.y.device, dtype=bool)
    id_edges[0:num_kept_edges] = True
    ood_edges = ~id_edges

    if hasattr(data, 'id_edges') and hasattr(data, 'ood_edges'):
        data.id_edges = data.id_edges | id_edges
        data.ood_edges = data.ood_edges | ood_edges
    else:
        data.id_edges = id_edges
        data.ood_edges = ood_edges

    return data, ind_perturbed


def random_attack_dice(data, ood_budget_per_graph=0.1, **_):
    """perturb graph's egdges using the DICE attack

    Args:
        data (torch_geometric.data.Data): data object representing the graph
        ood_budget_per_graph (float, optional): fraction of perturbed edges in the graph. Defaults to 0.1.

    Returns:
        Tuple[torch_geometric.data.Data, None]: tuple of perturbed graph data and None (for pipeline compatibility)
    """

    #performs a random attack on the graph structure
    #intended for a un-targeted setting
    #special case: add as many edges as deleted, i.e.
    #budget corresponds to perturbed edges
    #for each targeted community:
    #    - remove delta = budget[%] x intra-community edges
    #    - add delta cross-community edges vu, s.t. c_v != c_u
    #'''
    data = data.clone()

    # extract directed edges of undirected edge_index
    start, end = data.edge_index
    mask = start < end
    start, end = start[mask], end[mask]

    # extract nodes belonging to community
    num_classes = data.y.max() + 1
    nodes_per_class = [None] * num_classes
    for c in range(num_classes):
        nodes_per_class[c] = (data.y == c).nonzero()

    # extract edges being intra-community-edges
    deleted_edges = []
    inserted_start = []
    inserted_end = []

    # for each community
    for c in range(num_classes):
        intra_community_edges = ((data.y[start] == c) & (data.y[end] == c)).nonzero().squeeze()

        # find all nodes from other communities
        cross_community_nodes = []
        for c_j in range(num_classes):
            if c_j == c:
                continue
            cross_community_nodes.extend(nodes_per_class[c_j])

        # calulcate budget
        n_intra_edges = len(intra_community_edges)
        budget = int(ood_budget_per_graph * n_intra_edges)

        # sample deleted
        deleted_edges.extend(np.random.choice(intra_community_edges, budget, replace=False))

        # first: sample community nodes, then sample corresponding cross-community nodes
        n_c = nodes_per_class[c]
        community_nodes = np.random.choice(n_c.squeeze(), budget, replace=True)
        cross_community_nodes = np.random.choice(cross_community_nodes, budget, replace=True)

        for i, n in enumerate(community_nodes):
            inserted_start.append(n)
            inserted_end.append(cross_community_nodes[i])

    # finalize kept and new edges
    kept = torch.ones_like(start, dtype=bool)
    kept[deleted_edges] = False
    kept_start = start[kept]
    kept_end = start[kept]

    inserted_start = torch.LongTensor(inserted_start)
    inserted_end = torch.LongTensor(inserted_end)

    start = torch.cat([kept_start, inserted_start])
    end = torch.cat([kept_end, inserted_end])

    edge_index = torch.stack(
        [torch.cat([kept_start, kept_end], dim=0),
         torch.cat([kept_end, kept_start], dim=0)], dim=0)

    inserted_edges = torch.stack(
        [torch.cat([inserted_start, inserted_end], dim=0),
         torch.cat([inserted_end, inserted_start], dim=0)], dim=0)

    data.edge_index = torch.cat(
        [edge_index, inserted_edges], dim=1)

    # for compatibility with pipeline: masks for nodes
    # define ID / OOD masks for nodes
    ood_mask = torch.zeros_like(data.y, dtype=bool)
    id_mask = ~ood_mask
    data.ood_mask = ood_mask
    data.id_mask = id_mask

    condition = hasattr(data, 'ood_val_mask')
    condition = condition & hasattr(data, 'ood_test_mask')
    condition = condition & hasattr(data, 'id_val_mask')
    condition = condition & hasattr(data, 'id_test_mask')

    if condition:
        data.ood_val_mask = data.ood_val_mask | (ood_mask & data.val_mask)
        data.id_val_mask = data.id_val_mask | (id_mask & data.val_mask)
        data.ood_test_mask = data.ood_test_mask | (ood_mask & data.test_mask)
        data.id_test_mask = data.id_test_mask | (id_mask & data.test_mask)

    elif hasattr(data, 'train_mask'):
        data.ood_val_mask = ood_mask & data.val_mask
        data.id_val_mask = id_mask & data.val_mask
        data.ood_test_mask = ood_mask & data.test_mask
        data.id_test_mask = id_mask & data.test_mask

    # define masks for edges
    num_kept_edges = edge_index.size(1)

    id_edges = torch.zeros(data.edge_index.size(1), device=data.y.device, dtype=bool)
    id_edges[0:num_kept_edges] = True
    ood_edges = ~id_edges

    if hasattr(data, 'id_edges') and hasattr(data, 'ood_edges'):
        data.id_edges = data.id_edges | id_edges
        data.ood_edges = data.ood_edges | ood_edges
    else:
        data.id_edges = id_edges
        data.ood_edges = ood_edges

    return data, None


def random_edge_perturbations(data, ood_budget_per_graph=0.1, **_):
    """perturb graph's edges at random by moving edge's end point at random

    Args:
        data (torch_geometric.data.Data): data object representing the graph
        ood_budget_per_graph (float, optional): fraction of perturbed edges in the graph. Defaults to 0.1.

    Returns:
        Tuple[torch_geometric.data.Data, None]: tuple of perturbed graph data and None (for pipeline compatibility)
    """

    data = data.clone()

    num_nodes = data.y.size(0)
    num_edges = data.edge_index.size(1)
    num_perturbed_edges = int(ood_budget_per_graph * num_edges)

    # randomly sample edge-indices to replace target node
    edge_indices = np.random.choice(np.arange(num_edges), num_perturbed_edges, replace=False)
    target = np.random.choice(np.arange(num_nodes), num_perturbed_edges, replace=True)

    # for sampled edges: replace target node with sampled nodes
    data.edge_index[1, edge_indices] = torch.from_numpy(target).to(data.y.device)

    # for compatibility with pipeline: masks for nodes
    # define ID / OOD masks for nodes
    ood_mask = torch.zeros_like(data.y, dtype=bool)
    id_mask = ~ood_mask
    data.ood_mask = ood_mask
    data.id_mask = id_mask

    condition = hasattr(data, 'ood_val_mask')
    condition = condition & hasattr(data, 'ood_test_mask')
    condition = condition & hasattr(data, 'id_val_mask')
    condition = condition & hasattr(data, 'id_test_mask')

    if condition:
        data.ood_val_mask = data.ood_val_mask | (ood_mask & data.val_mask)
        data.id_val_mask = data.id_val_mask | (id_mask & data.val_mask)
        data.ood_test_mask = data.ood_test_mask | (ood_mask & data.test_mask)
        data.id_test_mask = data.id_test_mask | (id_mask & data.test_mask)

    elif hasattr(data, 'train_mask'):
        data.ood_val_mask = ood_mask & data.val_mask
        data.id_val_mask = id_mask & data.val_mask
        data.ood_test_mask = ood_mask & data.test_mask
        data.id_test_mask = id_mask & data.test_mask

    # define masks for edges
    id_edges = torch.ones(data.edge_index.size(1), device=data.y.device, dtype=bool)
    id_edges[edge_indices] = False
    ood_edges = ~id_edges

    if hasattr(data, 'id_edges') and hasattr(data, 'ood_edges'):
        data.id_edges = data.id_edges | id_edges
        data.ood_edges = data.ood_edges | ood_edges
    else:
        data.id_edges = id_edges
        data.ood_edges = ood_edges

    return data, None
