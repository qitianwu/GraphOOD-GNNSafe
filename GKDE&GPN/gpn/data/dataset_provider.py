import copy
import torch
import torch_geometric.data as td
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToSparseTensor
import numpy as np
import gpn.data as ud


class InMemoryDatasetProvider(td.InMemoryDataset):
    """InMemoryDatasetProvider

    Wrapper for a torch_geometric dataset which makes it compatible to our pipeline intended for usage with different OOD datasets.
    """

    def __init__(self, dataset):
        super().__init__()

        self.data_list = list(dataset)
        self._num_classes = dataset.num_classes
        self._num_features = dataset.num_features
        self._to_sparse = ToSparseTensor(
            remove_edge_index=True,
            fill_cache=True)

    @property
    def num_classes(self):
        return self._num_classes

    def set_num_classes(self, n_c):
        self._num_classes = n_c

    @property
    def num_features(self):
        return self._num_features

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def loader(self, batch_size=1, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def clone(self, shallow=False):
        self_clone = copy.copy(self)
        if not shallow:
            self_clone.data_list = [d.clone() for d in self.data_list]

        return self_clone

    def to(self, device, **kwargs):
        for i, l in enumerate(self.data_list):
            self.data_list[i] = l.to(device, **kwargs)

        return self

    def to_sparse(self):
        for i, l in enumerate(self.data_list):
            self.data_list[i] = self._to_sparse(l)

        return self


class OODInMemoryDatasetProvider(InMemoryDatasetProvider):
    """OODInMemoryDatasetProvider

    Wrapper which takes an existing InMemoryDatasetProvider to make it a perturbed dataset compatible with our pipeline.
    This dataset provider considers global graph perturbations, e.g. perturbations of a certain fraction of nodes or edges.
    """
    
    def perturb_dataset(self, **perturbation_kwargs):
        ood_type = perturbation_kwargs['ood_type']
        del perturbation_kwargs['ood_type']

        if ood_type == 'perturb_features':
            perturbation = ud.perturb_features

        elif ood_type == 'leave_out_classes':
            perturbation = ud.get_ood_split

        elif ood_type == 'random_attack_targeted':
            perturbation = ud.random_attack_targeted

        elif ood_type == 'random_attack_dice':
            perturbation = ud.random_attack_dice

        elif ood_type == 'random_edge_perturbations':
            perturbation = ud.random_edge_perturbations

        else:
            raise AssertionError

        for i, d in enumerate(self.data_list):

            d_p, n_c = perturbation(d, **perturbation_kwargs)

            self.data_list[i] = d_p

        if 'leave_out_classes' in ood_type:
            self.set_num_classes(n_c)


class OODIsolatedInMemoryDatasetProvider(InMemoryDatasetProvider):
    """OODIsolatedInMemoryDatasetProvider

    Wrapper which takes an existing InMemoryDatasetProvider to make it a perturbed dataset compatible with our pipeline.
    This dataset provider considers isolated graph perturbations, i.e. perturbations of one node at a time.
    """
    
    def __init__(self, tg_dataset, perturbation_type,
                 ood_noise_scale=1.0,
                 ood_perturbation_type='bernoulli_0.5',
                 ood_budget_per_node=0.25,
                 root='',
                 **_):

        assert len(tg_dataset) == 1
        super().__init__(tg_dataset)

        self.perturbation = None
        self.perturbation_kwargs = None

        NUM_NODES_ISOLATED = 100

        if perturbation_type == 'perturb_features':
            # for now: only on a fixed number of nodes to speed-up things
            test_indices = torch.nonzero(self.data_list[0].test_mask, as_tuple=False)
            random_indices = np.random.choice(range(len(test_indices)), NUM_NODES_ISOLATED, replace=False)
            test_indices = test_indices[random_indices]
            self.ind_perturbed = test_indices.squeeze().sort(dim=0).values

            self.perturbation = ud.perturb_features
            self.perturbation_kwargs = {
                'perturb_train_indices': False,
                'ood_noise_scale': ood_noise_scale,
                'ood_perturbation_type': ood_perturbation_type,
                'rood': root,
            }

        elif perturbation_type == 'random_attack_targeted':
            # for now: sample 40 nodes at random for comparison with existing
            # versions of random attacks
            test_indices = torch.nonzero(self.data_list[0].test_mask, as_tuple=False)
            random_indices = np.random.choice(range(len(test_indices)), NUM_NODES_ISOLATED, replace=False)
            test_indices = test_indices[random_indices]
            self.ind_perturbed = test_indices.squeeze().sort(dim=0).values

            self.perturbation = ud.random_attack_targeted
            self.perturbation_kwargs = {
                'perturb_train_indices': False,
                'ood_budget_per_node': ood_budget_per_node
            }

        else:
            raise AssertionError

    def __len__(self):
        # len corresponds to number of (test/val) nodes in graph
        return len(self.ind_perturbed)

    def __getitem__(self, index):
        # index corresponds to node number
        ind_perturbed = [self.ind_perturbed[index].item()]
        d_p, _ = self.perturbation(self.data_list[0], ind_perturbed=ind_perturbed, **self.perturbation_kwargs)
        return d_p
