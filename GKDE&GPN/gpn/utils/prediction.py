from __future__ import annotations

import torch
import attr
from .object import HalfFrozenObject


@attr.s(frozen=True)
class Prediction(HalfFrozenObject):
    """object specifying possible model predictions"""
    # softmax prediction
    soft: torch.Tensor = attr.ib(default=None)
    log_soft: torch.Tensor = attr.ib(default=None)
    aux_soft: torch.Tensor = attr.ib(default=None)
    aux_log_soft: torch.Tensor = attr.ib(default=None)

    # class prediction
    hard: torch.Tensor = attr.ib(default=None)

    # alpha prediction
    alpha: torch.Tensor = attr.ib(default=None)
    alpha_features: torch.Tensor = attr.ib(default=None)

    # hidden / latent variables
    x_hat: torch.Tensor = attr.ib(default=None)
    logits: torch.Tensor = attr.ib(default=None)
    logits_features: torch.Tensor = attr.ib(default=None)
    latent: torch.Tensor = attr.ib(default=None)
    latent_node: torch.Tensor = attr.ib(default=None)
    latent_features: torch.Tensor = attr.ib(default=None)
    hidden: torch.Tensor = attr.ib(default=None)
    hidden_features: torch.Tensor = attr.ib(default=None)

    # ensembles
    var_predicted: torch.Tensor = attr.ib(default=None)
    var: torch.Tensor = attr.ib(default=None)
    softs: torch.Tensor = attr.ib(default=None)

    # energy
    energy: torch.Tensor = attr.ib(default=None)
    energy_feaures: torch.Tensor = attr.ib(default=None)

    # structure
    p_c: torch.Tensor = attr.ib(default=None)
    p_u: torch.Tensor = attr.ib(default=None)
    p_uc: torch.Tensor = attr.ib(default=None)

    # log_prob / evidence scores
    chi: torch.Tensor = attr.ib(default=None)
    evidence: torch.Tensor = attr.ib(default=None)
    evidence_ft: torch.Tensor = attr.ib(default=None)
    evidence_nn: torch.Tensor = attr.ib(default=None)
    evidence_node: torch.Tensor = attr.ib(default=None)
    evidence_per_class: torch.Tensor = attr.ib(default=None)
    evidence_ft_per_class: torch.Tensor = attr.ib(default=None)

    ft_weight: torch.Tensor = attr.ib(default=None)
    nn_weight: torch.Tensor = attr.ib(default=None)

    log_ft: torch.Tensor = attr.ib(default=None)
    log_ft_per_class: torch.Tensor = attr.ib(default=None)
    log_nn: torch.Tensor = attr.ib(default=None)
    log_nn_per_class: torch.Tensor = attr.ib(default=None)
    log_node: torch.Tensor = attr.ib(default=None)

    # scores for prediction confidence
    prediction_confidence_aleatoric: torch.Tensor = attr.ib(default=None)
    prediction_confidence_epistemic: torch.Tensor = attr.ib(default=None)
    prediction_confidence_structure: torch.Tensor = attr.ib(default=None)

    # scores for sample confidence
    sample_confidence_aleatoric: torch.Tensor = attr.ib(default=None)
    sample_confidence_epistemic: torch.Tensor = attr.ib(default=None)
    sample_confidence_structure: torch.Tensor = attr.ib(default=None)
    sample_confidence_features: torch.Tensor = attr.ib(default=None)
    sample_confidence_neighborhood: torch.Tensor = attr.ib(default=None)

    # RGCN
    mu_1: torch.Tensor = attr.ib(default=None)
    mu_1p: torch.Tensor = attr.ib(default=None)
    mu_2: torch.Tensor = attr.ib(default=None)
    mu_2p: torch.Tensor = attr.ib(default=None)

    var_1: torch.Tensor = attr.ib(default=None)
    var_1p: torch.Tensor = attr.ib(default=None)
    var_2: torch.Tensor = attr.ib(default=None)
    var_2p: torch.Tensor = attr.ib(default=None)

    # BayesianGCN
    log_q: torch.Tensor = attr.ib(default=None)
    log_prior: torch.Tensor = attr.ib(default=None)

    # DUN
    act_vec: torch.Tensor = attr.ib(default=None)
    q: torch.Tensor = attr.ib(default=None)

    def collate(self, p_to_collate: Prediction) -> Prediction:
        for var_name, var_val in vars(self).items():
            if var_val is None:
                continue

            # get value from p
            if not isinstance(p_to_collate, (list, tuple)):
                p_to_collate = [p_to_collate]
            p_val = [getattr(p, var_name) for p in p_to_collate]
            self.set_value(var_name, torch.cat([var_val, *p_val]))

        return self

    def to(self, device, **kwargs) -> Prediction:
        for var_name, var_val in vars(self).items():
            if var_val is None:
                continue
            self.set_value(var_name, var_val.to(device, **kwargs))

        return self
