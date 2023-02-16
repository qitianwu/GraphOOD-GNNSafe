import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from gnns import *
import scipy.sparse
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd

class MaxLogits(nn.Module):
    def __init__(self, d, c, args):
        super(MaxLogits, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, c, dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            pred = torch.sigmoid(logits).unsqueeze(-1)
            pred = torch.cat([pred, 1- pred], dim=-1)
            max_logits = pred.max(dim=-1)[0]
            return max_logits.sum(dim=1)
        else:
            return logits.max(dim=1)[0]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_idx = dataset_ind.splits['train']
        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_idx]
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        return loss


class OE(nn.Module):
    def __init__(self, d, c, args):
        super(OE, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, c, dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            pred = torch.sigmoid(logits).unsqueeze(-1)
            pred = torch.cat([pred, 1- pred], dim=-1)
            max_logits = pred.max(dim=-1)[0]
            return max_logits.sum(dim=1)
        else:
            return logits.max(dim=1)[0]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_in_idx]
        logits_out = self.encoder(dataset_ood.x.to(device), dataset_ood.edge_index.to(device))[train_ood_idx]

        train_idx = dataset_ind.splits['train']
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        loss += 0.5 * -(logits_out.mean(1) - torch.logsumexp(logits_out, dim=1)).mean()
        return loss


class EnergyModel(nn.Module):
    def __init__(self, d, c, args):
        super(EnergyModel, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'sgc':
            self.encoder = SGC(in_channels=d, out_channels=c, hops=args.hops)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        return neg_energy

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_in_idx]
        logits_out = self.encoder(dataset_ood.x.to(device), dataset_ood.edge_index.to(device))[train_ood_idx]

        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in, dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        '''if args.dataset in ('proteins', 'ppi'):
            logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
            logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
        else:
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)
        if energy_in.shape[0] != energy_out.shape[0]:
            min_n = min(energy_in.shape[0], energy_out.shape[0])
            energy_in = energy_in[:min_n]
            energy_out = energy_out[:min_n]
        print(energy_in.mean().data, energy_out.mean().data)
        reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)
        # reg_loss = torch.mean(F.relu(energy_in - energy_out - args.m) ** 2)

        loss = sup_loss + args.lamda * reg_loss'''
        loss = sup_loss

        return loss


class EnergyProp(nn.Module):
    def __init__(self, d, c, args):
        super(EnergyProp, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'sgc':
            self.encoder = SGC(in_channels=d, out_channels=c, hops=args.hops)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, l=1, alpha=0.5):
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(l):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):

        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ('proteins', 'ppi'):
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        neg_energy_prop = self.propagation(neg_energy, edge_index, args.prop_layers, args.alpha)
        return neg_energy_prop[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)
        logits_in = self.encoder(x_in, edge_index_in)
        logits_out = self.encoder(x_out, edge_index_out)

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        '''if args.dataset in ('proteins', 'ppi'):
            logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
            logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
        else:
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)
        energy_prop_in = self.propagation(energy_in, edge_index_in, args.prop_layers, args.alpha)[train_in_idx]
        energy_prop_out = self.propagation(energy_out, edge_index_out, args.prop_layers, args.alpha)[train_ood_idx]

        if energy_prop_in.shape[0] != energy_prop_out.shape[0]:
            min_n = min(energy_prop_in.shape[0], energy_prop_out.shape[0])
            energy_prop_in = energy_prop_in[:min_n]
            energy_prop_out = energy_prop_out[:min_n]
        print(energy_prop_in.mean().data, energy_prop_out.mean().data)
        reg_loss = torch.mean(F.relu(energy_prop_in - args.m_in) ** 2 + F.relu(args.m_out - energy_prop_out) ** 2)
        # reg_loss = torch.mean(F.relu(energy_prop_in - energy_prop_out - args.m) ** 2)

        loss = sup_loss + args.lamda * reg_loss'''
        loss = sup_loss

        return loss

gpn_params = dict()
gpn_params["dim_hidden"] = 64
gpn_params["dropout_prob"] = 0.5
gpn_params["K"] = 10
gpn_params["add_self_loops"] = True
gpn_params["maf_layers"] = 0
gpn_params["gaussian_layers"] = 0
gpn_params["use_batched_flow"] = True
gpn_params["loss_reduction"] = 'sum'
gpn_params["approximate_reg"] = True
gpn_params["factor_flow_lr"] = None
gpn_params["flow_weight_decay"] = 0.0
gpn_params["pre_train_mode"] = 'flow'
gpn_params["alpha_evidence_scale"] = 'latent-new'
gpn_params["alpha_teleport"] = 0.1
gpn_params["entropy_reg"] = 0.0001
gpn_params["dim_latent"] = 32
gpn_params["radial_layers"] = 10
gpn_params["likelihood_type"] = None

from gpn.layers import APPNPPropagation
from gpn.layers import Density, Evidence
from gpn.utils import Prediction, apply_mask
from gpn.nn import uce_loss, entropy_reg

class GPN(nn.Module):
    def __init__(self, d, c, args):
        super(GPN, self).__init__()
        self.params = gpn_params
        self.params["dim_feature"] = d
        self.params["num_classes"] = c

        self.input_encoder = nn.Sequential(
            nn.Linear(d, self.params["dim_hidden"]),
            nn.ReLU(),
            nn.Dropout(p=self.params["dropout_prob"]))

        self.latent_encoder = nn.Linear(self.params["dim_hidden"], self.params["dim_latent"])

        use_batched = True if self.params["use_batched_flow"] else False
        self.flow = Density(
            dim_latent=self.params["dim_latent"],
            num_mixture_elements=c,
            radial_layers=self.params["radial_layers"],
            maf_layers=self.params["maf_layers"],
            gaussian_layers=self.params["gaussian_layers"],
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.params["alpha_evidence_scale"])

        self.propagation = APPNPPropagation(
            K=self.params["K"],
            alpha=self.params["alpha_teleport"],
            add_self_loops=self.params["add_self_loops"],
            cached=False,
            normalization='sym')

        self.detect_type = args.GPN_detect_type

        assert self.detect_type in ('Alea', 'Epist', 'Epist_wo_Net')
        assert self.params["pre_train_mode"] in ('encoder', 'flow', None)
        assert self.params['likelihood_type'] in ('UCE', 'nll_train', 'nll_train_and_val', 'nll_consistency', None)

    def reset_parameters(self):
        self.input_encoder = nn.Sequential(
            nn.Linear(self.params["dim_feature"], self.params["dim_hidden"]),
            nn.ReLU(),
            nn.Dropout(p=self.params["dropout_prob"]))

        self.latent_encoder = nn.Linear(self.params["dim_hidden"], self.params["dim_latent"])

        use_batched = True if self.params["use_batched_flow"] else False
        self.flow = Density(
            dim_latent=self.params["dim_latent"],
            num_mixture_elements=self.params["num_classes"],
            radial_layers=self.params["radial_layers"],
            maf_layers=self.params["maf_layers"],
            gaussian_layers=self.params["gaussian_layers"],
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.params["alpha_evidence_scale"])

        self.propagation = APPNPPropagation(
            K=self.params["K"],
            alpha=self.params["alpha_teleport"],
            add_self_loops=self.params["add_self_loops"],
            cached=False,
            normalization='sym')

    def forward(self, dataset, device):
        pred =  self.forward_impl(dataset, device)
        return pred.hard.unsqueeze(-1)

    def forward_impl(self, dataset, device):
        edge_index = dataset.edge_index.to(device) if dataset.edge_index is not None else dataset.adj_t.to(device)
        x = dataset.x.to(device)
        h = self.input_encoder(x)
        z = self.latent_encoder(h)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        if self.training:
            p_c = self.get_class_probalities(dataset).to(device)
            self.p_c = p_c
        else:
            p_c = self.p_c
        log_q_ft_per_class = self.flow(z) + p_c.view(1, -1).log()

        if '-plus-classes' in self.params["alpha_evidence_scale"]:
            further_scale = self.params["num_classes"]
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.params["dim_latent"],
            further_scale=further_scale).exp()

        alpha_features = 1.0 + beta_ft

        beta = self.propagation(beta_ft, edge_index)
        alpha = 1.0 + beta

        soft = alpha / alpha.sum(-1, keepdim=True)
        logits = None
        log_soft = soft.log()

        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # predictions and intermediary scores
            alpha=alpha,
            soft=soft,
            log_soft=log_soft,
            hard=hard,

            logits=logits,
            latent=z,
            latent_features=z,

            hidden=h,
            hidden_features=h,

            evidence=beta.sum(-1),
            evidence_ft=beta_ft.sum(-1),
            log_ft_per_class=log_q_ft_per_class,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=alpha_features.sum(-1),
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return pred

    def get_optimizer(self, lr: float, weight_decay: float):
        flow_lr = lr if self.params["factor_flow_lr"] is None else self.params["factor_flow_lr"] * lr
        flow_weight_decay = weight_decay if self.params["flow_weight_decay"] is None else self.params["flow_weight_decay"]

        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f'flow.{p[0]}' for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = torch.optim.Adam(flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay)
        model_optimizer = torch.optim.Adam(
            [{'params': flow_param_weights, 'lr': flow_lr, 'weight_decay': flow_weight_decay},
             {'params': params}],
            lr=lr, weight_decay=weight_decay)

        return model_optimizer, flow_optimizer

    def get_warmup_optimizer(self, lr: float, weight_decay: float):
        model_optimizer, flow_optimizer = self.get_optimizer(lr, weight_decay)

        if self.params["pre_train_mode"] == 'encoder':
            warmup_optimizer = model_optimizer
        else:
            warmup_optimizer = flow_optimizer

        return warmup_optimizer

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        train_in_idx = dataset_ind.splits['train']
        prediction = self.forward_impl(dataset_ind, device)
        y = dataset_ind.y[train_in_idx].to(device)
        alpha_train = prediction.alpha[train_in_idx]
        reg = self.params["entropy_reg"]
        return uce_loss(alpha_train, y, reduction=self.params["loss_reduction"]) + entropy_reg(alpha_train, reg,
                                                                                               approximate=True,
                                                                                               reduction=self.params[
                                                                                                   "loss_reduction"])

    def valid_loss(self, dataset_ind, device):
        val_idx = dataset_ind.splits['valid']
        prediction = self.forward_impl(dataset_ind, device)
        y = dataset_ind.y[val_idx].to(device)
        alpha_train = prediction.alpha[val_idx]
        reg = self.params["entropy_reg"]
        return uce_loss(alpha_train, y, reduction=self.params["loss_reduction"]) + entropy_reg(alpha_train, reg,
                                                                                               approximate=True,
                                                                                               reduction=self.params[
                                                                                                   "loss_reduction"])

    def detect(self, dataset, node_idx, device, args):
        pred = self.forward_impl(dataset, device)
        if self.detect_type == 'Alea':
            score = pred.sample_confidence_aleatoric[node_idx]
        elif self.detect_type == 'Epist':
            score = pred.sample_confidence_epistemic[node_idx]
        elif self.detect_type == 'Epist_wo_Net':
            score = pred.sample_confidence_features[node_idx]
        else:
            raise ValueError(f"Unknown detect type {self.detect_type}")

        return score

    def get_class_probalities(self, data):
        l_c = torch.zeros(self.params["num_classes"], device=data.x.device)
        train_idx = data.splits['train']
        y_train = data.y[train_idx]

        # calculate class_counts L(c)
        for c in range(self.params["num_classes"]):
            class_count = (y_train == c).int().sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        return p_c

'''sgcn_params = dict()
sgcn_params["seed"] = 42
sgcn_params["dim_hidden"] = 16
sgcn_params["dropout_prob"] = 0.5
sgcn_params["use_kernel"] = True
sgcn_params["lambda_1"] = 0.001
sgcn_params["teacher_training"] = True
sgcn_params["use_bayesian_dropout"] = False
sgcn_params["sample_method"] = 'log_evidence'
sgcn_params["num_samples_dropout"] = 10
sgcn_params["loss_reduction"] = None'''

from gpn.layers import GCNConv
import gpn.nn as unn
from gpn.nn import loss_reduce
import torch.distributions as D
from gpn.models.gdk import GDK
from gpn.utils import RunConfiguration, ModelConfiguration, DataConfiguration
from gpn.utils import TrainingConfiguration

class SGCN(nn.Module):
    def __init__(self, d, c, args):
        super(SGCN, self).__init__()
        self.params = dict()
        self.params = dict()
        self.params["seed"] = args.gkde_seed
        self.params["dim_hidden"] = args.gkde_dim_hidden
        self.params["dropout_prob"] = args.gkde_dropout_prob
        self.params["use_kernel"] = bool(args.gkde_use_kernel)
        self.params["lambda_1"] = args.gkde_lambda_1
        self.params["teacher_training"] = bool(args.gkde_teacher_training)
        self.params["use_bayesian_dropout"] = bool(args.gkde_use_bayesian_dropout)
        self.params["sample_method"] = args.gkde_sample_method
        self.params["num_samples_dropout"] = args.gkde_num_samples_dropout
        self.params["loss_reduction"] = args.gkde_loss_reduction

        self.params["dim_feature"] = d
        self.params["num_classes"] = c

        self.alpha_prior = None
        self.y_teacher = None

        self.conv1 = GCNConv(
            self.params["dim_feature"],
            self.params["dim_hidden"],
            cached=False,
            add_self_loops=True,
            normalization='sym')

        activation = []

        activation.append(nn.ReLU())
        activation.append(nn.Dropout(p=self.params["dropout_prob"]))

        self.activation = nn.Sequential(*activation)

        self.conv2 = GCNConv(
            self.params["dim_hidden"],
            self.params["num_classes"],
            cached=False,
            add_self_loops=True,
            normalization='sym')

        self.evidence_activation = torch.exp
        self.epoch = None

        self.detect_type = args.GPN_detect_type

        assert self.detect_type in ('Alea', 'Epist')

    def reset_parameters(self):
        self.alpha_prior = None
        self.y_teacher = None

        self.conv1 = GCNConv(
            self.params["dim_feature"],
            self.params["dim_hidden"],
            cached=False,
            add_self_loops=True,
            normalization='sym')

        activation = []

        activation.append(nn.ReLU())
        activation.append(nn.Dropout(p=self.params["dropout_prob"]))

        self.activation = nn.Sequential(*activation)

        self.conv2 = GCNConv(
            self.params["dim_hidden"],
            self.params["num_classes"],
            cached=False,
            add_self_loops=True,
            normalization='sym')

        self.evidence_activation = torch.exp
        self.epoch = None

    def forward(self, dataset, device):
        pred =  self.forward_impl(dataset, device)
        return pred.hard.unsqueeze(-1)

    def forward_impl(self, dataset, device):
        edge_index = dataset.edge_index.to(device) if dataset.edge_index is not None else dataset.adj_t.to(device)
        x = dataset.x.to(device)
        if self.training or (not self.params["use_bayesian_dropout"]):
            x = self.conv1(x, edge_index)
            x = self.activation(x)
            x = self.conv2(x, edge_index)
            evidence = self.evidence_activation(x)

        else:
            self_training = self.training
            self.train()
            samples = [None] * self.params["num_samples_dropout"]

            for i in range(self.params["num_samples_dropout"]):
                x = self.conv1(x, edge_index)
                x = self.activation(x)
                x = self.conv2(x, edge_index)
                samples[i] = x

            log_evidence = torch.stack(samples, dim=1)

            if self.params["sample_method"] == 'log_evidence':
                log_evidence = log_evidence.mean(dim=1)
                evidence = self.evidence_activation(log_evidence)

            elif self.params["sample_method"] == 'alpha':
                evidence = self.evidence_activation(log_evidence)
                evidence = evidence.mean(dim=1)

            else:
                raise AssertionError

            if self_training:
                self.train()
            else:
                self.eval()

        alpha = 1.0 + evidence
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
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=None,
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        if self.params["loss_reduction"] in ('sum', None):
            n_nodes = 1.0
            frac_train = 1.0

        else:
            n_nodes = dataset_ind.y.size(0)
            frac_train = dataset_ind.train_mask.float().mean()

        prediction = self.forward_impl(dataset_ind, device)

        alpha = prediction.alpha
        #n_nodes = data.y.size(0)
        #n_train = data.train_mask.sum()
        # bayesian risk of sum of squares
        alpha_train = alpha[dataset_ind.splits['train']]
        y = dataset_ind.y[dataset_ind.splits['train']]
        bay_risk = unn.bayesian_risk_sosq(alpha_train, y.to(device), reduction='sum')
        losses = {'BR': bay_risk * 1.0 / (n_nodes * frac_train)}

        # KL divergence w.r.t. alpha-prior from Gaussian Dirichlet Kernel
        if self.params["use_kernel"]:
            dirichlet = D.Dirichlet(alpha)
            alpha_prior = self.alpha_prior.to(alpha.device).detach()
            dirichlet_prior = D.Dirichlet(alpha_prior)
            KL_prior = D.kl.kl_divergence(dirichlet, dirichlet_prior)
            KL_prior = loss_reduce(KL_prior, reduction='sum')
            losses['KL_prior'] = self.params["lambda_1"] * KL_prior / n_nodes

        # KL divergence for teacher training
        if self.params["teacher_training"]:
            assert self.y_teacher is not None

            # currently only works for full-batch training
            # i.e. epochs == iterations
            if self.training:
                if self.epoch is None:
                    self.epoch = 0
                else:
                    self.epoch += 1

            y_teacher = self.y_teacher.to(prediction.soft.device).detach()
            lambda_2 = min(1.0, self.epoch * 1.0 / 200)
            categorical_pred = D.Categorical(prediction.soft)
            categorical_teacher = D.Categorical(y_teacher)
            KL_teacher = D.kl.kl_divergence(categorical_pred, categorical_teacher)
            KL_teacher = loss_reduce(KL_teacher, reduction='sum')
            losses['KL_teacher'] = lambda_2 * KL_teacher / n_nodes

        return losses['BR'] + losses['KL_prior'] + losses['KL_teacher']

    def valid_loss(self, dataset_ind, device):

        if self.params["loss_reduction"] in ('sum', None):
            n_nodes = 1.0
            frac_train = 1.0

        else:
            n_nodes = dataset_ind.y.size(0)
            frac_train = dataset_ind.splits['valid'].float().mean()

        prediction = self.forward_impl(dataset_ind, device)

        alpha = prediction.alpha
        #n_nodes = data.y.size(0)
        #n_train = data.train_mask.sum()
        # bayesian risk of sum of squares
        alpha_train = alpha[dataset_ind.splits['valid']]
        y = dataset_ind.y[dataset_ind.splits['valid']]
        bay_risk = unn.bayesian_risk_sosq(alpha_train, y.to(device), reduction='sum')
        losses = {'BR': bay_risk * 1.0 / (n_nodes * frac_train)}

        # KL divergence w.r.t. alpha-prior from Gaussian Dirichlet Kernel
        if self.params["use_kernel"]:
            dirichlet = D.Dirichlet(alpha)
            alpha_prior = self.alpha_prior.to(alpha.device)
            dirichlet_prior = D.Dirichlet(alpha_prior)
            KL_prior = D.kl.kl_divergence(dirichlet, dirichlet_prior)
            KL_prior = loss_reduce(KL_prior, reduction='sum')
            losses['KL_prior'] = self.params["lambda_1"] * KL_prior / n_nodes

        # KL divergence for teacher training
        if self.params["teacher_training"]:
            assert self.y_teacher is not None

            # currently only works for full-batch training
            # i.e. epochs == iterations
            if self.training:
                if self.epoch is None:
                    self.epoch = 0
                else:
                    self.epoch += 1

            y_teacher = self.y_teacher.to(prediction.soft.device)
            lambda_2 = min(1.0, self.epoch * 1.0 / 200)
            categorical_pred = D.Categorical(prediction.soft)
            categorical_teacher = D.Categorical(y_teacher)
            KL_teacher = D.kl.kl_divergence(categorical_pred, categorical_teacher)
            KL_teacher = loss_reduce(KL_teacher, reduction='sum')
            losses['KL_teacher'] = lambda_2 * KL_teacher / n_nodes

        return losses['BR'] + losses['KL_prior'] + losses['KL_teacher']

    def create_storage(self, dataset_ind, pretrained_model, device):
        # create storage for model itself

        # create kernel and load alpha-prior
        gdk_config = ModelConfiguration(
                model_name='GDK',
                num_classes=self.params["num_classes"],
                dim_features=self.params["dim_feature"],
                seed=self.params["seed"],
                init_no=1 # GDK only with init_no = 1
        )
        kernel = GDK(gdk_config)
        prediction = kernel(dataset_ind)
        self.alpha_prior = prediction.alpha.to(device)

        x = pretrained_model(dataset_ind, device)
        log_soft = F.log_softmax(x, dim=-1)
        soft = torch.exp(log_soft)
        self.y_teacher = soft.to(device)

    def detect(self, dataset, node_idx, device, args):
        pred = self.forward_impl(dataset, device)
        if self.detect_type == 'Alea':
            score = pred.sample_confidence_aleatoric[node_idx]
        elif self.detect_type == 'Epist':
            score = pred.sample_confidence_epistemic[node_idx]
        else:
            raise ValueError(f"Unknown detect type {self.detect_type}")

        return score

class ODIN(nn.Module):
    def __init__(self, d, c, args):
        super(ODIN, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, c, dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        odin_score = self.ODIN(dataset, node_idx, device, args.T, args.noise)
        torch.cuda.empty_cache()
        return torch.Tensor(-np.max(odin_score, 1))

    def ODIN(self, dataset, node_idx, device, temper, noiseMagnitude1):
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        data = dataset.x.to(device)
        data = Variable(data, requires_grad=True)
        edge_index = dataset.edge_index.to(device)
        outputs = self.encoder(data, edge_index)[node_idx]
        criterion = nn.CrossEntropyLoss()

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        # Using temperature scaling
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
        loss = criterion(outputs, labels)

        datagrad = torch.autograd.grad(loss, data)[0].detach()
        gradient = torch.sign(datagrad.data)
        # Normalizing the gradient to binary in {0, 1}
        #gradient = torch.ge(datagrad.data, 0)
        #gradient = (gradient.float() - 0.5) * 2

        '''gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
        gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
        gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)'''
        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))

        # Adding small perturbations to images
        tempInputs = torch.add(data.data.clone(), -noiseMagnitude1, gradient)
        outputs = self.encoder(Variable(tempInputs), edge_index)[node_idx]
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        return nnOutputs

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_idx = dataset_ind.splits['train']
        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_idx]
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        return loss

    '''def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_in_idx]
        logits_out = self.encoder(dataset_ood.x.to(device), dataset_ood.edge_index.to(device))[train_ood_idx]

        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in, dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        if args.dataset in ('proteins', 'ppi'):
            logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
            logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
        else:
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)
        if energy_in.shape[0] != energy_out.shape[0]:
            min_n = min(energy_in.shape[0], energy_out.shape[0])
            energy_in = energy_in[:min_n]
            energy_out = energy_out[:min_n]
        print(energy_in.mean().data, energy_out.mean().data)
        reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)
        # reg_loss = torch.mean(F.relu(energy_in - energy_out - args.m) ** 2)

        loss = sup_loss + args.lamda * reg_loss

        return loss'''


# noinspection PyUnreachableCode
class Mahalanobis(nn.Module):
    def __init__(self, d, c, args):
        super(Mahalanobis, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, c, dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, train_set, train_idx, test_set, node_idx, device, args):
        temp_list = self.encoder.feature_list(train_set.x.to(device), train_set.edge_index.to(device))[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        # print('get sample mean and covariance', count)
        num_classes = len(torch.unique(train_set.y))
        sample_mean, precision = self.sample_estimator(num_classes, feature_list, train_set, train_idx, device)
        in_score = self.get_Mahalanobis_score(test_set, node_idx, device, num_classes, sample_mean, precision, count-1, args.noise)
        return torch.Tensor(in_score)

    def get_Mahalanobis_score(self, test_set, node_idx, device,  num_classes, sample_mean, precision, layer_index, magnitude):
        '''
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index
        '''
        self.encoder.eval()
        Mahalanobis = []

        data, target = test_set.x.to(device), test_set.y[node_idx].to(device)
        edge_index = test_set.edge_index.to(device)
        data, target = Variable(data, requires_grad=True), Variable(target)

        out_features = self.encoder.intermediate_forward(data, edge_index, layer_index)[node_idx]
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        datagrad = autograd.grad(loss,data)[0]

        gradient = torch.ge(datagrad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        '''gradient.index_copy_(1, torch.LongTensor([0]).to(device),
                     gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).to(device),
                     gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).to(device),
                     gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7 / 255.0))'''

        tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = self.encoder.intermediate_forward(tempInputs, edge_index, layer_index)[node_idx]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())

        return np.asarray(Mahalanobis, dtype=np.float32)

    def sample_estimator(self, num_classes, feature_list, dataset, node_idx, device):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                 precision: list of precisions
        """
        import sklearn.covariance

        self.encoder.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct = 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        total = len(node_idx)
        output, out_features = self.encoder.feature_list(dataset.x.to(device), dataset.edge_index.to(device))
        output = output[node_idx]

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        target = dataset.y[node_idx].to(device)
        equal_flag = pred.eq(target).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(total):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
            for j in range(num_classes):
                if isinstance(list_features[out_count][j], int):
                    temp_list[j] = torch.tensor(list_features[out_count][j])
                else:
                    temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    if len((list_features[k][i] - sample_class_mean[k][i]).shape) == 1:
                        X = (list_features[k][i] - sample_class_mean[k][i]).unsqueeze(0)
                    else:
                        X = (list_features[k][i] - sample_class_mean[k][i])

                else:
                    try:
                        if len((list_features[k][i] - sample_class_mean[k][i]).shape) == 1:
                            X = torch.cat((X, (list_features[k][i] - sample_class_mean[k][i]).unsqueeze(0)), 0)
                        else:
                            X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                    except:
                        print(X.shape)
                        print((list_features[k][i] - sample_class_mean[k][i]).shape)
                        exit()

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            precision.append(temp_precision)

        # print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

        return sample_class_mean, precision

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):

        train_idx = dataset_ind.splits['train']
        logits_in = self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device))[train_idx]
        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        return loss