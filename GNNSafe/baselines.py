import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import *
import numpy as np

from torch.autograd import Variable
import torch.autograd as autograd

class MSP(nn.Module):
    def __init__(self, d, c, args):
        super(MSP, self).__init__()
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
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
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
            max_sp = pred.max(dim=-1)[0]
            return max_sp.sum(dim=1)
        else:
            sp = torch.softmax(logits, dim=-1)
            return sp.max(dim=1)[0]

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

        datagrad = autograd.grad(loss, data)[0]
        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(datagrad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        '''gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
        gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
        gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)'''
        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))

        # Adding small perturbations to images
        tempInputs = torch.add(data.data, -noiseMagnitude1, gradient)
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
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

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