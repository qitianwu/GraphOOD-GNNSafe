import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from logger import Logger_classify, Logger_detect, save_result
from data_utils import normalize, gen_normalized_adjs, evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, rand_splits, get_gpu_memory_map, count_parameters
from dataset import load_dataset
from parse import parser_add_main_args
from baselines import *
from gnnsafe import *

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_tr.y.shape) == 1:
    dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
if isinstance(dataset_ood_te, list):
    for data in dataset_ood_te:
        if len(data.y.shape) == 1:
            data.y = data.y.unsqueeze(1)
else:
    if len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

### get splits for all runs ###
if args.dataset in ['cora', 'citeseer', 'pubmed']:
    pass
else:
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

### print dataset info ###
c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]

print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
      + f"classes {c} | feats {d}")
print(f"ood tr dataset {args.dataset}: all nodes {dataset_ood_tr.num_nodes} | centered nodes {dataset_ood_tr.node_idx.shape[0]} | edges {dataset_ood_tr.edge_index.size(1)}")
if isinstance(dataset_ood_te, list):
    for i, data in enumerate(dataset_ood_te):
        print(f"ood te dataset {i} {args.dataset}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
else:
    print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")


### load method ###
if args.method == 'msp':
    model = MSP(d, c, args).to(device)
elif args.method in 'gnnsafe':
    model = GNNSafe(d, c, args).to(device)
elif args.method == 'OE':
    model = OE(d, c, args).to(device)
elif args.method == "ODIN":
    model = ODIN(d, c, args).to(device)
elif args.method == "Mahalanobis":
    model = Mahalanobis(d, c, args).to(device)

### loss function ###
if args.dataset in ('proteins', 'ppi'): # multi-label binary classification
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### metric for classification ###
if args.dataset in ('proteins', 'ppi', 'twitch'): # binary classification
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

### logger for result report ###
if args.mode == 'classify':
    logger = Logger_classify(args.runs, args)
else:
    logger = Logger_detect(args.runs, args)


model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    model.reset_parameters()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()

        if args.mode == 'classify':
            result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
            logger.add_result(run, result)

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%')
        else:
            result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)
            logger.add_result(run, result)

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'AUROC: {100 * result[0]:.2f}%, '
                      f'AUPR: {100 * result[1]:.2f}%, '
                      f'FPR95: {100 * result[2]:.2f}%, '
                      f'Test Score: {100 * result[-2]:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()


### Save results ###
if args.mode == 'detect':
    save_result(results, args)
