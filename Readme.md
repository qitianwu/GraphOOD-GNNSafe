# GNNSafe: Out-of-Distribution Detection for Graph Data

The official implementation for ICLR23 paper "Energy-based Out-of-Distribution Detection for Graph Neural Networks"

## What's news

[2023.02.09] We release the early version of our codes for reproducibility (more detailed info will be updated soon).

[2023.02.16] We provide the implementation for all baseline models used in the experiments.

```bibtex
      @inproceedings{wu2023gnnsafe,
      title = {Energy-based Out-of-Distribution Detection for Graph Neural Networks},
      author = {Qitian Wu and Yiting Chen and Chenxiao Yang and Junchi Yan},
      booktitle = {International Conference on Learning Representations (ICLR)},
      year = {2023}
      }
```

## Dependence

- Ubuntu 16.04.6
- Cuda 10.2
- Pytorch 1.9.0
- Pytorch Geometric 2.0.3

More information about required packages is listed in `requirements.txt`.

## Problem Settings

Out-of-distribution (OOD) detection on graph-structured data: Given a set of training nodes (inter-connected as a graph), 
one needs to train a robust classifier that can effectively identify OOD nodes (that have disparate distributions than training nodes) in the test set.
In the meanwhile, the classifier should maintain decent classification performance on in-distribution testing nodes. Different from image data, OOD detection on graph data needs to handle data inter-dependence as compared below.

<img width="600" alt="image" src="https://user-images.githubusercontent.com/22075007/219937529-f7d57dbc-ca9d-445f-ae27-f8c244cf9158.png">

OOD detection often has two specific problem settings, which we introduce in the following figures in comparison with standard supervised learning.

<img width="800" alt="image" src="https://user-images.githubusercontent.com/22075007/219937584-6627f89e-803f-49e6-b3ce-553db7529806.png">

- ***Supervised learning:*** the training and testing are based on data from the same distribution, i.e., in-distribution (IND) data. We use IND-Tr/IND-Val/IND-Te to denote the train/valid/test sets of in-distribution data.

- ***OOD detection w/o exposure***: the training is based on pure IND-Tr, and the model is evaluated by the performance of discriminating IND-Te and out-of-distribution (OOD) data in test set (short as OOD-Te).

- ***OOD detection w/ OOD exposure***: besides IND-Tr, the training stage is exposed to extra OOD data (short as OOD-Tr),
and the model is evaluated on OOD-Te and IND-Te.


## Datasets and Splits

For comprehensive evaluation, we introduce new benchmark settings for OOD detection on graphs.  
For , we follow different principles to create the data splits to introduce distribution shifts.

<img width="900" alt="image" src="https://user-images.githubusercontent.com/22075007/219937890-d0739791-8e5b-4dda-b4ea-8f5653728b10.png">


Due to different properties of different datasets, we use different ways for splitting.

- **Twitch** (multi-graph dataset): This dataset contains multiple sub-graphs. We use subgraph DE as IND, subgraph EN as OODTr and subgraphs ES, FR, RU as OODTe.

- **Arxiv** (dataset with context info): This dataset is a single graph where each node has a time label, i.e., when the paper is published.
 We follow [1] using the time as domain information for splitting nodes into IND, OODTr and OODTe.

- **Cora/Amazon/Coauthor** (standard dataset): Each of these datasets contain one single graph. We use the original data as IND, and follow the public splits for train/valid/test partition.
As for OOD data, we modified the original dataset to obtain OODTr and OODTe, with three different ways:

    - Structure manipulation: adopt stochastic block model to randomly generate a graph for OOD data.
    - Feature interpolation: use random interpolation to create node features for OOD data. 
    - Label leave-out: use nodes with partial classes as IND and leave out others for OODTr and OODTe.



## Model Implementation

There are four versions of our proposed model used in the experiments:

- `gnn_no_prop_no_reg` (***Energy*** in paper): the basic model using energy-based OOD detector

- `gnn_no_prop_use_reg` (***Energy FT*** in paper): the basic model with regularization loss on extra OOD exposure (i.e., using OODTr)

- `gnn_use_prop_no_reg` (***GNNSafe*** in paper): the basic model plus energy belief propagation

- `gnn_use_prop_use_reg` (***GNNSafe++*** in paper): the final model using both regularization and propagation

Since our model is agnostic to specific GNN architectures, we implement various off-the-shelf GNNs as the classifier backbone
(including MLP/GCN/GAT/JKNet/MixHop)

## How to run the code

1. Install the required packages according to `requirements.txt`.

2. The datasets we used are publicly available from Pytorch Geometric and OGB Package, and will be automatically downloaded when running our training scripts.

3. Follow the guidelines below for running the codes for different purposes.

### Reproducing main results (Table 1 and 2)
 
We provide the commands with hyper-parameters for all datasets in `GNNSafe/run.sh`. 
For example, for Cora with structure manipulation as OOD, one can run the following scripts for training and evaluation of baseline ***MSP*** and our models.
```shell 
    ### Cora with structure ood
    python main.py --method msp --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device 1
    python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device 1
    python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device 1
    python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --device 1
    python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device 1
```

### Discussion results (Fig. 1, 2 and 3)

Under the `GNNSafe` folder, we also release the scripts for hyper-parameter searching in `run_hyper_search.sh`, discussion on hyper-parameter sensitivity in `run_discuss.sh`,
and obtain results for energy visualization (Fig. 1 in paper) in `run_visualize.sh`. The figure plot code is provided in `plot.ipynb`.

### More usability for our codes

1. Our pipeline also supports supervised node classification by using `--mode classify`. For example,
```shell 
    # MaxLogits with GCN on Cora
    python main.py --method msp --backbone gcn --dataset cora --ood_type structure --mode classify --use_bn --device 1
    python main.py --method msp --backbone mlp --dataset cora --ood_type structure --mode classify --use_bn --device 1
```

2. We also provide the interface for other datasets (e.g., citeseer, pubmed, ogbn-proteins, etc.) in `GNNSafe/dataset.py`.

3. For more OOD types, one can refer our codes in `GNNSafe/dataset.py` and flexibly modify the parameters or generation codes.

If you have any question when running the codes or trying to do more research based on our implementation, feel free to contact me via echo740@sjtu.edu.cn

## Reference

If you found our codes and introduced data splits useful, please consider citing our paper:
 
```bibtex
  @inproceedings{wu2023gnnsafe,
  title = {Energy-based Out-of-Distribution Detection for Graph Neural Networks},
  author = {Qitian Wu and Yiting Chen and Chenxiao Yang and Junchi Yan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
  }
```

[1] Qitian Wu et al., Handling Distribution Shifts on Graphs: An Invariance Perspective. In ICLR2022.


