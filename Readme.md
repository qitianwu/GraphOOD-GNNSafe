# GNNSafe: Out-of-Distribution Detection for Graph Data

The official implementation for ICLR23 paper "Energy-based Out-of-Distribution Detection for Graph Neural Networks"

## What's news

[2023.02.09] We release the early version of our codes for reproducibility (more detailed info will be updated soon).

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

More information about required packages is listed in `requirement.txt`.

## Problem Settings

The task of out-of-distribution (OOD) detection is defined as: Given a set of training data samples, 
one needs to train a robust classifier that can effectively identify OOD samples (that have disparate distributions than training data) in the test set.
In the meanwhile, the classifier should maintain decent classification performance on in-distribution testing data.

There are two specific cases (with increasing difficulties) widely studied in the literature:

- ***OOD detection with training OOD exposure***: the training stage is exposed to extra OOD data (have disparate distributions from in-distribution training data),
and the model is evaluated on ***unseen*** OOD data in test set (testing OOD data often stem from disparate distributions than training OOD data).

- ***OOD detection without exposure***: the training is based on pure in-distribution data, and the model is evaluated on OOD data in test set.

## Datasets and Splits

For comprehensive evaluation, we introduce new benchmark settings for OOD detection on graphs. 
For each dataset, following the above protocols, we create three data portions:

- ***In-distribution data (IND)***: the data set used by traditional supervised learning. It is further split into training/validation/testing subsets (short as INDTr/INDVal/INDTe).
The training subset is for computing the supervised loss training the classifier; the testing (resp. validation) subset is used for evaluating (resp. early-stopping)
the classifier on the classification task. 

- ***OOD data for training (OODTr)***: the data set used as training OOD exposure and for computing the regularization loss for the OOD detector.

- ***OOD data for testing (OODTe)***: the data set used for evaluating the OOD detection model. In specific, the evaluation is based on discriminating the in-distribution testing data
and OOD testing data.

Due to different properties of different datasets, we use different ways for splitting.

- **Cora/Amazon/Coauthor** (standard dataset): Each of these datasets contain one single graph. We use the original data as IND, and follow the public splits for train/valid/test partition.
As for OOD data, we modified the original dataset to obtain OODTr and OODTe, with three different ways:

    - Structure manipulation: use all nodes in the original graph as IND
    and adopt stochastic block model to randomly generate a graph for OOD data. The OODTr and OODTe are generated independently.
    - Feature interpolation: use all nodes in the original graph as IND 
    and adopt random interpolation to create node features for OOD data. The OODTr and OODTe are generated independently.
    - Label leave-out: use nodes with partial classes as IND and leave out others for OODTr and OODTe.

- **Twitch** (multi-graph dataset): This dataset contains multiple sub-graphs. We use subgraph DE as IND, subgraph EN as OODTr and subgraphs ES, FR, RU as OODTe.
A recent work [1] shows that there exist significant distribution shifts among different sub-graphs, i.e., social networks from particular regions.

- **Arxiv** (dataset with context info): This dataset is a single graph where each node has a time label, i.e., when the paper is published.
 We follow [1] using the time as context for splitting domains: papers before 2015 as IID, from 2016 to 2017 as OODTr, and in 2018/2019/2020 as OODTe

## Model Implementation

There are four versions of our proposed model used in the experiments:

- `gnn_no_prop_no_reg` (***Energy*** in paper): the basic model using energy-based OOD detector

- `gnn_no_prop_use_reg` (***Energy FT*** in paper): the basic model with regularization loss on extra OOD exposure (i.e., using OODTr)

- `gnn_use_prop_no_reg` (***GNNSafe*** in paper): the basic model plus energy belief propagation

- `gnn_use_prop_use_reg` (***GNNSafe++*** in paper): the final model using both regularization and propagation

Since our model is agnostic to specific GNN architectures, we implement various off-the-shelf GNNs as the classifier backbone
(including MLP/GCN/GAT/JKNet/MixHop)

## How to run the code

### Reproducing comparative results

The training and evaluation for all datasets in one run can be finished within 1 hour, which is convenient for model development and further research. 
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

### Discussion results

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


