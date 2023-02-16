# Baseline implementation (GKDE & GPN)

## Dependence
The dependence are the same as GNNSafe
- Ubuntu 16.04.6
- Cuda 10.2
- Pytorch 1.9.0
- Pytorch Geometric 2.0.3

## Implementation details
We integrate GPN[1] and GKDE[2] in our pipeline based on the [official code](https://github.com/stadlmax/Graph-Posterior-Network) of [1]. 
We place the official implementation of [1] in `GKDE&GPN/gpn`. We use the default hyper-parameter settings as in [1]. 

## How to run the code
We release the scripts to run GPN and GKDE in `GKDE&GPN/run_gkde_gpn.sh`

## Reference
[1] Maximilian Stadler et al., Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification. 
 in NeurIPS 2021.

[2] X. Zhao, F. Chen, S. Hu, and J.-H. Cho. Uncertainty aware semi-supervised learning on graph
data. Advances in Neural Information Processing Systems, 2020.