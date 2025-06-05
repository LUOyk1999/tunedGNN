# 🔍 Research Series on Classic GNNs

| Benchmarking Series: Reassessing Classic GNNs | Paper |
| - | - |
| **_[Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification](https://github.com/LUOyk1999/tunedGNN) (NeurIPS 2024)_** | [Link](https://openreview.net/forum?id=xkljKdGe4E) |
| [Can Classic GNNs Be Strong Baselines for Graph-level Tasks?](https://github.com/LUOyk1999/GNNPlus) (ICML 2025) | [Link](https://arxiv.org/abs/2502.09263) | 

# Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification (NeurIPS 2024)

[![OpenReview](https://img.shields.io/badge/OpenReview-xkljKdGe4E-b31b1b.svg)](https://openreview.net/forum?id=xkljKdGe4E) [![arXiv](https://img.shields.io/badge/arXiv-2406.08993-b31b1b.svg)](https://arxiv.org/pdf/2406.08993)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-property-prediction-on-ogbn-proteins)](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-proteins?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-property-prediction-on-ogbn-products)](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-products?p=classic-gnns-are-strong-baselines-reassessing)

## Python environment setup with Conda

Tested with Python 3.7, PyTorch 1.12.1, and PyTorch Geometric 2.3.1, dgl 1.0.2.

```bash
pip install pandas
pip install scikit_learn
pip install numpy
pip install scipy
pip install einops
pip install ogb
pip install pyyaml
pip install googledrivedownloader
pip install networkx
pip install gdown
pip install matplotlib
```

## Overview

* `./medium_graph` Experiment code on medium graphs.

* `./large_graph` Experiment code on large graphs.

## Reference

If you find our codes useful, please consider citing our work

```
@inproceedings{
luo2024classic,
title={Classic {GNN}s are Strong Baselines: Reassessing {GNN}s for Node Classification},
author={Yuankai Luo and Lei Shi and Xiao-Ming Wu},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://openreview.net/forum?id=xkljKdGe4E}
}
```

## Poster

![gnn-min.png](https://raw.githubusercontent.com/LUOyk1999/images/refs/heads/main/images/gnn-min.png)
