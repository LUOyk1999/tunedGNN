# Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification (NeurIPS 2024)

[![arXiv](https://img.shields.io/badge/arXiv-2406.08993-b31b1b.svg)](https://arxiv.org/pdf/2406.08993)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-amazon-computers-1)](https://paperswithcode.com/sota/node-classification-on-amazon-computers-1?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-amazon-photo-1)](https://paperswithcode.com/sota/node-classification-on-amazon-photo-1?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-coauthor-cs)](https://paperswithcode.com/sota/node-classification-on-coauthor-cs?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-coauthor-physics)](https://paperswithcode.com/sota/node-classification-on-coauthor-physics?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-questions)](https://paperswithcode.com/sota/node-classification-on-questions?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-roman-empire)](https://paperswithcode.com/sota/node-classification-on-roman-empire?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-amazon-ratings)](https://paperswithcode.com/sota/node-classification-on-amazon-ratings?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-minesweeper)](https://paperswithcode.com/sota/node-classification-on-minesweeper?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-property-prediction-on-ogbn-proteins)](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-proteins?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-classification-on-pokec)](https://paperswithcode.com/sota/node-classification-on-pokec?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-property-prediction-on-ogbn-products)](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-products?p=classic-gnns-are-strong-baselines-reassessing) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classic-gnns-are-strong-baselines-reassessing/node-property-prediction-on-ogbn-arxiv)](https://paperswithcode.com/sota/node-property-prediction-on-ogbn-arxiv?p=classic-gnns-are-strong-baselines-reassessing)

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
@article{luo2024classic,
  title={Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification},
  author={Luo, Yuankai and Shi, Lei and Wu, Xiao-Ming},
  journal={arXiv preprint arXiv:2406.08993},
  year={2024}
}
```

## Poster

![gnn_poster-min.png](https://raw.githubusercontent.com/LUOyk1999/images/refs/heads/main/images/gnn_poster-min.png)
