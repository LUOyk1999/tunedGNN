# Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification

arxiv: https://arxiv.org/pdf/2406.08993

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

## Running classic GNNs on large graphs

```bash
cd large_graph

# running experiment on ogbn-arxiv (full batch training)
bash arxiv.sh

# running experiment on ogbn-proteins (mini-batch training) 
# Base code of ogbn-proteins sourced from https://github.com/AiRyunn/BoT (Yangkun Wang, Jiarui Jin, Weinan Zhang, Yong Yu, Zheng Zhang, and David Wipf. Bag of tricks for node classification with graph neural networks. arXiv preprint arXiv:2103.13355, 2021.)
# Modifications have been applied to adapt to our experiment
bash proteins.sh

# running experiment on ogbn-products (mini-batch training)
bash products.sh
```
