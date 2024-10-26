## Dataset

Chameleon and Squirrel: one can download the datasets from the google drive link below:
https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link (provided by Qitian Wu and Wentao Zhao and Chenxiao Yang and Hengrui Zhang and Fan Nie and Haitian Jiang and Yatao Bian and Junchi Yan, Simplifying and empowering transformers for large-graph representations. In Thirty-seventh Conference on Neural Information Processing Systems, 2023b.)

Download the geom-gcn folder, place it in `./data/` and unzip it. And we use the [new splits](https://github.com/yandex-research/heterophilous-graphs/tree/main) for Chameleon and Squirrel, that filter out the overlapped nodes.
Download `chameleon_filtered.npz`, put it into `./data/geom-gcn/chameleon/`.
Download `squirrel_filtered.npz`, put it into `./data/geom-gcn/squirrel/`.

## Run classic GNNs on medium graphs

GNNs:
```bash
sh run_gnn.sh 0
```

Results:
```bash
Dataset:

GNN lr hidden_channels layers dropout layer_norm batch_norm residual_connections

Cora：

GCN 0.001 512 3 0.7 False False False 85.10 ± 0.67 

GraphSAGE 0.001 256 3 0.7 False False False 83.88 ± 0.65 

GAT 0.001 512 3 0.2 False False True 84.46 ± 0.55 

Citeseer：

GCN 0.001 512 2 0.5 False False False 73.14 ± 0.67 

GraphSAGE 0.001 512 3 0.2 False False False 72.26 ± 0.55 

GAT 0.001 256 3 0.5 False False True 72.22 ± 0.84 

Pubmed：

GCN 0.005 256 2 0.7 False False False 81.12 ± 0.52

GraphSAGE 0.005 512 4 0.7 False False False 79.72 ± 0.50 

GAT 0.01 512 2 0.5 False False False 80.28 ± 0.64 

Computer:

GCN 0.001 512 3 0.5 True False False 93.99 ± 0.12

GraphSAGE 0.001 64 4 0.3 True False False 93.25 ± 0.14  

GAT 0.001 64 2 0.5 True False False 94.09 ± 0.37 

Photo:

GCN 0.001 256 6 0.5 True False True 96.10 ± 0.46 

GraphSAGE 0.001 64 6 0.2 True False True 96.78 ± 0.23 

GAT 0.001 64 3 0.5 True False True 96.60 ± 0.33 

CS:

GCN 0.001 512 2 0.3 True False True 96.17 ± 0.06

GraphSAGE 0.001 512 2 0.5 True False True 96.38 ± 0.11 

GAT 0.001 256 1 0.3 True False True 96.21 ± 0.14 

Physics:

GCN 0.001 64 2 0.3 True False True 97.46 ± 0.10 

GraphSAGE 0.001 64 2 0.7 False True True 97.19 ± 0.18 

GAT 0.001 256 2 0.7 False True True 97.25 ± 0.06 

WikiCS:

GCN 0.001 256 3 0.5 True False False 80.30 ± 0.62 

GraphSAGE 0.001 256 2 0.7 True False False 80.69 ± 0.31 

GAT 0.001 512 2 0.7 True False True 81.07 ± 0.54 

Squirrel:

GCN 0.01 256 4 0.7 False True True 45.01 ± 1.63 

GraphSAGE 0.01 256 3 0.7 False True True 40.78 ± 1.47 

GAT 0.005 512 7 0.5 False True True 41.73 ± 2.07 

Chameleon:

GCN 0.005 512 5 0.2 False False False 46.29 ± 3.40 

GraphSAGE 0.01 256 4 0.7 False True True 44.81 ± 4.74 

GAT 0.01 256 2 0.7 False True True 44.13 ± 4.17  

Amazon-Ratings:

GCN 0.001 512 4 0.5 False True True 53.80 ± 0.60 

GraphSAGE 0.001 512 9 0.5 False True True 55.40 ± 0.21 

GAT 0.001 512 4 0.5 False True True 55.54 ± 0.51 

Minesweeper:

GCN 0.01 64 12 0.2 False True True 97.86 ± 0.24 

GraphSAGE 0.01 64 15 0.2 False True True 97.77 ± 0.62 

GAT 0.01 64 15 0.2 False True True 97.73 ± 0.73 

Roman-Empire:

GCN 0.001 512 9 0.5 False True True 91.27 ± 0.20 

GraphSAGE 0.001 256 9 0.3 False True False 91.06 ± 0.27

GAT 0.001 512 10 0.3 False True True 90.63 ± 0.14 

Questions:

GCN 3e-05 512 10 0.3 False False True 79.02 ± 0.60 

GraphSAGE 3e-05 512 6 0.2 True False False 77.21 ± 1.28

GAT 3e-05 512 3 0.2 True False True 77.95 ± 0.51 
```
