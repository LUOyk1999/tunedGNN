#gcn
python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 512 --epochs 2000 --lr 0.0005 --runs 2 --local_layers 5 --bn --device 0 --res 

#sage
python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 256 --epochs 2000 --lr 0.0005 --runs 2 --local_layers 4 --bn --device 0 --res --sage

