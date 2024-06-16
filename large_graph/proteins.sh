python protein.py --gpu 1 --mpnn gat --n-layers 7
python protein.py --gpu 2 --mpnn gat --n-layers 6
python protein.py --gpu 2 --mpnn gate --n-layers 7
python protein.py --gpu 2 --mpnn sage --n-epochs 1000
python protein.py --gpu 2 --mpnn gcn --n-layers 3 --n-epochs 100