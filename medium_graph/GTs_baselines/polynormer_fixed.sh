## homophilic datasets
python main.py --dataset amazon-computer --hidden_channels 64 --epochs 1200 --lr 0.001 --runs 2 --local_layers 5  --global_layers 5  --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device 5 --save_result --model polynormer
python main.py --dataset amazon-photo --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 2 --local_layers 7  --global_layers 7  --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device $1 --save_result --model polynormer
python main.py --dataset coauthor-cs --hidden_channels 64 --epochs 1500 --lr 0.001 --runs 2 --local_layers 5  --global_layers 5  --weight_decay 5e-4 --dropout 0.3 --in_dropout 0.1 --num_heads 8 --device $1 --save_result --model polynormer
python main.py --dataset coauthor-physics --hidden_channels 32 --epochs 1500 --lr 0.001 --runs 2 --local_layers 5  --global_layers 5  --weight_decay 5e-4 --dropout 0.5 --in_dropout 0.1 --num_heads 8 --device $1 --save_result --model polynormer
python main.py --dataset wikics --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 2 --local_layers 7  --global_layers 7  --weight_decay 0.0 --dropout 0.5 --in_dropout 0.5 --num_heads 1 --device $1 --save_result --model polynormer

## heterophilic datasets
python main.py --dataset roman-empire --hidden_channels 64 --epochs 2500 --lr 0.001 --runs 2 --local_layers 10 --global_layers 10 --weight_decay 0.0 --dropout 0.3 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --device $1 --beta 0.5  --save_result --model polynormer
python main.py --dataset amazon-ratings --hidden_channels 256 --epochs 2500 --lr 0.001 --runs 2 --local_layers 10  --global_layers 10  --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 2 --device $1  --save_result--model polynormer
python main.py --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.001 --runs 2 --local_layers 10 --global_layers 10 --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 8 --metric rocauc --device $1 --save_result --model polynormer
python main.py --dataset questions --hidden_channels 64 --epochs 1500 --lr 3e-5 --runs 2 --local_layers 5  --global_layers 5  --weight_decay 0.0 --dropout 0.2 --global_dropout 0.5 --num_heads 8 --metric rocauc --device $1 --in_dropout 0.15 --beta 0.4 --pre_ln --save_result --model polynormer
