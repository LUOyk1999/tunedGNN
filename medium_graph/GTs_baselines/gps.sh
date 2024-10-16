
for layer in 1 2 3 4 5 6 7 8 9 10
do
for hidden_channels in 512 256 64
do
for dropout in 0.1 0.3 0.5 0.7
do
for head in 1 2 4
do

python main.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --layers $layer  --dropout $dropout  --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result --weight_decay 0.0
python main.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --layers $layer  --dropout $dropout  --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result --weight_decay 0.0
python main.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result --weight_decay 0.0

python main.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --local_layers $layer  --dropout $dropout --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result --weight_decay 0.0
python main.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --local_layers $layer  --dropout $dropout --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result --weight_decay 0.0
python main.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 2 --local_layers $layer  --dropout $dropout --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result --weight_decay 0.0

python main.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 2 --local_layers $layer  --dropout $dropout --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result --metric rocauc --weight_decay 0.0
python main.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 2 --local_layers $layer  --dropout $dropout --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result --metric rocauc --weight_decay 0.0
python main.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 2 --local_layers $layer  --dropout $dropout --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result --metric rocauc --weight_decay 0.0


python main.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 2 --local_layers $layer  --dropout 0.2 --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result --metric rocauc --weight_decay 0.0
python main.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 2 --local_layers $layer  --dropout 0.2 --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result --metric rocauc --weight_decay 0.0
python main.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 2 --local_layers $layer  --dropout 0.2 --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result --metric rocauc --weight_decay 0.0


python main.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.7 --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result --weight_decay 5e-5
python main.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.7 --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result --weight_decay 5e-5
python main.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.7 --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result --weight_decay 5e-5


python main.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.7 --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result --weight_decay 5e-5
python main.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.7 --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result --weight_decay 5e-5
python main.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.7 --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result --weight_decay 5e-5

python main.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.3 --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result
python main.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.3 --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result
python main.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.3 --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result

python main.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.5 --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result
python main.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.5 --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result
python main.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.5 --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result

python main.py --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.5 --num_heads $head --device $1 --model GPS --local_gnn_type GCN --save_result --weight_decay 0.0
python main.py --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.5 --num_heads $head --device $1 --model GPS --local_gnn_type GraphSAGE --save_result --weight_decay 0.0
python main.py --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 2 --local_layers $layer  --dropout 0.5 --num_heads $head --device $1 --model GPS --local_gnn_type GAT --save_result --weight_decay 0.0

done 
done 
done 
done 