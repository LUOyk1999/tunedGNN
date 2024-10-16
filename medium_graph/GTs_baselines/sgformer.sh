
for hidden_channels in 64 512 256
do
for layer in 1 2 3 4 5 6 7 8 9 10
do
for dropout in 0.1 0.3 0.5 0.7
do 
for head in 1
do
for graph_weight in 0.5 0.8
do

python main.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout  --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 0.0

python main.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 0.0

python main.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --metric rocauc --weight_decay 0.0

python main.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --metric rocauc --weight_decay 0.0

python main.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 5e-5

python main.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 5e-5

python main.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight

python main.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight

python main.py --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 0.0

# with jk

python main.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout  --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 0.0 --jk

python main.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 0.0 --jk

python main.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --metric rocauc --weight_decay 0.0 --jk

python main.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --metric rocauc --weight_decay 0.0 --jk

python main.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 5e-5 --jk

python main.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 5e-5 --jk

python main.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --jk

python main.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --jk

python main.py --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --layers $layer  --dropout $dropout --num_heads $head --device $1 --model sgformer --save_result --use_graph --use_act --alpha 0.5 --use_residual --use_bn --graph_weight $graph_weight --weight_decay 0.0 --jk

done 
done 
done 
done 
done