
for layer in 1 2 3 4 5 6 7 8 9 10
do
for dropout in 0.1 0.3 0.5 0.7
do
for hidden_channels in 64 256 512
do
for head in 2 4 8
do
for hop in 3 7 10
do

python main_nag.py --dataset amazon-ratings --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout  --num_heads $head --device $1 --model nagphormer --save_result --hops $hop --weight_decay 0.0

python main_nag.py --dataset roman-empire --hidden_channels $hidden_channels --epochs 2500 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nagphormer --save_result  --hops $hop --weight_decay 0.0

python main_nag.py --dataset minesweeper --hidden_channels $hidden_channels --epochs 2000 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nagphormer --save_result  --metric rocauc --hops $hop --weight_decay 0.0

python main_nag.py --dataset questions --hidden_channels $hidden_channels --epochs 1500 --lr 3e-5 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nagphormer --save_result  --metric rocauc --hops $hop --weight_decay 0.0

python main_nag.py --dataset amazon-computer --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nagphormer --save_result  --hops $hop --weight_decay 5e-5

python main_nag.py --dataset amazon-photo --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nagphormer --save_result  --hops $hop --weight_decay 5e-5

python main_nag.py --dataset coauthor-cs --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nagphormer --save_result  --hops $hop

python main_nag.py --dataset coauthor-physics --hidden_channels $hidden_channels --epochs 1500 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nagphormer --save_result  --hops $hop

python main_nag.py --dataset wikics --hidden_channels $hidden_channels --epochs 1000 --lr 0.001 --runs 3 --global_layers $layer  --dropout $dropout --num_heads $head --device $1 --model nagphormer --save_result  --hops $hop --weight_decay 0.0

done
done 
done 
done 
done