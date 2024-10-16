for hidden_channels in 64 128 256
do
for dropout in 0.1 0.3 0.5 0.7
do
for num_heads in 1 2 4
do
for layers in 1 2 3 4 5 6 7 8 9 10
do

# cora
python main.py --save_result --dataset cora --lr 0.01 --local_layers $layers --global_layers $layers \
    --hidden_channels $hidden_channels --weight_decay 5e-4 --dropout $dropout \
    --model polynormer \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --device $1 --runs 5 --num_heads $num_heads

# citeseer
python main.py --save_result --dataset citeseer --lr 0.005 --layers $layers \
    --hidden_channels $hidden_channels --weight_decay 0.01 --dropout $dropout \
    --model polynormer \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --device $1 --runs 5 --num_heads $num_heads

# pubmed
python main.py --save_result --dataset pubmed --lr 0.005 --layers $layers \
    --hidden_channels $hidden_channels --weight_decay 5e-4 --dropout $dropout \
     --rand_split_class --valid_num 500 --test_num 1000 \
     --model polynormer  \
    --rand_split_class --valid_num 500 --test_num 1000 \
    --seed 123 --device $1 --runs 5 --num_heads $num_heads

done
done
done
done



