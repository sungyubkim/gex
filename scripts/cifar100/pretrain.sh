# cifar, svhn pretrain
for seed in 0 1 2 3
do
    for corruption_ratio in 0.0 0.1 0.3
    do
        for dataset in cifar100
        do
            python3 -m gex.pretrain.main \
            --seed=${seed} \
            --dataset=${dataset} \
            --model=resnet_18 \
            --corruption_ratio=${corruption_ratio} 
        done
    done
done