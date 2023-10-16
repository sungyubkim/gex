for seed in 0 1 2 3
do
    for dataset in cifar100
    do
        # for if_method in la_fge tracinrp fscore el2n gexlr la_kfac randproj arnoldi random
        for if_method in la_fge tracinrp arnoldi randproj
        do
            if [ ${if_method} = 'tracinrp' ]
            then
                num_ens=5
            else
                num_ens=32
            fi
            
            python3 -m gex.pruning.main \
            --seed=${seed} \
            --dataset=${dataset} \
            --model=resnet_18 \
            --num_ens=${num_ens} \
            --ft_lr=0.01 \
            --ft_step=800 \
            --ft_lr_sched=constant \
            --if_method=${if_method}

            # # Train with sub-datasets
            # for num_filter in 5000 10000 15000 20000 25000
            # do
            #     python3 -m gex.pruning.main \
            #     --seed=${seed} \
            #     --dataset=${dataset} \
            #     --model=resnet_18 \
            #     --num_ens=${num_ens} \
            #     --ft_lr=0.01 \
            #     --ft_step=800 \
            #     --ft_lr_sched=constant \
            #     --if_method=${if_method} \
            #     --num_filter=${num_filter}
            # done
        done
    done
done