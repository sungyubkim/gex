for seed in 0 1 2 3
do
    for if_method in la_fge tracinrp fscore el2n la_kfac randproj arnoldi random
    do
        if [ ${if_method} = 'tracinrp' ]
        then
            num_ens=5
        else
            num_ens=32
        fi
        
        python3 -m gex.pruning.main \
        --seed=${seed} \
        --dataset=mnist \
        --model=vgg \
        --num_ens=${num_ens} \
        --ft_lr=0.05 \
        --ft_step=800 \
        --ft_lr_sched=cosine \
        --if_method=${if_method}

        # Train with sub-datasets
        for num_filter in 10000 20000 30000 40000
        do
            python3 -m gex.pruning.main \
            --seed=${seed} \
            --dataset=mnist \
            --model=vgg \
            --num_ens=${num_ens} \
            --ft_lr=0.05 \
            --ft_step=800 \
            --ft_lr_sched=cosine \
            --if_method=${if_method} \
            --num_filter=${num_filter}
        done
    done
done