# # cifar, svhn pretrain
# # for seed in 0 1 2 3
# for seed in 0
# do
#     # for corruption_ratio in 0.0 0.1 0.3
#     for corruption_ratio in 0.1
#     do
#         for dataset in cifar100
#         do
#             python3 -m gex.pretrain.main \
#             --seed=${seed} \
#             --dataset=${dataset} \
#             --model=resnet_18 \
#             --corruption_ratio=${corruption_ratio} \
#             --cutmix_alpha=0.4
#         done
#     done
# done

# # for corruption_ratio in 0.1 0.3
# for corruption_ratio in 0.1
# do
#     # for seed in 0 1 2 3
#     for seed in 0
#     do
#         for dataset in cifar10 cifar100
#         do
#             # for if_method in la_fge el2n tracinrp fscore cl knn la_kfac randproj arnoldi
#             for if_method in la_fge
#             do
#                 if [ ${if_method} = 'tracinrp' ]
#                 then
#                     num_ens=5
#                 else
#                     num_ens=32
#                 fi

#                 # python3 -m gex.noisy.main \
#                 # --seed=${seed} \
#                 # --dataset=${dataset} \
#                 # --model=resnet_18 \
#                 # --corruption_ratio=${corruption_ratio} \
#                 # --num_ens=${num_ens} \
#                 # --ft_lr=0.05 \
#                 # --ft_step=800 \
#                 # --ft_lr_sched=cosine \
#                 # --if_method=${if_method} \
#                 # --cutmix_alpha=0.4

#                 python3 -m gex.relabel.main \
#                 --seed=${seed} \
#                 --dataset=${dataset} \
#                 --model=resnet_18 \
#                 --corruption_ratio=${corruption_ratio} \
#                 --num_ens=${num_ens} \
#                 --ft_lr=0.05 \
#                 --ft_step=800 \
#                 --ft_lr_sched=cosine \
#                 --if_method=${if_method} \
#                 --cutmix_alpha=0.4
#             done
#         done
#     done
# done

# for corruption_ratio in 0.1 0.3
for corruption_ratio in 0.1
do
    # for seed in 0 1 2 3
    for seed in 0
    do
        for dataset in cifar10 cifar100
        do
            # for if_method in la_fge el2n tracinrp fscore cl knn la_kfac randproj arnoldi
            for if_method in la_fge
            do
                if [ ${if_method} = 'tracinrp' ]
                then
                    num_ens=5
                else
                    num_ens=32
                fi

                python3 -m gex.noisy.main \
                --seed=${seed} \
                --dataset=${dataset} \
                --model=resnet_18 \
                --corruption_ratio=${corruption_ratio} \
                --num_ens=${num_ens} \
                --ft_lr=0.05 \
                --ft_step=800 \
                --ft_lr_sched=cosine \
                --if_method=${if_method} \
                --cutmix_alpha=0.4 \
                --use_relabel=True
            done
        done
    done
done