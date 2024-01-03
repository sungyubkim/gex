python3 -m gex.pretrain.main \
    --dataset=mnist \
    --model=vgg \
    --corruption_ratio=0.1

python3 -m gex.noisy.main \
    --dataset=mnist \
    --model=vgg \
    --corruption_ratio=0.1 \
    --num_ens=8 \
    --ft_lr=0.05 \
    --ft_step=800 \
    --ft_lr_sched=cosine \
    --if_method=la_fge