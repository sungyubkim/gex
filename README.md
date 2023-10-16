# Geometric Ensemble for X (GEX)

## Supporting post-hoc methods

* Uncertainty estimation

    ```python
    from gex.laplace.inference import get_posterior
    list_of_trainer = get_posterior(trainer, dataset_opt)
    ```

    * Full curvature (only for smalle nets & datasets): `--la_method=full`
    * Kronecker-Factored Approximate Curvature (K-FAC) : `--la_method=kfac`
    * Randomize-Then-Optimize (rto) : `--la_method=rto`
    * Fast Geometric Ensemble (FGE) : `--la_method=fge`

* Influence estimations

    ```python
    from gex.influence.estimate import compute_influence
    # to compute influence kernel (N_tr, N_te) between train-test
    influence_kernel = compute_influence(trainer, dataset_tr, dataset_te, dataset_opt \
    , self_influence=False)
    # to compute self-influence (N_tr) for train dataset
    influence_kernel = compute_influence(trainer, dataset_tr, dataset_te, dataset_opt \
    , self_influence=True)
    ```

    * LiSSA (`--if_method=lissa`)
    * K-FAC (`--if_method=kfac`)
    * Random Projection (`--if_method=randproj`)
    * TracIn Random Projection (`--if_method=tracinrp`)
    * Arnoldi (`--if_method=arnoldi`)
    * Laplace approximation with full curvature (`--if_method=la_full`)
    * Laplace approximation with K-FAC (`--if_method=la_kfac`)
    * Geometric Ensemble (`--if_method=la_fge`)

## Our key motivation for influence estimation

* Observation 1: Influence function is predictive covariance of Linearized Laplace (LL) posterior:

$$
\mathcal{I}(z,z') = \mathrm{Cov}(\ell^\mathrm{lin}(z,\theta), \ell^\mathrm{lin}(z', \theta))
$$

* Observation 2: Self-influence is predictive variance of of Linearized Laplace (LL) posterior:

$$
\mathcal{I}(z,z) = \mathrm{Var}(\ell^\mathrm{lin}(z,\theta))
$$

## How to use this repo?

* Pull docker image for dependency

```shell
docker pull sungyubkim/jax:ntk-0.4.2
```

* Run docker image

```shell
docker run -p 8080:8080/tcp -it --rm --gpus all \
--ipc=host -v $PWD:/root -w /root \
sungyubkim/jax:ntk-0.4.2
```

* To run a single python file, 

```shell
# to pre-train NN
python3 -m gex.pretrain.main \
    --dataset=mnist \
    --model=vgg \
    --corruption_ratio=0.1

# to estimate influence of pre-trained NN
python3 -m gex.noisy.main \
    --dataset=mnist \
    --model=vgg \
    --corruption_ratio=0.1 \
    --num_ens=8 \
    --ft_lr=0.05 \
    --ft_step=800 \
    --ft_lr_sched=cosine \
    --if_method=la_fge
```

* To run multiple python files at once with `./gex/{task}/total.sh`

```shell
bash gex/mnist/total.sh
```

* Basically, results files (e.g., log, checkpoints, plots) will be saved in 

```shell
./gex/{task}/result/{pretrain_hyperparameter_settings}/{posthoc_hyperparameter_settings}
```