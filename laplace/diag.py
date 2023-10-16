import os

import numpy as np
import jax
import jax.numpy as jnp
import optax
from absl import flags
from tqdm import tqdm

from gex.utils import ckpt, metrics, mp, tool

FLAGS = flags.FLAGS

def loss_fn_proto(params, trainer, batch, rng):
    
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)
    if FLAGS.use_connect:
        new_params = vec_params * params
    else:
        new_params = params
    new_params = unravel_fn(new_params)
    
    logit = tool.forward(new_params, trainer, batch['x'], train=False)
    if FLAGS.problem_type=='reg':
        loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1).mean()
    elif FLAGS.problem_type=='cls':
        loss = - (jax.nn.log_softmax(logit) * optax.smooth_labels(batch['y'], FLAGS.label_smooth)).sum(axis=-1).mean()
    elif FLAGS.problem_type=='multitask':
        log_prob = jax.nn.log_sigmoid(logit)
        log_not_prob = jax.nn.log_sigmoid(-logit)
        label = (1-FLAGS.label_smooth) * batch['y'] + FLAGS.label_smooth * (1-batch['y'])
        loss = - (label * log_prob + (1-label) * log_not_prob).sum(axis=-1)
        
    return loss

def get_posterior_diag(trainer, dataset_opt, *args, **kwargs):
    
    # parsing settings
    rng = jax.random.PRNGKey(42)
    if 'loss_fn' in kwargs:
        loss_fn = kwargs['loss_fn']
    else:
        loss_fn = loss_fn_proto
    if 'ft_step' in kwargs:
        ft_step = kwargs['ft_step']
    else:
        ft_step = FLAGS.ft_step
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
    else:
        pretrain_dir = None
        
    hess_fn_p = jax.pmap(jax.hessian(loss_fn))
    trainer = mp.unreplicate(trainer)
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)
    
    print(f'Compute Hess')
    hess = 0.
    pbar = tqdm(range(ft_step))
    for _ in pbar:
        rng, rng_ = jax.random.split(rng)
        batch_tr = next(dataset_opt)
        if FLAGS.use_connect:
            params = jnp.ones_like(vec_params)
        else:
            params = vec_params
            
        hess += hess_fn_p(
            mp.replicate(params), 
            mp.replicate(trainer), 
            batch_tr, 
            mp.replicate(rng_),
            )
    hess = hess / ft_step
    # remove parallel axis
    hess = jnp.mean(hess, axis=0)
    
    print(f'Decompose Hessian')
    L, U = np.linalg.eigh(hess)
    L = L
    id_positive = L > 0
    L = L[id_positive]
    U = U[:,id_positive]
    
    print(f'Sample {FLAGS.num_ens} members')    
    posterior = []
    rng, rng_ = jax.random.split(rng)
    pert = jax.random.normal(rng, (len(vec_params), FLAGS.num_ens)) # (P, M)
    pert = U @ np.diag(L**(-0.5)) @ (U.T @ pert) # (P, M)
    
    trainer_ft = tool.init_trainer_ft_lin(trainer)
    for i in range(FLAGS.num_ens):
        trainer_ft = trainer_ft.replace(params=unravel_fn(pert[:,i]), offset=trainer.params)
        posterior.append(trainer_ft)
        
    if pretrain_dir is not None:
        from gex.laplace.inference import get_posthoc_hparams
        posthoc_dir = get_posthoc_hparams(pretrain_dir, True)
        ckpt.check_dir(f'{posthoc_dir}/curvature')
        np.save(f'{posthoc_dir}/curvature/L.npy', L)
        np.save(f'{posthoc_dir}/curvature/U.npy', U)
        ckpt.save_ens(posterior, posthoc_dir, 'full')
        
    return posterior