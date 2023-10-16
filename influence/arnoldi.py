from random import random
from absl import flags
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

from gex.utils import ckpt, metrics, mp, tool

FLAGS = flags.FLAGS

@jax.grad
def grad_fn(params, trainer, batch, rng):
    _, unravel_fn = tool.params_to_vec(trainer.params, True)
    logit = tool.forward(unravel_fn(params), trainer, batch['x'], rng=rng, train=False)
    if FLAGS.problem_type=='reg':
        loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1).mean()
    elif FLAGS.problem_type=='cls':
        loss = - (jax.nn.log_softmax(logit) * optax.smooth_labels(batch['y'], FLAGS.label_smooth)).sum(axis=-1).mean()
    elif FLAGS.problem_type=='multitask':
        log_prob = jax.nn.log_sigmoid(logit)
        log_not_prob = jax.nn.log_sigmoid(-logit)
        label = (1-FLAGS.label_smooth) * batch['y'] + FLAGS.label_smooth * (1-batch['y'])
        loss = - (label * log_prob + (1-label) * log_not_prob).sum(axis=-1).mean()
    return loss

@jax.pmap
@jax.grad
def hvp_fn(params, v, trainer, batch, rng):
    grad = grad_fn(params, trainer, batch, rng)
    gvp = (grad * v).sum()
    return gvp

def loss_fn(params, trainer, batch, rng):
    _, unravel_fn = tool.params_to_vec(trainer.params, True)
    logit = tool.forward(unravel_fn(params), trainer, batch['x'], rng=rng, train=False)
    if FLAGS.problem_type=='reg':
        loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1)
    elif FLAGS.problem_type=='cls':
        loss = - (jax.nn.log_softmax(logit) * optax.smooth_labels(batch['y'], FLAGS.label_smooth)).sum(axis=-1)
    elif FLAGS.problem_type=='multitask':
        log_prob = jax.nn.log_sigmoid(logit)
        log_not_prob = jax.nn.log_sigmoid(-logit)
        label = (1-FLAGS.label_smooth) * batch['y'] \
                + FLAGS.label_smooth * (1-batch['y'])
        loss = - (label * log_prob + (1-label) * log_not_prob).sum(axis=-1)
    return loss

@jax.pmap
def jvp(params, pert, trainer, batch, rng):
    f = lambda x: loss_fn(x, trainer, batch, rng)
    res = jax.jvp(f, [params], [pert])[1]
    return res

def arnoldi(trainer_r, dataset_tr, rng):
    trainer = mp.unreplicate(trainer_r)
    vec_params = tool.params_to_vec(trainer.params)
    vec_params_r = mp.replicate(vec_params)

    proj = []
    appr_mat = np.zeros((FLAGS.arnoldi_iter, FLAGS.arnoldi_iter-1))
    rng, rng_ = jax.random.split(rng)
    vec0 = jax.random.normal(rng_, vec_params.shape)
    vec0 = vec0 / jnp.sqrt((vec0**2).sum())
    proj.append(vec0)
    rng = jax.random.PRNGKey(42)
    
    for n in tqdm(range(FLAGS.arnoldi_iter-1)):
        rng, rng_ = jax.random.split(rng)
        batch_tr = next(dataset_tr)
        vec = hvp_fn(
            vec_params_r, 
            mp.replicate(proj[n]), 
            trainer_r, 
            batch_tr,
            mp.replicate(rng_),
            )
        vec = np.array(jnp.mean(vec, axis=0))
        
        for j, proj_vec in enumerate(proj):
            appr_mat[j, n] = (vec @ proj_vec)
            vec = vec - appr_mat[j, n] * proj_vec
            
        new_norm = np.sqrt((vec**2).sum())
        
        if new_norm < 1e-6:
            appr_mat[n+1, n] = 0
            vec = np.zeros_like(vec)
            proj.append(vec)
            break
            
        appr_mat[n+1, n] = new_norm
        vec = vec / appr_mat[n+1, n]
        vec = np.array(vec)
        proj.append(vec)
    
    return appr_mat, proj

def distill(appr_mat, proj):
    
    appr_mat = appr_mat[:-1, :]
    n = appr_mat.shape[0]

    # Make appr_mat Hermitian and tridiagonal when force_hermitian=True.
    for i in range(n):
        for j in range(n):
            if i - j > 1 or j - i > 1:
                appr_mat[i, j] = 0
    # Make Hermitian.
    appr_mat = .5 * (appr_mat + appr_mat.T)
    
    # Get eigenvalues / vectors for Hermitian matrix.
    eigvals, eigvecs = np.linalg.eigh(appr_mat)
    # Sort the eigvals by absolute value.
    # idx = np.argsort(np.abs(eigvals))
    # eigvals = eigvals[idx]
    # eigvecs = eigvecs[:, idx]
    
    reduced_projections = change_basis(
        eigvecs[:, -FLAGS.arnoldi_dim:],
        proj[:-1],
    )
            
    return eigvals[-FLAGS.arnoldi_dim:], reduced_projections

def change_basis(matrix, proj):
    out = []
    for j in range(matrix.shape[1]):
        element = np.zeros_like(proj[0])
        for i in range(matrix.shape[0]):
            element = element + matrix[i, j] * proj[i]
        out.append(element)
    return np.array(out)

def proj_dataset(dataset, num_samples, rand_proj, vec_params, trainer_r):
    pbar = tqdm(range(FLAGS.arnoldi_dim))
    dataset_jvp = np.zeros((FLAGS.arnoldi_dim, num_samples))
    vec_params_r = mp.replicate(vec_params)
    step_per_epoch = np.ceil(num_samples / FLAGS.batch_size_device).astype(int)
    rng = jax.random.PRNGKey(42)
    for i in pbar:
        rand_proj_i_r = mp.replicate(rand_proj[i])
        for b in range(step_per_epoch):
            # project batch onto random component i
            rng, rng_ = jax.random.split(rng)
            batch_tr = next(dataset)
            batch_jvp = jvp(
                vec_params_r, 
                rand_proj_i_r, 
                trainer_r, 
                batch_tr,
                mp.replicate(rng_),
                )
            batch_jvp = batch_jvp.reshape(-1) # (batch_size,)
            batch_tr_idx = batch_tr['idx'].reshape(-1)
            dataset_jvp[i, batch_tr_idx] = batch_jvp
    return dataset_jvp

def compute_influence_arnoldi(trainer_r, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    '''
    Compute influence function with Arnoldi iteration.
    
    References
    [1] Schioppa, Andrea, et al. "Scaling up influence functions." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.
    '''
    rng = jax.random.PRNGKey(42)
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
    else:
        pretrain_dir = None
    dataset_opt = kwargs['dataset_opt']
    num_train = kwargs['num_train']
    num_test = kwargs['num_test']
    
    trainer = mp.unreplicate(trainer_r)
    vec_params = tool.params_to_vec(trainer.params)
    appr_mat, proj = arnoldi(trainer_r, dataset_opt, rng)
    eigval_list, eigvec_list = distill(appr_mat, proj)
    # (arnoldi_dim), (arnoldi_dim, n_params)
    
    dataset_tr_jvp = proj_dataset(dataset_tr, num_train, eigvec_list, vec_params, trainer_r) 
    # (randproj_dim, n_dataset_tr)
    
    if self_influence:
        influence = np.diag((1/eigval_list)**0.5) @ dataset_tr_jvp
        influence = (influence**2).mean(axis=0)
    else:
        # compute reduced cross influence to prevent OOM issue.
        dataset_te_jvp = proj_dataset(dataset_te, num_test, eigvec_list, vec_params, trainer_r) 
        # (randproj_dim, n_dataset_te)
        dataset_te_jvp = dataset_te_jvp.mean(axis=1, keepdims=True)
        # (randproj_dim, 1)
        influence = (
            (dataset_te_jvp.T @ np.diag(1/eigval_list)) @ dataset_tr_jvp
            ).mean(axis=0)
        
    return influence