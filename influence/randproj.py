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

def loss_fn(params, trainer, batch, rng):
    _, unravel_fn = tool.params_to_vec(trainer.params, True)
    logit = tool.forward(unravel_fn(params), trainer, batch['x'], rng=rng, train=False)
    if FLAGS.problem_type=='reg':
        loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1)
    elif FLAGS.problem_type=='cls':
        loss = - (jax.nn.log_softmax(logit)*optax.smooth_labels(batch['y'], FLAGS.label_smooth)).sum(axis=-1)
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

def proj_dataset(dataset, num_samples, rand_proj, vec_params, trainer_r):
    pbar = tqdm(range(FLAGS.randproj_dim))
    dataset_jvp = np.zeros((FLAGS.randproj_dim, num_samples))
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

def compute_influence_randproj(trainer_r, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    '''
    Compute influence function with two approximations.
    1. Approximate Hessian to simple identity matrix. (H ≈ I)
    2. Sample unbiased estimator of G^T G = I where G is d x p matrix (d << p) This can reduce computation of instance-wise backprop. to mini-batch-wise backprop with JVP.
    
    References
    [1] Appendix E of Pruthi, Garima, et al. "Estimating training data influence by tracing gradient descent." Advances in Neural Information Processing Systems 33 (2020): 19920-19930.
    [2] Section 4 of Schioppa, Andrea, et al. "Scaling up influence functions." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.
    '''
    rng = jax.random.PRNGKey(42)
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
    else:
        pretrain_dir = None
    num_train = kwargs['num_train']
    num_test = kwargs['num_test']
    
    trainer = mp.unreplicate(trainer_r)
    vec_params = tool.params_to_vec(trainer.params)
    rng, rng_ = jax.random.split(rng)
    rand_proj = jax.random.normal(rng_, (FLAGS.randproj_dim, *vec_params.shape))
    
    dataset_tr_jvp = proj_dataset(dataset_tr, num_train, rand_proj, vec_params, trainer_r) 
    # (randproj_dim, n_dataset_tr)
    
    if self_influence:
        influence = (dataset_tr_jvp**2).mean(axis=0)
    else:
        # compute reduced cross influence to prevent OOM issue.
        dataset_te_jvp = proj_dataset(dataset_te, num_test, rand_proj, vec_params, trainer_r) # (randproj_dim, n_dataset_te)
        dataset_te_jvp = dataset_te_jvp.mean(axis=1, keepdims=True)
        # (randproj_dim, 1)
        influence = (dataset_te_jvp.T @ dataset_tr_jvp).mean(axis=0)
        
    return influence