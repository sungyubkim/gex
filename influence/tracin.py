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

@jax.pmap
@jax.grad
def grad_fn(params, trainer, batch, rng):
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
    return loss.sum()

def proj_dataset(dataset, num_samples, vec_params, trainer_r):
    dataset_jvp = np.zeros(num_samples)
    vec_params_r = mp.replicate(vec_params)
    step_per_epoch = np.ceil(num_samples / FLAGS.batch_size_device).astype(int)
    rng = jax.random.PRNGKey(42)
    for b in tqdm(range(step_per_epoch)):
        # project batch onto random component i
        batch_tr = next(dataset)
        len_batch = batch_tr['x'].shape[1]
        batch_jvp = []
        for i in range(len_batch):
            rng, rng_ = jax.random.split(rng)
            sub_batch = {k: v[:,i:i+1] for k, v in batch_tr.items()}
            grad = grad_fn(vec_params_r, trainer_r, sub_batch, mp.replicate(rng_))
            grad_norm_sq = np.array((grad**2).sum(axis=-1))
            batch_jvp.append(grad_norm_sq)
        batch_jvp = np.stack(batch_jvp, axis=1)
        batch_jvp = batch_jvp.reshape(-1) # (batch_size,)
        batch_tr_idx = batch_tr['idx'].reshape(-1)
        dataset_jvp[batch_tr_idx] = batch_jvp
    return dataset_jvp

def compute_influence_tracin(trainer_r, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    '''
    Compute influence function with TracInRP (random projection)
    Note that we do not restrict trainer_list to pre-train trajectory in this function.
    
    References
    [1] Pruthi, Garima, et al. "Estimating training data influence by tracing gradient descent." Advances in Neural Information Processing Systems 33 (2020): 19920-19930.
    '''
    
    rng = jax.random.PRNGKey(42)
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
    else:
        pretrain_dir = None
    num_train = kwargs['num_train']
    num_test = kwargs['num_test']
    
    trainer = mp.unreplicate(trainer_r)
    if FLAGS.tracinrp_fge:
        from gex.laplace.inference import get_posterior
        trainer_list = get_posterior(
            trainer_r,
            la_method='fge',
            pretrain_dir=pretrain_dir,
        )
    else:
        trainer_list = ckpt.load_ens(
            trainer, 
            FLAGS.num_ens, 
            pretrain_dir, 
            'traj',
            )
    vec_params = tool.params_to_vec(trainer.params)
    
    pbar = tqdm(trainer_list)
    influence = 0.
    for t, trainer in enumerate(pbar):
        trainer_r = mp.replicate(trainer)
        vec_params = tool.params_to_vec(trainer.params)
        dataset_tr_jvp = proj_dataset(dataset_tr, num_train, vec_params, trainer_r)
        
        influence += dataset_tr_jvp
    
    influence = influence / len(trainer_list)
        
    return influence