from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from absl import flags
FLAGS = flags.FLAGS

from gex.utils import ckpt, metrics, mp, tool

def avg_list_of_trainer(list_of_trainer, dataset_opt=None):
    '''
    Average list of trainer to a single trainer (not replicated).
    [1] Izmailov, Pavel, et al. "Averaging weights leads to wider optima and better generalization." arXiv preprint arXiv:1803.05407 (2018).
    '''
    num_ens = len(list_of_trainer)
    res = 0.
    for i in range(num_ens):
        vec_params, unravel_fn = tool.params_to_vec(list_of_trainer[i].params, True)
        res += vec_params / num_ens
    avg_params = unravel_fn(res)
    trainer = list_of_trainer[0].replace(params=avg_params)
    if FLAGS.use_bn:
        trainer = mp.unreplicate(tool.update_bn(mp.replicate(trainer), dataset_opt))
    return trainer

def gauss_posterior(list_of_trainer, sample_num, dataset_opt=None):
    '''
    Generate Gaussian samples from list of trainer as list of trainer
    [1] Maddox, Wesley J., et al. "A simple baseline for bayesian uncertainty in deep learning." Advances in Neural Information Processing Systems 32 (2019).
    '''
    num_ens = len(list_of_trainer)
    # gather params
    vec_params_list = []
    for i in range(num_ens):
        vec_params, unravel_fn = tool.params_to_vec(list_of_trainer[i].params, True)
        vec_params_list.append(vec_params)
    vec_params_list = jnp.stack(vec_params_list)
    vec_params_avg = vec_params_list.mean(axis=0, keepdims=True)
    vec_params_std = vec_params_list.std(axis=0)
    
    result = []
    rng = jax.random.PRNGKey(0)
    for i in tqdm(range(sample_num)):
        rng, rng_diag, rng_low_rank = jax.random.split(rng, 3)
        
        pert_diag = vec_params_std * jax.random.normal(rng_diag, vec_params.shape) / np.sqrt(2)
        pert_low_rank = ((vec_params_list - vec_params_avg) * jax.random.normal(rng_low_rank, (num_ens,1))).sum(axis=0) / np.sqrt(2 * (num_ens - 1))
        pert = pert_diag + pert_low_rank
        
        member = list_of_trainer[0].replace(
            params = unravel_fn(vec_params_avg[0] + pert))
        if FLAGS.use_bn:
            member = mp.unreplicate(tool.update_bn(mp.replicate(member), dataset_opt))
        result.append(member)
    return result

def get_statistics(list_of_trainer):
    vec_params_list = []
    for i in range(len(list_of_trainer)):
        vec_params, unravel_fn = tool.params_to_vec(list_of_trainer[i].params, True)
        vec_params_list.append(vec_params)
    vec_params_list = jnp.stack(vec_params_list)
    vec_params_avg = vec_params_list.mean(axis=0)
    cov_low_rank = (vec_params_list - vec_params_avg.reshape(1,-1))
    return vec_params_avg, cov_low_rank

def fge_to_swa(posterior, posthoc_dir=None, dataset_opt=None):
    print(f'Compute SWA')
    trainer_swa = mp.replicate(avg_list_of_trainer(posterior, dataset_opt))
    if posthoc_dir is not None:
        ckpt.save_ckpt(trainer_swa, posthoc_dir)
    return trainer_swa

def fge_to_swag(posterior, posthoc_dir=None, dataset_opt=None):
    print(f'Compute SWAG')
    posterior_swag = gauss_posterior(
        posterior, 
        FLAGS.swag_num_ens,
        dataset_opt,
        )
    if posthoc_dir is not None:
        ckpt.save_ens(posterior_swag, posthoc_dir, 'swag')
    return posterior_swag