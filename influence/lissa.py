from absl import flags
import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from time import time

from gex.utils import ckpt, metrics, mp, tool

FLAGS = flags.FLAGS

@jax.grad
def samplewise_grad_fn(params, trainer, batch):
    _, unravel_fn = tool.params_to_vec(trainer.params, True)
    logit = tool.forward(unravel_fn(params), trainer, batch['x'], train=False)
    if FLAGS.problem_type=='reg':
        loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1).mean()
    elif FLAGS.problem_type=='cls':
        loss = - (jax.nn.log_softmax(logit)*optax.smooth_labels(batch['y'], FLAGS.label_smooth)).sum(axis=-1).mean()
    return loss

@jax.grad
def grad_fn(params, trainer, batch):
    _, unravel_fn = tool.params_to_vec(trainer.params, True)
    logit = tool.forward(unravel_fn(params), trainer, batch['x'], train=False)
    if FLAGS.problem_type=='reg':
        loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1).mean()
    elif FLAGS.problem_type=='cls':
        loss = - (jax.nn.log_softmax(logit)*optax.smooth_labels(batch['y'], FLAGS.label_smooth)).sum(axis=-1).mean()
    wd = 0.5 * (tool.params_to_vec(params)**2).sum()
    return loss + FLAGS.weight_decay * wd

@jax.pmap
@jax.grad
def hvp_fn(params, v, trainer, batch):
    grad = grad_fn(params, trainer, batch)
    gvp = (grad * v).sum()
    return gvp

def inverse_hvp_lissa(v, trainer, dataset_opt, num_samples=1):
    '''
    Inverse Hessian-Vector product (IHVP) computation with LiSSA
    [1] https://github.com/awslabs/aws-cv-unique-information/blob/master/sample_info/modules/influence_functions.py#L66:~:text=def-,inverse_hvp_lissa,-(model%2C
    [2] https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py#L475:~:text=def-,get_inverse_hvp_lissa,-(self%2C
    '''
    
    vec_params = tool.params_to_vec(mp.unreplicate(trainer).params)
    result = None
    for i in range(num_samples):
        cur_estimate = v
        for _ in range(FLAGS.lissa_iter):
            batch = next(dataset_opt)
            hvp = hvp_fn(mp.replicate(vec_params), cur_estimate, trainer, batch)
            cur_estimate = v + (1. - FLAGS.lissa_damping) * cur_estimate \
                            - hvp/FLAGS.lissa_scale
        if result is None:
            result = cur_estimate/FLAGS.lissa_scale
        else:
            result = result + cur_estimate/FLAGS.lissa_scale
    result = result/num_samples
    return result

def compute_influence_lissa(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    
    dataset_opt = kwargs['dataset_opt']
    vec_params = tool.params_to_vec(mp.unreplicate(trainer).params)
    samplewise_grad_fn_p = jax.pmap(samplewise_grad_fn)
    
    pbar = tqdm(dataset_tr)
    influence = []
    for b, batch_tr in enumerate(pbar):
        if b==1:
            # remove first iteration to exclude compile time
            start_time = time()
        influence.append([]) # (b)
        for i in range(batch_tr['x'].shape[1]):
            instance_tr = {'x':batch_tr['x'][:,i][:,np.newaxis], 'y':batch_tr['y'][:,i][:,np.newaxis]}
            grad_tr_sample = samplewise_grad_fn_p(mp.replicate(vec_params), trainer, instance_tr) # (N_device, p)
            influence[b].append([]) # (b, i)
            for d in range(batch_tr['x'].shape[0]):
                ihvp = inverse_hvp_lissa(mp.replicate(grad_tr_sample[d]), trainer, dataset_opt) 
                ihvp = ihvp.mean(axis=0) # (p)
                if self_influence:
                    elem = (ihvp * grad_tr_sample).sum()
                    influence[b][i].append(elem)
                    pbar.set_postfix({'influence':f'{elem:.4f}'})
                    # (b, i, d)
                else:
                    influence[b][i].append([]) # (b, i, d)
                    for b_, batch_te in enumerate(dataset_te):
                        influence[b][i][d].append([]) # (b, i, d, b_)
                        for i_ in range(batch_te['x'].shape[1]):
                            instance_te = {'x':batch_te['x'][:,i_][:,np.newaxis], 'y':batch_te['y'][:,i_][:,np.newaxis]}
                            grad_te_sample = samplewise_grad_fn_p(mp.replicate(vec_params), trainer, instance_te) # (N_device, p)
                            influence[b][i][d][b_].append([])
                            # (b, i, d, b_, i_)
                            for d_ in range(batch_te['x'].shape[0]):
                                elem = (ihvp * grad_te_sample[d_]).sum()
                                influence[b][i][d][b_][i_].append(elem)
                                pbar.set_postfix({'influence':f'{elem:.4f}'})
                                # (b, i, d, b_, i_, d_)
                        
    influence = np.array(influence)
    if self_influence:
        # (b, i, d) -> (b * i * d)
        influence = influence.reshape(-1)
    else:
        # (b, i, d, b_, i_, d_) -> (b * i * d, b_ * i_ * d)
        b, i, d, b_, i_, d_ = influence.shape
        influence = influence.reshape(b, i, d, b_ * i_ * d_).reshape(b * i * d, b_ * i_ * d)
    end_time = time()
    print(f'Computation time except compile : {end_time - start_time:.4} s')
    return influence