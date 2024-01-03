from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from absl import flags
from tqdm import tqdm
import optax
from time import time
import os
import pickle

from gex.utils import ckpt, metrics, mp, tool
# import kfac_jax
# from kfac_jax._src import utils

FLAGS = flags.FLAGS

def get_posterior_kfac(trainer, dataset_opt, *args, **kwargs):
    
    # parsing settings
    rng = jax.random.PRNGKey(42)
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    else:
        alpha = FLAGS.alpha
    if 'sigma' in kwargs:
        sigma = kwargs['sigma']
    else:
        sigma = FLAGS.sigma
    num_train = kwargs['num_train']
    problem_type = FLAGS.problem_type
    if 'ft_step' in kwargs:
        ft_step = kwargs['ft_step']
    else:
        ft_step = FLAGS.ft_step
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
    else:
        pretrain_dir = None
    
    trainer_ft = tool.init_trainer_ft_lin(mp.unreplicate(trainer))
    num_devices = jax.device_count()
    
    # construct K-FAC pre-conditioner
    def loss_fn(params, batch):
        f_j = partial(tool.forward, train=False)
        logit = f_j(params, mp.unreplicate(trainer), batch['x'])
        if problem_type=='reg':
            loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1).mean()
            kfac_jax.register_squared_error_loss(logit, batch['y'])
        elif problem_type=='cls':
            loss = - (jax.nn.log_softmax(logit)*batch['y']).sum(axis=-1).mean()
            kfac_jax.register_softmax_cross_entropy_loss(logit, batch['y'])
        return loss
    
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=(sigma/alpha)**2/num_train,
        learning_rate_schedule=optax.constant_schedule(0.0),
        momentum_schedule=optax.constant_schedule(0.0),
        damping_schedule=optax.constant_schedule(0.0),
        multi_device=True,
        estimation_mode='fisher_gradients'
    )
    params = trainer.params
    opt_state = optimizer.init(
        params, 
        mp.replicate(rng), 
        batch=next(dataset_opt),
        )
    
    print(f'Construct K-FAC pre-condition for {ft_step} mini-batches')    
    for step in tqdm(range(ft_step)):
        if step==1:
            # remove first iteration to exclude compile time
            start_time = time()
        rng, rng_ = jax.random.split(rng)
        batch = next(dataset_opt)
        params, opt_state, stats = optimizer.step(
            params, opt_state, mp.replicate(rng_), 
            batch=batch, global_step_int=step
        )
    opt_state.damping = mp.replicate([0.,])
    end_time = time()
    print(f'Cache time except compile : {end_time - start_time:.4} s')
    
    estimator = optimizer._estimator
    estimator_state = opt_state.estimator_state
    
    num_stage = int(np.ceil(float(FLAGS.num_ens)/num_devices))
    posterior = []
    for i in tqdm(range(num_stage)):
        rng, rng_ = jax.random.split(rng)
        pert = sample(
                estimator,
                estimator_state,
                trainer,
                jax.random.split(rng_, num_devices),
                (sigma/alpha)**2/num_train,
        )
        for j in range(num_devices):
            params = jax.tree_util.tree_map(
                lambda x:x*sigma/np.sqrt(num_train), mp.unreplicate(pert,j))
            if FLAGS.use_connect: 
                params = jax.tree_util.tree_map(
                    lambda x,y:x*y, params, trainer_ft.offset)
            trainer_ft = trainer_ft.replace(params=params)
            posterior.append(trainer_ft)
            
    # if pretrain_dir is not None:
    #     from gex.laplace.inference import get_posthoc_hparams
    #     posthoc_dir = get_posthoc_hparams(pretrain_dir, True)
    #     ckpt.save_ens(posterior, posthoc_dir, 'kfac')
    
    return posterior

@partial(jax.pmap, static_broadcasted_argnums=(0, 4))
def sample(estimator, estimator_state, trainer, rng, identity_weight):
    
    power = -0.5
    blocks_params = estimator.params_vector_to_blocks_vectors(trainer.params)
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)
    vectors = jax.random.normal(rng, vec_params.shape, jnp.float32)
    vectors = unravel_fn(vectors)
    blocks_vectors = estimator.params_vector_to_blocks_vectors(vectors)
    identity_weight = utils.to_tuple_or_repeat(identity_weight, estimator.num_blocks)
    
    result = []
    for i in range(estimator.num_blocks):
        block = estimator.blocks[i]
        block_state = estimator_state.blocks_states[i]
        block_param = blocks_params[i]
        block_vector = blocks_vectors[i]
        block_identity_weight = identity_weight[i]
        
        if hasattr(block_state, 'diagonal_factors'):
            res = diag_matpower(block, block_state, block_param, block_vector, block_identity_weight, power)
        
        elif hasattr(block_state, 'inputs_factor'):
            res = kfac_matpower_norm(block, block_state, block_param, block_vector, block_identity_weight, power)
            
        elif hasattr(block_state, 'matrix'):
            res = full_matpower(block, block_state, block_param, block_vector, block_identity_weight, power)
            
        result.append(res)
        
    result = estimator.blocks_vectors_to_params_vector(result)
    
    return result

def diag_matpower(block, state, param, vector, identity_weight, power):
    scale = 100 # Since Orphan layers use sq. of avg. grad. instead of avg. of sq. of grad. we scale for NaiveDiagonal.
    if FLAGS.use_connect:
        factors = tuple(f.value * (p**2) * scale + identity_weight 
                        for p, f in zip(param, state.diagonal_factors))
    else:
        factors = tuple(f.value * scale + identity_weight 
                        for f in state.diagonal_factors)
    assert len(factors) == len(vector)
    result = tuple(jnp.power(f, power) * v for f, v in zip(factors, vector))
    return result

def full_matpower(block, state, param, vector, identity_weight, power):
    param =  block.parameters_list_to_single_vector(param)
    vector = block.parameters_list_to_single_vector(vector)
    
    matrix = state.matrix.value
    if FLAGS.use_connect:
        param = jnp.diag(param)
        matrix = (param @ (matrix @ param))
    s, q = utils.safe_psd_eigh(matrix)
    
    result = jnp.matmul(jnp.transpose(q), vector)
    result = jnp.power(s + identity_weight, power) * result
    result = jnp.matmul(q, result)
    
    return block.single_vector_to_parameters_list(result)
    

# approximation with isotroical assumption
def kfac_matpower_norm(block, state, param, vector, identity_weight, power):
    param =  block.parameters_shaped_list_to_single_matrix(param)
    vector = block.parameters_shaped_list_to_single_matrix(vector)
    
    aat = state.inputs_factor.value
    ggt = state.outputs_factor.value
    s_i, q_i = utils.safe_psd_eigh(aat)
    s_o, q_o = utils.safe_psd_eigh(ggt)
    
    if FLAGS.use_connect:
        param_norm = jnp.linalg.norm(param, ord='fro')
        coef = param_norm**2
    else:
        coef = 1
        
    eigenvalues = coef * jnp.outer(s_i, s_o) + identity_weight
    eigenvalues = jnp.power(eigenvalues, power)
    result = utils.kronecker_eigen_basis_mul_v(q_o, q_i, eigenvalues, vector)
    result = block.single_matrix_to_parameters_shaped_list(result)
            
    return result

# approximation with evd
def kfac_matpower_evd(block, state, param, vector, identity_weight, power):
    param =  block.parameters_shaped_list_to_single_matrix(param)
    vector = block.parameters_shaped_list_to_single_matrix(vector)
    s_i, q_i = utils.safe_psd_eigh(state.inputs_factor.value)
    s_o, q_o = utils.safe_psd_eigh(state.outputs_factor.value)
    eigenvalues = jnp.outer(s_i, s_o)
    
    if FLAGS.use_connect:
        # compute at once : high space complexity
        # proj = jnp.expand_dims(q_i, (1,3)) * jnp.expand_dims(q_o, (0,2))
        # damping = ((proj*jnp.expand_dims(param**(-1),(2,3)))**2).sum(axis=(2,3))
        
        # comput along (input_dim, output_dim) : high time complexity
        # damping = jnp.zeros_like(eigenvalues)
        # for m in range(len(s_i)):
        #     for n in range(len(s_o)):
        #         proj = jnp.outer(q_i[:,m],q_o[:,n])
        #         damping = damping.at[m, n].set(((param**(-1)*proj)**2).sum())
        
        # compute along input dim : middle time & space complexity
        damping = jnp.zeros_like(eigenvalues)
        for m in range(len(s_i)):
            proj = jnp.expand_dims(q_i[:,m],(1,2))*jnp.expand_dims(q_o, 0)
            damping += ((jnp.expand_dims(param,2)**(-1) * proj)**2).sum(axis=2)
        
        identity_weight = damping * identity_weight
                
    eigenvalues = jnp.outer(s_i, s_o) + identity_weight
    eigenvalues = jnp.power(eigenvalues, power)
    result = utils.kronecker_eigen_basis_mul_v(q_o, q_i, eigenvalues, vector)
    if FLAGS.use_connect:
        result = result / param
    result = block.single_matrix_to_parameters_shaped_list(result)
    return result

# approximation with svd
def kfac_matpower_svd(block, state, param, vector, identity_weight, power):
    param =  block.parameters_shaped_list_to_single_matrix(param)
    vector = block.parameters_shaped_list_to_single_matrix(vector)
    
    if FLAGS.use_connect:
        u, s, vt = jnp.linalg.svd(param, full_matrices=False)
        result = 0
                
        coef = s[0] * s[0]
        aat = u[:,0].reshape(-1,1) * u[:,0].reshape(1,-1) 
        aat = state.inputs_factor.value * aat
        ggt = vt[0,:].reshape(-1,1) * vt[0,:].reshape(1,-1)
        ggt = state.outputs_factor.value * ggt
        s_i, q_i = utils.safe_psd_eigh(aat)
        s_o, q_o = utils.safe_psd_eigh(ggt)

        eigenvalues = coef * jnp.outer(s_i, s_o) + identity_weight
        eigenvalues = jnp.power(eigenvalues, power)
        
        result += utils.kronecker_eigen_basis_mul_v(q_o, q_i, eigenvalues, vector)
                
        result = block.single_matrix_to_parameters_shaped_list(result)
    else:
        s_i, q_i = utils.safe_psd_eigh(state.inputs_factor.value)
        s_o, q_o = utils.safe_psd_eigh(state.outputs_factor.value)
        eigenvalues = jnp.outer(s_i, s_o) + identity_weight
        eigenvalues = jnp.power(eigenvalues, power)
        result = utils.kronecker_eigen_basis_mul_v(q_o, q_i, eigenvalues, vector)
        result = block.single_matrix_to_parameters_shaped_list(result)
            
    return result

def save(ckpt_dir: str, state) -> None:
 ckpt.check_dir(ckpt_dir)
 with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
   for x in jax.tree_util.tree_leaves(state):
     np.save(f, x, allow_pickle=False)

 tree_struct = jax.tree_util.tree_map(lambda t: 0, state)
 with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
   pickle.dump(tree_struct, f)

def rertore(ckpt_dir):
 with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
   tree_struct = pickle.load(f)
 
 leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
 with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
   flat_state = [np.load(f) for _ in leaves]

 return jax.tree_util.tree_unflatten(treedef, flat_state)