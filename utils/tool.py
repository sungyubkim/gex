import tensorflow as tf
import os
import numpy as np
import random
from typing import Any, Callable
from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
import haiku as hk
import optax

from absl import flags
from jax.flatten_util import ravel_pytree

import haiku as hk
import jmp

flags.DEFINE_enum('problem_type', 'cls', ['reg', 'cls', 'multitask'],
help='type of problem (reg: regression, cls: classification)')
flags.DEFINE_float('label_smooth', 0.0, help='label smoothing coefficient')
flags.DEFINE_bool('use_bn', True, help='use Batch Normalization layer')
flags.DEFINE_float('pert_scale_sam', 0.0, help='perturbation scale of SAM')
flags.DEFINE_enum('plot_fmt', 'pdf', ['png', 'pdf'], help='plot file format')
flags.DEFINE_integer('dpi', 100, help='dot-per-inch for plots')
FLAGS = flags.FLAGS

def make_forward(model):
    
    def _forward(*args, **kwargs):
        return model()(*args, **kwargs)
    
    return hk.transform_with_state(_forward)

class Trainer(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    init_fn: Callable = struct.field(pytree_node=False) 
    tx: Callable = struct.field(pytree_node=False)
    params: Any = None
    state: Any = None
    opt_state: Any = None
    
    @classmethod
    def create(cls, *, init_fn, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(step=0, init_fn=init_fn, apply_fn=apply_fn, params=params, tx=tx, opt_state=opt_state, **kwargs)
    
    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step+1, params=new_params, opt_state=new_opt_state, **kwargs)
    
def select_tree(pred: jnp.ndarray, a, b):
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_map(partial(jax.lax.select, pred), a, b)
    
class TrainerPert(Trainer):
    offset : Any = None
    
def init_trainer_ft(trainer, rng=None, batch=None):
    
    if rng is not None:
        # re-initialize the model
        new_params, state = trainer.init_fn(rng, batch['x'], train=True, print_shape=False)
        trainer = trainer.replace(params=new_params, state=state)
        
    # fine-tune the model
    if FLAGS.ft_lr_sched=='constant':
        # tx = optax.chain(optax.sgd(learning_rate=FLAGS.ft_lr, momentum=0.9))
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate=FLAGS.ft_lr, momentum=0.9),
        )
    elif FLAGS.ft_lr_sched=='cosine':
        tx = optax.chain(optax.sgd(learning_rate=optax.cosine_decay_schedule(FLAGS.ft_lr, FLAGS.ft_step), momentum=0.9))
            
    opt_state = tx.init(trainer.params)
    trainer_ft = trainer.replace(
        step=0,
        tx=tx,
        opt_state=opt_state,
    )
    return trainer_ft

init_trainer_ft_p = jax.pmap(init_trainer_ft)

def init_trainer_ft_lin(trainer, init_params=None):
    vec_params, unravel_fn = params_to_vec(trainer.params, True)
    if init_params is None:
        init_params = jnp.zeros_like(vec_params)
    if FLAGS.ft_lr_sched=='constant':
        tx = optax.chain(optax.sgd(learning_rate=FLAGS.ft_lr, momentum=0.9))
    elif FLAGS.ft_lr_sched=='cosine':
        tx = optax.chain(optax.sgd(learning_rate=optax.cosine_decay_schedule(FLAGS.ft_lr, FLAGS.ft_step), momentum=0.9))
    trainer_ft = TrainerPert.create(
        init_fn=trainer.init_fn,
        apply_fn=trainer.apply_fn,
        state=trainer.state,
        offset=trainer.params,
        params=unravel_fn(init_params),
        tx=tx,
    )
    return trainer_ft

def compress_trainer_ft(trainer_ft, trainer):
    trainer = trainer.replace(
        params=jax.tree_util.tree_map(lambda x,y:x+y, trainer_ft.offset, trainer_ft.params),
        batch_stats=trainer_ft.batch_stats,
    )
    return trainer

def params_to_vec(param, unravel=False):
    vec_param, unravel_fn = ravel_pytree(param)
    if unravel:
        return vec_param, unravel_fn
    else:
        return vec_param
    
def get_logit_dataset(trainer, dataset, num_samples, num_classes, train=False, rng=None, fn_type=None):
    step_per_epoch = np.ceil(num_samples / FLAGS.batch_size_device).astype(int)
    num_devices = jax.device_count()
    if rng is None:
        rng = jax.random.PRNGKey(42)
        
    pred = np.zeros((num_samples, num_classes))
    label = np.zeros((num_samples, num_classes))
    for b in range(step_per_epoch):
        batch = next(dataset)
        rng, rng_ = jax.random.split(rng)
        rng_ = jax.random.split(rng_, num_devices)
        # select function type
        if fn_type=='lin':
            p = forward_lin_p(trainer.offset, trainer.params, trainer, batch['x'], rng_, train, False)
        elif fn_type=='pert':
            p = forward_pert_p(trainer.offset, trainer.params, trainer, batch['x'], rng_, train)
        else:
            p = forward_p(trainer.params, trainer, batch['x'], rng_, train)
        
        batch_idx = batch['idx'].reshape(-1)
        pred[batch_idx] = p.reshape(-1, num_classes)
        label[batch_idx] = batch['y'].reshape(-1, num_classes)
        
    return pred, label

def forward(params, trainer, input_, rng=None, train=True):
    res, _ = trainer.apply_fn(params, trainer.state, rng, input_, train)
    return res

forward_p = jax.pmap(forward, static_broadcasted_argnums=(4,))

def forward_lin(params, pert, trainer, input_, rng=None, train=True, separate=False):
    # linearized forward with perturbation
    f = lambda x: forward(x, trainer, input_, rng=rng, train=train)
    pred_zero, pred_first = jax.jvp(f, [params], [pert])
    if separate:
        return pred_zero, pred_first
    else:
        return pred_zero + pred_first

forward_lin_p = jax.pmap(forward_lin, static_broadcasted_argnums=(5,6))

def forward_pert(params, pert, trainer, input_, rng=None, train=True):
    # perturbed forward with perturbation
    return forward(jax.tree_util.tree_map(lambda x,y:x+y, params, pert), trainer, input_, rng=rng, train=train)

forward_pert_p = jax.pmap(forward_pert, static_broadcasted_argnums=(5,))

def update_bn_batch(trainer, batch, rng, use_offset):
    if use_offset:
        params = trainer.offset
    else:
        params = trainer.params
    _, state = trainer.apply_fn(params, trainer.state, rng, batch['x'], True)
    trainer = trainer.replace(state=state)
    return trainer

update_bn_batch_p = jax.pmap(update_bn_batch, axis_name='batch', static_broadcasted_argnums=(3,))

def update_bn(trainer, dataset, num_step=50):
    # update bn statstic with repeatable dataset.
    # Izmailov, Pavel, et al. "Averaging weights leads to wider optima and better generalization." arXiv preprint arXiv:1803.05407 (2018).
    # Maddox, Wesley J., et al. "A simple baseline for bayesian uncertainty in deep learning." Advances in Neural Information Processing Systems 32 (2019).
    rng = jax.random.PRNGKey(42)
    num_devices = jax.device_count()
    for _ in range(num_step):
        batch = next(dataset)
        rng, rng_ = jax.random.split(rng)
        trainer = update_bn_batch_p(trainer, batch, jax.random.split(rng_, num_devices), hasattr(trainer, 'offset'))
    return trainer

def set_seed(seed, deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1' # slow but reproducible
        
class bcolors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    VIOLET = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'