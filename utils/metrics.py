from functools import partial
from itertools import combinations
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax

import numpy as np
from absl import flags
from tqdm import tqdm
from gex.utils import tool

flags.DEFINE_integer('iter_max', 100, 
help='number of iteration for Hutchison method')
flags.DEFINE_enum('unc_type', 'logit_var', ['ent', 'logit_var', 'prob_var'], 
help='type of uncertainty to be computed()')
FLAGS = flags.FLAGS

def get_calibration(prob, label, num_bins=10):
    acc = (jnp.argmax(prob,axis=-1)==jnp.argmax(label,axis=-1)).mean()
    ece = expected_calibration_error(
            num_bins, pred=prob, labels_true=label.argmax(axis=-1)
        )
    nll = -(jnp.log(prob) * label).sum(axis=-1).mean()
    brier = ((prob - label)**2).sum(axis=-1).mean()
    return acc, ece, nll, brier

def compute_unc(logit):
    if FLAGS.problem_type=='reg':
        var = jnp.sum(jnp.var(logit, axis=0), axis=-1)
        return var
    else:
        if FLAGS.unc_type=='logit_var':
            # logit = jax.nn.log_softmax(logit, axis=-1)
            var = jnp.sum(jnp.var(logit, axis=0), axis=-1)
            return var
        elif FLAGS.unc_type=='prob_var':
            logit = jax.nn.softmax(logit, axis=-1)
            var = jnp.sum(jnp.var(logit, axis=0), axis=-1)
            return var
        elif FLAGS.unc_type=='ent':
            ens_prob = jnp.mean(jax.nn.softmax(logit, axis=-1),axis=0)
            ens_log_prob = jnp.log(ens_prob)
            ent = jnp.sum( - ens_log_prob * ens_prob, axis=-1)
            return ent

@partial(jax.pmap, static_broadcasted_argnums=(3, 4))
def log_batch(trainer, batch, rng, fn_type=None, loss_type=None):
    if fn_type=='lin':
        logit = tool.forward_lin(trainer.offset, trainer.params, trainer, batch['x'], rng, train=False, separate=False)
    elif fn_type=='pert':
        logit = tool.forward_pert(trainer.offset, trainer.params, trainer, batch['x'], rng, train=False)
    else:
        logit = tool.forward(trainer.params, trainer, batch['x'], rng, train=False)
    
    if loss_type is not None:
        problem_type = loss_type
    else:
        problem_type = FLAGS.problem_type
    
    if problem_type=='reg':
        prob = logit
        loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1)
    elif problem_type=='cls':
        prob = jax.nn.softmax(logit, axis=-1)
        loss = - (jax.nn.log_softmax(logit) * optax.smooth_labels(batch['y'], FLAGS.label_smooth)).sum(axis=-1)
    elif problem_type=='multitask':
        prob = jax.nn.sigmoid(logit)
        # pretraining of ViT/MLP-Mixer uses sigmoid cross-entropy
        # https://github.com/google-research/vision_transformer/issues/34
        log_prob = jax.nn.log_sigmoid(logit)
        log_not_prob = jax.nn.log_sigmoid(-logit)
        # label smoothing for sigmoid cross-entropy
        # https://github.com/google-research/vision_transformer/issues/89#issuecomment-853307136
        label = (1-FLAGS.label_smooth) * batch['y'] + FLAGS.label_smooth * (1-batch['y'])
        loss = - (label * log_prob + (1-label) * log_not_prob).sum(axis=-1)
        
    brier = ((prob - batch['y'])**2).sum(axis=-1)
    acc = (jnp.argmax(logit,axis=-1)==jnp.argmax(batch['y'], axis=-1)).astype(int)
    return loss, acc, brier

def log_dataset(state, dataset, num_samples, fn_type=None, loss_type=None):
    step_per_epoch = np.ceil(num_samples / FLAGS.batch_size_device).astype(int)
    rng = jax.random.PRNGKey(FLAGS.seed)
    num_devices = jax.device_count()
    
    loss = np.zeros((num_samples,))
    acc = np.zeros((num_samples,))
    brier = np.zeros((num_samples,))
    for _ in range(step_per_epoch):
        batch = next(dataset)
        rng, rng_ = jax.random.split(rng)
        loss_b, acc_b, brier_b = log_batch(state, batch, jax.random.split(rng_, num_devices), fn_type, loss_type)
        
        batch_idx = batch['idx'].reshape(-1)
        loss[batch_idx] = loss_b.reshape(-1)
        acc[batch_idx] = acc_b.reshape(-1)
        brier[batch_idx] = brier_b.reshape(-1)
        
    return {'loss' : loss, 'acc' : acc, 'brier' : brier}

@partial(jax.pmap, static_broadcasted_argnums=(0,))
def tr_hess_batch_p(loss_fn, state, batch):
    # Hutchinson's method for estimating trace of Hessian
    rng = jax.random.PRNGKey(FLAGS.seed)
    # redefine loss for HVP computation
    loss_fn_ = lambda params, inputs, targets : loss_fn(
        params, 
        state, 
        {'x' : inputs, 'y' : targets},
        False,
        )[0]
    def body_fn(_, carrier):
        res, rng = carrier
        rng, rng_r = jax.random.split(rng)
        v = jax.random.rademacher(
            rng_r, 
            (ravel_pytree(state.params)[0].size,), 
            jnp.float32,
            )
        Hv = optax.hvp(loss_fn_, v, state.params, batch['x'], batch['y'])
        Hv = ravel_pytree(Hv)[0] / batch['x'].shape[0]
        vHv = jnp.vdot(v, Hv)
        res += vHv / FLAGS.iter_max
        return res, rng
    res, rng = jax.lax.fori_loop(0, FLAGS.iter_max, body_fn, (0, rng))
    return res

def tr_hess_batch(loss_fn, state, batch):
    tr_hess = tr_hess_batch_p(loss_fn, state, batch)
    tr_hess = np.mean(jax.device_get(tr_hess))
    return tr_hess

def tr_hess_dataset(loss_fn, state, dataset):
    tr_hess_total = 0.
    n_total = 0
    for batch in tqdm(dataset):
        batch_shape = batch['x'].shape
        n = batch_shape[0] * batch_shape[1]
        tr_hess = tr_hess_batch(loss_fn, state, batch)
        tr_hess_total += tr_hess * n
        n_total += n
    tr_hess_total /= n_total
    return tr_hess_total

@jax.pmap
def tr_ntk_batch_p(state, batch):
    # Hutchinson's method for estimating trace of NTK
    rng = jax.random.PRNGKey(FLAGS.seed)
    # redefine forward for JVP computation
    def f(params):
        return state.apply_fn(
            {'params' : params, 'batch_stats': state.batch_stats},
            batch['x'], 
            train=False,
        )
    _, f_vjp = jax.vjp(f, state.params)
    def body_fn(_, carrier):
        res, rng = carrier
        _, rng = jax.random.split( rng )
        v = jax.random.rademacher(
        rng, 
        (batch['x'].shape[0], batch['y'].shape[-1]),
        jnp.float32,
        )
        j_p = ravel_pytree(f_vjp(v))[0]
        tr_ntk= jnp.sum(jnp.square(j_p)) / batch['x'].shape[0]
        res += tr_ntk / FLAGS.iter_max
        return res, rng
    a = jax.lax.fori_loop(0, FLAGS.iter_max, body_fn, (0.,rng))
    res, rng = a
    return res

def tr_ntk_batch(state, batch):
    tr_ntk = tr_ntk_batch_p(state, batch)
    tr_ntk = np.mean(jax.device_get(tr_ntk))
    return tr_ntk

def tr_ntk_dataset(state, dataset):
    tr_ntk_total = 0.
    n_total = 0
    for batch in tqdm(dataset):
        batch_shape = batch['x'].shape
        n = batch_shape[0] * batch_shape[1]
        tr_ntk = tr_ntk_batch(state, batch)
        tr_ntk_total += tr_ntk * n
        n_total += n
    tr_ntk_total /= n_total
    return tr_ntk_total

def compute_ratio_err(pred_list, label):
    # diversity metrics 1. ratio-error
    ratio_err = []
    for comb in combinations(range(pred_list.shape[0]), 2):
        i,j = comb
        err_i = (
            jnp.argmax(pred_list[i],axis=-1) != jnp.argmax(label,axis=-1)
            ).astype(jnp.float32)
        err_j = (
            jnp.argmax(pred_list[j],axis=-1) != jnp.argmax(label,axis=-1)
            ).astype(jnp.float32)
        same = (err_i * err_j).sum()
        diff = (1.-(1.-err_i)*(1.-err_j)).sum() - same
        ratio_err.append(diff / same)
    ratio_err = np.mean(ratio_err)
    return ratio_err

def compute_q_stat(pred_list, label):
    # diversity metrics 2. q-stat
    q_stat = []
    for comb in combinations(range(pred_list.shape[0]), 2):
        i,j = comb
        err_i = (
            jnp.argmax(pred_list[i],axis=-1) != jnp.argmax(label,axis=-1)
            ).astype(jnp.float32)
        err_j = (
            jnp.argmax(pred_list[j],axis=-1) != jnp.argmax(label,axis=-1)
            ).astype(jnp.float32)
        n_00 = (err_i * err_j).sum()
        n_01 = (err_i * (1.-err_j)).sum()
        n_10 = ((1.-err_i) * err_j).sum()
        n_11 = ((1.-err_i) * (1.-err_j)).sum()
        q_stat.append((n_11 * n_00 - n_01 * n_10)/(n_11 * n_00 + n_01 * n_10))
    q_stat = np.mean(q_stat)
    return q_stat

def compute_cc(pred_list, label):
    # diversity metrics 3. correlation coefficient
    cc = []
    for comb in combinations(range(pred_list.shape[0]), 2):
        i,j = comb
        err_i = (
            jnp.argmax(pred_list[i],axis=-1) != jnp.argmax(label,axis=-1)
            ).astype(jnp.float32)
        err_j = (
            jnp.argmax(pred_list[j],axis=-1) != jnp.argmax(label,axis=-1)
            ).astype(jnp.float32)
        cc.append(jnp.corrcoef(err_i, err_j))
    cc = np.mean(cc)
    return cc

def compute_disagree(pred_list, label):
    # diversity metric 4. disagree
    disagree = []
    for comb in combinations(range(pred_list.shape[0]), 2):
        i,j = comb
        dis = (
            jnp.argmax(pred_list[i],axis=1) != jnp.argmax(pred_list[j],axis=1)
        ).astype(jnp.float32).mean()
        disagree.append(dis)
    disagree = np.mean(disagree)
    return disagree

# ece computation with probability (not logit)
# from https://github.com/tensorflow/probability/blob/bedb9aac14500546a41380d48112d40a16cc88c0/tensorflow_probability/python/stats/calibration.py#L203
import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps

def _compute_calibration_bin_statistics(
    num_bins, pred=None, labels_true=None, labels_predicted=None):
  if labels_predicted is None:
    pred_y = tf.argmax(pred, axis=1, output_type=labels_true.dtype)
  else:
    pred_y = labels_predicted
  correct = tf.cast(tf.equal(pred_y, labels_true), tf.int32)
  prob_y = tf.gather(
      pred, pred_y[:, tf.newaxis], batch_dims=1)  # p(pred_y | x)
  prob_y = tf.reshape(prob_y, (ps.size(prob_y),))
  bins = tf.hirtogram_fixed_width_bins(prob_y, [0.0, 1.0], nbins=num_bins)
  event_bin_counts = tf.math.bincount(
      correct * num_bins + bins,
      minlength=2 * num_bins,
      maxlength=2 * num_bins)
  event_bin_counts = tf.reshape(event_bin_counts, (2, num_bins))
  pmean_observed = tf.math.unsorted_segment_sum(prob_y, bins, num_bins)
  tiny = np.finfo(dtype_util.as_numpy_dtype(pred.dtype)).tiny
  pmean_observed = pmean_observed / (
      tf.cast(tf.reduce_sum(event_bin_counts, axis=0), pred.dtype) + tiny)
  return event_bin_counts, pmean_observed

def expected_calibration_error(num_bins, pred=None, labels_true=None,
                               labels_predicted=None, name=None):
  with tf.name_scope(name or 'expected_calibration_error'):
    pred = tf.convert_to_tensor(pred, tf.float32)
    labels_true = tf.convert_to_tensor(labels_true, tf.int32)
    if labels_predicted is not None:
      labels_predicted = tf.convert_to_tensor(labels_predicted)
    event_bin_counts, pmean_observed = _compute_calibration_bin_statistics(
        num_bins, pred=pred, labels_true=labels_true,
        labels_predicted=labels_predicted)
    event_bin_counts = tf.cast(event_bin_counts, tf.float32)
    bin_n = tf.reduce_sum(event_bin_counts, axis=0)
    pbins = bin_n / tf.reduce_sum(bin_n)
    tiny = np.finfo(np.float32).tiny
    pcorrect = event_bin_counts[1, :] / (bin_n + tiny)
    ece = tf.reduce_sum(pbins * tf.abs(pcorrect - pmean_observed))
  return float(ece)
