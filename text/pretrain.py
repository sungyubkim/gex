# restrict tf only for cpu (we only use tf as a dataloader)
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

from functools import partial
from copy import deepcopy
from typing import OrderedDict
from absl import app, flags
import jax 
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
np.set_printoptions(precision=3, linewidth=100, suppress=True)

from gex.utils import ckpt, metrics, mp, tool
from gex.utils.tool import bcolors
from gex.models import swem
from gex.pretrain.hparams import get_pretrain_hparams

from gex.text.dbpedia import TextDataset
import tensorflow_text as tf_text
import requests
from tensorflow_text import SentencepieceTokenizer

FLAGS = flags.FLAGS

@partial(jax.jit, static_argnums=(2,3))
def init_trainer(rng, batch, num_classes, num_train):
    
    model_method = getattr(swem, 'SWEM')
    net = partial(
        model_method,
        name=FLAGS.model,
        num_classes=num_classes,
        )
        
    forward = tool.make_forward(net)
    params, state = forward.init(rng, batch, train=True, print_shape=True)
    if FLAGS.optimizer=='sgd':
        tx = optax.chain(
            optax.sgd(
                learning_rate=create_lr_sched(num_train),
                momentum=0.9,
                ),
        )
    elif FLAGS.optimizer=='adamw':
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=create_lr_sched(num_train),
                weight_decay=FLAGS.weight_decay,
            )
        )
    update_freq = FLAGS.batch_size_total // FLAGS.batch_size_device
    tx = optax.MultiSteps(tx, update_freq)
    trainer = tool.Trainer.create(
        init_fn=forward.init, 
        apply_fn=forward.apply, 
        params=params,
        state=state,
        tx=tx,
    )
    return trainer

def create_lr_sched(num_train):
    step_per_epoch = np.ceil(float(num_train) / FLAGS.batch_size_total).astype(int)
    total_step = FLAGS.num_epochs * step_per_epoch
    warmup_step = int(FLAGS.warmup_ratio * FLAGS.num_epochs) * step_per_epoch
    return optax.warmup_cosine_decay_schedule(0.0, FLAGS.peak_lr, warmup_step, total_step)

def loss_fn(params, pert, wd_coef, trainer, batch, rng):
    vec_params, unravel_fn = tool.params_to_vec(params, True)
    new_params = vec_params + pert
    logit = tool.forward(unravel_fn(new_params), trainer, batch['x'], rng=rng, train=True)
    
    if FLAGS.problem_type=='reg':
        loss = 0.5 * ((logit - batch['y'])**2).sum(axis=-1)
    elif FLAGS.problem_type=='cls':
        loss = - (jax.nn.log_softmax(logit) * optax.smooth_labels(batch['y'], FLAGS.label_smooth)).sum(axis=-1)
    elif FLAGS.problem_type=='multitask':
        # pretraining of ViT/MLP-Mixer uses sigmoid cross-entropy
        # https://github.com/google-research/vision_transformer/issues/34
        log_prob = jax.nn.log_sigmoid(logit)
        log_not_prob = jax.nn.log_sigmoid(-logit)
        # label smoothing for sigmoid cross-entropy
        # https://github.com/google-research/vision_transformer/issues/89#issuecomment-853307136
        label = (1-FLAGS.label_smooth) * batch['y'] + FLAGS.label_smooth * (1-batch['y'])
        loss = - (label * log_prob + (1-label) * log_not_prob).sum(axis=-1)
    acc = (jnp.argmax(logit,axis=-1)==jnp.argmax(batch['y'], axis=-1)).astype(int)

    wd = 0.5 * (new_params**2).sum()
    loss_ = loss.mean() 
    if FLAGS.optimizer=='sgd':
        loss_ = loss_ + wd_coef * wd
    return loss_, (loss, acc, wd)

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=3)
def opt_step(trainer, batch, rng, sync_grad=True):
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)

    grad_fn = jax.grad(loss_fn, has_aux=True)
    # compute grad
    grad, (loss_b, acc_b, wd_b) = grad_fn(
        trainer.params,
        jnp.zeros_like(vec_params),
        0.0 if FLAGS.pert_scale_sam > 0.0 else FLAGS.weight_decay,
        trainer,
        batch,
        rng,
        )
    grad = tool.params_to_vec(grad)
    grad_norm = jnp.sqrt((grad**2).sum())

    # compute sam grad and log
    if FLAGS.pert_scale_sam > 0.0:
        grad_sam, (loss_a, acc_a, wd_a) = grad_fn(
            trainer.params,
            FLAGS.pert_scale_sam * (grad/grad_norm),
            FLAGS.weight_decay,
            trainer,
            batch,
            rng,
            )
        grad_sam = tool.params_to_vec(grad_sam)
        grad_sam_norm = jnp.sqrt((grad_sam**2).sum())
        grad = grad_sam
        log = [
            ('loss_sgd', loss_b),
            ('loss_sam', loss_a),
            ('acc_sgd', acc_b),
            ('acc_sam', acc_a),
            ('grad_sgd', grad_norm),
            ('grad_sam', grad_sam_norm),
        ]
    else:
        log = [
            ('loss_sgd', loss_b),
            ('acc_sgd', acc_b),
            ('grad_sgd', grad_norm),
        ]
    log = OrderedDict(log)
    
    # update NN
    if sync_grad:
        grad = jax.lax.pmean(grad, axis_name='batch')
    trainer = trainer.apply_gradients(grads=unravel_fn(grad))
    return log, trainer

def plot_metric(metric, pretrain_dir, metric_name):
    fig, ax = plt.subplots()
    bins = np.linspace(
        np.min(metric),
        np.max(metric),
        100,
    )
    ax.hist([metric], bins=bins, color='teal', alpha=0.5, edgecolor='white', log=True)
    plt.title(f'{metric_name}', fontdict = {'fontsize' : 10})
    ckpt.check_dir(f'{pretrain_dir}/figs')
    plt.savefig(f'{pretrain_dir}/figs/{metric_name}.{FLAGS.plot_fmt}', dpi=FLAGS.dpi)
    plt.clf()

def main(_):
    tool.set_seed(seed=FLAGS.seed)
    num_devices = jax.device_count()
    batch_dims = (num_devices, FLAGS.batch_size_device//num_devices)

    ds = TextDataset()

    pretrain_hparams = get_pretrain_hparams()
    pretrain_dir = get_pretrain_hparams(is_dir=True)

    # define pseudo-random number generator
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, rng_ = jax.random.split(rng)

    # initialize network and optimizer
    # jax.random.normal(rng_, (1, *ds.img_shape)), 
    trainer = init_trainer(
        rng_, 
        jax.random.choice(rng_, 1000, (1,6)),
        ds.num_classes,
        ds.num_train,
        )
    if FLAGS.eval:
        trainer = ckpt.rertore_checkpoint(
            pretrain_dir,
            trainer,
            )
    trainer = mp.replicate(trainer)

    # train model
    train_opt = ds.load_dataset(
        batch_dims=batch_dims,
        split='train',
        shuffle=True,
        )
    test_eval = ds.load_dataset(
        batch_dims=batch_dims, 
        split='test',
        shuffle=False,
        )
    
    step_per_epoch_tr = np.ceil(float(ds.num_train) / FLAGS.batch_size_device).astype(int)
    save_epoch = int(0.5 * FLAGS.num_epochs)
    save_freq = save_epoch // FLAGS.num_ckpt
    lr_sched = create_lr_sched(ds.num_train)

    if not(FLAGS.eval):
        loss_tr = np.zeros((ds.num_train,), dtype=float)
        fscore = np.zeros((ds.num_train,), dtype=int)
        prev_acc = np.zeros((ds.num_train,), dtype=int)
        
        pbar = tqdm(range(1,FLAGS.num_epochs+1))
        for epoch in pbar:
            res_list = []
            for _ in range(step_per_epoch_tr):
                batch_tr = next(train_opt)
                rng, rng_ = jax.random.split(rng)
                log, trainer = opt_step(trainer, batch_tr, jax.random.split(rng_, num_devices), True)
                
                # update forgetting score
                batch_idx = batch_tr['idx'].reshape(-1)
                acc = log['acc_sgd'].reshape(-1)
                fscore[batch_idx] += (prev_acc[batch_idx] > acc).astype(int)
                prev_acc[batch_idx] = acc
                loss_tr[batch_idx] = log['loss_sgd'].reshape(-1)
                
                log = OrderedDict([(k,np.mean(v)) for k,v in log.items()])
                log.update({'epoch': epoch})
                log.move_to_end('epoch', last=False)
                res_list.append(log)
                pbar.set_postfix(ckpt.pretty_log(log))

            if (epoch%FLAGS.log_freq)==0:
                trainer = tool.update_bn(trainer, train_opt)
                ckpt.save_ckpt(trainer, pretrain_dir)
                res = ckpt.reduce_dict_list(res_list)
                
                log_test = metrics.log_dataset(trainer, test_eval, ds.num_test)
                acc_test = np.mean(log_test['acc'])
                res['acc_test'] = acc_test
                
                lr = lr_sched(mp.unreplicate(trainer.step))
                res['lr'] = float(lr)
                
                ckpt.print_log(ckpt.pretty_log(res))
                ckpt.dict2tsv(res, f'{pretrain_dir}/log.tsv')
                
                # log
                ckpt.array2tsv(loss_tr, f'{pretrain_dir}/loss_tr.tsv')
                plot_metric(loss_tr, pretrain_dir, f'loss_tr_{epoch}')
                ckpt.array2tsv(log_test['loss'], f'{pretrain_dir}/loss_test.tsv')
                plot_metric(log_test['loss'], pretrain_dir, f'loss_test_{epoch}')
                ckpt.array2tsv(fscore, f'{pretrain_dir}/fscore.tsv')
                plot_metric(fscore, pretrain_dir, f'fscore_{epoch}')
                
            # ckpt for tracin
            if epoch > save_epoch and (epoch - save_epoch) % save_freq == 0:
                i = (epoch - save_epoch) // save_freq
                ckpt.save_ckpt(trainer, f'{pretrain_dir}/traj/{i-1}')
        
        log = {
        'pretrain': pretrain_hparams,
        'acc_test': f'{acc_test:.4f}',
        }
        ckpt.dict2tsv(log, f'./gex/result/{FLAGS.dataset}/pretrain.tsv')


if __name__ == "__main__":
    app.run(main)