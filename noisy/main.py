from typing import OrderedDict
from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
np.set_printoptions(precision=3, linewidth=100, suppress=True)
from sklearn.metrics import roc_auc_score, average_precision_score

from gex.pretrain.main import init_trainer, opt_step
from gex.pretrain.hparams import get_pretrain_hparams
from gex.datasets.image import ImageDataset
from gex.datasets.dbpedia import TextDataset
from gex.utils import ckpt, metrics, mp, tool
from gex.influence.estimate import compute_influence, get_influence_hparams

FLAGS = flags.FLAGS

def main(_):
    tool.set_seed(seed=FLAGS.seed)
    num_devices = jax.device_count()
    batch_dims = (num_devices, FLAGS.batch_size_device//num_devices)
    
    pretrain_hparams = get_pretrain_hparams()
    pretrain_dir = get_pretrain_hparams(True)
    influence_hparams = get_influence_hparams(pretrain_dir)
    influence_dir = get_influence_hparams(pretrain_dir, True)
    
    # define pseudo-random number generator
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, rng_ = jax.random.split(rng)
    
    # make datasets.
    if FLAGS.dataset == 'dbpedia':
        ds = TextDataset()
    else:
        ds = ImageDataset()
        
    train_opt = ds.load_dataset(
        batch_dims=batch_dims,
        split='train',
        shuffle=True,
        augment=True,
        )
    train_eval = ds.load_dataset(
        batch_dims=batch_dims,
        split='train',
        shuffle=False,
        augment=False,
        )
    test_eval = ds.load_dataset(
        batch_dims=batch_dims, 
        split='validation' if (FLAGS.dataset=='imagenet2012') else 'test',
        shuffle=False,
        augment=False,
        )
    
    # initialize network and optimizer
    if FLAGS.dataset == 'dbpedia':
        trainer = init_trainer(
        rng_, 
        jax.random.choice(rng_, 1000, (1,6)),
        ds.num_classes,
        ds.num_train,
        )
    else:
        trainer = init_trainer(
            rng_, 
            jax.random.normal(rng_, (1, *ds.img_shape)), 
            ds.num_classes,
            ds.num_train,
            )
    trainer = ckpt.load_ckpt(pretrain_dir, trainer)
    trainer = mp.replicate(trainer)
    
    influence = compute_influence(trainer, train_eval, test_eval, 
                                    FLAGS.self_influence,
                                    opt_step=opt_step,
                                    dataset_opt=train_opt,
                                    pretrain_dir=pretrain_dir,
                                    num_train=ds.num_train,
                                    num_test=ds.num_test,
                                    num_classes=ds.num_classes, # only for knn/cl
                                    )
    label = np.zeros(ds.num_train, dtype=np.int32)
    # check existence of corrupted index
    if hasattr(ds, 'corrupted_idx'):
        label[ds.corrupted_idx] = 1
        
        if FLAGS.self_influence:
            auc = roc_auc_score(label, influence)
            ap = average_precision_score(label, influence)
        else:
            auc = roc_auc_score(label, -influence)
            ap = average_precision_score(label, -influence)
        print(f'{influence_hparams} AUC : {auc:.4f}, AP : {ap:.4f}')
        
        fig, ax = plt.subplots()
        bins = np.linspace(
            np.min(influence),
            np.max(influence),
            # 30,
            100,
        )
        ax.hist([influence[(1-label).astype(bool)]], bins=bins, label='clean', color='teal', alpha=0.5, edgecolor='white', log=True)
        ax.hist([influence[label.astype(bool)]], bins=bins, label='corrupted', color='salmon', alpha=0.5, stacked=True, edgecolor='white', log=True)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.title(f'AUC : {auc:.4f}, AP : {ap:.4f}', fontdict = {'fontsize' : 10})
        plt.legend()
        plt.savefig(f'{influence_dir}/noisy.{FLAGS.plot_fmt}', dpi=FLAGS.dpi)
        plt.clf()
        
        log = {
            'pretrain': pretrain_hparams,
            'if_method': influence_hparams,
            'AUC': f'{auc:.4f}',
            'AP': f'{ap:.4f}',
            }
        ckpt.dict2tsv(log, f'./gex/result/{FLAGS.dataset}/noisy.tsv')
    
if __name__=='__main__':
    app.run(main)