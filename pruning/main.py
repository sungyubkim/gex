from typing import OrderedDict
from absl import app, flags
from tqdm import tqdm
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

from gex.pretrain.main import init_trainer, opt_step, create_lr_sched
from gex.pretrain.hparams import get_pretrain_hparams
from gex.datasets.image import ImageDataset
from gex.utils import ckpt, metrics, mp, tool
from gex.influence.estimate import compute_influence, get_influence_hparams
from gex.utils.tool import bcolors

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
    
    if FLAGS.num_filter > 0:
        
        if FLAGS.iterative_pruning:
            df = pd.read_csv(f'{influence_dir}/influence_{FLAGS.num_filter-5000}.tsv', sep='\t')
        else:
            df = pd.read_csv(f'{influence_dir}/influence_0.tsv', sep='\t')
        influence = df.values[:,-1]
            
        # make datasets.
        train_opt = ds.load_dataset(
            batch_dims=batch_dims,
            split='train',
            shuffle=True,
            augment=True,
            num_filter=FLAGS.num_filter, 
            summarization_score=influence,
            )
        # initialize network and optimizer
        trainer = init_trainer(
            rng_,
            jax.random.normal(rng_, (1, *ds.img_shape)), 
            ds.num_classes,
            ds.num_train_filtered,
            )
        trainer = mp.replicate(trainer)
        
        pbar = tqdm(range(1,FLAGS.num_epochs+1))
        lr_sched = create_lr_sched(ds.num_train_filtered)
        step_per_epoch_tr = np.ceil(float(ds.num_train_filtered) / FLAGS.batch_size_device).astype(int)
        update_freq = FLAGS.batch_size_total // FLAGS.batch_size_device
        for epoch in pbar:
            res_list = []
            
            for _ in range(step_per_epoch_tr):
                batch_tr = next(train_opt)
                rng, rng_ = jax.random.split(rng)
                log, trainer = opt_step(trainer, batch_tr, jax.random.split(rng_, num_devices), True)
                
                log = mp.unreplicate(log)
                log = OrderedDict([(k,np.mean(v)) for k,v in log.items()])
                log.update({'epoch': epoch})
                log.move_to_end('epoch', last=False)
                res_list.append(log)
                pbar.set_postfix(ckpt.pretty_log(log))

            if (epoch%FLAGS.log_freq)==0:
                trainer = tool.update_bn(trainer, train_opt)
                res = ckpt.reduce_dict_list(res_list)
                
                log_test = metrics.log_dataset(trainer, test_eval, ds.num_test)
                acc_test = np.mean(log_test['acc'])
                res['acc_test'] = acc_test
                
                lr = lr_sched(mp.unreplicate(trainer.step)//update_freq)
                res['lr'] = float(lr)
                
                ckpt.print_log(ckpt.pretty_log(res))
                fname = f'filter_{FLAGS.num_filter}'
                ckpt.dict2tsv(res, f'{influence_dir}/{fname}_log.tsv')
                
        log = {
        'pretrain': pretrain_hparams,
        'if_method': influence_hparams,
        'num_filter': FLAGS.num_filter,
        'acc_test': f'{acc_test:.4f}',
        }
        ckpt.dict2tsv(log, f'./gex/result/{FLAGS.dataset}/pruning.tsv')
        
        if FLAGS.iterative_pruning:
            influence = compute_influence(trainer, train_eval, test_eval, True,
                                        opt_step=opt_step,
                                        dataset_opt=train_opt,
                                        pretrain_dir=pretrain_dir,
                                        num_train=ds.num_train,
                                        num_test=ds.num_test,
                                        num_class=ds.num_classes,
                                        )
            # mask filtered samples
            influence = np.where(ds.left, influence, -np.inf)
            df = pd.DataFrame(influence, columns=['influence'])
            df.to_csv(f'{influence_dir}/influence_{FLAGS.num_filter}.tsv', sep='\t', index=False)
            
            fig, ax = plt.subplots()
            influence = influence[np.isfinite(influence)]
            bins = np.linspace(
                np.min(influence),
                np.max(influence),
                100,
            )
            ax.hist([influence], bins=bins, color='teal', alpha=0.5, edgecolor='white', log=True)
            plt.title(f'Self-Influence', fontdict = {'fontsize' : 10})
            plt.savefig(f'{influence_dir}/hist_{FLAGS.num_filter}.{FLAGS.plot_fmt}', dpi=FLAGS.dpi)
            plt.clf()
        
    else:
        # initialize network and optimizer
        trainer = init_trainer(
            rng_,
            jax.random.normal(rng_, (1, *ds.img_shape)), 
            ds.num_classes,
            ds.num_train,
            )
        trainer = ckpt.load_ckpt(pretrain_dir, trainer)
        trainer = mp.replicate(trainer)
        
        influence = compute_influence(trainer, train_eval, test_eval, True,
                                      opt_step=opt_step,
                                      dataset_opt=train_opt,
                                      pretrain_dir=pretrain_dir,
                                      num_train=ds.num_train,
                                      num_test=ds.num_test,
                                      num_classes=ds.num_classes,
                                      )
        
        fig, ax = plt.subplots()
        influence = influence[np.isfinite(influence)]
        bins = np.linspace(
            np.min(influence),
            np.max(influence),
            100,
        )
        ax.hist([influence], bins=bins, color='teal', alpha=0.5, edgecolor='white', log=True)
        plt.title(f'Self-Influence', fontdict = {'fontsize' : 10})
        plt.savefig(f'{influence_dir}/hist_{FLAGS.num_filter}.{FLAGS.plot_fmt}', dpi=FLAGS.dpi)
        plt.clf()
    
if __name__=='__main__':
    app.run(main)