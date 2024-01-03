from tqdm import tqdm
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
from skimage.filters import threshold_otsu

from gex.pretrain.main import init_trainer, opt_step
from gex.pretrain.hparams import get_pretrain_hparams
from gex.datasets.image import ImageDataset
from gex.utils import ckpt, metrics, mp, tool
from gex.influence.estimate import compute_influence, get_influence_hparams

flags.DEFINE_bool('relabel_self_influence', True, help='use self-influence for relabeling')
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
    
    # initialize network and optimizer
    trainer = init_trainer(
        rng_,
        jax.random.normal(rng_, (1, *ds.img_shape)), 
        ds.num_classes,
        ds.num_train,
        )
    trainer = ckpt.load_ckpt(pretrain_dir, trainer)
    trainer = mp.replicate(trainer)
    
    logit_tr, label_tr = tool.get_logit_dataset(trainer, train_eval, ds.num_train, ds.num_classes)
    if FLAGS.problem_type=='cls':
        prob_tr = jax.nn.softmax(logit_tr)
    elif FLAGS.problem_type=='multitask':
        prob_tr = jax.nn.sigmoid(logit_tr)
    
    influence = compute_influence(trainer, train_eval, test_eval, 
                                    FLAGS.self_influence,
                                    opt_step=opt_step,
                                    dataset_opt=train_opt,
                                    pretrain_dir=pretrain_dir,
                                    num_train=ds.num_train,
                                    num_test=ds.num_test,
                                    num_classes=ds.num_classes, # only for knn/cl
                                    )
    # if FLAGS.dataset=='imagenet2012':
    #     threshold = np.sort(influence)[-int(0.05*len(influence))]
    # else:
    #     threshold = threshold_otsu(influence)
    threshold = threshold_otsu(influence)
    harmful_instance = (influence > threshold).astype(float)
        
    num_harmful = harmful_instance.sum()
    print(f'Compute new label for {num_harmful} instances with threshold {threshold:.4f}')
    
    # if FLAGS.problem_type=='cls':
    #     prob_noisy = (prob_tr * label_tr).sum(axis=1, keepdims=True)
    #     new_label = (1-harmful_instance).reshape(-1,1) * label_tr + harmful_instance.reshape(-1,1) * (np.log((1-prob_noisy)**(1/(ds.num_classes-1))+1e-12) / (np.log(prob_tr+1e-12) + 1e-12)) * (1-label_tr)
    # else:
    #     new_label = (1-harmful_instance).reshape(-1,1) * label_tr + harmful_instance.reshape(-1,1) * prob_tr
    new_label = prob_tr
        
    train_opt = ds.load_dataset(
        batch_dims=batch_dims,
        split='train',
        shuffle=True,
        augment=True,
        new_label=new_label,
        )
    
    print(f'Start retraining')
    
    trainer = init_trainer(
        rng_,
        jax.random.normal(rng_, (1, *ds.img_shape)), 
        ds.num_classes,
        ds.num_train,
        )
    trainer = mp.replicate(trainer)
    
    pbar = tqdm(range(1,FLAGS.num_epochs+1))
    step_per_epoch_tr = np.ceil(float(ds.num_train) / FLAGS.batch_size_device).astype(int)
    for epoch in pbar:
        res_list = []
        for _ in range(step_per_epoch_tr):
            batch_tr = next(train_opt)
            rng, rng_ = jax.random.split(rng)
            log, trainer = opt_step(trainer, batch_tr, jax.random.split(rng_, num_devices), True)
            log = mp.unreplicate(log)
            log = OrderedDict([(k,f'{np.mean(v):.2f}') for k,v in log.items()])
            log.update({'epoch': epoch})
            log.move_to_end('epoch', last=False)
            res_list.append(log)
            pbar.set_postfix(log)

        if (epoch%FLAGS.log_freq)==0:
            if FLAGS.use_bn and 'resnet' in FLAGS.model:
                trainer = tool.update_bn(trainer, train_opt)
            res = ckpt.reduce_dict_list(res_list)
            
            log_test = metrics.log_dataset(trainer, test_eval, ds.num_test)
            acc_test = np.mean(log_test['acc'])
            res['acc_test'] = acc_test
            ckpt.dict2tsv(res, f'{influence_dir}/relabel_log.tsv')
            ckpt.save_ckpt(trainer, f'{influence_dir}/relabel')
    
    log = {
        'pretrain': pretrain_hparams,
        'if_method': influence_hparams,
        'acc_test': f'{acc_test:.4f}',
        }
    ckpt.dict2tsv(log, f'./gex/result/{FLAGS.dataset}/relabel.tsv')
    
    
if __name__=='__main__':
    app.run(main)