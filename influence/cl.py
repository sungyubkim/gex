from genericpath import isdir
from random import gauss, random
from absl import flags
import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from time import time
import os
import pandas as pd
from typing import OrderedDict

from gex.pretrain.main import init_trainer, opt_step
from gex.datasets.image import ImageDataset
from gex.datasets.dbpedia import TextDataset
from gex.utils import ckpt, metrics, mp, tool
from cleanlab.rank import get_label_quality_scores

FLAGS = flags.FLAGS

def compute_influence_cl(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    
    from gex.influence.estimate import get_influence_hparams
    pretrain_dir = kwargs['pretrain_dir']
    posthoc_dir = get_influence_hparams(pretrain_dir, True)
    num_train = kwargs['num_train']
    num_classes = kwargs['num_classes']
    
    if FLAGS.dataset == 'dbpedia':
        ds = TextDataset()
    else:
        ds = ImageDataset()
    
    num_devices = jax.device_count()
    batch_dims = (num_devices, FLAGS.batch_size_device//num_devices)
    step_per_epoch_tr = np.ceil(float(ds.num_train) / FLAGS.batch_size_device).astype(int)
    
    # define pseudo-random number generator
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, rng_ = jax.random.split(rng)
    
    print(f'Pre-train for Confident Learning')
    for i in range(2):
        if i==0:
            sub_split_tr = 0
            sub_split_val = 1
        else:
            sub_split_tr = 1
            sub_split_val = 0
            
        # make datasets.
        train_repeat = ds.load_dataset(
            batch_dims=batch_dims,
            split='train',
            sub_split=sub_split_tr,
            shuffle=True,
            augment=True,
            )
        eval_val = ds.load_dataset(
            batch_dims=batch_dims,
            split='train',
            sub_split=sub_split_val,
            shuffle=False,
            augment=False,
            ) 
        eval_test = ds.load_dataset(
            batch_dims=batch_dims, 
            split='test',
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
        trainer = mp.replicate(trainer)
        
        pbar = tqdm(range(1,FLAGS.num_epochs+1))
        for epoch in pbar:
            res_list = []
            for _ in range(step_per_epoch_tr):
                batch_tr = next(train_repeat)
                rng, rng_ = jax.random.split(rng)
                log, trainer = opt_step(trainer, batch_tr, jax.random.split(rng_, num_devices), True)
                log = mp.unreplicate(log)
                log = OrderedDict([(k,f'{np.mean(v):.2f}') for k,v in log.items()])
                log.update({'epoch': epoch})
                log.move_to_end('epoch', last=False)
                res_list.append(log)
                pbar.set_postfix(log)

            if (epoch%FLAGS.log_freq)==0:
                trainer = tool.update_bn(trainer, train_repeat)
                ckpt.save_ckpt(trainer, f'{posthoc_dir}/{i}')
                res = ckpt.reduce_dict_list(res_list)
                log_val = metrics.log_dataset(trainer, eval_val, ds.num_train)
                acc_val = np.mean(log_val['acc'])
                res['acc_val'] = f'{acc_val:.2f}'
                log_test = metrics.log_dataset(trainer, eval_test, ds.num_test)
                acc_test = np.mean(log_test['acc'])
                res['acc_test'] = f'{acc_test:.2f}'
                ckpt.dict2tsv(res, f'{posthoc_dir}/{i}/log.tsv')
                
    print(f'Estimate Confident Learning score')
    logit_val_total = np.zeros((num_train, num_classes))
    label_val_total = np.zeros((num_train, num_classes))
    
    for i in range(2):
        # define pseudo-random number generator
        rng = jax.random.PRNGKey(FLAGS.seed)
        rng, rng_ = jax.random.split(rng)
        
        if i==0:
            sub_split_tr = 0
            sub_split_val = 1
        else:
            sub_split_tr = 1
            sub_split_val = 0
            
        eval_val = ds.load_dataset(
            batch_dims=batch_dims,
            split='train',
            sub_split=sub_split_val,
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
        trainer = ckpt.load_ckpt(f'{posthoc_dir}/{i}', trainer)
        trainer = mp.replicate(trainer)
        logit_val, label_val = tool.get_logit_dataset(trainer, eval_val, num_train, num_classes)
        logit_val_total += logit_val
        label_val_total += label_val
    
    print(f'Compute label quality')
    label_quality = get_label_quality_scores(
        np.array(jnp.argmax(label_val, axis=1)),
        np.array(jax.nn.softmax(logit_val, axis=1)),
    )
    influence = 1. - label_quality
    
    return influence