from typing import OrderedDict
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

from gex.utils import ckpt, metrics, mp, tool
from gex.datasets.image import ImageDataset
from gex.datasets.dbpedia import TextDataset
from gex.pretrain.main import init_trainer

FLAGS = flags.FLAGS

def compute_influence_el2n(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    
    rng = jax.random.PRNGKey(42)
    num_devices = jax.device_count()
    
    dataset_opt = kwargs['dataset_opt']
    opt_step = kwargs['opt_step']
    
    if FLAGS.dataset == 'dbpedia':
        ds = TextDataset()
    else:
        ds = ImageDataset()
    step_per_epoch = np.ceil(float(ds.num_train) / FLAGS.batch_size_device).astype(int)
    
    # score_list = []
    score_accum = np.zeros(ds.num_train)
    for repeat in range(FLAGS.el2n_repeat):
        
        rng, rng_ = jax.random.split(rng)
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
        
        # pretrain for small epochs
        pbar = tqdm(range(1, FLAGS.el2n_epoch+1))
        for epoch in pbar:
            for step in range(step_per_epoch):
                batch = next(dataset_opt)
                rng, rng_ = jax.random.split(rng)
                log, trainer = opt_step(trainer, batch, jax.random.split(rng_, num_devices), True)
                
                log = OrderedDict([(k,f'{np.mean(v):.2f}') for k,v in log.items()])
                log.update({'epoch': epoch})
                log.move_to_end('epoch', last=False)
                pbar.set_postfix(log)
        
        if FLAGS.use_bn:
            trainer = tool.update_bn(trainer, dataset_opt)
        
        log = metrics.log_dataset(trainer, dataset_tr, ds.num_train)
        score_accum += log['brier']
    
    return score_accum / FLAGS.el2n_repeat