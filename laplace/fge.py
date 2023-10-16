import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import Any, OrderedDict
from tqdm import tqdm
from absl import flags
from time import time

from gex.utils import ckpt, metrics, mp, tool
from gex.laplace.utils import get_statistics, fge_to_swa

FLAGS = flags.FLAGS

def get_posterior_fge(trainer, dataset_opt, *args, **kwargs):
    
    # parsing settings
    opt_step = kwargs['opt_step']
    rng = jax.random.PRNGKey(42)
    num_devices = jax.device_count()
    sync_grad = not(FLAGS.ft_local)
    if FLAGS.ft_local:
        num_stage = int(np.ceil(float(FLAGS.num_ens)/num_devices))
    else:
        num_stage = FLAGS.num_ens
    if 'ft_step' in kwargs:
        ft_step = kwargs['ft_step']
    else:
        ft_step = FLAGS.ft_step
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
    else:
        pretrain_dir = None
    
    print(f'Start {FLAGS.num_ens} ensemble training')
    posterior = []
    trainer_ft = trainer
    
    num_train = kwargs['num_train']
    loss_accumulation_list = []
    loss_accumulation = np.zeros(num_train)
    
    for i in range(num_stage):

        trainer_ft = tool.init_trainer_ft_p(trainer_ft)
            
        pbar = tqdm(range(ft_step))
        for step in pbar:
            if i==0 and step==1:
                # remove first iteration to exclude compile time
                start_time = time()
            batch_tr = next(dataset_opt)
            rng, rng_ = jax.random.split(rng)
            log, trainer_ft = opt_step(
                trainer_ft,
                batch_tr,
                jax.random.split(rng_, num_devices),
                sync_grad,
                )
            
            # update loss_diff
            for m in range(num_devices):
                batch_idx = batch_tr['idx'][m]
                if FLAGS.pert_scale_sam > 0:
                    loss = log['loss_sam'][m]
                else:
                    loss = log['loss_sgd'][m]
                loss_accumulation[batch_idx] += loss
            
            # check loss_accumulation is all filled
            if (loss_accumulation != 0).all():
                loss_accumulation_list.append(loss_accumulation)
                loss_accumulation = np.zeros(num_train)
            
            log = OrderedDict([(k,f'{np.mean(v):.2f}') for k,v in log.items()])
            log.update({'stage': i})
            log.move_to_end('stage', last=False)
            pbar.set_postfix(log)
            
        end_time = time()
        print(f'Cache time except compile : {end_time - start_time:.4} s')

        print(f'Post-processing members')
        if FLAGS.use_bn:
            trainer_ft = tool.update_bn(trainer_ft, dataset_opt)
        if FLAGS.ft_local:
            # add multiple local models
            for ens_mem in tqdm(range(num_devices)):
                member = mp.replicate(mp.unreplicate(trainer_ft, ens_mem))
                posterior.append(mp.unreplicate(member))
        else:
            # add single model
            posterior.append(mp.unreplicate(trainer_ft))
    
    if pretrain_dir is not None:
        from gex.laplace.inference import get_posthoc_hparams
        posthoc_dir = get_posthoc_hparams(pretrain_dir, True)
        fge_to_swa(posterior, posthoc_dir, dataset_opt)
        ckpt.save_ens(posterior, posthoc_dir, 'fge')
        
    return posterior