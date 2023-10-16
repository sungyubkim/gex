from genericpath import isdir
from random import gauss, random
from absl import flags
import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import gex.laplace.utils as la_utils
from time import time
import os
import pandas as pd

from gex.utils import ckpt, metrics, mp, tool
from gex.laplace.inference import get_posterior

FLAGS = flags.FLAGS

def compute_influence_laplace(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs):
        
    # load/generate sampled models
    la_method = FLAGS.if_method.split('_')[1]
    posterior = get_posterior(trainer, la_method=la_method, *args, **kwargs)
    num_train = kwargs['num_train']
    num_test = kwargs['num_test']
    
    loss_tr_list = compute_loss(posterior, dataset_tr, num_train, la_method) # (N_tr, M)    
    loss_tr_orig = np.maximum(metrics.log_dataset(trainer, dataset_tr, num_train)['loss'], FLAGS.gex_eps) # (N_tr,)
    loss_tr_deviation = loss_tr_list - loss_tr_orig.reshape(-1,1) # (N_tr, M)
        
    if self_influence:
        influence = (loss_tr_deviation**2).mean(axis=-1) # (N_tr,)
    else:
        loss_te_list = compute_loss(posterior, dataset_te, num_test, la_method).mean(axis=0).reshape(1, -1) # (1, M)
        loss_te_orig = np.maximum(metrics.log_dataset(trainer, dataset_te, num_test)['loss'].mean(), 0.05) # (1,)
        loss_te_deviation = loss_te_list - loss_te_orig # (1, M)
        influence = (loss_te_deviation * loss_tr_deviation).mean(axis=-1) # (N_tr,)
    return influence
    
def compute_loss(posterior, dataset, num_samples, la_method):
    loss_list = []
    for i in tqdm(range(len(posterior))):
    # for i in tqdm(range(int(FLAGS.memo))):
        member = mp.replicate(posterior[i])
        use_lin = not(la_method=='fge')
        if use_lin:
            log = metrics.log_dataset(member, dataset, num_samples, fn_type='lin')
        else:
            log = metrics.log_dataset(member, dataset, num_samples)
        loss_list.append(log['loss']) # (N,)
        
    loss_list = jnp.stack(loss_list, -1) # (N, M)
    return loss_list