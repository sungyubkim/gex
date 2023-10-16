from typing import OrderedDict
from absl import app, flags
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gex.utils import ckpt, metrics, mp, tool

FLAGS = flags.FLAGS

def compute_influence_knn(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    
    pretrain_dir = kwargs['pretrain_dir']
    num_train = kwargs['num_train']
    num_classes = kwargs['num_classes']
    result = []
    
    for member in range(5):
        
        trainer = mp.unreplicate(trainer)
        trainer = ckpt.load_ckpt(f'{pretrain_dir}/traj/{member}', trainer)
        trainer = mp.replicate(trainer)
        
        logit_tr, label_tr = tool.get_logit_dataset(trainer, dataset_tr, num_train, num_classes)
        
        batch_size = 100
        k = logit_tr.shape[0]//logit_tr.shape[1]
        
        filter_list = []
        pbar = tqdm(range(logit_tr.shape[0]//batch_size))
        for i in pbar:
            # find k-nearest logits
            tgt = logit_tr[i*batch_size:i*batch_size+batch_size,np.newaxis] 
            # (batch_size, 1, num_class)
            
            dist = ((tgt - logit_tr[np.newaxis])**2).sum(axis=-1) 
            # (batch_size, n_tr)
            
            sorted_idx = np.argsort(dist, axis=-1)[:,:k] 
            # (batch_size, knn_k)
            
            label_sorted = label_tr[sorted_idx] 
            # (batch_size, knn_k, num_class)
            
            label_knn = label_sorted.mean(axis=1) 
            # (batch_size, num_class)
            
            label_true = label_tr[i*batch_size:i*batch_size+batch_size] 
            # (batch_size, num_class)
            
            is_neq = ((label_knn-label_true)**2).sum(axis=-1) 
            # (batch_size,)
            
            filter_list.append(is_neq)
            
        filter_list = np.concatenate(filter_list, axis=0)
        result.append(filter_list)
    
    result = np.stack(result, axis=0)
    result = result.mean(axis=0)
        
    return result