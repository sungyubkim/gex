import os
import jax
import haiku as hk
import flax
from flax.training import checkpoints
from flax.training.checkpoints import restore_checkpoint as load_ckpt
import numpy as np
import pickle
from collections import OrderedDict
from gex.utils.tool import bcolors

from gex.utils import mp

def save_ckpt(trainer, path):
    if jax.process_index() == 0:
        trainer = jax.device_get(mp.unreplicate(trainer))
        checkpoints.save_checkpoint(path, trainer, trainer.step, overwrite=True)

def save_ens(list_of_state, dest_dir, group_name='fge'):
    # save list of state as ckpt file
    num_ens = len(list_of_state)
    for i in range(num_ens):
        save_ckpt(mp.replicate(list_of_state[i]),f'{dest_dir}/{group_name}/{i}')
    return None

def load_ens(state, num_ens, dest_dir, group_name='fge'):
    # load list of state from ckpt file
    result = []
    for i in range(num_ens):
        state = load_ckpt(f'{dest_dir}/{group_name}/{i}', state)
        result.append(state)
    return result

def check_dir(folder_path):
    # save path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
def pretty_log(log):
    res = []
    for k,v in log.items():
        if type(v)==int:
           res.append((k,v))
        else:
            res.append((k,f'{np.mean(v):.4f}'))
    return OrderedDict(res)

def print_log(log):
    print(f'{bcolors.BLUE}')
    print(' '.join("{:<8}|".format(k) for k in log.keys()))
    print(' '.join("{:<8}|".format(v) for v in log.values()))
    print(f'{bcolors.END}')

def dict2tsv(res, file_name):
    for k, v in res.items():
        if type(v)==str:
            res[k] = v
        elif k=='epoch':
            res[k] = v
        else:
            res[k] = f'{v:.4f}'
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('\t'.join(list(res.keys())))
            f.write('\n')

    with open(file_name, 'a') as f:
        f.write('\t'.join([str(r) for r in list(res.values())]))
        f.write('\n')
        
def array2tsv(log, file_name):
    with open(file_name, 'ab') as f:
        np.savetxt(f, log.reshape(1,-1), delimiter='\t', fmt='%.2f')

def reduce_dict_list(dict_list):
    res = {}
    keys = dict_list[0].keys()
    for k in keys:
        if k=='epoch':
            res[k] = dict_list[0][k]
        else:
            v = []
            for d in dict_list:
                v.append(float(d[k]))
            v = np.mean(v)
            res[k] = float(v)
    return res