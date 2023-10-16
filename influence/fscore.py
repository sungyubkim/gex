import numpy as np
import pandas as pd

def compute_influence_fscore(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    
    pretrain_dir = kwargs['pretrain_dir']
    # fscore = pd.read_csv(f'{pretrain_dir}/fscore.tsv', sep='\t').values[:,-1]
    # fscore = pd.read_csv(f'{pretrain_dir}/fscore.tsv', sep='\t').values[-1]
    fscore = np.genfromtxt(fname=f"{pretrain_dir}/fscore.tsv", delimiter='\t')[-1]
    
    return fscore