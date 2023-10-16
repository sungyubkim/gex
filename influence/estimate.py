import os
import numpy as np
import pandas as pd
from code import compile_command
from absl import flags

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
np.set_printoptions(precision=3, linewidth=100, suppress=True)

from gex.utils import ckpt
from gex.utils.tool import bcolors
from gex.influence.lissa import compute_influence_lissa
from gex.influence.randproj import compute_influence_randproj
from gex.influence.arnoldi import compute_influence_arnoldi
from gex.influence.tracinrp import compute_influence_tracinrp
from gex.influence.tracin import compute_influence_tracin
from gex.influence.laplace import compute_influence_laplace
from gex.influence.fscore import compute_influence_fscore
from gex.influence.el2n import compute_influence_el2n
from gex.influence.random import compute_influence_random
from gex.influence.cl import compute_influence_cl
from gex.influence.knn import compute_influence_knn

flags.DEFINE_bool('self_influence', True, 'compute self-influence')
flags.DEFINE_enum('if_method', 
                  'randproj',
                  ['fscore',
                   'el2n',
                   'lissa', 
                   'gexlr', 
                   'randproj', 
                   'arnoldi', 
                   'tracin',
                   'tracinrp', 
                   'la_fge', 
                   'la_kfac', 
                   'cl',
                   'knn',
                   'random',
                   ],
                  help='method to estimate influence function')
# hyper-parameters for K-FAC
flags.DEFINE_integer('kfac_iter', 1000, help='max iteration of K-FAC')
flags.DEFINE_float('kfac_alpha', 0.01,
help='damping coef. of K-FAC (equivalent to alpha)')
flags.DEFINE_float('kfac_sigma', 0.01, 
help='damping coef. of K-FAC (equivalent to sigma)')
# hyper-parameters for LiSSA
flags.DEFINE_integer('lissa_iter', 10, help='max iteration of lissa')
flags.DEFINE_float('lissa_scale', 10.0, help='scale hyper-parameter')
flags.DEFINE_float('lissa_damping', 0.0, help='damping hyper-parameter')
# hyper-parameters for RandProj
flags.DEFINE_integer('randproj_dim', 20, help='dimension of random projection')
# hyper-parameters for TracInRP
flags.DEFINE_bool('tracinrp_fge', False, help='whether to use fge for tracinrp')
flags.DEFINE_integer('tracinrp_dim', 20, help='dimension of random projection')
# hyper-parameters for Arnoldi
flags.DEFINE_integer('arnoldi_dim', 20, help='dimension of random projection')
flags.DEFINE_integer('arnoldi_iter', 50, help='dimension of random projection')
# hyper-parameters for GEX
flags.DEFINE_float('gex_eps', 0.05, help='tau for GEX')
# hyper-parameters for EL2N
flags.DEFINE_integer('el2n_repeat', 5, help='number of repeat for el2n estimation')
flags.DEFINE_integer('el2n_epoch', 10, help='number of epochs for el2n estimation')
# hyper-parameters for pruning
flags.DEFINE_integer('num_filter', 0, help='number of filtered samples')
flags.DEFINE_bool('iterative_pruning', False, help='whether to use iterative pruning')
flags.DEFINE_string('memo', 'default', help='memo for the experiment')
FLAGS = flags.FLAGS

def compute_influence(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    if 'if_method' in kwargs:
        if_method = kwargs['if_method']
    else:
        if_method = FLAGS.if_method
        
    # load influence 
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
        influence_dir = get_influence_hparams(pretrain_dir, True)
        if os.path.isfile(f'{influence_dir}/influence_{FLAGS.num_filter}.tsv'):
            influence = pd.read_csv(f'{influence_dir}/influence_{FLAGS.num_filter}.tsv', sep='\t').values[:,0]
            return influence
    
    if 'la_' in FLAGS.if_method:
        estimate_method = compute_influence_laplace
    else:
        estimate_method = globals()[f'compute_influence_{if_method}']
        
    influence = estimate_method(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs)
        
    # save influence
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
        influence_dir = get_influence_hparams(pretrain_dir, True)
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
        
    return influence

def get_influence_hparams(pretrain_dir, is_dir=False):
    posthoc_hparams = [
        FLAGS.if_method,
    ]
    if FLAGS.if_method=='kfac':
        posthoc_hparams.extend([FLAGS.kfac_iter, FLAGS.kfac_alpha, FLAGS.kfac_sigma])
    if FLAGS.if_method=='lissa':
        posthoc_hparams.extend([FLAGS.lissa_iter])
    if FLAGS.if_method=='randproj':
        posthoc_hparams.extend([FLAGS.randproj_dim])
    if FLAGS.if_method=='arnoldi':
        posthoc_hparams.extend([FLAGS.arnoldi_dim, FLAGS.arnoldi_iter])
    if FLAGS.if_method=='tracinrp':
        posthoc_hparams.extend([FLAGS.tracinrp_dim])
    if FLAGS.if_method=='gexlr':
        posthoc_hparams.extend([FLAGS.gexlr_dim])
    if FLAGS.if_method=='el2n':
        posthoc_hparams.extend([FLAGS.el2n_repeat, FLAGS.el2n_epoch])
    if 'la_' in FLAGS.if_method:
        if 'fge' in FLAGS.if_method:
            la_hparams = [
                FLAGS.num_ens,
                FLAGS.ft_step,
                FLAGS.ft_lr,
                FLAGS.ft_lr_sched,
                FLAGS.gex_eps,
            ]
            if FLAGS.ft_local:
                la_hparams.append('para')
        else:
            la_hparams = [
                FLAGS.num_ens,
                FLAGS.ft_step,
                FLAGS.alpha,
                FLAGS.sigma,
                FLAGS.gex_eps,
            ]
            if FLAGS.use_connect:
                la_hparams.append('connect')
        posthoc_hparams.extend(la_hparams)
    if not(FLAGS.self_influence):
        posthoc_hparams.append('cross')
    if FLAGS.iterative_pruning:
        posthoc_hparams.append('iter')
    posthoc_hparams = '_'.join(map(str, posthoc_hparams))
    posthoc_hparams = f'{posthoc_hparams}_{FLAGS.memo}'
    if is_dir:
        posthoc_dir = f'{pretrain_dir}/influence/{posthoc_hparams}'
        ckpt.check_dir(posthoc_dir)
        print(f'{bcolors.YELLOW}influence dir: {posthoc_dir}{bcolors.END}')
        return posthoc_dir
    else:
        return posthoc_hparams