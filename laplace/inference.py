import os
from absl import flags
from copy import deepcopy
import jax

from gex.utils import ckpt, metrics, mp, tool
# from gex.laplace.fge import get_posterior_fge
from gex.laplace.fge import get_posterior_fge
from gex.laplace.full import get_posterior_full
from gex.laplace.kfac import get_posterior_kfac
from gex.laplace.diag import get_posterior_diag
from gex.utils.tool import bcolors

# inference hyper-parameters
flags.DEFINE_enum('la_method', 'fge', [
                    'fge', 
                    'rto', 
                    'lr',
                    'full', 
                    'kfac', 
                    'diag', 
                    'll', 
                    ], 
                  help='inference method for LA)')
flags.DEFINE_bool('use_connect', False, help='use connectivity or parameter')
flags.DEFINE_integer('num_ens', 32, help='number of ensemble members')
# hyper-parameters for FGE
flags.DEFINE_bool('ft_local', True, help='use local fine-tuning')
flags.DEFINE_integer('ft_step', 800, help='step number of posterior sampling')
flags.DEFINE_float('ft_lr', 0.05, help='peak lr for fine-tuning')
flags.DEFINE_enum('ft_lr_sched', 'constant', ['cosine', 'constant'], help='fine-tuning LR scheduler')
# prior hyper-parameters (only for LA)
flags.DEFINE_float('alpha', 0.0001, help='prior scale')
flags.DEFINE_float('sigma', 0.0001, help='observational noise scale')
# LR hyper-parameters
flags.DEFINE_integer('lr_iter', 100, help='number of iterations for LA-LR')
flags.DEFINE_integer('lr_dim', 90, help='prior scale')
flags.DEFINE_integer('start_rank', 0, help='')
flags.DEFINE_integer('num_rank', 10, help='')
FLAGS = flags.FLAGS

def get_posterior(trainer, *args, **kwargs):
    if 'la_method' in kwargs:
        la_method = kwargs['la_method']
    else:
        la_method = FLAGS.la_method
        
    # load sampled models
    if 'pretrain_dir' in kwargs:
        pretrain_dir = kwargs['pretrain_dir']
        posthoc_dir = get_posthoc_hparams(pretrain_dir, True)
        if os.path.isdir(f'{posthoc_dir}/{la_method}'):
            if 'fge' in la_method:
                trainer_ft = tool.init_trainer_ft_p(trainer)
            else:
                trainer_ft = tool.init_trainer_ft_lin(mp.unreplicate(trainer))
            posterior = ckpt.load_ens(
                trainer_ft, 
                FLAGS.num_ens, 
                posthoc_dir,
                f'{la_method}',
                )
            return posterior
        
    # generate sampled models
    la_method_fn = globals()[f'get_posterior_{la_method}']
    if la_method=='kfac':
        # Since kfac_jax automatically remove trainer, we backup trainer.
        trainer = deepcopy(trainer)
    posterior = la_method_fn(trainer, *args, **kwargs)
    return posterior

def get_posthoc_hparams(pretrain_dir, is_dir=False):

    if 'la' in FLAGS.if_method:
        la_method = FLAGS.if_method.split('_')[1]
    else:
        la_method = FLAGS.la_method
        
    posthoc_hparams = [la_method]
    if 'fge' in la_method:
        la_hparams = [
            FLAGS.num_ens,
            FLAGS.ft_step,
            FLAGS.ft_lr,
            FLAGS.ft_lr_sched,
        ]
        if FLAGS.ft_local:
            la_hparams.append('para')
    else:
        la_hparams = [
            FLAGS.num_ens,
            FLAGS.ft_step,
            FLAGS.alpha,
            FLAGS.sigma,
        ]
        if FLAGS.use_connect:
            la_hparams.append('connect')
    posthoc_hparams.extend(la_hparams)
    posthoc_hparams = '_'.join(map(str, posthoc_hparams))
    if is_dir:
        posthoc_dir = f'{pretrain_dir}/laplace/{posthoc_hparams}'
        ckpt.check_dir(posthoc_dir)
        print(f'{bcolors.BLUE}posthoc dir: {posthoc_dir}{bcolors.END}')
        return posthoc_dir
    else:
        return posthoc_hparams