from absl import flags
from gex.utils import ckpt
from gex.utils.tool import bcolors

# additional hyper-parameters
flags.DEFINE_integer('num_epochs', 200, help='epoch number of pre-training')
flags.DEFINE_string('model', 'resnet_18', help='model architecture')
flags.DEFINE_enum('dataset', 'cifar10', ['mnist', 'cifar10', 'cifar100', 'svhn_cropped', 'hetero','imagenet2012', 'cifar10_n', 'cifar100_n', 'dbpedia'], help='training dataset')
flags.DEFINE_integer('seed', 0, help='random number seed')
flags.DEFINE_bool('eval', False, help='do not training')
flags.DEFINE_integer('log_freq',10, help='(epoch) frequency of logging')

# tunable hparams for generalization
flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'adamw'], help='optimizer')
flags.DEFINE_float('weight_decay', 0.0005, help='l2 regularization coeffcient')
flags.DEFINE_float('peak_lr', 0.4, help='peak learning during learning rate schedule')
flags.DEFINE_float('warmup_ratio', 0.1, help='warmup ratio for training epochs')
flags.DEFINE_integer('batch_size_device', 1024, help='gpu-wise batch size')
flags.DEFINE_integer('batch_size_total', 1024, help='total batch size')
flags.DEFINE_integer('num_ckpt', 5, help='number of ckpts for TracIn')
FLAGS = flags.FLAGS

def get_pretrain_hparams(is_dir=False):
    hparams = [
        FLAGS.model,
        FLAGS.optimizer,
        FLAGS.num_epochs,
        FLAGS.peak_lr,
        FLAGS.weight_decay,
        FLAGS.batch_size_total,
        ]
    if FLAGS.label_smooth > 0.0:
        hparams.extend(['ls', FLAGS.label_smooth])
    if FLAGS.pert_scale_sam > 0.0:
        hparams.extend(['sam', FLAGS.pert_scale_sam])
    if FLAGS.mixup_alpha > 0.0:
        hparams.extend(['mixup', FLAGS.mixup_alpha])
    if FLAGS.cutmix_alpha > 0.0:
        hparams.extend(['cutmix', FLAGS.cutmix_alpha])
    if FLAGS.corruption_ratio > 0.0:
        hparams.extend(['corrupted', FLAGS.corruption_ratio])
    if FLAGS.problem_type != 'cls':
        hparams.append(FLAGS.problem_type)
    hparams.append(FLAGS.seed)
    hparams = '_'.join(map(str, hparams))
    pretrain_dir = f'./gex/result/{FLAGS.dataset}/{hparams}'
    ckpt.check_dir(pretrain_dir)
    if is_dir:
        print(f'{bcolors.YELLOW}pretrain dir: {pretrain_dir}{bcolors.END}')
        return pretrain_dir
    else:
        return hparams