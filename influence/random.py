from functools import partial
from typing import OrderedDict
from absl import flags
import jax
import jax.numpy as jnp
from numpy.random import default_rng
import numpy as np
from tqdm import tqdm
from time import time
import optax

from gex.utils import ckpt, metrics, mp, tool

FLAGS = flags.FLAGS

def compute_influence_random(trainer, dataset_tr, dataset_te, self_influence, *args, **kwargs):
    num_train = kwargs['num_train']
    rng = default_rng(FLAGS.seed)
    result = rng.normal(size=(num_train,))
    return result