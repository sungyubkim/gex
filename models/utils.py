from typing import Optional, Any, Iterable
from functools import partial
import jax
import jax.numpy as jnp
import haiku as hk

import einops
from dataclasses import dataclass

@dataclass
class DropPath(hk.Module):
    dropout_rate : 0.0
    
    def __call__(self, x, train):
        dropout_rate = self.dropout_rate if train else 0.
        keep_rate = 1 - dropout_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = hk.next_rng_key()
        mask = keep_rate + jax.random.uniform(rng, shape)
        mask = jnp.floor(mask)
        return (x / keep_rate) * mask