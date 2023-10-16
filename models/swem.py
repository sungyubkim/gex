from typing import Any, Callable, Tuple, Sequence

ModuleType = Any

import jax
import haiku as hk
from dataclasses import dataclass
from functools import partial

@dataclass
class SWEM(hk.Module):
    name : str
    num_classes : int
    dropout_rate : float = 0.3
    
    def __call__(self, x, train=True, print_shape=False):
        
        mask = x > 0
        
        if print_shape:
            print('print shape of rep.')
            print(x.shape, 'input')
            
        x = hk.Embed(
            vocab_size=1000,
            embed_dim=512,
            w_init=hk.initializers.UniformScaling(0.001),
        )(x)
        if print_shape:
            print(x.shape, 'embedding')
            # (batch, seq, dim)
        
        x = x * mask.reshape(mask.shape + (1,))
        if print_shape:
            print(x.shape, 'padding')
            # (batch, seq, dim)
        
        x = x.sum(axis=1) / mask.sum(axis=1).reshape(mask.shape[0], 1) 
        if print_shape:
            print(x.shape, 'pooled')
            # (batch, dim)
        
        if train:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        x = hk.Linear(
            output_size=512,
            name=f'linear',
        )(x)
        x = jax.nn.relu(x)
        if train:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        x = hk.Linear(
                output_size=self.num_classes,
                name=f'head',
            )(x)
        if print_shape:
            print(x.shape, 'head')
            
        return x