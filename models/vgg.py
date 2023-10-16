from typing import Any, Callable, Tuple, Sequence

ModuleType = Any

import jax
import haiku as hk
from dataclasses import dataclass
from functools import partial

@dataclass
class VGG(hk.Module):
    name : str
    num_classes : int
    num_filters : Sequence[int] = (64, 128, 256, 512)
    eps : float=1e-5
    
    def __call__(self, x, train=True, print_shape=False):
        conv = partial(
            hk.Conv2D,
            with_bias=False,
        )
        norm = partial(
            hk.BatchNorm,
            create_scale=True,
            create_offset=True,
            decay_rate=0.9,
            eps=self.eps,
        )
        
        if print_shape:
            print('print shape of rep.')
            print(x.shape, 'input')
            
        for i, filter_size in enumerate(self.num_filters):
            for j in range(2):
                x = conv(
                    output_channels=filter_size,
                    kernel_shape=3,
                    stride=1,
                    name=f'block_{i}_{j}',
                )(x)
                x = norm(name=f'block_{i}_{j}_bn')(x, train)
                x = jax.nn.relu(x)
                if print_shape:
                    print(x.shape, f'block_{i}_{j}')
            x = hk.max_pool(x, window_shape=2, strides=2, padding='SAME')
            
        x = x.mean(axis=(1,2))
        x = hk.Linear(
                output_size=self.num_classes,
                name=f'head',
            )(x)
        if print_shape:
            print(x.shape, 'head')
        return x