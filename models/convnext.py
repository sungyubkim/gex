import imp
from typing import Optional, Any, Iterable
from functools import partial
import jax
import jax.numpy as jnp
import haiku as hk

import einops
from dataclasses import dataclass
from gex.models.utils import DropPath

initializer = hk.initializers.TruncatedNormal(0.02)

@dataclass
class ConvNextBlock(hk.Module):
    dim : int = 256
    dropout_rate : float = 0.0
    layer_scale_init_value : float = 1e-6
    norm : Any = None
    name : Optional[str] = None
    
    def __call__(self, x, train):
        y = hk.DepthwiseConv2D(
            channel_multiplier=1,
            kernel_shape=7,
            name=f'depthwise_conv',
            w_init=initializer,
        )(x)
        y = self.norm(name=f'norm')(y)
        y = hk.Linear(
            output_size=4*self.dim,
            name=f'pointwise_conv_0',
            w_init=initializer,
        )(y)
        y = jax.nn.gelu(y)
        y = hk.Linear(
            output_size=self.dim,
            name='pointwise_conv_1',
            w_init=initializer,
        )(y)
        if self.layer_scale_init_value > 0:
            gamma = hk.get_parameter(f'gamma', (self.dim,), init=hk.initializers.Constant(self.layer_scale_init_value))
            y = gamma * y
        y = x + DropPath(self.dropout_rate)(y, train)
        return y

@dataclass
class ConvNext(hk.Module):
    name : str
    depths : Iterable = (3, 3, 9, 3)
    dims : Iterable = (96, 192, 384, 768)
    dropout_rate : float = 0.0
    layer_scale_init_value : float = 1e-6
    head_init_scale : float = 1.0
    num_classes : int = 1000
    eps : float = 1e-5
    
    def __call__(self, x, train=True, print_shape=False):
        norm = partial(
            hk.LayerNorm,
            axis=-1,
            create_scale=True,
            create_offset=True,
            eps=self.eps,
        )
        dropout_rates = jnp.linspace(0, self.dropout_rate, sum(self.depths))
        curr = 0
        
        if print_shape:
            print('print shape of rep.')
            print(x.shape, 'input')
            
        x = hk.Conv2D(
            output_channels=self.dims[0],
            kernel_shape=4,
            stride=4,
            padding='VALID',
            name=f'emb',
            w_init=initializer,
        )(x)
        x = norm()(x)
        if print_shape:
            print(x.shape, 'embedding')
            
        for j in range(self.depths[0]):
            x = ConvNextBlock(
                dim=self.dims[0],
                dropout_rate=dropout_rates[curr + j],
                layer_scale_init_value=self.layer_scale_init_value,
                norm=norm,
                name=f'block_0_{j}',
            )(x, train)
            if print_shape:
                print(x.shape, f'block_0_{j}')
        curr += self.depths[0]
        
        for i in range(1, 4):
            x = norm()(x)
            x = hk.Conv2D(
                output_channels=self.dims[i],
                kernel_shape=2,
                stride=2,
                name=f'block_{i}_downsample',
                w_init=initializer,
            )(x)
            
            for j in range(self.depths[i]):
                x = ConvNextBlock(
                    dim=self.dims[i],
                    dropout_rate=dropout_rates[curr + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    norm=norm,
                    name=f'block_{i}_{j}',
                )(x, train)
                if print_shape:
                    print(x.shape, f'block_{i}_{j}')
            curr += self.depths[i]
            
        x = x.mean(axis=(1,2))
        x = norm(name=f'pre_head_ln')(x)
        if print_shape:
            print(x.shape, 'pre_head')
            
        x = hk.Linear(
            output_size=self.num_classes,
            w_init=hk.initializers.Constant(0.),
            b_init=hk.initializers.Constant(-10.), 
            name=f'head',
            )(x)
        if print_shape:
            print(x.shape, 'final')
        
        return x
    
ConvNext_t = partial(
    ConvNext,
    depths=(3, 3, 9, 3),
    dims=(96, 192, 384, 768),
    dropout_rate=0.1,
)

ConvNext_s = partial(
    ConvNext,
    depths=(3, 3, 27, 3),
    dims=(96, 192, 384, 768),
    dropout_rate=0.4,
)

ConvNext_b = partial(
    ConvNext,
    depths=(3, 3, 27, 3),
    dims=(128, 256, 512, 1024),
    dropout_rate=0.5,
)

ConvNext_l = partial(
    ConvNext,
    depths=(3, 3, 27, 3),
    dims=(192, 384, 768, 1536),
    dropout_rate=0.5,
)

ConvNext_h = partial(
    ConvNext,
    depths=(3, 3, 27, 3),
    dims=(256, 512, 1024, 2048),
    dropout_rate=0.5,
)