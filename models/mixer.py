from typing import Optional, Any
from functools import partial
import jax
import jax.numpy as jnp
import haiku as hk

import einops
from dataclasses import dataclass
from gex.models.utils import DropPath

@dataclass
class MlpBlock(hk.Module):
    mlp_dim : int
    name : Optional[str] = None
    
    def __call__(self, x):
        y = hk.Linear(self.mlp_dim)(x)
        y = jax.nn.gelu(y)
        y = hk.Linear(x.shape[-1])(y)
        return y

@dataclass
class MixerBlock(hk.Module):
    tokens_mlp_dim : int
    channels_mlp_dim : int
    dropout_rate : float
    norm : Any
    name : Optional[str] = None
    
    def __call__(self, x, train):
        y = self.norm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + DropPath(self.dropout_rate)(y, train)
        y = self.norm()(x)
        y = MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)
        y = x + DropPath(self.dropout_rate)(y, train)
        return y

@dataclass
class MLPMixer(hk.Module):
    name : str
    patch_size : int
    num_classes : int
    num_blocks : int
    hidden_size : int
    tokens_mlp_dim : int
    channels_mlp_dim : int
    dropout_rate : float
    eps : float = 1e-6
    
    def __call__(self, x, train=True, print_shape=False):
        norm = partial(
            hk.LayerNorm,
            axis=-1,
            create_scale=True,
            create_offset=True,
            eps=self.eps,
        )
        dropout_rates = jnp.linspace(0, self.dropout_rate, self.num_blocks)
        
        if print_shape:
            print('print shape of rep.')
            print(x.shape, 'input')
            
        x = hk.Conv2D(
            output_channels=self.hidden_size,
            kernel_shape=self.patch_size,
            stride=self.patch_size,
            padding='SAME',
            name=f'emb',
            )(x)
        if print_shape:
            print(x.shape, 'embedding')
            
        x = einops.rearrange(x, 'n h w c -> n (h w) c')
        if print_shape:
            print(x.shape, 'reshape')
        
        for i in range(self.num_blocks):
            x = MixerBlock(
                tokens_mlp_dim=self.tokens_mlp_dim, 
                channels_mlp_dim=self.channels_mlp_dim,
                dropout_rate=dropout_rates[i],
                norm=norm,
                )(x, train)
            if print_shape:
                print(x.shape, f'block_{i}')
        
        x = norm(name=f'pre_head_ln')(x)
        x = x.mean(axis=1)
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
    
Mixer_s = partial(
    MLPMixer,
    num_blocks = 8,
    hidden_size = 512,
    tokens_mlp_dim = 256,
    channels_mlp_dim = 2048,
    dropout_rate=0.1,
)
Mixer_b = partial(
    MLPMixer,
    num_blocks = 12,
    hidden_size = 768,
    tokens_mlp_dim = 384,
    channels_mlp_dim = 3072,
    dropout_rate=0.1,
)
Mixer_l = partial(
    MLPMixer,
    num_blocks = 24,
    hidden_size = 1024,
    tokens_mlp_dim = 512,
    channels_mlp_dim = 4096,
    dropout_rate=0.1,
)
Mixer_h = partial(
    MLPMixer,
    num_blocks = 32,
    hidden_size = 1280,
    tokens_mlp_dim = 640,
    channels_mlp_dim = 5120,
    dropout_rate=0.1,
)