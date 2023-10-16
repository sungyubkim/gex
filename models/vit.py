from typing import Optional, Any, Callable, Tuple, Sequence
from functools import partial
import jax
import jax.numpy as jnp
import haiku as hk
from dataclasses import dataclass

xavier_uniform = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    
@dataclass
class MlpBlock(hk.Module):
    hidden_size : int
    mlp_dim : int
    dropout_rate : float
    name : Optional[str] = None
        
    def __call__(self, x, train):
        dropout_rate = self.dropout_rate if train else 0.
        x = hk.Linear(self.mlp_dim, w_init=xavier_uniform)(x)
        # (n, 1 + h * w, mlp_dim)
        x = jax.nn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        x = hk.Linear(self.hidden_size, w_init=xavier_uniform)(x)
        # (n, 1 + h * w, c)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        return x
    
@dataclass
class SelfAttention(hk.Module):
    num_heads : int
    key_size : int
    dropout_rate : float
    name : Optional[str] = None
    
    def __call__(self, x, train):
        dropout_rate = self.dropout_rate if train else 0.
        
        n, l, hd = x.shape
        h = self.num_heads
        d = self.key_size
        assert hd == h * d
        
        q = hk.Linear(self.key_size * self.num_heads, w_init=xavier_uniform, name='q')(x).reshape(n, l, h, d)
        k = hk.Linear(self.key_size * self.num_heads, w_init=xavier_uniform, name='k')(x).reshape(n, l, h, d)
        v = hk.Linear(self.key_size * self.num_heads, w_init=xavier_uniform, name='v')(x).reshape(n, l, h, d)
        
        qkt = jnp.einsum('n q h d, n k h d -> n q k h', q, k) / jnp.sqrt(self.key_size)
        attn = jax.nn.softmax(qkt, axis=2) # softmax over keys
        attn = hk.dropout(hk.next_rng_key(), dropout_rate, attn)
        
        x = jnp.einsum('n q k h, n k h d -> n q h d', attn, v)
        x = x.reshape(n, l, hd)
        
        x = hk.Linear(self.key_size * self.num_heads, w_init=xavier_uniform, name='o')(x)
        return x
        
    
@dataclass
class Encoder1DBlock(hk.Module):
    num_heads : int
    hidden_size : int
    mlp_dim : int
    dropout_rate : float
    norm : Any
    name : Optional[str] = None
        
    def __call__(self, x, train):
        dropout_rate = self.dropout_rate if train else 0.
        y = self.norm(name=f'norm_0')(x)
        y = SelfAttention(
            num_heads=self.num_heads,
            key_size=64,
            dropout_rate=0,
            name=f'self_attn',
        )(y, train)
        # attending over sequence of (1 + h * w) patches
        # (n, 1 + h * w, hidden_size)
        y = hk.dropout(hk.next_rng_key(), dropout_rate, y)
        x = x + y
        
        y = self.norm(name=f'norm_1')(x)
        y = MlpBlock(
            hidden_size=self.hidden_size,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            name=f'mlp_block',
        )(y, train)
        
        return x + y
        
@dataclass
class Encoder(hk.Module):
    num_layers : int
    num_heads : int
    hidden_size : int
    mlp_dim : int
    dropout_rate : float
    norm : Any
    name : Optional[str] = None
        
    def __call__(self, x, train):
        for layer in range(self.num_layers):
            x = Encoder1DBlock(
                    hidden_size=self.hidden_size,
                    mlp_dim=self.mlp_dim,
                    dropout_rate=self.dropout_rate,
                    num_heads=self.num_heads,
                    norm=self.norm,
                    name=f'encoder_block_{layer}',
                )(x, train)
            # (n, 1 + h * w, hidden_size)
        encoded = self.norm(name='norm')(x)
        return encoded

@dataclass
class ViT(hk.Module):
    name : str
    patch_size : int
    num_classes : int
    num_heads : int
    hidden_size : int
    mlp_dim : int
    dropout_rate : float
    num_layers : int = 12
    representation_size : int = None
    eps : float = 1e-5
    pool_type : str = 'cls_token'

    def __call__(self, x, train=True, print_shape=False):
        dropout_rate = self.dropout_rate if train else 0.
        norm = partial(
            hk.LayerNorm,
            axis=-1,
            create_scale=True,
            create_offset=True,
            eps=self.eps,
        )
        
        if print_shape:
            print('print shape of rep.')
            print(x.shape, 'input')
            
        x = hk.Conv2D(
            self.hidden_size,
            self.patch_size,
            self.patch_size,
            padding='VALID',
            name=f'emb',
            )(x)
        if print_shape:
            print(x.shape, 'embedding')
        # (n, h, w, hidden_size)
        
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c]) # (n, h * w, hidden_size)
        if print_shape:
            print(x.shape, 'reshape')
        
        pos_embedding = hk.get_parameter(f'pos_emb', (1, h * w, c), init=hk.initializers.RandomNormal(0.02))
        x = x + pos_embedding
        if print_shape:
            print(x.shape, 'pos')
        # (n, h * w, hidden_size)
        
        cls_token = hk.get_parameter(f'cls_token', (1, 1, c), init=jnp.zeros)
        cls_token = jnp.tile(cls_token, [n, 1, 1])
        x = jnp.concatenate([cls_token, x], axis=1)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        if print_shape:
            print(x.shape, 'cls')
        # (n, 1 + h * w, hidden_size)
        
        x = Encoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            norm=norm,
            name=f'encoder',
            )(x, train)
        if print_shape:
            print(x.shape, 'transformer')
        # (n, 1 + h * w, hidden_size)
        
        if self.pool_type=='cls_token':
            x = x[:, 0]
        elif self.pool_type=='gap':
            x = x.mean(axis=1)
            
        if print_shape:
            print(x.shape, 'token')
        # (n, hidden_size)
        
        if self.representation_size is not None:
            x = hk.Linear(self.representation_size, name='pre_logits')(x)
            x = jax.nn.tanh(x)
            print(x.shape, 'pre_logits')
        
        # init. of w & b for last-layer : https://github.com/google-research/vision_transformer/issues/34
        x = hk.Linear(
            output_size=self.num_classes,
            w_init=hk.initializers.Constant(0.),
            b_init=hk.initializers.Constant(-10.), 
            name=f'head',
            )(x)
        
        if print_shape:
            print(x.shape, 'final')
        # (n, num_classes)

        return x
    
ViT_s = partial(
    ViT,
    hidden_size = 384,
    mlp_dim = 1536,
    num_heads = 6,
    dropout_rate = 0.0,
)
ViT_b = partial(
    ViT,
    hidden_size = 768,
    mlp_dim = 3072,
    num_heads = 12,
    dropout_rate = 0.1,
)
ViT_l = partial(
    ViT,
    hidden_size = 1024,
    mlp_dim = 4096,
    num_heads = 16,
    dropout_rate = 0.1,
)
ViT_h = partial(
    ViT,
    hidden_size = 1280,
    mlp_dim = 5120,
    num_heads = 16,
    dropout_rate = 0.1,
)