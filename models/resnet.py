from typing import Any, Callable, Tuple, Sequence

ModuleType = Any

import jax
import haiku as hk
from dataclasses import dataclass
from functools import partial

@dataclass
class Block(hk.Module):
    name : str
    conv : ModuleType
    norm : ModuleType
    act : Callable
    filters : int
    stride : int = 1
    
    def __call__(self, x, train):
        y = x
        y = self.conv(output_channels=self.filters, kernel_shape=3, name=f'{self.name}_conv_0')(y)
        y = self.norm(name=f'{self.name}_norm_0')(y, train)
        y = self.act(y)
        y = self.conv(output_channels=self.filters, kernel_shape=3, stride=self.stride, name=f'{self.name}_conv_1')(y)
        y = self.norm(name=f'{self.name}_norm_1')(y, train)
        
        if x.shape != y.shape:
            x = self.conv(output_channels=self.filters, kernel_shape=1, stride=self.stride, name=f'{self.name}_shorcut')(x)
            x = self.norm(name=f'{self.name}_norm_shortcut')(x, train)
        
        return self.act(x + y)
    
@dataclass
class Bottleneck(hk.Module):
    name : str
    conv : ModuleType
    norm : ModuleType
    act : Callable
    filters : int
    stride : int = 1
    
    def __call__(self, x, train):
        y = x
        y = self.conv(output_channels=self.filters, kernel_shape=1, name=f'{self.name}_conv_0')(y)
        y = self.norm(name=f'{self.name}_norm_0')(y, train)
        y = self.act(y)
        y = self.conv(output_channels=self.filters, kernel_shape=3, stride=self.stride, name=f'{self.name}_conv_1')(y)
        y = self.norm(name=f'{self.name}_norm_1')(y, train)
        y = self.act(y)
        y = self.conv(output_channels=self.filters * 4, kernel_shape=1, name=f'{self.name}_conv_2')(y)
        y = self.norm(name=f'{self.name}_norm_2')(y, train)
        
        if x.shape != y.shape:
            x = self.conv(output_channels=self.filters * 4, kernel_shape=1, stride=self.stride, name=f'{self.name}_shorcut')(x)
            x = self.norm(name=f'{self.name}_norm_shortcut')(x, train)
        
        return self.act(x + y)
    
@dataclass
class ResNet(hk.Module):
    name : str
    block_cls : ModuleType
    stage_sizes : Sequence[int]
    num_classes : int
    strides : Sequence[int] = (1, 2, 2, 2)
    num_filters : Sequence[int] = (64, 128, 256, 512)
    act : Callable = jax.nn.relu
    eps : float=1e-5
    imagenet : bool = False
    
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
        
        if self.imagenet:
            x = conv(
				output_channels=64,
				kernel_shape=7,
				stride=2,
				name=f'embedding_conv',
			)(x)
            x = norm(name='embedding_bn')(x, train)
            x = self.act(x)
            x = hk.max_pool(x, window_shape=3, strides=2, padding='SAME')
        else:
            x = conv(
				output_channels=64,
				kernel_shape=3,
				stride=1,
				name=f'embedding',
			)(x)
            x = norm(name='embedding_bn')(x, train)
            x = self.act(x)
        if print_shape:
            print(x.shape, 'embedding')
        
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # only the first block of block_group follows stride
                # other blocks are stride 1.
                stride = (1 if (j > 0) else self.strides[i])
                x = self.block_cls(
                    name=f'block_{i}_{j}',
                    conv=conv,
                    norm=norm,
                    filters=self.num_filters[i],
                    stride=stride,
                    act=self.act
                    )(x, train)
                if print_shape:
                    print(x.shape, f'block_{i}_{j}')
        
        x = x.mean(axis=(1,2))
        if print_shape:
            print(x.shape, 'pre_logits')
            
        if self.imagenet:
            x = hk.Linear(
                output_size=self.num_classes,
                w_init=hk.initializers.Constant(0.),
                b_init=hk.initializers.Constant(-10.), 
                name=f'head',
                )(x)
        else:
            x = hk.Linear(
                output_size=self.num_classes,
                name=f'head',
            )(x)
        if print_shape:
            print(x.shape, 'head')
        return x
    
ResNet18 = partial(
    ResNet,
    stage_sizes=[2, 2, 2, 2],
    block_cls=Block,
    )
ResNet34 = partial(
    ResNet,
    stage_sizes=[3, 4, 6, 3],
    block_cls=Block,
    )
ResNet50 = partial(
    ResNet,
    stage_sizes=[3, 4, 6, 3],
    block_cls=Bottleneck,
    )
ResNet101 = partial(
    ResNet,
    stage_sizes=[3, 4, 23, 3],
    block_cls=Bottleneck,
    )
ResNet152 = partial(
    ResNet,
    stage_sizes=[3, 8, 36, 3],
    block_cls=Bottleneck,
    )
ResNet200 = partial(
    ResNet,
    stage_sizes=[3, 24, 36, 3],
    block_cls=Bottleneck,
    )