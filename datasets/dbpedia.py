# ref.: https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
from copy import deepcopy
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.random import default_rng
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import keras_cv
import jax
import jax.numpy as jnp
import flax
from gex.utils.tool import bcolors
import tensorflow_text as tf_text
import requests
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

import os

from absl import app, flags
FLAGS = flags.FLAGS

class TextDataset:
    
    def __init__(self):
        
        self.num_classes = 14
        self.num_train = 560_000
        self.num_test = 70_000
        url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_oss_model.model?raw=true"
        sp_model = requests.get(url).content
        self.tokenizer = tf_text.SentencepieceTokenizer(
            sp_model, 
            out_type=tf.int32
            )

        
    def load_dataset(
        self,
        batch_dims, 
        split='train',
        sub_split=None,
        shuffle=True,
        repeat=True,
        augment=True,
    ):
        # load dataset
        ds = tfds.load('huggingface:dbpedia_14/dbpedia_14', 
                       split=split,
                       shuffle_files=False,
                       )
        
        total_batch_size = np.prod(batch_dims)
        
        def add_idx(i, sample):
            sample['idx'] = i
            return sample
        ds = ds.enumerate().map(add_idx)
        
        # Apply label corruption (corruption_ratio > 0)
        print(f'{bcolors.CYAN}')
        rng = default_rng(FLAGS.seed)
        if (split=='train') and (FLAGS.corruption_ratio > 0):
            # get corrupted labels for the first time
            if not(hasattr(self, 'corrupted_label')):
                # select corrupted idx
                self.corrupted_idx = np.sort(rng.choice(self.num_train, int(self.num_train * FLAGS.corruption_ratio), replace=False))
                print(f'Corrupted idx (head) : {self.corrupted_idx[:5]}')
                
                # get corrupted labels
                self.clean_label = np.array(list(tfds.as_numpy(ds.map(lambda x:x['label']))))
                self.corrupted_label = deepcopy(self.clean_label)
                random_label = rng.choice(self.num_classes, len(self.corrupted_idx))
                self.corrupted_label[self.corrupted_idx] = random_label
                self.corrupted_idx = np.arange(self.num_train)[self.clean_label != self.corrupted_label]
                print(f'Clean labels : {[self.clean_label[self.corrupted_idx[i]] for i in range(5)]} -> Corrupted labels :{[self.corrupted_label[self.corrupted_idx[i]] for i in range(5)]}')
            
            # overwrite corrupted labels
            corrupted_label = tf.constant(self.corrupted_label)
            def corrupt(sample):
                sample['label'] = corrupted_label[sample['idx']]
                return sample
            ds = ds.map(corrupt)
        print(f'{bcolors.END}')
        
        # Apply sub-split (sub_split is not None)
        if sub_split is not None:
            ds = ds.shard(num_shards=2, index=sub_split)
        
        ds = ds.cache()
        
        # Shuffle and repeat
        if shuffle:
            ds = ds.shuffle(50_000)
        if repeat:
            ds = ds.repeat()
        
        # Decode and crop images
        def preprocess(sample):
            label = tf.one_hot(sample['label'], self.num_classes, 1., 0.)
            return {'x':sample['content'], 'y':label, 'idx':sample['idx']}
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Split into batches
        ds = ds.batch(total_batch_size, drop_remainder=False)
        
        # Reshape batch for multi-gpu training
        def batch_reshape(batch):
            for k,v in batch.items():
                batch[k] = tf.reshape(v, batch_dims+v.shape[1:])
                if k == 'x':
                    batch[k] = self.tokenizer.tokenize(batch[k]).to_tensor()
            return batch
        ds = ds.map(batch_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Prefetch dataset
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return self.prefetch(ds)
    
    def prefetch(self, dataset, n_prefetch=2):
        ds_iter = iter(dataset)
        ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x), ds_iter)
        if n_prefetch:
            ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
        return ds_iter
    
def test(_):
    dataset = TextDataset()
    ds = dataset.load_dataset([2, 2], split='train')
    for i in range(10):
        batch = next(ds)
        print(i, batch['x'].shape, batch['y'].shape, batch['idx'].shape)
        print(batch['x'])
        if i > 0:
            break
    
if '__main__' == __name__:
    app.run(test)