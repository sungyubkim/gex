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

from absl import flags
flags.DEFINE_float('corruption_ratio', 0.0, help='ratio of corrupted samples')
flags.DEFINE_float('class_balance_ratio', 0.0,  help='LB of imbalance ratio')
flags.DEFINE_float('randaug_magnitude', 0.0, help='magnitude for Randaug')
flags.DEFINE_float('mixup_alpha', 0.0, help='alpha for mixup')
flags.DEFINE_float('cutmix_alpha', 0.0, help='alpha for cutmix')
FLAGS = flags.FLAGS

class ImageDataset:

    def __init__(self):
        
        if FLAGS.dataset == 'mnist':
            self.img_shape = (28, 28, 1)
            self.crop_padding = 4
            self.num_classes = 10
            self.num_train = 60000
            self.num_test = 10000
        elif FLAGS.dataset == 'cifar10':
            self.img_shape = (32, 32, 3)
            self.crop_padding = 4
            self.num_classes = 10
            self.num_train = 50000
            self.num_test = 10000
        elif FLAGS.dataset == 'cifar100':
            self.img_shape = (32, 32, 3)
            self.crop_padding = 4
            self.num_classes = 100
            self.num_train = 50000
            self.num_test = 10000
        elif FLAGS.dataset == 'svhn_cropped':
            self.img_shape = (32, 32, 3)
            self.crop_padding = 4
            self.num_classes = 10
            self.num_train = 73257
            self.num_test = 26032
        elif FLAGS.dataset == 'hetero':
            self.img_shape = (32, 32, 3)
            self.crop_padding = 4
            self.num_classes = 10
            self.num_train = 50000
            self.num_test = 10000
        elif FLAGS.dataset == 'cifar10_n':
            self.img_shape = (32, 32, 3)
            self.crop_padding = 4
            self.num_classes = 10
            self.num_train = 50000
            self.num_test = 10000
        elif FLAGS.dataset == 'cifar100_n':
            self.img_shape = (32, 32, 3)
            self.crop_padding = 4
            self.num_classes = 100
            self.num_train = 50000
            self.num_test = 10000
        elif FLAGS.dataset == 'imagenet2012':
            self.img_shape = (224, 224, 3)
            self.crop_padding = 32
            self.num_classes = 1000
            self.num_train = 1281167
            self.num_test = 50000
        
    def load_dataset(
        self, 
        batch_dims, 
        split='train',
        sub_split=None,
        shuffle=True,
        augment=True, 
        repeat=True,
        num_filter=0,
        summarization_score=None,
        new_label=None,
        ):
        '''
        Load and preprocess dataset.
        '''
        
        # Load dataset
        if FLAGS.dataset == 'imagenet2012':
            ds = tfds.load(
                    "{}:5.*.*".format(FLAGS.dataset),
                    data_dir='./tensorflow_datasets',
                    split=split,
                    shuffle_files=False, # do not shuffle files before reading
                    decoders={'image': tfds.decode.SkipDecoding()},
            )
        elif FLAGS.dataset == 'hetero':
            if split == 'train':
                ds_mnist = tfds.load(
                        "MNIST:3.*.*",
                        data_dir='./tensorflow_datasets',
                        split='train[:45000]',
                        # split='train[:25000]',
                )
                def gray_to_rgb(batch):
                    img, label = batch['image'], batch['label']
                    img = tf.image.grayscale_to_rgb(img)
                    img = tf.image.resize(img, (32, 32))
                    return {'image':img, 'label':label}
                ds_mnist = ds_mnist.map(gray_to_rgb)
                ds_svhn = tfds.load(
                    "svhn_cropped:3.0.0",
                    data_dir='./tensorflow_datasets',
                    split='train[:5000]'
                    # split='train[:25000]',
                )
                def resize(batch):
                    img, label = batch['image'], batch['label']
                    img = tf.image.resize(img, (32, 32))
                    return {'image':img, 'label':label}
                ds_svhn = ds_svhn.map(resize)
                ds = ds_mnist.concatenate(ds_svhn)
            else:
                ds = tfds.load(
                        "svhn_cropped:3.0.0",
                        data_dir='./tensorflow_datasets',
                        split='test[:10000]', 
                )
        else:
            ds = tfds.load(
                    "{}:*.*.*".format(FLAGS.dataset),
                    data_dir='./tensorflow_datasets',
                    split=split,
                    shuffle_files=False,
            )
            # ds_numpy = tfds.as_numpy(ds)
            # for ex in ds_numpy:
            #     print(ex['image'].shape, ex['label'].shape);input()
        total_batch_size = np.prod(batch_dims)
        
        # Register sample id for logging
        if (split=='train') and (FLAGS.dataset == 'cifar10_n'):
            df = pd.read_csv('./tensorflow_datasets/downloads/manual/CIFAR-10_human_annotations.csv')
            self.clean_label = df['clean_label'].values
            self.corrupted_label = df['worse_label'].values
            self.corrupted_idx = np.arange(self.num_train)[self.clean_label != self.corrupted_label]
            print(f'Clean labels : {[self.clean_label[self.corrupted_idx[i]] for i in range(5)]} -> Corrupted labels :{[self.corrupted_label[self.corrupted_idx[i]] for i in range(5)]}')
            def add_idx(i, sample):
                return {'images':sample['image'], 'labels':sample['worse_label'], 'idx':i}
        elif (split=='train') and (FLAGS.dataset == 'cifar100_n'):
            df = pd.read_csv('./tensorflow_datasets/downloads/manual/CIFAR-100_human_annotations.csv')
            self.clean_label = df['clean_label'].values
            self.corrupted_label = df['noise_label'].values
            self.corrupted_idx = np.arange(self.num_train)[self.clean_label != self.corrupted_label]
            print(f'Clean labels : {[self.clean_label[self.corrupted_idx[i]] for i in range(5)]} -> Corrupted labels :{[self.corrupted_label[self.corrupted_idx[i]] for i in range(5)]}')
            def add_idx(i, sample):
                return {'images':sample['image'], 'labels':sample['noise_label'], 'idx':i}
        else:
            def add_idx(i, sample):
                return {'images':sample['image'], 'labels':sample['label'], 'idx':i}
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
                self.clean_label = np.array(list(tfds.as_numpy(ds.map(lambda x:x['labels']))))
                self.corrupted_label = deepcopy(self.clean_label)
                random_label = rng.choice(self.num_classes, len(self.corrupted_idx))
                self.corrupted_label[self.corrupted_idx] = random_label
                self.corrupted_idx = np.arange(self.num_train)[self.clean_label != self.corrupted_label]
                print(f'Clean labels : {[self.clean_label[self.corrupted_idx[i]] for i in range(5)]} -> Corrupted labels :{[self.corrupted_label[self.corrupted_idx[i]] for i in range(5)]}')
            
            # overwrite corrupted labels
            corrupted_label = tf.constant(self.corrupted_label)
            def corrupt(sample):
                sample['labels'] = corrupted_label[sample['idx']]
                return sample
            ds = ds.map(corrupt)
            
        # Apply dataset pruning (num_filter > 0)
        if (split=='train') and (num_filter > 0) and (summarization_score is not None):
            # get left samples for the first time
            if not(hasattr(self, 'left')):
                if FLAGS.if_method=='fscore':
                    summarization_score = summarization_score.astype(np.float64)
                    summarization_score += rng.uniform(0.0, 0.5, summarization_score.shape)
                
                class_idx = np.array(list(tfds.as_numpy(ds.map(lambda sample:sample['labels']))))
                pruned_sample_per_class = int(FLAGS.class_balance_ratio * (num_filter / self.num_classes))
                class_filter_idx = np.full((self.num_train,), False)
                print(f'Class-wise pruning {pruned_sample_per_class * self.num_classes} samples')
                for i in range(self.num_classes):
                    class_thershold = np.sort(summarization_score[class_idx==i])[pruned_sample_per_class]
                    class_filter_idx_i = (class_idx==i) & (summarization_score < class_thershold)
                    class_filter_idx = class_filter_idx | (class_filter_idx_i)
                print(f'Class-wise pruned samples : {np.sum(class_filter_idx)}')
                
                pruned_sample_global = int((1-FLAGS.class_balance_ratio) * num_filter)
                print(f'Global pruning {pruned_sample_global} samples')
                global_threshold = np.sort(summarization_score[~class_filter_idx])[pruned_sample_global]
                global_filter_idx = (~class_filter_idx) & (summarization_score < global_threshold)
                print(f'Globally pruned samples : {np.sum(global_filter_idx)}')
                
                self.left = (~(class_filter_idx | global_filter_idx)).astype(int)
                print(f'Left samples : {np.sum(self.left)}')
                self.left = tf.constant(self.left)
            
            def mark(sample):
                sample['mark'] = self.left[sample['idx']]
                return sample
            ds = ds.map(mark).filter(lambda sample: sample['mark'] == 1)
            self.num_train_filtered = self.num_train - num_filter
            num_train_ = ds.reduce(0, lambda x,_: x+1).numpy()
            
            assert self.num_train_filtered==num_train_
        print(f'{bcolors.END}')
        
        # Apply sub-split (sub_split is not None)
        if sub_split is not None:
            ds = ds.shard(num_shards=2, index=sub_split)
        
        # Cache dataset (except for imagenet2012)
        if FLAGS.dataset != 'imagenet2012':
            ds = ds.cache()
            
        # Shuffle and repeat
        if shuffle:
            ds = ds.shuffle(min(self.num_train, 256_000))
        if repeat:
            ds = ds.repeat()
        
        # Decode and crop images
        def preprocess(example):
            image = self._preprocess_image(example['images'], augment)
            label = tf.one_hot(example['labels'], self.num_classes, 1., 0.)
            return {'x':image, 'y':label, 'idx':example['idx']}
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            
        # Apply relabeling (new_label is not None)
        if new_label is not None:
            new_label = tf.constant(new_label)
            def change_label(example):
                example['y'] = new_label[example['idx']]
                return example
            ds = ds.map(change_label, num_parallel_calls=tf.data.AUTOTUNE)    
        
        # Split into batches
        ds = ds.batch(total_batch_size, drop_remainder=False)
        
        # Apply augmentation
        def image_aug(batch):
            img, label, idx = batch['x'], batch['y'], batch['idx']
            # apply random augmentation
            if FLAGS.randaug_magnitude > 0:
                rand_augment = keras_cv.layers.RandAugment(
                    value_range=(-4., 4.),
                    magnitude=FLAGS.randaug_magnitude,
                )
                img = rand_augment(img)
            # apply cutmix and mixup
            if FLAGS.cutmix_alpha > 0:
                cut_mix = keras_cv.layers.CutMix(FLAGS.cutmix_alpha)
                outputs = cut_mix({'images':img, 'labels':label})
                img, label = outputs['images'], outputs['labels']
            if FLAGS.mixup_alpha > 0:
                mix_up = keras_cv.layers.MixUp(FLAGS.mixup_alpha)
                outputs = mix_up({'images':img, 'labels':label})
                img, label = outputs['images'], outputs['labels']
            outputs = {'x':img, 'y':label, 'idx':idx}
            return outputs
        
        if augment:
            ds = ds.map(image_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
        # Reshape batch for multi-gpu training
        def batch_reshape(batch):
            for k,v in batch.items():
                batch[k] = tf.reshape(v, batch_dims+v.shape[1:])
            return batch
        ds = ds.map(batch_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Prefetch dataset
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return self.prefetch(ds, 2)
    
    def _preprocess_image(self, image_bytes, augment):
        '''
        Preprocess image bytes to image tensor. 
        Dependency: _decode_and_random_crop and _decode_and_center_crop
        '''
        # Decode image and crop image (random crop or center crop).
        if FLAGS.dataset == 'imagenet2012':
            if augment:
                image = self._decode_and_random_crop(image_bytes)
                image = tf.image.random_flip_left_right(image)
            else:
                image = self._decode_and_center_crop(image_bytes)
            assert image.dtype == tf.uint8
        else:
            image = image_bytes
            if augment:
                padding = [[4,4],[4,4],[0,0]]
                image = tf.pad(image, padding, mode='REFLECT')
                image = tf.image.random_crop(image, self.img_shape)
                if not(FLAGS.dataset in ['mnist', 'svhn_cropped', 'hetero']):
                    image = tf.image.random_flip_left_right(image)
        # Resize image to self.img_shape.
        image = tf.image.resize(image, [self.img_shape[0], self.img_shape[1]], tf.image.ResizeMethod.BICUBIC)
        # Normalize image.
        image = image / 255.0
        image = (image - 0.5) / 0.25
        return image
    
    def _decode_and_random_crop(self, image_bytes):
        """Make a random crop of 224."""
        jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        image = self._dirtorted_bounding_box_crop(
                image_bytes,
                jpeg_shape=jpeg_shape,
                bbox=bbox,
                min_object_covered=0,
                aspect_ratio_range=(3 / 4, 4 / 3),
                area_range=(0.05, 1.0),
                max_attempts=10)
        if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
            # If the random crop failed fall back to center crop.
            image = self._decode_and_center_crop(image_bytes, jpeg_shape)
        return image
    
    def _dirtorted_bounding_box_crop(
        self,
        image_bytes: tf.Tensor,
        *,
        jpeg_shape: tf.Tensor,
        bbox: tf.Tensor,
        min_object_covered: float,
        aspect_ratio_range: Tuple[float, float],
        area_range: Tuple[float, float],
        max_attempts: int,
        ) -> tf.Tensor:
        """Generates cropped_image using one of the bboxes randomly dirtorted."""
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
                jpeg_shape,
                bounding_boxes=bbox,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
        return image

    def _decode_and_center_crop(
        self,
        image_bytes: tf.Tensor,
        jpeg_shape: Optional[tf.Tensor] = None,
        ) -> tf.Tensor:
        """Crops to center of image with padding then scales."""
        if jpeg_shape is None:
            jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
        image_height = jpeg_shape[0]
        image_width = jpeg_shape[1]

        padded_center_crop_size = tf.cast(
            ((224 / (224 + 32)) * tf.cast(tf.minimum(image_height, image_width), tf.float32)),
            tf.int32)

        offset_height = ((image_height - padded_center_crop_size) + 1) // 2
        offset_width = ((image_width - padded_center_crop_size) + 1) // 2
        crop_window = tf.stack([offset_height, offset_width,padded_center_crop_size, padded_center_crop_size])
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
        return image
    
    def prefetch(self, dataset, n_prefetch):
        ds_iter = iter(dataset)
        ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x), ds_iter)
        if n_prefetch:
            ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
        return ds_iter