import tensorflow as tf
from data_loader import DataLoader
from misc.preprocessGt import 
from utils import preprocessData
from tf.contrib.image import rotate
import numpy as np
import math
import random

def preprocess(image, label):
    seed = random.randint(1, 101)
    random_rot_angle1 = tf.random.uniform(shape=image.shape[0], minval=0, maxval=16, dtype=tf.dtypes.int16)
    random_rot_angle2 = tf.random.uniform(shape=image.shape[0], minval=345, maxval=360, dtype=tf.dtypes.int16)
    if seed > 50:
        random_rot_angle = random_rot_angle
    else:
        random_rot_angle = random_rot_angle2

    random_rot_angle = random_rot_angle * math.pi / 180
    image = rotate(image, random_rot_angle)
    label = rotate(label, random_rot_angle)
    
    if seed > 50:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)

    return image, label

def _parse_function(image, label):
    image, label = preprocess(image, label)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, tf.cast(label, tf.float32)


def input_fn(is_training, is_eager, X, Y, params):
    
    num_samples = X.shape[0]

    def parse_fn(f, l): return _parse_function(f, l)

    if is_training:
        if is_eager:
            dataset = (tf.data.Dataset.from_tensor_slices((X, Y))
                    .shuffle(num_samples)
                    .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                    .batch(params.batch_size)
                    .prefetch(1)
                    )
        else:
            features_placeholder = tf.placeholder(X.dtype, X.shape)
            labels_placeholder = tf.placeholder(Y.dtype, Y.shape)
            dataset = (tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
                    .shuffle(num_samples)
                    .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                    .batch(params.batch_size)
                    .prefetch(1)
            )
            iterator = dataset.make_initializable_iterator()
            images, labels = iterator.get_next()
            iterator_init_op = iterator.initializer

            return {"images": images,
                    "labels": labels, 
                    "X_placeholder": features_placeholder, 
                    "Y_placeholder": labels_placeholder,
                    "X": X, 
                    "Y": Y,
                    "iterator_init_op": iterator_init_op
                    }
    else:
        dataset = (X, Y)
        

    return {"dataset": dataset}
