import tensorflow as tf
from tensorflow.python.keras import backend
import json
import logging
import shutil
import os
import random
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage.interpolation import rotate
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance
#import Augmentor



def focal_loss_softmax(labels, logits, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    # print(backend.int_shape(labels))

    y_pred = tf.nn.softmax(logits, axis=-1)
    # print(backend.int_shape(y_pred))

    eps = backend.epsilon()
    y_pred = backend.clip(y_pred, eps, 1. - eps)

    labels = tf.one_hot(tf.squeeze(labels), depth=y_pred.shape[-1])
    L = -labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L = tf.reduce_sum(L, axis=-1)
    return L, y_pred


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def delete_dir_content(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def hausdorf_distance(a, b):
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])


def shuffle(imgs, gts):
    np.random.seed(42)
    np.random.shuffle(imgs)
    np.random.seed(42)
    np.random.shuffle(gts)

    return imgs, gts

def batch_img_generator(imgs, gts, num_epochs=1, batch_size=1, is_preprocess=True):
    i = 0
    epoch = 1
    imgs, gts = shuffle(imgs, gts)
    
    while epoch <= num_epochs:
        if i >= imgs.shape[0]:

            epoch += 1
            i = 0
            imgs, gts = shuffle(imgs, gts)
        
        start = i
        end = imgs.shape[0] if i + batch_size >= imgs.shape[0] else i + batch_size
        
        batch_imgs = imgs[start:end]
        batch_gts = gts[start:end]

        yield batch_imgs, batch_gts, epoch
   
