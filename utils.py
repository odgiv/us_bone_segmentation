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
import Augmentor



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


def batch_img_generator(imgs, gts, batch_size=1, is_preprocess=True):
    i = 0
    while True:
        if i == 0:
            np.random.seed(42)
            np.random.shuffle(imgs)
            np.random.seed(42)
            np.random.shuffle(gts)
        batch_imgs = imgs[i:batch_size]
        batch_gts = gts[i:batch_size]

        i += batch_size

        if i >= imgs.shape[0]:
            i = 0

        yield(batch_imgs, batch_gts)

def rotate_img(image, angle):
    # # image: np.ndarray
    # # angle: float
    # #
    # # grab the dim of the image and then determine the center
    # (h, w) = image.shape[:2]
    # (cX, cY) = (w//2, h//2)
    # # grab the rotation matrix
    # M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    # # cos = np.abs(M[0, 0])
    # # sin = np.abs(M[0, 1])

    # # # compute the new bounding dim of the image
    # # nW = int((h*sin) + (w*cos))
    # # nH = int((h*cos) + (w*sin))

    # # M[0, 2] += (nW/2)-cX
    # # M[1, 2] += (nH/2)-cY

    # image = cv.warpAffine(image, M, (w, h)) # nW, nH
    image = rotate(image, angle)
    return image

def vert_flip(img):
    return cv.flip(img, 1)

def random_sharpness(pil_image, range_of_factors=(0.0, 2.0)):
    sharpness_factor = random.uniform(*range_of_factors)
    sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = sharpness_enhancer.enhance(sharpness_factor)
    return pil_image


def random_color(pil_image, range_of_factors=(0.0, 1.0)):
    color_factor = random.uniform(*range_of_factors)
    color_enhancer = ImageEnhance.Color(pil_image)
    pil_image = color_enhancer.enhance(color_factor)
    return pil_image


def random_contrast(pil_image, range_of_factors=(0.5, 1.5)):
    contrast_factor = random.uniform(*range_of_factors)
    contrast_enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = contrast_enhancer.enhance(contrast_factor)
    return pil_image


def random_brightness(pil_image, range_of_factors=(0.5, 1.5)):
    brightness_factor = random.uniform(*range_of_factors)
    brightness_enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = brightness_enhancer.enhance(brightness_factor)
    return pil_image


def chain_random_image_enhancements(image):
    # image: np array
    pil_image = Image.fromarray(image)
    pil_image = random_sharpness(pil_image)
    pil_image = random_color(pil_image)
    pil_image = random_contrast(pil_image)
    pil_image = random_brightness(pil_image)
    return np.array(pil_image)


def preprocessData(img, gt, rotation_angle_range=(-10, 10)):
    b = random.choice([True, False])
    if b:
        img = vert_flip(img)
        gt = vert_flip(gt)
    rotation_angle = np.random.choice(list(range(*rotation_angle_range)), 1)
    img = rotate_img(img, rotation_angle[0])
    gt = rotate_img(gt, rotation_angle[0])

    return img, gt
    # p = Augmentor.Pipeline()
    # p.rotate(0.5, max_left_rotation=10, max_right_rotation=10)
    # p.flip_left_right(0.5)
    # p.zoom_random(0.5, percentage_area=0.9)
    # return p.keras_generator_from_array(images, labels, scaled=False, batch_size=batch_size)    