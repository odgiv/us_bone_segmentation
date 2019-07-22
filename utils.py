import tensorflow as tf
from tensorflow.python.keras import backend
import json
import logging
import shutil
import os
import random
import numpy as np
import cv2 as cv
import pprint
from PIL import Image, ImageEnhance
import math
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import rotate
from albumentations.augmentations.transforms import ElasticTransform

def get_crop_shape(target, refer):
    """
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D
    https://stackoverflow.com/questions/41925765/keras-cropping2d-changes-color-channel
    """
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

    

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


def img_and_mask_generator(x, y, batch_size=1, shuffle=True):
    data_gen_args = dict(
        rescale=1./255
    )
        
    image_data_generator = ImageDataGenerator(**data_gen_args)
    mask_data_generator = ImageDataGenerator(**data_gen_args)

    seed = 1
    if isinstance(x, np.ndarray):
        image_gen = image_data_generator.flow(x, batch_size=batch_size, seed=seed, shuffle=shuffle)
        mask_gen = mask_data_generator.flow(y, batch_size=batch_size, seed=seed, shuffle=shuffle)
    else:
        image_gen = image_data_generator.flow_from_directory(x, batch_size=batch_size, seed=seed, shuffle=shuffle, class_mode=None, color_mode="grayscale", target_size=(465, 381))
        mask_gen = mask_data_generator.flow_from_directory(y, batch_size=batch_size, seed=seed, shuffle=shuffle, class_mode=None, color_mode="grayscale", target_size=(465, 381))

    return zip(image_gen, mask_gen)

   
def augmented_img_and_mask_generator(x, y, batch_size):
    rs = np.random.RandomState()

    # elastic = ElasticTransform(p=1)

    def preprocessing(img):
        return img #elastic(img)


    data_gen_args = dict(
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=10,
        width_shift_range=0.2, 
        height_shift_range=0.1,
        shear_range=0.2,
        preprocessing_function = preprocessing,
        rescale=1./255,
        fill_mode="constant",
        cval=0
    )

    img_gen_args = dict(data_gen_args)
    #img_gen_args["brightness_range"]=(0.5, 1.5)
    # img_gen_args["rescale"]=1./255
        
    mask_gen_args = dict(data_gen_args)

    print("Data generation arguments:")
    pprint(data_gen_args)
        
    image_data_generator = ImageDataGenerator(**img_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_gen_args)

    seed = 1
    print(x, y)
    image_gen = image_data_generator.flow_from_directory(x, batch_size=batch_size, seed=seed, class_mode=None, color_mode="grayscale", target_size=(465, 381))
    mask_gen = mask_data_generator.flow_from_directory(y, batch_size=batch_size, seed=seed, class_mode=None, color_mode="grayscale", target_size=(465, 381))

    return zip(image_gen, mask_gen)
