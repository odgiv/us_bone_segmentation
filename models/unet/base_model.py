'''
@author: Odgiiv Khuurkhunkhuu
@email: odgiiv_kh[at]gmail
@create date: 2019-01-10
'''

import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, concatenate, ZeroPadding2D
from utils import focal_loss_softmax


def unet_conv2d(nb_filters, kernel=(3, 3), activation="relu", padding="same"):
    conv2d_1 = Conv2D(nb_filters, kernel, activation=activation, padding=padding)
    conv2d_2 = Conv2D(nb_filters, kernel, activation=activation, padding=padding)
    return Sequential([conv2d_1, conv2d_2])

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


class Unet(Model):

    def __init__(self, num_classes=2):
        super(Unet, self).__init__()

        print("Creating Unet model.")

        self.conv1 = unet_conv2d(32)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2 = unet_conv2d(64)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3 = unet_conv2d(128)
        self.pool3 = MaxPooling2D(pool_size=(2, 2))

        self.conv4 = unet_conv2d(256)
        self.pool4 = MaxPooling2D(pool_size=(2, 2))

        self.center = unet_conv2d(512)

        self.up_conv5 = UpSampling2D(size=(2, 2))
        self.conv6 = unet_conv2d(256)

        self.up_conv6 = UpSampling2D(size=(2, 2))
        self.conv7 = unet_conv2d(128)

        self.up_conv7 = UpSampling2D(size=(2, 2))
        self.conv8 = unet_conv2d(64)

        self.up_conv8 = UpSampling2D(size=(2, 2))
        self.conv9 = unet_conv2d(32)

        self.conv10 = Conv2D(num_classes, (1, 1))


    def call(self, inputs):
        
        concat_axis = 3
        seg_conv1 = self.conv1(inputs)
        seg_pool1 = self.pool1(seg_conv1)

        seg_conv2 = self.conv2(seg_pool1)
        seg_pool2 = self.pool2(seg_conv2)

        seg_conv3 = self.conv3(seg_pool2)
        seg_pool3 = self.pool3(seg_conv3)

        seg_conv4 = self.conv4(seg_pool3)
        seg_pool4 = self.pool4(seg_conv4)

        seg_center = self.center(seg_pool4)

        seg_up_conv5 = self.up_conv5(seg_center)
        ch, cw = get_crop_shape(seg_conv4, seg_up_conv5)
        seg_crop_conv4 = Cropping2D(cropping=(ch, cw))(seg_conv4)
        seg_up6 = concatenate([seg_up_conv5, seg_crop_conv4], axis=concat_axis)
        seg_conv6 = self.conv6(seg_up6)

        seg_up_conv6 = self.up_conv6(seg_conv6)
        ch, cw = get_crop_shape(seg_conv3, seg_up_conv6)
        seg_crop_conv3 = Cropping2D(cropping=(ch, cw))(seg_conv3)
        seg_up7 = concatenate([seg_up_conv6, seg_crop_conv3], axis=concat_axis)
        seg_conv7 = self.conv7(seg_up7)

        seg_up_conv7 = self.up_conv7(seg_conv7)
        ch, cw = get_crop_shape(seg_conv2, seg_up_conv7)
        seg_crop_conv2 = Cropping2D(cropping=(ch, cw))(seg_conv2)
        seg_up8 = concatenate([seg_up_conv7, seg_crop_conv2], axis=concat_axis)
        seg_conv8 = self.conv8(seg_up8)

        seg_up_conv8 = self.up_conv8(seg_conv8)
        ch, cw = get_crop_shape(seg_conv1, seg_up_conv8)
        seg_crop_conv1 = Cropping2D(cropping=(ch, cw))(seg_conv1)
        seg_up9 = concatenate([seg_up_conv8, seg_crop_conv1], axis=concat_axis)
        seg_conv9 = self.conv9(seg_up9)

        ch, cw = get_crop_shape(inputs, seg_conv9)
        seg_conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(seg_conv9)
        seg_conv10 = self.conv10(seg_conv9)
        
        return seg_conv10


    def model_fn(self, mode, inputs, params, reuse=False):
        is_training = (mode == 'train')

        uNet = Unet()

        model_spec = inputs
        model_spec['unet'] = uNet
        
        return model_spec
