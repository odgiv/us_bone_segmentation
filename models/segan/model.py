'''
@author: Odgiiv Khuurkhunkhuu
@email: odgiiv_kh[gmail]
@created date: 2019-01-25
'''

import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Lambda, Input, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, Concatenate, ZeroPadding2D, BatchNormalization, Activation, add, multiply, LeakyReLU, Flatten
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras import Model
from tensorflow.python.keras.losses import mean_absolute_error as mea
import numpy as np


def conv_lrelu(nb_filters, kernel=(4, 4), stride=(2, 2)):
    conv2d = Conv2D(nb_filters, kernel, stride, padding="same")
    lrelu = LeakyReLU()
    return Sequential([conv2d, lrelu])


def conv_bn_lrelu(nb_filters, kernel=(4, 4), stride=(2, 2)):
    conv2d = Conv2D(nb_filters,  kernel, stride, padding="same")
    bnorm = BatchNormalization()
    lrelu = LeakyReLU()
    return Sequential([conv2d, bnorm, lrelu])


def up_conv(nb_filters, kernel=(3, 3), stride=(1, 1)):
    upsample2d = UpSampling2D(size=(2, 2))
    conv2d = Conv2D(nb_filters, kernel, stride, padding="same")
    return Sequential([upsample2d, conv2d])


def up_conv_bn_relu(nb_filters, kernel=(3, 3), stride=(1, 1)):
    upsample2d = UpSampling2D(size=(2, 2))
    conv2d = Conv2D(nb_filters, kernel, stride, padding="same")
    bnorm = BatchNormalization()
    relu = Activation('relu')
    return Sequential([upsample2d, conv2d, bnorm, relu])


class SegmentorNet(Model):

    def get_crop_shape(self, target, refer):
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

        return ((ch1, ch2), (cw1, cw2))

    def __init__(self):
        super(SegmentorNet, self).__init__()

        print("Creating SegAN model.")

        self.num_filters = [64, 128, 256, 512]

        self.conv_lrelu1 = conv_lrelu(self.num_filters[0])
        self.conv_bn_lrelu1 = conv_bn_lrelu(self.num_filters[1])
        self.conv_bn_lrelu2 = conv_bn_lrelu(self.num_filters[2])
        self.conv_bn_lrelu3 = conv_bn_lrelu(self.num_filters[3])
        self.up_conv_bn_relu1 = up_conv_bn_relu(self.num_filters[2])
        self.up_conv_bn_relu2 = up_conv_bn_relu(self.num_filters[1])
        self.up_conv_bn_relu3 = up_conv_bn_relu(self.num_filters[0])
        self.up_conv1 = up_conv(1)

    def call(self, inputs):
        # self.conv_lrelu(inputs, self.num_filters[0])
        seg_conv1 = self.conv_lrelu1(inputs)
        seg_conv2 = self.conv_bn_lrelu1(seg_conv1)
        seg_conv3 = self.conv_bn_lrelu2(seg_conv2)  # , self.num_filters[2])
        seg_center = self.conv_bn_lrelu3(seg_conv3)  # , self.num_filters[3])

        seg_up_con4 = self.up_conv_bn_relu1(seg_center)
        seg_up_con5 = self.up_conv_bn_relu2(seg_up_con4)
        seg_up_con6 = self.up_conv_bn_relu3(seg_up_con5)
        pred = self.up_conv1(seg_up_con6)
        # print(pred.shape)
        ch, cw = self.get_crop_shape(pred, inputs)
        pred = Cropping2D(cropping=((ch, cw)))(pred)
        
        # pred = self.cropping_2d(pred, input)
        # pred = Lambda(lambda target, refer: self.get_crop_shape(
        #     target, refer), arguments={'refer': input})(pred)

        # pred = Activation("sigmoid")(pred)
        return pred


class CriticNet(Model):
    def __init__(self):
        super(CriticNet, self).__init__()

        self.num_filters = [64, 128, 256]

        self.conv_lrelu1 = conv_lrelu(self.num_filters[0])
        self.conv_bn_lrelu1 = conv_bn_lrelu(self.num_filters[1])
        self.conv_bn_lrelu2 = conv_bn_lrelu(self.num_filters[2])

    def call(self, inputs):
        cri_conv1 = self.conv_lrelu1(inputs)
        cri_conv2 = self.conv_bn_lrelu1(cri_conv1)
        cri_conv3 = self.conv_bn_lrelu2(cri_conv2)
        features = Concatenate()([Flatten()(inputs), Flatten()(
            cri_conv1), Flatten()(cri_conv2), Flatten()(cri_conv3)])
        return features


class SegAN(Model):

    def __init__(self):
        super(SegAN, self).__init__()

    def model_fn(self, mode, inputs={}):

        is_training = (mode == "train")

        segNet = SegmentorNet()
        criNet = CriticNet()

        # if not is_training:
        #     segNet.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        #     segNet.fit(x=np.zeros((1,1,1,1)), y=np.zeros((1,1,1,1)), epochs=0, steps_per_epoch=0)
        #     segNet.load_weights(params.save_weights_path + 'segan_best_weights.h5')

        model_spec = inputs
        model_spec["segmentor_net"] = segNet
        model_spec["critic_net"] = criNet

        return model_spec
