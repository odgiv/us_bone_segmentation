'''
@author: Odgiiv Khuurkhunkhuu
@email: odgiiv_kh[gmail]
@create date: 2019-01-18
'''
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Lambda, Input, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, concatenate, ZeroPadding2D, BatchNormalization, Activation, add, multiply
from utils import focal_loss_softmax, mean_IU, mean_iou
import sys

from base_model import Unet


class AttentionalUnet(Unet):

    def resize_bilinear(self, g, x):
        return Lambda(lambda g, x: tf.image.resize_bilinear(g, size=(x.shape[1], x.shape[2]), align_corners=True), arguments={'x': x})(g)
    
    def AttentionBlock(self, x, g, num_filters, is_batchnorm=False):
        # x: 29x46x256, g: 14x23x256;
        x = Conv2D(num_filters, (2,2), strides=(2,2))(x) # 14x23x_num_filters
        g = Conv2D(num_filters, (1,1), use_bias=True)(g) # 14x23x_num_filters
        # up_g = tf.image.resize_bilinear(g, size=(x.shape[1], x.shape[2]), align_corners=True)
        up_g = self.resize_bilinear(g, x)
        add_xg = add([x, up_g])
        act_xg = Activation('relu')(add_xg)
        act_xg = Conv2D(1, (1,1), use_bias=True)(act_xg)
        act_xg = Activation('sigmoid')(act_xg) # 14x23x_num_filters
        # up_xg = tf.image.resize_bilinear(act_xg, size=(x.shape[1], x.shape[2]), align_corners=True)
        up_xg = self.resize_bilinear(act_xg, x)
        mul_x_up_xg = multiply([x, up_xg])

        result = Conv2D(x.shape[-1], (1,1))(mul_x_up_xg)
        result = BatchNormalization()(result)
        return result

    def GatingSignal(self, input, num_filters, is_batchnorm=False):
        x = Conv2D(num_filters, (1,1))(input)
        if is_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
        
    def build_model(self, inputs, num_classes):
        concat_axis = 3

        inputs = Input(tensor=inputs["images"]) # 233x369 as an example
        conv1 = self.UnetConv2D(inputs, 32) # 233x369x32 
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1) # 116x184x32

        conv2 = self.UnetConv2D(pool1, 64) # 116x184x64
        pool2 = MaxPooling2D(pool_size=(2,2))(conv2) # 58x92x64

        conv3 = self.UnetConv2D(pool2, 128) # 58x92x128
        pool3 = MaxPooling2D(pool_size=(2,2))(conv3) # 29x46x128

        conv4 = self.UnetConv2D(pool3, 256) # 29x46x256
        pool4 = MaxPooling2D(pool_size=(2,2))(conv4) # 14x23x256

        center = self.UnetConv2D(pool4, 512) # 14x23x512

        gating1 = self.GatingSignal(center, 256) # 14x23x256
        attn1 = self.AttentionBlock(conv4, gating1, 512) # 14x23x512
        up_conv5 = UpSampling2D(size=(2,2))(center) # 28x46x512
        ch, cw = self.get_crop_shape(up_conv5, attn1)
        attn1 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(attn1)
        up6 = concatenate([up_conv5, attn1], axis=concat_axis)
        conv6 = self.UnetConv2D(up6, 256) # 28x46x256

        gating2 = self.GatingSignal(conv6, 256) # 
        attn2 = self.AttentionBlock(conv3, gating2, 256) # 
        up_conv6 = UpSampling2D(size=(2,2))(conv6)
        ch, cw = self.get_crop_shape(up_conv6, attn2)
        attn2 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(attn2)
        up7 = concatenate([up_conv6, attn2], axis=concat_axis)
        conv7 = self.UnetConv2D(up7, 128)

        gating3 = self.GatingSignal(conv7, 128) # 
        attn3 = self.AttentionBlock(conv2, gating3, 128) # 
        up_conv7 = UpSampling2D(size=(2,2))(conv7)
        ch, cw = self.get_crop_shape(up_conv7, attn3)
        attn3 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(attn3)
        up8 = concatenate([up_conv7, attn3], axis=concat_axis)
        conv8 = self.UnetConv2D(up8, 64)

        gating4 = self.GatingSignal(conv8, 64) # 
        attn4 = self.AttentionBlock(conv1, gating4, 64) # 
        up_conv8 = UpSampling2D(size=(2,2))(conv8)
        ch, cw = self.get_crop_shape(up_conv8, attn4)
        attn4 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(attn4)
        up9 = concatenate([up_conv8, attn4], axis=concat_axis)
        conv9 = self.UnetConv2D(up9, 32)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D(num_classes, (1,1))(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model

