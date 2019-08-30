"""
Attention U-Net model
"""
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Lambda, Input, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, concatenate, ZeroPadding2D, BatchNormalization, Activation, Add, Multiply
from tensorflow.python.keras.regularizers import l2
import sys
from utils import get_crop_shape, unet_conv2d


def gating_signal(num_filters, is_batchnorm=True):
    conv2d = Conv2D(num_filters, (1,1))
    batch_norm = BatchNormalization()
    relu = Activation('relu')
    if is_batchnorm:
        return Sequential([conv2d, batch_norm, relu])
    else:
        return Sequential([conv2d, relu])


class SubAttentionBlock(Model):

    def __init__(self, num_filters):
        super(SubAttentionBlock, self).__init__()

        # self.gating = gating_signal(256)
        self.att_conv1 = Conv2D(num_filters, (2,2), strides=(2,2))
        self.att_conv2 = Conv2D(num_filters, (1,1), use_bias=True)

        self.att_add = Add()

        self.att_relu = Activation('relu')
        self.att_conv3 = Conv2D(1, (1,1), use_bias=True)
        self.att_sigm = Activation('sigmoid')
        self.att_multiply = Multiply()

        self.att_conv4 = Conv2D(int(num_filters/2), (1,1))
        self.att_batch_norm = BatchNormalization()


    def call(self, inputs):
        down_pool = inputs[0]
        # down_conv = inputs[1]
        gating = inputs[1]

        # gating = self.gating(down_conv)
        att_conv1 = self.att_conv1(down_pool)
        att_conv2 = self.att_conv2(gating)

        att_resized_gating = tf.image.resize_bilinear(att_conv2, size=(int(att_conv1.get_shape()[1]), int(att_conv1.get_shape()[2])), align_corners=True)

        att_add = self.att_add([att_conv1, att_resized_gating])
        att_relu = self.att_relu(att_add)
        att_conv3 = self.att_conv3(att_relu)
        att_sigm = self.att_sigm(att_conv3)

        att_resized_sigm = tf.image.resize_bilinear(att_sigm, size=(int(down_pool.get_shape()[1]), int(down_pool.get_shape()[2])), align_corners=True)

        att_mul = self.att_multiply([down_pool, att_resized_sigm])

        att_conv4 = self.att_conv4(att_mul)

        att_batch_norm = self.att_batch_norm(att_conv4)

        return att_batch_norm


class AttentionalUnet(Model):

    def __init__(self, num_classes=2, l2_value=0.0):
        super(AttentionalUnet, self).__init__()

        print("Creating AttentionalUnet model.")

        self.conv1 = unet_conv2d(64, kernel_regularizer=l2(l2_value))
        self.pool1 = MaxPooling2D(pool_size=(2,2))
        
        self.conv2 = unet_conv2d(128, kernel_regularizer=l2(l2_value)) 
        self.pool2 = MaxPooling2D(pool_size=(2,2))

        self.conv3 = unet_conv2d(256, kernel_regularizer=l2(l2_value))
        self.pool3 = MaxPooling2D(pool_size=(2,2))

        self.conv4 = unet_conv2d(512, kernel_regularizer=l2(l2_value))
        self.pool4 = MaxPooling2D(pool_size=(2,2))

        self.center = unet_conv2d(1024, kernel_regularizer=l2(l2_value))

        self.gating = gating_signal(128)

        # AttentionBlock1 layers

        self.att1 = SubAttentionBlock(256)

        # AttentionBlock end

        self.up_conv5 = UpSampling2D(size=(2,2))
        self.up_conv6 = unet_conv2d(512, kernel_regularizer=l2(l2_value))

        # AttentionBlock2 layers
        
        self.att2 = SubAttentionBlock(128)
        # AttentionBlock end

        self.up_conv7 = UpSampling2D(size=(2,2))
        self.up_conv8 = unet_conv2d(256, kernel_regularizer=l2(l2_value))

        # AttentionBlock3 layers

        self.att3 = SubAttentionBlock(64)
        # AttentionBlock end

        self.up_conv9 = UpSampling2D(size=(2,2))
        self.up_conv10 = unet_conv2d(128, kernel_regularizer=l2(l2_value))

        # AttentionBlock4 layers
        
        self.att4 = SubAttentionBlock(32)
        # AttentionBlock end

        self.up_conv11 = UpSampling2D(size=(2,2))
        self.up_conv12 = unet_conv2d(64, kernel_regularizer=l2(l2_value))

        self.up_conv13 = Conv2D(num_classes, (1,1))


    def call(self, inputs):
        concat_axis = 3

        down_conv1 = self.conv1(inputs)
        down_pool1 = self.pool1(down_conv1)

        down_conv2 = self.conv2(down_pool1)
        down_pool2 = self.pool2(down_conv2)

        down_conv3 = self.conv3(down_pool2)
        down_pool3 = self.pool3(down_conv3)

        down_conv4 = self.conv4(down_pool3)
        down_pool4 = self.pool4(down_conv4)
   
        center = self.center(down_pool4)        

        # attention1 part

        gating = self.gating(center)

        # att1_batch_norm = self.att1([down_conv4, center])
        att1_batch_norm = self.att1([down_conv4, gating])

        att1_up_conv5 = self.up_conv5(center)

        ch, cw = get_crop_shape(att1_batch_norm, att1_up_conv5)

        attn1_zero_pad = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(att1_up_conv5)
        up_concat1 = concatenate([att1_batch_norm, attn1_zero_pad], axis=concat_axis) 
        up_conv6 = self.up_conv6(up_concat1)

        # attention2 part

        # att2_batch_norm = self.att2([down_conv3, up_conv6])
        att2_batch_norm = self.att2([down_conv3, gating])

        att2_up_conv5 = self.up_conv7(up_conv6)

        ch, cw = get_crop_shape(att2_batch_norm, att2_up_conv5)

        attn2_zero_pad = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(att2_up_conv5)
        up_concat2 = concatenate([att2_batch_norm, attn2_zero_pad], axis=concat_axis) 
        up_conv7 = self.up_conv8(up_concat2)

        # attention3 part

        # att3_batch_norm = self.att3([down_conv2, up_conv7])
        att3_batch_norm = self.att3([down_conv2, gating])

        att3_up_conv5 = self.up_conv9(up_conv7)

        ch, cw = get_crop_shape(att3_batch_norm, att3_up_conv5)

        attn3_zero_pad = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(att3_up_conv5)
        up_concat3 = concatenate([att3_batch_norm, attn3_zero_pad], axis=concat_axis) 
        up_conv8 = self.up_conv10(up_concat3)

        # attention4 part

        # att4_batch_norm = self.att4([down_conv1, up_conv8])
        att4_batch_norm = self.att4([down_conv1, gating])

        att4_up_conv5 = self.up_conv11(up_conv8)

        ch, cw = get_crop_shape(att4_batch_norm, att4_up_conv5)

        attn4_zero_pad = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(att4_up_conv5)
        up_concat4 = concatenate([att4_batch_norm, attn4_zero_pad], axis=concat_axis) 
        up_conv9 = self.up_conv12(up_concat4)

        # Last part
        ch, cw = get_crop_shape(inputs, up_conv9)
        up_conv10 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(up_conv9)
        
        output = self.up_conv13(up_conv10)

        return output
