'''
@author: Odgiiv Khuurkhunkhuu
@email: odgiiv_kh[gmail]
@create date: 2019-01-18
'''
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Lambda, Input, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, concatenate, ZeroPadding2D, BatchNormalization, Activation, Add, Multiply
import sys
from base_model import unet_conv2d

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


def gating_signal(num_filters, is_batchnorm=False):
    conv2d = Conv2D(num_filters, (1,1))
    batch_norm = BatchNormalization()
    relu = Activation('relu')
    if is_batchnorm:
        return Sequential([conv2d, batch_norm, relu])
    else:
        return Sequential([conv2d, relu])

class AttentionalUnet(Model):

    def __init__(self, num_classes=2):
        super(AttentionalUnet, self).__init__()

        print("Creating AttentionalUnet model.")

        self.conv1 = unet_conv2d(32)
        self.pool1 = MaxPooling2D(pool_size=(2,2))
        
        self.conv2 = unet_conv2d(64)
        self.pool2 = MaxPooling2D(pool_size=(2,2))

        self.conv3 = unet_conv2d(128)
        self.pool3 = MaxPooling2D(pool_size=(2,2))

        self.conv4 = unet_conv2d(256)
        self.pool4 = MaxPooling2D(pool_size=(2,2))

        self.center = unet_conv2d(512)

        # AttentionBlock1 layers
        self.gating1 = gating_signal(256)
        self.att1_conv1 = Conv2D(512, (2,2), strides=(2,2))
        self.att1_conv2 = Conv2D(512, (1,1), use_bias=True)

        self.att1_add = Add()

        self.att1_relu = Activation('relu')
        self.att1_conv3 = Conv2D(1, (1,1), use_bias=True)
        self.att1_sigm = Activation('sigmoid')
        self.att1_multiply = Multiply()

        self.att1_conv4 = Conv2D(256, (1,1))
        self.att1_batch_norm = BatchNormalization()
        # AttentionBlock end

        self.up_conv5 = UpSampling2D(size=(2,2))
        self.up_conv6 = unet_conv2d(256)

        # AttentionBlock2 layers
        self.gating2 = gating_signal(256)
        self.att2_conv1 = Conv2D(256, (2,2), strides=(2,2))
        self.att2_conv2 = Conv2D(256, (1,1), use_bias=True)

        self.att2_add = Add()

        self.att2_relu = Activation('relu')
        self.att2_conv3 = Conv2D(1, (1,1), use_bias=True)
        self.att2_sigm = Activation('sigmoid')
        self.att2_multiply = Multiply()

        self.att2_conv4 = Conv2D(128, (1,1))
        self.att2_batch_norm = BatchNormalization()
        # AttentionBlock end

        self.up_conv7 = UpSampling2D(size=(2,2))
        self.up_conv8 = unet_conv2d(128)

        # AttentionBlock3 layers
        self.gating3 = gating_signal(256)
        self.att3_conv1 = Conv2D(128, (2,2), strides=(2,2))
        self.att3_conv2 = Conv2D(128, (1,1), use_bias=True)

        self.att3_add = Add()

        self.att3_relu = Activation('relu')
        self.att3_conv3 = Conv2D(1, (1,1), use_bias=True)
        self.att3_sigm = Activation('sigmoid')
        self.att3_multiply = Multiply()

        self.att3_conv4 = Conv2D(64, (1,1))
        self.att3_batch_norm = BatchNormalization()
        # AttentionBlock end

        self.up_conv9 = UpSampling2D(size=(2,2))
        self.up_conv10 = unet_conv2d(64)

        # AttentionBlock4 layers
        self.gating4 = gating_signal(256)
        self.att4_conv1 = Conv2D(64, (2,2), strides=(2,2))
        self.att4_conv2 = Conv2D(64, (1,1), use_bias=True)

        self.att4_add = Add()

        self.att4_relu = Activation('relu')
        self.att4_conv3 = Conv2D(1, (1,1), use_bias=True)
        self.att4_sigm = Activation('sigmoid')
        self.att4_multiply = Multiply()

        self.att4_conv4 = Conv2D(32, (1,1))
        self.att4_batch_norm = BatchNormalization()
        # AttentionBlock end

        self.up_conv11 = UpSampling2D(size=(2,2))
        self.up_conv12 = unet_conv2d(32)

        self.up_conv13 = Conv2D(num_classes, (1,1))

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
        gating1 = self.gating1(center)

        att1_conv1 = self.att1_conv1(down_pool4)
        att1_conv2 = self.att1_conv2(gating1)

        att1_resized_gating = tf.image.resize_bilinear(att1_conv2, size=(int(att1_conv1.get_shape()[1]), int(att1_conv1.get_shape()[2])), align_corners=True)

        att1_add = self.att1_add([att1_conv1, att1_resized_gating])
        att1_relu = self.att1_relu(att1_add)
        att1_conv3 = self.att1_conv3(att1_relu)
        att1_sigm = self.att1_sigm(att1_conv3)

        att1_resized_sigm = tf.image.resize_bilinear(att1_sigm, size=(int(down_pool4.get_shape()[1]), int(down_pool4.get_shape()[2])), align_corners=True)

        att1_mul = self.att1_multiply([down_pool4, att1_resized_sigm])

        att1_conv4 = self.att1_conv4(att1_mul)

        att1_batch_norm = self.att1_batch_norm(att1_conv4)

        att1_up_conv5 = self.up_conv5(center)

        ch, cw = get_crop_shape(att1_up_conv5, att1_batch_norm)

        attn1_zero_pad = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(att1_batch_norm)
        up_concat1 = concatenate([att1_up_conv5, attn1_zero_pad], axis=concat_axis) 
        up_conv6 = self.up_conv6(up_concat1)

        # attention2 part
        gating2 = self.gating2(up_conv6)

        att2_conv1 = self.att2_conv1(down_pool3)
        att2_conv2 = self.att2_conv2(gating2)

        att2_resized_gating = tf.image.resize_bilinear(att2_conv2, size=(int(att2_conv1.get_shape()[1]), int(att2_conv1.get_shape()[2])), align_corners=True)

        att2_add = self.att2_add([att2_conv1, att2_resized_gating])
        att2_relu = self.att2_relu(att2_add)
        att2_conv3 = self.att2_conv3(att2_relu)
        att2_sigm = self.att2_sigm(att2_conv3)

        att2_resized_sigm = tf.image.resize_bilinear(att2_sigm, size=(int(down_pool3.get_shape()[1]), int(down_pool3.get_shape()[2])), align_corners=True)

        att2_mul = self.att2_multiply([down_pool3, att2_resized_sigm])

        att2_conv4 = self.att2_conv4(att2_mul)

        att2_batch_norm = self.att2_batch_norm(att2_conv4)

        att2_up_conv5 = self.up_conv7(up_conv6)

        ch, cw = get_crop_shape(att2_up_conv5, att2_batch_norm)

        attn2_zero_pad = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(att2_batch_norm)
        up_concat2 = concatenate([att2_up_conv5, attn2_zero_pad], axis=concat_axis) 
        up_conv7 = self.up_conv8(up_concat2)

        # attention3 part
        gating3 = self.gating3(up_conv7)

        att3_conv1 = self.att3_conv1(down_pool2)
        att3_conv2 = self.att3_conv2(gating3)

        att3_resized_gating = tf.image.resize_bilinear(att3_conv2, size=(int(att3_conv1.get_shape()[1]), int(att3_conv1.get_shape()[2])), align_corners=True)

        att3_add = self.att3_add([att3_conv1, att3_resized_gating])
        att3_relu = self.att3_relu(att3_add)
        att3_conv3 = self.att3_conv3(att3_relu)
        att3_sigm = self.att3_sigm(att3_conv3)

        att3_resized_sigm = tf.image.resize_bilinear(att3_sigm, size=(int(down_pool2.get_shape()[1]), int(down_pool2.get_shape()[2])), align_corners=True)

        att3_mul = self.att2_multiply([down_pool2, att3_resized_sigm])

        att3_conv4 = self.att3_conv4(att3_mul)

        att3_batch_norm = self.att3_batch_norm(att3_conv4)

        att3_up_conv5 = self.up_conv9(up_conv7)

        ch, cw = get_crop_shape(att3_up_conv5, att3_batch_norm)

        attn3_zero_pad = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(att3_batch_norm)
        up_concat3 = concatenate([att3_up_conv5, attn3_zero_pad], axis=concat_axis) 
        up_conv8 = self.up_conv10(up_concat3)

        # attention4 part
        gating4 = self.gating4(up_conv8)

        att4_conv1 = self.att4_conv1(down_pool1)
        att4_conv2 = self.att4_conv2(gating4)

        att4_resized_gating = tf.image.resize_bilinear(att4_conv2, size=(int(att4_conv1.get_shape()[1]), int(att4_conv1.get_shape()[2])), align_corners=True)

        att4_add = self.att4_add([att4_conv1, att4_resized_gating])
        att4_relu = self.att4_relu(att4_add)
        att4_conv3 = self.att4_conv3(att4_relu)
        att4_sigm = self.att4_sigm(att4_conv3)

        att4_resized_sigm = tf.image.resize_bilinear(att4_sigm, size=(int(down_pool1.get_shape()[1]), int(down_pool1.get_shape()[2])), align_corners=True)

        att4_mul = self.att4_multiply([down_pool1, att4_resized_sigm])

        att4_conv4 = self.att4_conv4(att4_mul)

        att4_batch_norm = self.att4_batch_norm(att4_conv4)

        att4_up_conv5 = self.up_conv11(up_conv8)

        ch, cw = get_crop_shape(att4_up_conv5, att4_batch_norm)

        attn4_zero_pad = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(att4_batch_norm)
        up_concat4 = concatenate([att4_up_conv5, attn4_zero_pad], axis=concat_axis) 
        up_conv9 = self.up_conv12(up_concat4)

        ch, cw = get_crop_shape(inputs, up_conv9)
        up_conv10 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(up_conv9)
        output = self.up_conv13(up_conv10)

        return output

    def model_fn(self, mode, inputs):
        is_training = (mode == 'train')

        uNet = AttentionalUnet()

        model_spec = inputs
        model_spec['unet'] = uNet
        
        return model_spec
