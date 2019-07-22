'''
@author: Odgiiv Khuurkhunkhuu
@email: odgiiv_kh[at]gmail
@create date: 2019-01-10
'''

import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, concatenate, ZeroPadding2D, ReLU, BatchNormalization, Dropout
from tensorflow.python.keras.regularizers import l2
from utils import get_crop_shape, hausdorf_distance
import numpy as np


def unet_conv2d(nb_filters, kernel=(3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.0), use_batch_norm=False, drop_rate=0.0):
    conv2d_1 = Conv2D(nb_filters, kernel, padding=padding, activation="relu", kernel_regularizer=kernel_regularizer)
    conv2d_2 = Conv2D(nb_filters, kernel, padding=padding, activation="relu", kernel_regularizer=kernel_regularizer)
    seq1 = [conv2d_1]
    seq2 = [conv2d_2]

    if use_batch_norm:
        batch1 = BatchNormalization()
        batch2 = BatchNormalization()
        seq1 += [batch1]
        seq2 += [batch2]

    if drop_rate > 0:
        drop1 = Dropout(drop_rate)
        drop2 = Dropout(drop_rate)  
        seq1 += [drop1]
        seq2 += [drop2]

    return Sequential(seq1 + seq2)


class Unet(Model):

    def __init__(self, num_classes=2, l2=0.0):
        super(Unet, self).__init__()

        print("Creating Unet model.")

        self.conv1 = unet_conv2d(64, kernel_regularizer=l2(l2))
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2 = unet_conv2d(128, kernel_regularizer=l2(l2))
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3 = unet_conv2d(256, kernel_regularizer=l2(l2))
        self.pool3 = MaxPooling2D(pool_size=(2, 2))

        self.conv4 = unet_conv2d(512, kernel_regularizer=l2(l2))
        self.pool4 = MaxPooling2D(pool_size=(2, 2))

        self.center = unet_conv2d(1024, kernel_regularizer=l2(l2))

        self.up_conv5 = UpSampling2D(size=(2, 2))
        self.conv6 = unet_conv2d(512, kernel_regularizer=l2(l2))

        self.up_conv6 = UpSampling2D(size=(2, 2))
        self.conv7 = unet_conv2d(256, kernel_regularizer=l2(l2))

        self.up_conv7 = UpSampling2D(size=(2, 2))
        self.conv8 = unet_conv2d(128, kernel_regularizer=l2(l2))

        self.up_conv8 = UpSampling2D(size=(2, 2))
        self.conv9 = unet_conv2d(64, kernel_regularizer=l2(l2))

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

    
    def evaluate(valid_gen, segmentor_net, steps_per_valid_epoch, model_name):
        IoUs = []
        valid_loss_avg = tf.contrib.eager.metrics.Mean()
        hds = tf.contrib.eager.metrics.Mean()
        logging.info("Starting validation...")
        current_val_step = 0
        for imgs, labels in valid_gen:

            labels[labels >= 0.5] = 1
            labels[labels < 0.5] = 0
            labels = labels.astype("uint8")

            imgs = tf.image.convert_image_dtype(imgs, tf.float32)

            pred = segmentor_net(imgs)
            pred_np = pred.numpy()

            pred_np = np.argmax(pred_np, axis=-1)
            pred_np = np.expand_dims(pred_np, -1)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=pred)
            valid_loss_avg(loss)

            pred_locations = np.argwhere(pred_np == 1)
            label_locations = np.argwhere(labels == 1)

            hd = hausdorf_distance(pred_locations, label_locations)
            hds(hd)

            for x in range(imgs.shape[0]):
                IoU = np.sum(pred_np[x][gt[x] == 1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x] == 1]))
                ##print(IoU)
                #print(np.sum(pred_np[x] ==1))
                IoUs.append(IoU)
            
            current_val_step += 1
            if current_val_step == steps_per_valid_epoch:
                break

        IoUs = np.array(IoUs, dtype=np.float64)
        mIoU = np.mean(IoUs, axis=0)

        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar("val_avg_hd", hds.result())
            tf.contrib.summary.scalar("val_avg_loss", valid_loss_avg.result())
            tf.contrib.summary.scalar("val_avg_IoU", mIoU)    
        
        print("loss valid avg {0:.4f}, mIoU on validation set: {1:.4f}".format(valid_loss_avg.result(), mIoU))

        return mIoU, valid_loss_avg.result()    
