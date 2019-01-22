'''
@author: Odgiiv Khuurkhunkhuu
@email: odgiiv_kh[gmail]
@create date: 2019-01-10
'''

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, concatenate, ZeroPadding2D
from utils import focal_loss_softmax, mean_IU, mean_iou


class Unet:
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

        return (ch1, ch2), (cw1, cw2)

    def UnetConv2D(self, input, filters, kernel=(3, 3), activation="relu", padding="same"):
        conv = Conv2D(filters, kernel, activation=activation, padding=padding)(input)
        conv = Conv2D(filters, kernel, activation=activation, padding=padding)(conv)
        return conv


    def build_model(self, inputs, num_classes):
        """U-net model
        Args:
            inputs: (dict) contains the inputs of the graph (features, labels ...)
            this can be `tf.placeholder` or outputs of `tf.data`
        """
        concat_axis = 3

        inputs = Input(tensor=inputs["images"])
        conv1 = self.UnetConv2D(inputs, 32)
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        conv2 = self.UnetConv2D(pool1, 64)
        pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

        conv3 = self.UnetConv2D(pool2, 128)
        pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

        conv4 = self.UnetConv2D(pool3, 256)
        pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

        center = self.UnetConv2D(pool4, 512)

        up_conv5 = UpSampling2D(size=(2,2))(center)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = self.UnetConv2D(up6, 256)

        up_conv6 = UpSampling2D(size=(2,2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = self.UnetConv2D(up7, 128)

        up_conv7 = UpSampling2D(size=(2,2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = self.UnetConv2D(up8, 64)

        up_conv8 = UpSampling2D(size=(2,2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = self.UnetConv2D(up9, 32)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D(num_classes, (1,1))(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model


    def model_fn(self, mode, inputs, params, reuse=False):
        is_training = (mode == 'train')
        labels = inputs["labels"]
        labels = tf.cast(labels, tf.int32)

        if is_training:
            # with tf.variable_scope('segmetation_models', reuse=reuse):
            model = self.build_model(inputs, params.num_classes)
            # prediction = model.outputs[0]
        else: 
            model = inputs["model"]
            model.inputs[0] = Input(tensor=inputs["images"])

        prediction = model.outputs[0]

        _loss, prediction = focal_loss_softmax(labels=labels, logits=prediction)
        loss = tf.reduce_mean(_loss)

        prediction = tf.argmax(prediction, axis=-1, output_type=tf.int32)
        prediction = tf.expand_dims(prediction, -1)

        print(prediction.shape)
        print(labels.shape)

        # miu, iu = mean_IU(prediction, tf.squeeze(labels)),
        
        mean_iou, conf_mat = tf.metrics.mean_iou(labels, prediction, num_classes=2)

        if is_training:    
            
            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
            global_step = tf.train.get_or_create_global_step()
            train_op = optimizer.minimize(loss, global_step=global_step)

        # with tf.variable_scope("metrics"):
        metrics = {
            'mean_iou': tf.metrics.mean_iou(labels, prediction, num_classes=2),
            'loss': tf.metrics.mean(_loss)
        }
        # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('mean_iou', mean_iou)
        tf.summary.image('train_image', inputs['images'])
        tf.summary.image('label', tf.cast(255 * labels, tf.uint8))
        tf.summary.image('predicted_label', tf.cast(255 * prediction, tf.uint8))

        model_spec = inputs
        model_spec['variable_init_op'] = tf.global_variables_initializer()
        model_spec['local_variable_init_op'] = tf.local_variables_initializer()
        model_spec['model'] = model
        model_spec['prediction'] = prediction
        model_spec['loss'] = loss
        model_spec['mean_iou'] = mean_iou
        model_spec['conf_mat'] = conf_mat
        model_spec['metrics_init_op'] = tf.variables_initializer(metric_variables)
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()     

        if is_training:
            model_spec['train_op'] = train_op

        return model_spec
