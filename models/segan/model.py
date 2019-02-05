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

        self.num_filters = [64, 128, 256, 512]

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

    # def cropping_2d(self, target, refer):
    #     return

    def conv_lrelu(self, x, nb_filters, kernel=(4, 4), stride=(2, 2)):
        x = Conv2D(nb_filters, kernel, stride, padding="same")(x)
        return LeakyReLU()(x)

    def conv_bn_lrelu(self, x, nb_filters, kernel=(4, 4), stride=(2, 2)):
        x = Conv2D(nb_filters,  kernel, stride, padding="same")(x)
        x = BatchNormalization()(x)
        return LeakyReLU()(x)

    def up_conv(self, x, nb_filters, kernel=(3, 3), stride=(1, 1)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(nb_filters, kernel, stride, padding="same")(x)
        return x

    def up_conv_bn_relu(self, x, nb_filters, kernel=(3, 3), stride=(1, 1)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(nb_filters, kernel, stride, padding="same")(x)
        x = BatchNormalization()(x)
        return Activation('relu')(x)

    def segmentor(self, input):
        seg_inputs = input
        seg_conv1 = self.conv_lrelu(seg_inputs, self.num_filters[0])
        seg_conv2 = self.conv_bn_lrelu(seg_conv1, self.num_filters[1])
        seg_conv3 = self.conv_bn_lrelu(seg_conv2, self.num_filters[2])
        seg_center = self.conv_bn_lrelu(seg_conv3, self.num_filters[3])
        seg_up_con4 = self.up_conv_bn_relu(seg_center, self.num_filters[2])
        seg_up_con5 = self.up_conv_bn_relu(
            seg_up_con4, self.num_filters[1])
        seg_up_con6 = self.up_conv_bn_relu(
            seg_up_con5, self.num_filters[0])
        pred = self.up_conv(seg_up_con6, 1)

        ch, cw = self.get_crop_shape(pred, input)
        pred = Cropping2D(cropping=((ch, cw)))(pred)

        # pred = self.cropping_2d(pred, input)
        # pred = Lambda(lambda target, refer: self.get_crop_shape(
        #     target, refer), arguments={'refer': input})(pred)

        pred = Activation("sigmoid")(pred)

        return Model([seg_inputs], [pred], name="segmentor_net")

    def critic(self, input):
        cri_inputs = input
        cri_conv1 = self.conv_lrelu(cri_inputs, self.num_filters[0])
        cri_conv2 = self.conv_bn_lrelu(cri_conv1, self.num_filters[1])
        cri_conv3 = self.conv_bn_lrelu(cri_conv2, self.num_filters[2])
        features = Concatenate()([Flatten()(cri_inputs), Flatten()(
            cri_conv1), Flatten()(cri_conv2), Flatten()(cri_conv3)])
        return features

    def build_model(self, inputs, params):
        # images = inputs["images"]
        # targets = inputs["labels"]
        # segmentor_net = self.segmentor((params.img_h, params.img_w, params.img_c))  # images
        # shared_critic_net = self.critic(
        #     (params.img_h, params.img_w, params.img_c))

        # masked_input_seg = multiply([images, segmentor_net.outputs[0]])
        # critic_net_output_for_seg = shared_critic_net(masked_input_seg)

        # masket_input_gt = multiply([images, targets])
        # critic_net_output_for_target = shared_critic_net(masket_input_gt)

        return {
            "segmentor_net": SegmentorNet(),
            "critic_net": CriticNet(),
            # "critic_net_output_for_seg": critic_net_output_for_seg,
            # "critic_net_output_for_target": critic_net_output_for_target
        }

    def model_fn(self, mode, inputs, params):

        is_training = (mode == "train")
        # labels = tf.cast(inputs["labels"], tf.int32)

        networks = self.build_model(inputs, params)

        # critic_net_output_for_segmentor = networks["critic_net_output_for_seg"]
        # critic_net_output_for_target = networks["critic_net_output_for_target"]

        # seg_prediction = segmentor_net.outputs[0]
        # seg_prediction_binary = tf.round(seg_prediction)

        # seg_loss = tf.reduce_mean(
        #     tf.abs(critic_net_output_for_segmentor - critic_net_output_for_target))
        # cri_loss = - \
        #     tf.reduce_mean(
        #         tf.abs(critic_net_output_for_segmentor - critic_net_output_for_target))

        # if is_training:
        #     optimizerG = tf.train.AdamOptimizer(
        #         learning_rate=params.learning_rate)
        #     optimizerD = tf.train.AdamOptimizer(
        #         learning_rate=params.learning_rate)

        #     global_step = tf.train.get_or_create_global_step()
        #     seg_train_op = optimizerG.minimize(
        #         seg_loss, global_step=global_step)
        #     cri_train_op = optimizerD.minimize(
        #         cri_loss, global_step=global_step)

        # tf.summary.scalar("seg_loss", seg_loss)
        # tf.summary.scalar("cri_loss", cri_loss)
        # tf.summary.image("train_image", inputs["images"])
        # # tf.summary.image("label", tf.cast(255 * labels, tf.uint8))
        # tf.summary.image("predicted_label", tf.cast(
        #     255 * seg_prediction_binary, tf.uint8))

        model_spec = inputs
        # model_spec["variable_init_op"] = tf.global_variables_initializer()
        # model_spec["local_variable_init_op"] = tf.local_variables_initializer()
        model_spec["segmentor_net"] = networks["segmentor_net"]
        model_spec["critic_net"] = networks["critic_net"]

        # model_spec["seg_loss"] = seg_loss
        # model_spec["cri_loss"] = cri_loss

        # model_spec["seg_prediction"] = seg_prediction
        # model_spec['summary_op'] = tf.summary.merge_all()

        # if is_training:
        #     model_spec["seg_train_op"] = seg_train_op,
        #     model_spec["cri_train_op"] = cri_train_op,

        return model_spec
