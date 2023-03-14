# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout, Conv1D
from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D


class Conv1dBloc(Layer):
    """Elementary 1D-convolution block for TempCNN encoder"""

    def __init__(self, filters_nb, kernel_size, drop_val, **kwargs):
        super(Conv1dBloc, self).__init__(**kwargs)
        self.conv1D = Conv1D(filters_nb, kernel_size,
                             padding="same", kernel_initializer='he_normal')
        self.batch_norm = BatchNormalization()
        self.act = Activation('relu')
        self.output_ = Dropout(drop_val)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        conv1d = self.conv1D(inputs)
        batch_norm = self.batch_norm(conv1d, training=training)
        act = self.act(batch_norm)
        return self.output_(act, training=training)


class TempCnnEncoder(Layer):
    """"Encoder of SITS on temporal dimension"""

    def __init__(self, drop_val=0.5, **kwargs):
        super(TempCnnEncoder, self).__init__(**kwargs)
        self.conv_bloc1 = Conv1dBloc(64, 5, drop_val)
        self.conv_bloc2 = Conv1dBloc(64, 5, drop_val)
        self.conv_bloc3 = Conv1dBloc(64, 5, drop_val)
        self.flatten = layers.Flatten()

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        conv1 = self.conv_bloc1(inputs, training=training)
        conv2 = self.conv_bloc2(conv1, training=training)
        conv3 = self.conv_bloc3(conv2, training=training)
        flatten = self.flatten(conv3)
        return flatten


class PatchEncoder(Layer):
    """"Encoder of VHR on temporal dimension"""

    def __init__(self, drop_val=0.5, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.conv2D_1 = Conv2D(128, 5, strides=(1, 1), padding="same",
                               kernel_initializer='he_normal')
        self.batch_norm_1 = BatchNormalization()
        self.act_1 = Activation('relu')
        self.drop_out_1 = Dropout(drop_val)
        self.conv2D_2 = Conv2D(128, 5, strides=(1, 1), padding="same",
                               kernel_initializer='he_normal')
        self.batch_norm_2 = BatchNormalization()
        self.act_2 = Activation('relu')
        self.drop_out_2 = Dropout(drop_val)
        self.conv2D_3 = Conv2D(128, 5, strides=(1, 1), padding="same",
                               kernel_initializer='he_normal')
        self.batch_norm_3 = BatchNormalization()
        self.act_3 = Activation('relu')
        self.drop_out_3 = Dropout(drop_val)
        self.global_avg_pooling = GlobalAveragePooling2D()

    def call(self, inputs, training=False):
        conv2d_1 = self.conv2D_1(inputs)
        bn_1 = self.batch_norm_1(conv2d_1, training=training)
        act_1 = self.act_1(bn_1)
        do_1 = self.drop_out_1(act_1, training=training)
        conv2d_2 = self.conv2d_2(do_1)
        bn_2 = self.batch_norm_2(conv2d_2, training=training)
        act_2 = self.act_2(bn_2)
        do_2 = self.drop_out_2(act_2, training=training)
        conv2d_3 = self.conv2d_3(do_2)
        bn_3 = self.batch_norm_3(conv2d_3, training=training)
        act_3 = self.act_3(bn_3)
        do_3 = self.drop_out_3(act_3, training=training)
        return self.global_avg_pooling(do_3)


class Classifier(Layer):
    """"Generic classifier for concatenated features"""

    def __init__(self, nb_class, nb_units, drop_val=0.5, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.dense = Dense(nb_units)
        self.batch_norm = BatchNormalization()
        self.act = Activation('relu')
        self.dropout = Dropout(drop_val)
        self.output_ = Dense(nb_class, activation="softmax")

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        dense = self.dense(inputs)
        batch_norm = self.batch_norm(dense, training=training)
        act = self.act(batch_norm)
        dropout = self.dropout(act, training=training)
        return self.output_(dropout)


class AuxiliaryClassifier(Layer):
    """"Auxiliary classifier for each branch encoder"""

    def __init__(self, nb_class, **kwargs):
        super(AuxiliaryClassifier, self).__init__(**kwargs)
        self.output_ = Dense(nb_class, activation="softmax")

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        return self.output_(inputs)


@tf.custom_gradient
def gradient_reverse(x, lamb_da=1.0):
    y = tf.identity(x)

    def custom_grad(dy):
        return lamb_da * -dy, None

    return y, custom_grad


class GradReverse(Layer):
    """"Gradient reversal layer (GRL)"""

    def __init__(self):
        super().__init__()

    def call(self, x, lamb_da=1.0):
        return gradient_reverse(x, lamb_da)


class M3SPADAModel(keras.Model):
    """"M3SPADAModel model is composed of
    a feature extractor for times series: ts_encoder
    a feature extractor for VHR images: vhr_encoder
    a label predictor/classifier for concatenated features: fusion_classifier
    a label predictor/classifier for ts features: ts_classifier
    a label predictor/classifier for vhr features: vhr_classifier
    a domain predictor/classifier: domain_classifier
    a GRL to connect concatenated features and domain predictor/classifier
    """

    def __init__(self, nb_class, nb_units, drop_val=0.5, **kwargs):
        super(M3SPADAModel, self).__init__(**kwargs)
        self.ts_encoder = TempCnnEncoder()
        self.ts_classifier = AuxiliaryClassifier(nb_class)
        self.vhr_encoder = PatchEncoder(drop_val)
        self.vhr_classifier = AuxiliaryClassifier(nb_class)
        self.concat = Concatenate()
        self.fusion_classifier = Classifier(nb_class, nb_units)
        self.grl = GradReverse()
        self.domain_classifier = Classifier(2, 256)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, lamb_da=1.0):
        ts_input, vhr_input = inputs
        ts_enc_out = self.ts_encoder(ts_input, training=training)
        ts_classifier_out = self.ts_classifier(ts_enc_out)
        vhr_enc_out = self.vhr_encoder(vhr_input, training=training)
        vhr_classifier_out = self.vhr_classifier(vhr_enc_out)
        concat_out = self.concat([ts_enc_out, vhr_enc_out])
        fusion_classifier_out = self.fusion_classifier(concat_out,
                                                       training=training)
        grl = self.grl(concat_out, lamb_da)
        domain_classifier_out = self.domain_classifier(grl, training=training)
        return concat_out, ts_classifier_out, vhr_classifier_out, \
            fusion_classifier_out, domain_classifier_out
