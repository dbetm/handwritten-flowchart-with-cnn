# -*- coding: utf-8 -*-

"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image
Recognition](https://arxiv.org/abs/1409.1556)
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

from . ROI_pooling_conv import ROIPoolingConv


class CNN(object):
    """Build CNN architecture, VGG16 Keras model."""

    def __init__(self, num_anchors, rois, num_classes):
        super(CNN, self).__init__()

        self.weights_input_path= "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        self.num_anchors = num_anchors
        self.input_rois, self.num_rois = rois
        self.num_classes = num_classes

    @staticmethod
    def get_img_output_length(width, height):
        """Resize image dim, by -16x factor."""
        def get_output_length(input_length):
            return input_length // 16

        return get_output_length(width), get_output_length(height)

    def build_nn_base(self, input_tensor=None):
        """Build the VGG16 convolutional layers."""

        # Determine proper input shape (Backend TensorFlow)
        input_shape = (None, None, 3)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        bn_axis = 3

        # Build the convolutionals layers
        x = self.__get_conv_blocks(img_input)
        return x

    def __get_conv_blocks(self, img_input):
        """Add some convolutional blocks layers for CNN."""
        kernel_conv = (3, 3)

        # Block 1
        x = Conv2D(
            64,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block1_conv1'
        )(img_input)
        x = Conv2D(
            64,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block1_conv2'
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(
            128,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block2_conv1'
        )(x)
        x = Conv2D(128,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block2_conv2'
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(
            256,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block3_conv1'
        )(x)
        x = Conv2D(
            256,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block3_conv2'
        )(x)
        x = Conv2D(
            256,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block3_conv3'
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block4_conv1'
        )(x)
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block4_conv2'
        )(x)
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block4_conv3'
        )(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block5_conv1'
        )(x)
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block5_conv2'
        )(x)
        x = Conv2D(
            512,
            kernel_conv,
            activation='relu',
            padding='same',
            name='block5_conv3'
        )(x)

        return x

    def create_rpn(self, base_layers):
        """Create the Region Proposal Network."""

        x = Conv2D(512,
            (3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='normal',
            name='rpn_conv1'
        )(base_layers)

        x_class = Conv2D(self.num_anchors,
            (1, 1),
            activation='sigmoid',
            kernel_initializer='uniform',
            name='rpn_out_class'
        )(x)
        x_regr = Conv2D(self.num_anchors * 4,
            (1, 1),
            activation='linear',
            kernel_initializer='zero',
            name='rpn_out_regress'
        )(x)

        return [x_class, x_regr, base_layers]

    def build_classifier(self, base_layers, num_classes=21):
        """Build the classifier (top layers for architecture)."""

        pooling_regions = 7
        input_shape = (self.num_rois, 7, 7, 512)

        out_roi_pool = ROIPoolingConv(
                            pooling_regions,
                            self.num_rois
                        )([base_layers, self.input_rois])
        # Build Full-Connected
        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        out = TimeDistributed(Dropout(0.5))(out)
        # Define the class of the object using softmax function
        out_class = TimeDistributed(
                        Dense(
                            num_classes,
                            activation='softmax',
                            kernel_initializer='zero'
                        ),
                        name='dense_class_{}'.format(num_classes)
                    )(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(
                        Dense(
                            4 * (num_classes-1),
                            activation='linear',
                            kernel_initializer='zero'
                        ),
                        name='dense_regress_{}'.format(num_classes)
                    )(out)

        return [out_class, out_regr]
