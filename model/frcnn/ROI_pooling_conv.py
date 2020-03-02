# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.engine.topology import Layer
import keras.backend as K


class ROIPoolingConv(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for
    Visual Recognition.
    """
    """K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        4D tensor with shape:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """
    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(ROIPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        """Define the number of channels"""
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        """Define the shape of the output."""
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        """Generate the final output."""
        
        assert(len(x) == 2)
        # Recovery the img, and number of regions of interest
        img = x[0]
        rois = x[1]
        # Get the input shape such the dims of the image
        input_shape = K.shape(img)

        outputs = []
        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)
            num_pool_regions = self.pool_size

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            rs = tf.image.resize(
                img[:, y:y+h, x:x+w, :],
                (self.pool_size, self.pool_size)
            )
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(
            final_output,
            (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)
        )
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
