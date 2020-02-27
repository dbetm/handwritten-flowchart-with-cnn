#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

class LossesCalculator(object):
	"""Class with functions for calculate the loss of the RPN and classifier."""
	# Statics attrs
	lambda_rpn_regr = 1.0
	lambda_rpn_class = 1.0
	lambda_cls_regr = 1.0
	lambda_cls_class = 1.0
	epsilon = None
	num_classes = None
	num_anchors = None

	def __init__(self, num_classes, num_anchors, epsilon=1e-4):
		super(LossesCalculator, self).__init__()
		LossesCalculator.epsilon = epsilon
		LossesCalculator.num_classes = num_classes
		LossesCalculator.num_anchors = num_anchors

	@staticmethod
	def rpn_loss_regr():
		def rpn_loss_regr_fixed_num(y_true, y_pred):
			x = y_true[:, :, :, 4 * LossesCalculator.num_anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
			return LossesCalculator.lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * LossesCalculator.num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(LossesCalculator.epsilon + y_true[:, :, :, :4 * LossesCalculator.num_anchors])

		return rpn_loss_regr_fixed_num

	@staticmethod
	def rpn_loss_cls():
		def rpn_loss_cls_fixed_num(y_true, y_pred):
			return LossesCalculator.lambda_rpn_class * K.sum(y_true[:, :, :, :LossesCalculator.num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, LossesCalculator.num_anchors:])) / K.sum(LossesCalculator.epsilon + y_true[:, :, :, :LossesCalculator.num_anchors])

		return rpn_loss_cls_fixed_num

	@staticmethod
	def class_loss_regr():
		def class_loss_regr_fixed_num(y_true, y_pred):
			num_classes = LossesCalculator.num_classes - 1
			x = y_true[:, :, 4*num_classes:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
			return LossesCalculator.lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(LossesCalculator.epsilon + y_true[:, :, :4*num_classes])
		return class_loss_regr_fixed_num

	@staticmethod
	def class_loss_cls(y_true, y_pred):
		return LossesCalculator.lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
