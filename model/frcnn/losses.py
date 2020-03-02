# -*- coding: utf-8 -*-

from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf


class LossesCalculator(object):
	"""Class with functions for calculate the loss of the RPN and classifier."""

	# Static attrs
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
		"""Calculate loss for regression in RPN."""

		def rpn_loss_regr_fixed_num(y_true, y_pred):
			anchors = LossesCalculator.num_anchors
			x = y_true[:, :, :, 4 * anchors:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

			lbda_rpn_regr = LossesCalculator.lambda_rpn_regr
			eps = LossesCalculator.epsilon
			y_sel = y_true[:, :, :, :4 * anchors]
			sum1 = K.sum(y_sel * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)))
			sum2 = K.sum(eps + y_true[:, :, :, :4 * anchors])
			return lbda_rpn_regr * sum1 / sum2

		return rpn_loss_regr_fixed_num

	@staticmethod
	def rpn_loss_cls():
		"""Calculate loss for classification task in RPN."""

		def rpn_loss_cls_fixed_num(y_true, y_pred):

			lbda_rpn_cls = LossesCalculator.lambda_rpn_class
			anchors = LossesCalculator.num_anchors
			eps = LossesCalculator.epsilon
			binary_crossentropy = K.binary_crossentropy(
				y_pred[:, :, :, :],
				y_true[:, :, :, anchors:]
			)
			sum1 = K.sum(y_true[:, :, :, :anchors] * binary_crossentropy)
			sum2 = K.sum(eps + y_true[:, :, :, :anchors])
			return lbda_rpn_cls * sum1 / sum2

		return rpn_loss_cls_fixed_num

	@staticmethod
	def class_loss_regr():
		"""Calculate loss for regression in classifier."""

		def class_loss_regr_fixed_num(y_true, y_pred):
			num_classes = LossesCalculator.num_classes - 1
			x = y_true[:, :, 4*num_classes:] - y_pred
			x_abs = K.abs(x)
			x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

			lbda_cls_regr = LossesCalculator.lambda_cls_regr
			eps = LossesCalculator.epsilon
			y_sel = y_true[:, :, :4*num_classes]
			sum1 = K.sum(y_sel * (x_bool * (0.5*x*x) + (1-x_bool) * (x_abs-0.5)))
			sum2 = K.sum(eps + y_true[:, :, :4*num_classes])

			return lbda_cls_regr * sum1 / sum2

		return class_loss_regr_fixed_num

	@staticmethod
	def class_loss_cls(y_true, y_pred):
		"""Calculate loss for classification task in classifier."""

		lbda_cls_class = LossesCalculator.lambda_cls_class
		mean = K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
		return lbda_cls_class * mean
