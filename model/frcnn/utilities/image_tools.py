# -*- coding: utf-8 -*-
import cv2
import numpy as np

class ImageTools(object):
	"""Methods for get required format to the images."""

	def __init__(self):
		super(ImageTools, self).__init__()

	@staticmethod
	def get_new_img_size(width, height, img_min_side=600):
		"""Resize the image, based on a minimum side."""

		if width <= height:
			f = float(img_min_side) / width
			new_height = int(f * height)
			new_width = img_min_side
		else:
			f = float(img_min_side) / height
			new_width = int(f * width)
			new_height = img_min_side

		return new_width, new_height

	@staticmethod
	def format_img(img, config):
		img_min_side = float(config.im_size)
		(height,width,_) = img.shape

		new_width, new_height = ImageTools.get_new_img_size(
			width,
			height,
			img_min_side
		)
		new_width = int(new_width)
		new_height = int(new_height)
		# factor to resize over x
		fx = width / float(new_width)
		# factor to resize over y
		fy = height / float(new_height)
		# Resize the image
		img = cv2.resize(
			img,
			(new_width, new_height),
			interpolation=cv2.INTER_CUBIC,
		)
		img = img[:, :, (2, 1, 0)]
		img = img.astype(np.float32)
		# Substract channels means
		img[:, :, 0] -= config.img_channel_mean[0]
		img[:, :, 1] -= config.img_channel_mean[1]
		img[:, :, 2] -= config.img_channel_mean[2]
		img /= config.img_scaling_factor
		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)
		# return imagen and factors
		return img, fx, fy
