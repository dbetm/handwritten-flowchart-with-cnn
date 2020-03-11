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
	def get_format_img_size(image, config):
		"""Resize image, apply substract channels means and
		return ratio.
		"""

		img_min_side = float(config.im_size)

		(height, width, _) = image.shape

		# Resize the image according to the smaller side
		if width <= height:
			ratio = img_min_side / width
			new_height = int(ratio * height)
			new_width = int(img_min_side)
		else:
			ratio = img_min_side / height
			new_width = int(ratio * width)
			new_height = int(img_min_side)
		image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

		image = ImageTools.format_img_channels(image, config)

		return image, ratio

	@staticmethod
	def format_img(img, config):
		"""Resize image, apply substract channels means and
		return factors for x-side and y-side.
		"""

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
		img = ImageTools.format_img_channels(img, config)
		# return imagen and factors
		return img, fx, fy

	@staticmethod
	def format_img_channels(img, config):
		"""Format the image channels based on configuration."""

		img = img[:, :, (2, 1, 0)]
		img = img.astype(np.float32)
		# Substract channels means
		img[:, :, 0] -= config.img_channel_mean[0]
		img[:, :, 1] -= config.img_channel_mean[1]
		img[:, :, 2] -= config.img_channel_mean[2]
		img /= config.img_scaling_factor
		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)

		return img

	@staticmethod
	def get_real_coordinates(ratio, x1, y1, x2, y2):
		"""Method to transform the coordinates of the bounding box to
		its original size.
		"""

		real_x1 = int(round(x1 // ratio))
		real_y1 = int(round(y1 // ratio))
		real_x2 = int(round(x2 // ratio))
		real_y2 = int(round(y2 // ratio))

		return (real_x1, real_y1, real_x2, real_y2)
